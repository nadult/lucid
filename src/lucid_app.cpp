// Copyright (C) Krzysztof Jakubowski <nadult@fastmail.fm>
// This file is part of LucidRaster. See license.txt for details.

#include "lucid_app.h"

#include "lucid_renderer.h"
#include "meshlet.h"
#include "path_tracer.h"
#include "pbr_renderer.h"
#include "scene_setup.h"
#include "simple_renderer.h"

#include "scene_convert.h"

#include <fwk/any_config.h>
#include <fwk/gfx/camera_variant.h>
#include <fwk/gfx/canvas_2d.h>
#include <fwk/gfx/image.h>
#include <fwk/gfx/shader_compiler.h>
#include <fwk/gui/imgui.h>
#include <fwk/gui/widgets.h>
#include <fwk/io/file_system.h>
#include <fwk/math/axis_angle.h>
#include <fwk/perf/analyzer.h>
#include <fwk/perf/exec_tree.h>
#include <fwk/perf/manager.h>
#include <fwk/sys/input.h>

#include <fwk/vulkan/vulkan_command_queue.h>
#include <fwk/vulkan/vulkan_device.h>
#include <fwk/vulkan/vulkan_image.h>
#include <fwk/vulkan/vulkan_instance.h>
#include <fwk/vulkan/vulkan_memory_manager.h>
#include <fwk/vulkan/vulkan_pipeline.h>
#include <fwk/vulkan/vulkan_swap_chain.h>
#include <fwk/vulkan/vulkan_window.h>

// TODO: device lost crash when resizing window with path tracer

FilePath mainPath() {
	return platform == Platform::msvc ? FilePath::current().get() : executablePath().parent();
}

string dataPath(string file_name) { return mainPath() / "data" / file_name; }

// TODO: move it somewhere else...
void VertexArray::getDefs(VPipelineSetup &setup, bool with_tangents) {
	setup.vertex_attribs = {{vertexAttrib<float3>(0, 0), vertexAttrib<IColor>(1, 1),
							 vertexAttrib<float2>(2, 2), vertexAttrib<u32>(3, 3)}};
	setup.vertex_bindings = {{vertexBinding<float3>(0), vertexBinding<IColor>(1),
							  vertexBinding<float2>(2), vertexBinding<u32>(3)}};

	if(with_tangents) {
		setup.vertex_attribs.emplace_back(vertexAttrib<u32>(4, 4));
		setup.vertex_bindings.emplace_back(vertexBinding<u32>(4));
	}
}

PVRenderPass guiRenderPass(VDeviceRef device) {
	auto sc_format = device->swapChain()->format();
	auto color_sync = VColorSync(VLoadOp::load, VStoreOp::store, VImageLayout::general,
								 VImageLayout::present_src);
	return device->getRenderPass({{sc_format, 1, color_sync}});
}

LucidApp::LucidApp(VWindowRef window, VDeviceRef device)
	: m_window(window), m_device(device), m_gui_render_pass(guiRenderPass(device)),
	  m_gui(device, window, m_gui_render_pass, {GuiStyleMode::mini}),
	  m_cam_control(Plane3F(float3(0, 1, 0), 0.0f)), m_lighting(SceneLighting::makeDefault()) {

	//m_device->memory().setLogging(VMemoryDomain::device,
	//							  VMemoryBlockType::slab | VMemoryBlockType::unmanaged);
	ShaderCompilerSetup sc_setup;
	auto shader_config = getShaderConfig(*device);
	sc_setup.vulkan_version = device->version();
	sc_setup.source_dirs.emplace_back(dataPath("shaders"));
	sc_setup.spirv_cache_dir = dataPath(format("spirv_%", shader_config.build_name));
#ifndef NDEBUG
	sc_setup.generate_assembly = true;
	sc_setup.debug_info = true;
#endif

	m_shader_compiler.emplace(sc_setup);
	SimpleRenderer::addShaderDefs(*device, *m_shader_compiler, shader_config);
	LucidRenderer::addShaderDefs(*device, *m_shader_compiler, shader_config);
	PathTracer::addShaderDefs(*device, *m_shader_compiler, shader_config);
	PbrRenderer::addShaderDefs(*device, *m_shader_compiler, shader_config);

	if(perf::Manager::instance())
		m_perf_analyzer.emplace();

	OrbitingCamera cam({}, 10.0f, 0.5f, 0.8f);
	m_cam_control.setTarget(cam);
	m_cam_control.finishAnim();
	m_cam_control.o_config.params.depth = {1.0f / 16.0f, 1024.0f};
	m_cam_control.o_config.rotation_filter = [](const InputEvent &ev) {
		return ev.mouseButtonPressed(InputButton::right);
	};

	m_setups.emplace_back(new BoxesSetup());
	m_setups.emplace_back(new PlanesSetup());
	for(auto scene_name : LoadedSetup::findAll())
		m_setups.emplace_back(new LoadedSetup(scene_name));
	selectSetup(0);
}

LucidApp::~LucidApp() = default;

static string configPath() { return dataPath("../lucid_config.xml"); }

void LucidApp::setConfig(const AnyConfig &config) {
	m_rendering_mode = config.get("rendering_mode", m_rendering_mode);
	m_lucid_opts = config.get("trans_opts", m_lucid_opts);
	m_wireframe_mode = config.get("wireframe", m_wireframe_mode);
	m_show_stats = config.get("show_stats", m_show_stats);
	m_select_stats_tab = config.get("selected_stats_tab", -1);
	if(auto *scene_name = config.get<string>("scene"))
		selectSetup(*scene_name);
	if(config.get("show_perf_analyzer", true)) {
		if(!m_perf_analyzer)
			m_perf_analyzer.emplace();
		if(auto *sub = config.subConfig("perf_analyzer"))
			m_perf_analyzer->setConfig(*sub);
	} else {
		m_perf_analyzer.reset();
	}

	if(auto *sub = config.subConfig("gui"))
		m_gui.setConfig(*sub);
	if(auto *lighting = config.subConfig("lighting"))
		m_lighting.setConfig(*lighting);
	m_cam_control.load(config);
}

Maybe<AnyConfig> LucidApp::loadConfig() {
	Maybe<AnyConfig> config;
	auto path = configPath();
	if(access(path)) {
		auto doc = std::move(XmlDocument::load(path).get());
		config = AnyConfig::load(doc.child("config"), true).get();
		config->printErrors();
	}
	return config;
}

void LucidApp::saveConfig() const {
	AnyConfig out;
	if(access(configPath())) {
		auto doc = std::move(XmlDocument::load(configPath()).get());
		out = AnyConfig::load(doc.child("config"), true).get();
	}

	out.set("rendering_mode", m_rendering_mode);
	out.set("trans_opts", m_lucid_opts);
	out.set("wireframe", m_wireframe_mode);
	bool is_maximized = m_window->isMaximized();
	out.set("window_rect", is_maximized ? m_window->restoredRect() : m_window->rect());
	out.set("window_maximized", is_maximized);
	out.set("show_stats", m_show_stats);
	out.set("selected_stats_tab", m_selected_stats_tab);
	out.set("show_perf_analyzer", !!m_perf_analyzer);
	if(m_setup_idx != -1)
		out.set("scene", m_setups[m_setup_idx]->name);
	if(m_perf_analyzer)
		out.set("perf_analyzer", m_perf_analyzer->config());
	out.set("gui", m_gui.config());
	out.set("lighting", m_lighting.config());
	m_cam_control.save(out);

	XmlDocument doc;
	out.save(doc.addChild("config"));
	doc.save(configPath()).check();
}

void LucidApp::selectSetup(ZStr name) {
	for(int i : intRange(m_setups))
		if(m_setups[i]->name == name) {
			selectSetup(i);
			break;
		}
}

void LucidApp::selectSetup(int idx) {
	DASSERT(m_setups.inRange(idx));
	if(m_setup_idx == idx)
		return;
	if(m_setup_idx != -1)
		m_setups[m_setup_idx]->camera = m_cam_control.current();
	auto &setup = *m_setups[idx];
	if(auto result = setup.updateScene(m_device); !result) {
		result.error().print();
		return;
	}
	if(auto cam = setup.camera)
		m_cam_control.setTarget(*cam);
	m_cam_control.finishAnim();
	m_lucid_opts.setIf(LucidRenderOpt::additive_blending, setup.render_config.additive_blending);
	m_setup_idx = idx;
	m_scene_frame_id = 0;
}

void LucidApp::switchView() {
	if(m_setup_idx == -1)
		return;
	auto &setup = *m_setups[m_setup_idx];
	if(!setup.views.empty()) {
		setup.view_id = (setup.view_id + 1) % setup.views.size();
		m_cam_control.setTarget(setup.views[setup.view_id]);
		m_cam_control.finishAnim();
	}
}

bool LucidApp::updateViewport() {
	auto viewport = IRect(m_device->swapChain()->size());
	bool changed = m_viewport != viewport;
	m_viewport = viewport;
	if(changed)
		m_cam_control.o_config.params.viewport = m_viewport;
	return changed;
}

Ex<void> LucidApp::updateRenderer() {
	PERF_SCOPE();

	bool do_update = m_lucid_renderer && m_lucid_renderer->opts() != m_lucid_opts;
	if(updateViewport())
		do_update = true;

	if(!do_update && m_last_time - m_last_shader_update_time > 0.5) {
		m_last_shader_update_time = m_last_time;
		auto update_list = m_shader_compiler->updateList();
		vector<ShaderDefId> used_shaders;
		if(m_simple_renderer)
			used_shaders = m_simple_renderer->shaderDefIds();
		if(m_pbr_renderer)
			insertBack(used_shaders, m_pbr_renderer->shaderDefIds());
		if(m_lucid_renderer)
			insertBack(used_shaders, m_lucid_renderer->shaderDefIds());
		if(m_path_tracer)
			insertBack(used_shaders, m_path_tracer->shaderDefIds());
		makeSortedUnique(used_shaders);
		if(setIntersection(used_shaders, update_list))
			do_update = true;
	}

	if(do_update) {
		m_lucid_renderer.reset();
		m_simple_renderer.reset();
		m_pbr_renderer.reset();
		m_path_tracer.reset();
		m_device->waitForIdle();
		m_last_shader_update_time = m_last_time;
		m_scene_frame_id = 0;
	}

	if(m_rendering_mode == RenderingMode::path_trace &&
	   !(m_device->features() & VDeviceFeature::ray_tracing)) {
		m_rendering_mode = RenderingMode::simple;
	}

	auto swap_chain = m_device->swapChain();

	if(m_rendering_mode == RenderingMode::pbr && !m_pbr_renderer)
		m_pbr_renderer = EX_PASS(construct<PbrRenderer>(*m_device, *m_shader_compiler, m_viewport,
														swap_chain->format()));

	if(isOneOf(m_rendering_mode, RenderingMode::simple, RenderingMode::mixed) && !m_simple_renderer)
		m_simple_renderer = EX_PASS(construct<SimpleRenderer>(*m_device, *m_shader_compiler,
															  m_viewport, swap_chain->format()));

	if(isOneOf(m_rendering_mode, RenderingMode::lucid, RenderingMode::mixed) && !m_lucid_renderer)
		m_lucid_renderer = EX_PASS(construct<LucidRenderer>(
			*m_device, *m_shader_compiler, swap_chain->format(), m_lucid_opts, m_viewport.size()));

	if(m_rendering_mode == RenderingMode::path_trace && !m_path_tracer) {
		auto format = m_device->swapChain()->format();
		m_path_tracer = EX_PASS(construct<PathTracer>(*m_device, *m_shader_compiler, format,
													  m_path_tracer_opts, m_viewport.size()));
	}

	return {};
}

Ex<> LucidApp::updateEnvMap() {
	if(m_lighting.env_map_path.empty())
		return {};

	auto time = getTime();
	auto panorama = EX_PASS(loadExr(m_lighting.env_map_path));
	auto vimage = EX_PASS(VulkanImage::createAndUpload(*m_device, panorama));
	m_lighting.env_map = VulkanImageView::create(vimage);

	printf("EXR loading time: %.2f sec\n", getTime() - time);
	return {};
}

static void showStatsRows(CSpan<StatsRow> rows, ZStr title, int label_width) {
	if(ImGui::BeginTable(title.c_str(), 2, ImGuiTableFlags_RowBg)) {
		ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, label_width);
		ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch);

		for(auto &row : rows) {
			ImGui::TableNextRow();
			ImGui::TableSetColumnIndex(0);
			ImGui::Text("%s", row.label.c_str());
			ImGui::TableSetColumnIndex(1);
			ImGui::Text("%s", row.value.c_str());
			if(!row.tooltip.empty() && row.tooltip != row.value && ImGui::IsItemHovered())
				ImGui::SetTooltip("%s", row.tooltip.c_str());
		}
		ImGui::EndTable();
	}
}

static string formatSize(long long value) {
	if(value >= 32 * 1000 * 1000)
		return stdFormat("%.0fM", double(value) / (1000 * 1000));
	if(value > 32 * 1000)
		return stdFormat("%.0fK", double(value) / 1000);
	return toString(value);
};

void LucidApp::showSceneStats(const Scene &scene) {
	int num_degenerate_quads = 0;
	for(auto &mesh : scene.meshes)
		num_degenerate_quads += mesh.num_degenerate_quads;
	auto bbox = scene.bounding_box;

	vector<StatsRow> rows;
	rows = {
		{"resolution", toString(m_viewport.size()), ""},
		{"mesh instances", toString(scene.meshes.size()), ""},
		{"triangles", formatSize(scene.numTris()), toString(scene.numTris())},
		{"vertices", formatSize(scene.numVerts()), toString(scene.numVerts())},
		{"quads", formatSize(scene.numQuads()), toString(scene.numQuads())},
		{"degenerate quads",
		 stdFormat("%d (%.2f %%)", num_degenerate_quads,
				   double(num_degenerate_quads) / scene.numQuads() * 100.0),
		 ""},

		{"bbox min", stdFormat("(%.2f, %.2f, %.2f)", bbox.min().x, bbox.min().y, bbox.min().z), ""},
		{"bbox max", stdFormat("(%.2f, %.2f, %.2f)", bbox.max().x, bbox.max().y, bbox.max().z),
		 ""}};

	showStatsRows(rows, "scene_stats", 90 * m_gui.dpiScale());
}

void LucidApp::showRasterStats(const Scene &scene) {
	auto groups = m_lucid_renderer->getStats();

	for(int i : intRange(groups)) {
		auto &group = groups[i];
		if(!group.title.empty())
			ImGui::Text("%s", group.title.c_str());
		showStatsRows(group.rows, format("_group%", i), group.label_width * m_gui.dpiScale());
		if(i + 1 < groups.size()) {
			ImGui::Separator();
			ImGui::Separator();
		}
	}
}

void LucidApp::showStatsMenu(const Scene &scene) {
	ImGui::Begin("Statistics", &m_show_stats);
	ImGuiTabBarFlags tab_bar_flags = ImGuiTabBarFlags_None;
	if(ImGui::BeginTabBar("stats_bar", tab_bar_flags)) {
		auto flags = m_select_stats_tab == 0 ? ImGuiTabItemFlags_SetSelected : 0;
		if(ImGui::BeginTabItem("Scene stats", nullptr, flags)) {
			showSceneStats(scene);
			m_selected_stats_tab = 0;
			ImGui::EndTabItem();
		}

		flags = m_select_stats_tab == 1 ? ImGuiTabItemFlags_SetSelected : 0;
		if(ImGui::BeginTabItem("Rasterizer stats", nullptr, flags)) {
			showRasterStats(scene);
			m_selected_stats_tab = 1;
			ImGui::EndTabItem();
		}
		ImGui::EndTabBar();
		m_select_stats_tab = -1;
	}
	ImGui::End();
}

void LucidApp::doMenu() {
	PERF_SCOPE();

	ImGui::SetNextWindowSize({240 * m_gui.dpiScale(), 0});
	ImGui::Begin("Lucid rasterizer tools", nullptr, ImGuiWindowFlags_NoResize);

	auto setup_idx = m_setup_idx;
	auto names = transform(m_setups, [](auto &setup) { return setup->name.c_str(); });
	if(m_gui.selectIndex("Scene:", setup_idx, names))
		selectSetup(setup_idx);
	auto &setup = *m_setups[m_setup_idx];
	setup.doMenu(m_device);

	auto *scene = setup.scene ? setup.scene.get() : nullptr;

	if(ImGui::Button("Rasterizer options", {120 * m_gui.dpiScale(), 0}))
		ImGui::OpenPopup("render_opts");
	ImGui::SameLine();
	if(ImGui::Button("Statistics", {99 * m_gui.dpiScale(), 0})) {
		ImGui::SetWindowFocus("Statistics");
		m_show_stats = true;
	}
	if(ImGui::Button("Other tools", {120 * m_gui.dpiScale(), 0}))
		ImGui::OpenPopup("other_tools");
	ImGui::SameLine();
	if(ImGui::Button("Filtering options", {99 * m_gui.dpiScale(), 0}))
		ImGui::OpenPopup("filter_opts");

	if(ImGui::BeginPopup("render_opts")) {
		m_gui.selectFlags(m_lucid_opts);
		setup.render_config.additive_blending = m_lucid_opts & LucidRenderOpt::additive_blending;
		ImGui::EndPopup();
	}

	if(ImGui::BeginPopup("other_tools")) {
		if(ImGui::Button("Use FPP camera")) {
			auto new_cam = FppCamera::closest(m_cam_control.currentCamera());
			m_cam_control.setTarget(new_cam);
		}
		if(ImGui::Button("Save configuration"))
			saveConfig();
		ImGui::SliderFloat("Move speed", &m_cam_control.o_config.move_multiplier, 0.05f, 10.0f);
		if(ImGui::Button("Print camera pos")) {
			auto camera = m_cam_control.current();
			if(FppCamera *fpp_cam = camera)
				print("FppCamera{%, %, %}\n", fpp_cam->pos, fpp_cam->forward_xz, fpp_cam->rot_vert);
			if(OrbitingCamera *orbit_cam = camera)
				print("OrbitingCamera{%, %, %, %}\n", orbit_cam->center, orbit_cam->distance,
					  orbit_cam->rot_horiz, orbit_cam->rot_vert);
		}
		ImGui::InputFloat("Square weight", &m_square_weight);
		//if(setup.scene && ImGui::Button("Generate meshlets")) {
		//	m_test_meshlets = true;
		//}
		if(perf::Manager::instance() && !m_perf_analyzer)
			if(ImGui::Button("Show performance analyzer"))
				m_perf_analyzer.emplace();
		ImGui::EndPopup();
	}

	if(ImGui::BeginPopup("filter_opts")) {
		auto &sampler = setup.render_config.sampler_setup;
		m_gui.selectEnum("Magnification filter", sampler.mag_filter);
		m_gui.selectEnum("Minification filter", sampler.min_filter);

		bool mipmapping_enabled = !!sampler.mipmap_filter;
		if(ImGui::Checkbox("Mipmapping", &mipmapping_enabled)) {
			if(mipmapping_enabled)
				sampler.mipmap_filter = VTexFilter::nearest;
			else
				sampler.mipmap_filter = none;
		}

		if(mipmapping_enabled)
			m_gui.selectEnum("Mipmap filter", *sampler.mipmap_filter);

		auto label = format("Anisotropy: %", (int)sampler.max_anisotropy_samples);
		if(ImGui::Button(label.c_str())) {
			auto &limits = m_device->physInfo().properties.limits;
			int aniso = int(sampler.max_anisotropy_samples) * 2;
			int max_aniso = min(128, int(limits.maxSamplerAnisotropy));
			if(aniso > max_aniso)
				aniso = 1;
			sampler.max_anisotropy_samples = aniso;
		}
		ImGui::EndPopup();
	}

	m_gui.selectEnum("Rendering mode", m_rendering_mode);
	ImGui::Checkbox("Back-face culling", &setup.render_config.backface_culling);
	ImGui::SameLine();

	if(ImGui::Checkbox("Additive blending", &setup.render_config.additive_blending))
		m_lucid_opts.setIf(LucidRenderOpt::additive_blending,
						   setup.render_config.additive_blending);

	ImGui::Checkbox("Wireframe mode (only simple rendering)", &m_wireframe_mode);

	// TODO: different opacity for different scenes ?
	int label_size = (int)ImGui::CalcTextSize("Scene opacity").x;
	ImGui::SetNextItemWidth(220 * m_gui.dpiScale() - label_size);
	ImGui::SliderFloat("Scene opacity", &setup.render_config.scene_opacity, 0.0f, 1.0f);

	auto warning = [](ZStr text) {
		ImGui::TextColored((ImVec4)FColor(ColorId::red), "%s", text.c_str());
	};

	if(m_lucid_renderer && scene->numQuads() > m_lucid_renderer->maxSceneQuads())
		warning(format(
			"Scene has too many quads (%K, max:%K)\nfor Lucid rasterizer (Not enough GPU memory).",
			scene->numQuads() / 1024, m_lucid_renderer->maxSceneQuads() / 1024));
	if(!isOneOf(m_rendering_mode, RenderingMode::pbr, RenderingMode::path_trace) &&
	   !scene->hasSimpleTextures())
		warning("PBR scenes require PBR or PathTrace renderers.");
	if(!(m_device->features() & VDeviceFeature::ray_tracing))
		warning("Ray-tracing not available on this device.");
	if(m_device->physInfo().deviceType() != VPhysicalDeviceType::discrete_gpu)
		warning("Not running on a discrete GPU.");

	ImGui::End();

	if(m_show_stats && scene)
		showStatsMenu(*scene);
	//ImGui::ShowDemoWindow();
}

bool LucidApp::handleInput(vector<InputEvent> events, float time_diff) {
	m_mouse_pos = none;

	events = m_cam_control.handleInput(std::move(events));
	for(const auto &event : events) {
		if(event.keyDown(InputKey::esc)) {
			return false;
		}
		if(event.keyDown(InputKey::f11)) {
			auto flags =
				m_window->isFullscreen() ? VWindowFlags() : VWindowFlag::fullscreen_desktop;
			m_window->setFullscreen(flags);
		}

		if(event.keyDown(InputKey::space))
			switchView();

		if(event.isMouseOverEvent())
			m_mouse_pos = (float2)event.mousePos();

		if(event.mouseButtonDown(InputButton::right) && m_is_picking_block) {
			m_is_picking_block = false;
		}

		if(event.mouseButtonDown(InputButton::left) && m_setup_idx != -1) {
			if(m_is_picking_block) {
				m_is_final_pick = true;
				continue;
			}

			auto pos = float2(event.mousePos());
			auto seg = m_cam_control.currentCamera().screenRay(pos);

			m_picked_pos = none;
			auto &setup = m_setups[m_setup_idx];
			if(setup && setup->scene) {
				if(auto result = setup->scene->intersect(seg))
					m_picked_pos = result->pos;
			}
			//DUMP(m_picked_pos);
		}
	}

	if(m_is_picking_block && m_mouse_pos) {
		int2 pos = int2(*m_mouse_pos);
		pos.y = m_viewport.height() - pos.y;
		// TODO
	}

	return true;
}

bool LucidApp::tick(float time_diff) {
	PERF_SCOPE();
	m_cam_control.tick(time_diff, false);

	vector<InputEvent> events;
	events = m_window->inputEvents();

	TextFormatter title;
	title("Lucid rasterizer res:%", m_window->size());
	if(auto dpi_scale = m_window->dpiScale(); dpi_scale > 1.0f)
		title(" dpi_scale:%", dpi_scale);
	m_window->setTitle(title.text());

	if(m_test_meshlets) {
		if(m_setup_idx != -1) {
			auto &setup = m_setups[m_setup_idx];
			if(setup && setup->scene)
				meshletTest(*setup->scene, m_square_weight);
		}
		m_test_meshlets = false;
	}

	// TODO: handleMenus  function ?
	m_gui.beginFrame(*m_window);
	doMenu();
	if(m_perf_analyzer) {
		bool show = true;
		m_perf_analyzer->doMenu(show);
		if(!show)
			m_perf_analyzer.reset();
	}
	events = m_gui.finishFrame(*m_window);
	auto result = handleInput(events, time_diff);
	updatePerfStats();

	if(m_verify_lucid_info && m_lucid_renderer)
		m_lucid_renderer->verifyInfo();
#ifndef NDEBUG
	m_device->memory().validate();
#endif

	return result;
}

void LucidApp::clearScreen(const RenderContext &ctx) {
	auto &cmds = m_device->cmdQueue();
	PERF_GPU_SCOPE(cmds);
	auto swap_chain = m_device->swapChain();
	auto sc_format = swap_chain->format();
	auto color_sync =
		VColorSync(VLoadOp::clear, VStoreOp::store, VImageLayout::undefined, VImageLayout::general);
	auto rpass = m_device->getRenderPass({{sc_format, 1, color_sync}});
	auto framebuffer = ctx.device.getFramebuffer({swap_chain->acquiredImage()});
	cmds.beginRenderPass(framebuffer, rpass, none, {ctx.config.background_color});
	cmds.endRenderPass();
	cmds.barrier(VPipeStage::all_graphics, VPipeStage::all_graphics, VAccess::color_att_write,
				 VAccess::memory_read);
}

void LucidApp::drawScene() {
	auto &cmds = m_device->cmdQueue();
	PERF_GPU_SCOPE(cmds);
	auto cam = m_cam_control.currentCamera();
	auto &setup = *m_setups[m_setup_idx];
	auto swap_chain = m_device->swapChain();

	auto frustum = cam.frustum();
	auto draws = setup.scene->draws(frustum);
	auto tex_pair = setup.scene->textureAtlasPair();

	RenderContext ctx{*setup.scene,
					  *m_device,
					  setup.render_config,
					  setup.scene->verts,
					  setup.scene->tris_ib,
					  setup.scene->quads_ib,
					  draws,
					  setup.scene->materials,
					  tex_pair.first,
					  tex_pair.second,
					  m_lighting,
					  frustum,
					  cam};

	for(auto &material : ctx.materials)
		material.opacity *= setup.render_config.scene_opacity;
	if(setup.render_config.scene_opacity < 1.0)
		for(auto &dc : ctx.dcs)
			dc.opts &= ~DrawCallOpt::is_opaque;

	clearScreen(ctx);

	bool is_pbr = !setup.scene->hasSimpleTextures();
	if(!is_pbr && isOneOf(m_rendering_mode, RenderingMode::simple, RenderingMode::mixed))
		m_simple_renderer->render(ctx, m_wireframe_mode).check();
	if(!is_pbr && isOneOf(m_rendering_mode, RenderingMode::lucid, RenderingMode::mixed))
		if(setup.scene->numQuads() <= m_lucid_renderer->maxSceneQuads())
			m_lucid_renderer->render(ctx);
	if(m_rendering_mode == RenderingMode::pbr && m_pbr_renderer)
		m_pbr_renderer->render(ctx, m_wireframe_mode).check();
	if(m_rendering_mode == RenderingMode::path_trace && m_path_tracer)
		m_path_tracer->render(ctx);
	m_scene_frame_id++;
}

void LucidApp::drawFrame() {
	PERF_SCOPE();

	m_device->beginFrame().check();
	auto swap_chain = m_device->swapChain();
	if(swap_chain->status() == VSwapChainStatus::image_acquired) {
		drawScene();

		auto &cmds = m_device->cmdQueue();
		PERF_GPU_SCOPE(cmds, "ImGuiWrapper::drawFrame");
		auto fb = m_device->getFramebuffer({m_device->swapChain()->acquiredImage()});
		cmds.beginRenderPass(fb, m_gui_render_pass, none);
		m_gui.drawFrame(*m_window, cmds.bufferHandle());
		cmds.endRenderPass();
	}
	m_device->finishFrame().check();
	m_gui.endFrame();
}

bool LucidApp::mainLoop() {
	perf::nextFrame();
	perf::Manager::instance()->getNewFrames();

	auto cur_time = getTime();
	float time_diff = m_last_time < 0.0 ? 1.0f / 60.0f : (cur_time - m_last_time);
	m_last_time = cur_time;

	if(!tick(time_diff))
		return false;

	updateRenderer().check();
	drawFrame();

	return true;
}

vector<LucidApp::StatPoint> LucidApp::selectPerfPoints() const {
	auto *manager = perf::Manager::instance();
	if(!manager || !m_gather_perf_stats)
		return {};
	auto &exec_tree = manager->execTree();

	array<ZStr, 3> selection[] = {
		{"lucid", "LucidRenderer::render", ""},
		{"setup", "LucidRenderer::quadSetup", ""},
		{"bins", "LucidRenderer::computeBins", ""},
		{"raster_low", "LucidRenderer::rasterLow", ""},
		{"raster_hi", "LucidRenderer::rasterHigh", ""},
		{"simple", "SimpleRenderer::render", ""},
	};

	vector<StatPoint> out;
	for(int i = 0; i < arraySize(selection); i++) {
		int point_id = -1;
		for(int p = 1; p < perf::numPoints(); p++) {
			auto *point = perf::pointInfo(perf::PointId(p));
			if(selection[i][1] == point->func.name && selection[i][2] == point->tag) {
				point_id = p;
				break;
			}
		}

		Maybe<perf::ExecId> exec_id;
		for(int e : intRange(exec_tree.nodes)) {
			auto &node = exec_tree.nodes[e];
			if(node.point_id == point_id && node.gpu_time_id) {
				exec_id = perf::ExecId(e);
				break;
			}
		}

		if(!exec_id) {
			print("Invalid/missing perf point selected for stats: % [%]\n", selection[i][1],
				  selection[i][2]);
			continue;
		}

		out.emplace_back(*exec_id, selection[i][0]);
	}

	return out;
}

void LucidApp ::updatePerfStats() {
	auto *manager = perf::Manager::instance();
	if(!manager || m_setup_idx == -1 || !m_gather_perf_stats)
		return;

	m_stats.resize(m_setups.size());
	auto frames = manager->frames();
	if(!frames)
		return;

	if(m_setup_idx != m_prev_setup_idx || m_rendering_mode == RenderingMode::simple) {
		m_skip_frame_id = frames.back().frame_id + 1;
		m_prev_setup_idx = m_setup_idx;
	}

	// TODO: check for m_scene_frame_id is a hack; Rendering should work properly from first frame...
	if(m_setup_idx != -1 && frames.back().frame_id > m_skip_frame_id && m_scene_frame_id > 5)
		m_stats[m_setup_idx].emplace_back(frames.back());
}

// TODO: generalize, move to libfwk, use in LucidRenderer
void print2DArray(CSpan<ZStr> column_names, CSpan<ZStr> row_names, CSpan<vector<string>> rows) {
	DASSERT(row_names.size() == rows.size());
	vector<int> col_widths = transform(column_names, [](auto &name) { return name.size(); });

	// TODO: limit to console sizes

	for(auto &row : rows) {
		DASSERT(row.size() == column_names.size());
		for(int i : intRange(row))
			col_widths[i] = max(col_widths[i], (int)row[i].size());
	}

	int row_name_width = 0;
	for(auto &row_name : row_names)
		row_name_width = max(row_name_width, row_name.size());

	int max_name_width = 20;
	for(auto &width : col_widths)
		width = min(width, max_name_width);
	row_name_width = min(row_name_width, max_name_width);

	auto print_spaces = [](int len) {
		for(int i = 0; i < len; i++)
			printf(" ");
	};

	print_spaces(row_name_width + 2);
	for(int i : intRange(column_names)) {
		auto &name = column_names[i];
		int empty_width = col_widths[i] - name.size();
		print_spaces(empty_width / 2);
		printf("%s", name.c_str());
		if(i + 1 < column_names.size())
			print_spaces(empty_width - empty_width / 2 + 2);
	}
	printf("\n");

	for(int r = 0; r < row_names.size(); r++) {
		print_spaces(row_name_width - row_names[r].size());
		printf("%s: ", row_names[r].c_str());

		for(int c = 0; c < col_widths.size(); c++) {
			auto &value = rows[r][c];
			print_spaces(col_widths[c] - value.size());
			printf("%s", value.c_str());
			if(c + 1 < col_widths.size())
				print_spaces(2);
		}
		printf("\n");
	}
}

void LucidApp::printPerfStats() {
	if(!m_perf_analyzer || !m_gather_perf_stats)
		return;
	auto points = selectPerfPoints();
	if(!points)
		return;

	vector<ZStr> column_names = transform(points, [](auto &point) { return point.short_name; });
	vector<ZStr> row_names;
	vector<vector<string>> rows;
	int lucid_id = indexOf(column_names, "lucid");
	int simple_id = indexOf(column_names, "simple");
	column_names.emplace_back("ratio");

	for(int i : intRange(m_setups)) {
		if(!m_stats[i])
			continue;

		perf::Analyzer::FrameRange range;
		m_perf_analyzer->computeRange(range, m_stats[i]);

		int lucid_value = 0, simple_value = 0;
		vector<string> row;
		for(auto &point : points) {
			auto &stat = range.rows[point.exec_id];
			if(stat.num_instances == 0) {
				row.emplace_back();
				continue;
			}
			int value = (int)round(stat.gpu_avg / 1000.0);
			if(row.size() == lucid_id)
				lucid_value = value;
			else if(row.size() == simple_id)
				simple_value = value;
			row.emplace_back(stdFormat("%d", value));
		}

		if(allOf(row, ""))
			continue;

		string ratio;
		if(lucid_value != 0 && simple_value != 0)
			ratio = stdFormat("%.2f", double(lucid_value) / simple_value);
		row.emplace_back(ratio);

		row_names.emplace_back(m_setups[i]->name);
		rows.emplace_back(row);
	}

	if(rows) {
		print("Average GPU timings shown; unit: microseconds; Resolution: %\n", m_viewport.size());
		print2DArray(column_names, row_names, rows);
	}
}

bool LucidApp::mainLoop(VulkanWindow &window, void *this_ptr) {
	auto *lucid_app = ((LucidApp *)this_ptr);
	DASSERT(&*lucid_app->m_window == &window);
	return lucid_app->mainLoop();
}
