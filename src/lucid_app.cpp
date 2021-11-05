#include "lucid_app.h"

#include "meshlet.h"
#include "scene_setup.h"
#include "simple_renderer.h"

#include <fwk/any_config.h>
#include <fwk/gfx/camera_variant.h>
#include <fwk/gfx/draw_call.h>
#include <fwk/gfx/font_factory.h>
#include <fwk/gfx/gl_device.h>
#include <fwk/gfx/gl_format.h>
#include <fwk/gfx/gl_framebuffer.h>
#include <fwk/gfx/gl_texture.h>
#include <fwk/gfx/image.h>
#include <fwk/gfx/line_buffer.h>
#include <fwk/gfx/material_set.h>
#include <fwk/gfx/mesh.h>
#include <fwk/gfx/opengl.h>
#include <fwk/gfx/renderer2d.h>
#include <fwk/io/file_system.h>
#include <fwk/math/axis_angle.h>
#include <fwk/math/quat.h>
#include <fwk/math/random.h>
#include <fwk/math/rotation.h>
#include <fwk/menu/helpers.h>
#include <fwk/menu_imgui.h>
#include <fwk/perf/analyzer.h>
#include <fwk/perf/exec_tree.h>
#include <fwk/perf/manager.h>
#include <fwk/sys/input.h>

// TODO: save imgui settings

string dataPath(string file_name) { return executablePath().parent() / "data" / file_name; }

LucidApp::LucidApp()
	: m_imgui(GlDevice::instance(), ImGuiStyleMode::mini),
	  m_cam_control(Plane3F(float3(0, 1, 0), 0.0f)), m_lighting(SceneLighting::makeDefault()) {
	m_filtering_params.magnification = TextureFilterOpt::linear;
	m_filtering_params.minification = TextureFilterOpt::linear;
	m_filtering_params.mipmap = TextureFilterOpt::linear;
	if(perf::Manager::instance())
		m_perf_analyzer.emplace();

	OrbitingCamera cam({}, 10.0f, 0.5f, 0.8f);
	m_cam_control.setTarget(cam);
	m_cam_control.finishAnim();
	m_cam_control.o_config.params.depth = {1.0f / 16.0f, 1024.0f};
	m_cam_control.o_config.rotation_filter = [](const InputEvent &ev) {
		return ev.mouseButtonPressed(InputButton::right);
	};

	auto font_path = dataPath("LiberationSans-Regular.ttf");
	m_font.emplace(move(FontFactory().makeFont(font_path, 14, false).get()));

	m_setups.emplace_back(new BoxesSetup());
	for(auto scene_name : LoadedSetup::findAll())
		m_setups.emplace_back(new LoadedSetup(scene_name));
	selectSetup(0);

	updateViewport();
	updateRenderer();
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
	if(m_perf_analyzer)
		if(auto *sub = config.subConfig("perf_analyzer"))
			m_perf_analyzer->setConfig(*sub);
	if(auto *sub = config.subConfig("imgui"))
		m_imgui.setConfig(*sub);
	m_cam_control.load(config);
}

Maybe<AnyConfig> LucidApp::loadConfig() {
	Maybe<AnyConfig> config;
	auto path = configPath();
	if(access(path)) {
		auto doc = move(XmlDocument::load(path).get());
		config = AnyConfig::load(doc.child("config"), true).get();
		config->printErrors();
	}
	return config;
}

void LucidApp::saveConfig() const {
	AnyConfig out;
	if(access(configPath())) {
		auto doc = move(XmlDocument::load(configPath()).get());
		out = AnyConfig::load(doc.child("config"), true).get();
	}

	out.set("rendering_mode", m_rendering_mode);
	out.set("trans_opts", m_lucid_opts);
	out.set("wireframe", m_wireframe_mode);
	out.set("window_rect", GlDevice::instance().windowRect());
	out.set("show_stats", m_show_stats);
	out.set("selected_stats_tab", m_selected_stats_tab);

	if(m_setup_idx != -1)
		out.set("scene", m_setups[m_setup_idx]->name);
	if(m_perf_analyzer)
		out.set("perf_analyzer", m_perf_analyzer->config());
	out.set("imgui", m_imgui.config());
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
	if(auto result = m_setups[idx]->updateScene(); !result) {
		result.error().print();
		return;
	}
	if(auto cam = m_setups[idx]->camera)
		m_cam_control.setTarget(*cam);
	m_cam_control.finishAnim();
	m_setup_idx = idx;
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
	auto viewport = IRect(GlDevice::instance().windowSize());
	bool changed = m_viewport != viewport;
	m_viewport = viewport;
	if(changed)
		m_cam_control.o_config.params.viewport = m_viewport;
	return changed;
}

void LucidApp::updateRenderer() {
	bool do_update = !m_lucid_renderer || m_lucid_renderer->opts() != m_lucid_opts;
	if(updateViewport())
		do_update = true;

	for(auto entry : findFiles(dataPath("trans"))) {
		if(entry.path.fileExtension() == "shader") {
			auto time = lastModificationTime(entry.path);
			auto &ref = m_shader_times[entry.path];
			if(time && ref < *time) {
				do_update = true;
				ref = *time;
			}
		}
	}

	if(do_update) {
		m_depth_buffer.emplace(GlFormat::depth32, m_viewport.size(), 1);
		m_depth_buffer->setWrapping(TextureWrapOpt::clamp_to_edge);
		m_clear_fbo.emplace(none, m_depth_buffer);

		m_lucid_renderer.reset();
		m_simple_renderer.reset();
		m_simple_renderer = move(construct<SimpleRenderer>(m_viewport).get());
		m_lucid_renderer = move(construct<LucidRenderer>(m_lucid_opts, m_viewport.size()).get());
	}
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

	showStatsRows(rows, "scene_stats", 90);
}

void LucidApp::showRasterStats(const Scene &scene) {
	auto groups = m_lucid_renderer->getStats();
	for(int i : intRange(groups)) {
		auto &group = groups[i];
		if(!group.title.empty())
			ImGui::Text("%s", group.title.c_str());
		showStatsRows(group.rows, format("_group%", i), group.label_width);
		if(i + 1 < groups.size())
			ImGui::Separator();
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

void LucidApp::doMenu(Renderer2D &renderer_2d) {
	ImGui::SetNextWindowSize({240, 0});
	ImGui::Begin("Lucid rasterizer tools", nullptr, ImGuiWindowFlags_NoResize);

	auto setup_idx = m_setup_idx;
	auto names = transform(m_setups, [](auto &setup) { return setup->name.c_str(); });
	if(menu::selectIndex("Scene:", setup_idx, names))
		selectSetup(setup_idx);
	auto &setup = *m_setups[m_setup_idx];
	setup.doMenu();

	auto *scene = setup.scene ? setup.scene.get() : nullptr;

	if(ImGui::Button("Rasterizer options", {120, 0}))
		ImGui::OpenPopup("render_opts");
	ImGui::SameLine();
	if(ImGui::Button("Statistics", {99, 0})) {
		ImGui::SetWindowFocus("Statistics");
		m_show_stats = true;
	}
	if(ImGui::Button("Other tools", {120, 0}))
		ImGui::OpenPopup("other_tools");
	ImGui::SameLine();
	if(ImGui::Button("Filtering options", {99, 0}))
		ImGui::OpenPopup("filter_opts");

	if(ImGui::BeginPopup("render_opts")) {
		menu::selectFlags(m_lucid_opts);
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
		if(ImGui::Button("Print size histograms")) {
			m_lucid_renderer->printHistograms();
		}
		if(ImGui::Button("Raster masks snapshot")) {
			auto image = m_lucid_renderer->masksSnapshot();
			image.saveTGA(format("%_masks.tga", setup.name)).ignore();
		}
		ImGui::InputFloat("Square weight", &m_square_weight);
		if(setup.scene && ImGui::Button("Generate meshlets")) {
			m_test_meshlets = true;
		}
		if(scene && ImGui::Button("Print materials")) {
			for(auto &mat : scene->materials)
				print("%\n", mat.description());
		}
		ImGui::EndPopup();
	}

	if(ImGui::BeginPopup("filter_opts")) {
		menu::selectEnum("Magnification filter", m_filtering_params.magnification);
		menu::selectEnum("Minification filter", m_filtering_params.minification);

		bool mipmapping_enabled = !!m_filtering_params.mipmap;
		if(menu::Checkbox("Mipmapping", &mipmapping_enabled)) {
			if(mipmapping_enabled)
				m_filtering_params.mipmap = TextureFilterOpt::nearest;
			else
				m_filtering_params.mipmap = none;
		}

		if(mipmapping_enabled)
			menu::selectEnum("Mipmap filter", *m_filtering_params.mipmap);

		auto label = format("Anisotropy: %", (int)m_filtering_params.max_anisotropy_samples);
		if(ImGui::Button(label.c_str())) {
			int aniso = int(m_filtering_params.max_anisotropy_samples) * 2;
			int max_aniso = min(128, gl_info->limits[GlLimit::max_texture_anisotropy]);
			if(aniso > max_aniso)
				aniso = 1;
			m_filtering_params.max_anisotropy_samples = aniso;
		}
		ImGui::EndPopup();
	}

	menu::selectEnum("Rendering mode", m_rendering_mode);
	ImGui::Checkbox("Back-face culling", &setup.render_config.backface_culling);
	ImGui::SameLine();
	ImGui::Checkbox("Additive blending", &setup.render_config.additive_blending);
	ImGui::Checkbox("Wireframe mode (only simple rendering)", &m_wireframe_mode);

	// TODO: different opacity for different scenes ?
	int labels_size[] = {(int)ImGui::CalcTextSize("Scene opacity").x,
						 (int)ImGui::CalcTextSize("Scene color").x};
	ImGui::SetNextItemWidth(220 - labels_size[0]);
	ImGui::SliderFloat("Scene opacity", &setup.render_config.scene_opacity, 0.0f, 1.0f);
	ImGui::SetNextItemWidth(220 - labels_size[0]);
	ImGui::ColorEdit3("Scene color", setup.render_config.scene_color.v, 0);
	ImGui::End();

	if(m_show_stats && scene)
		showStatsMenu(*scene);
	//ImGui::ShowDemoWindow();
}

bool LucidApp::handleInput(vector<InputEvent> events, float time_diff) {
	m_mouse_pos = none;

	events = m_cam_control.handleInput(move(events));
	for(const auto &event : events) {
		if(event.keyDown(InputKey::esc)) {
			return false;
		}
		if(event.keyDown(InputKey::f11)) {
			auto &gl_device = GlDevice::instance();

			auto flags =
				gl_device.isWindowFullscreen() ? GlDeviceFlags() : GlDeviceOpt::fullscreen_desktop;
			gl_device.setWindowFullscreen(flags);
		}

		if(event.keyDown(InputKey::space))
			switchView();

		if(event.isMouseOverEvent())
			m_mouse_pos = (float2)event.mousePos();
		if(event.mouseButtonDown(InputButton::left) && m_setup_idx != -1) {
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
	return true;
}

bool LucidApp::tick(float time_diff) {
	PERF_SCOPE();
	m_cam_control.tick(time_diff, false);

	auto &device = GlDevice::instance();
	vector<InputEvent> events;
	events = device.inputEvents();

	if(m_test_meshlets) {
		if(m_setup_idx != -1) {
			auto &setup = m_setups[m_setup_idx];
			if(setup && setup->scene)
				meshletTest(*setup->scene, m_square_weight);
		}
		m_test_meshlets = false;
	}

	// TODO: handleMenus  function ?
	m_imgui.beginFrame(device);
	if(m_perf_analyzer) {
		bool show = true;
		m_perf_analyzer->doMenu(show);
		if(!show)
			m_perf_analyzer.reset();
	}
	events = m_imgui.finishFrame(device);
	auto result = handleInput(events, time_diff);
	updatePerfStats();

	return result;
}

void LucidApp::drawScene() {
	PERF_GPU_SCOPE();
	auto cam = m_cam_control.currentCamera();

	auto proj_mat = cam.projectionMatrix();
	auto view_mat = cam.viewMatrix();
	auto &setup = *m_setups[m_setup_idx];

	{
		// TODO: renderer3d should draw to depth buffer
		// TODO: what objects to show ?
		PERF_GPU_SCOPE("Clear buffers");
		m_clear_fbo->bind();
		glClearDepth(1.0f);
		glClearStencil(0);
		glStencilMask(0xff);
		glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
		GlFramebuffer::unbind();
	}

	auto frustum = cam.frustum();
	auto draws = setup.scene->draws(frustum);
	auto tex_pair = setup.scene->textureAtlasPair();
	for(auto &tex : {tex_pair.first, tex_pair.second}) {
		if(tex)
			tex->setFiltering(m_filtering_params);
	}

	RenderContext ctx{setup.render_config,
					  setup.scene->mesh_vao,
					  setup.scene->quads_ib,
					  draws,
					  setup.scene->materials,
					  tex_pair.first,
					  tex_pair.second,
					  m_lighting,
					  frustum,
					  cam,
					  {},
					  m_depth_buffer,
					  {}};
	for(auto &material : ctx.materials) {
		material.opacity *= setup.render_config.scene_opacity;
		material.diffuse *= setup.render_config.scene_color;
	}
	if(setup.render_config.scene_opacity < 1.0)
		for(auto &dc : ctx.dcs)
			dc.opts &= ~DrawCallOpt::is_opaque;

	if(m_rendering_mode != RenderingMode::trans)
		m_simple_renderer->render(ctx, m_wireframe_mode);
	if(m_rendering_mode != RenderingMode::simple)
		m_lucid_renderer->render(ctx);
}

bool LucidApp::mainLoop(GlDevice &device) {
	perf::nextFrame();
	perf::Manager::instance()->getNewFrames();

	clearColor(IColor(0, 30, 30));
	clearDepth(1.0f);

	auto cur_time = getTime();
	float time_diff = m_last_time < 0.0 ? 1.0f / 60.0f : (cur_time - m_last_time);
	m_last_time = cur_time;

	updateRenderer();
	auto result = tick(time_diff);
	drawScene();

	Renderer2D renderer_2d(m_viewport, Orient2D::y_down);
	doMenu(renderer_2d);
	renderer_2d.render();

	{
		PERF_GPU_SCOPE("ImGuiWrapper::drawFrame");
		m_imgui.drawFrame(device);
	}

	glFlush();

	return result;
}

vector<LucidApp::StatPoint> LucidApp::selectPerfPoints() const {
	auto *manager = perf::Manager::instance();
	if(!manager || !m_gather_perf_stats)
		return {};
	auto &exec_tree = manager->execTree();

	array<ZStr, 3> selection[] = {
		{"total", "LucidRenderer::render", ""},
		{"setup", "LucidRenderer::setupQuads", ""},
		{"bins", "LucidRenderer::computeBins", ""},
		{"tiles", "LucidRenderer::computeTiles", ""},
		{"raster_masks", "LucidRenderer::rasterizeMasks", ""},
		{"sort_masks", "LucidRenderer::sortMasks", ""},
		{"final_raster", "LucidRenderer::rasterizeFinal", ""},
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
			print("Invalid perf point selected for stats: %s [%s]\n", selection[i][1].c_str(),
				  selection[i][2].c_str());
			return {};
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
	if(m_setup_idx != -1 && frames.back().frame_id > m_skip_frame_id)
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
			printf("%s", value.c_str());
			if(c + 1 < col_widths.size())
				print_spaces(col_widths[c] - value.size() + 2);
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

	for(int i : intRange(m_setups)) {
		if(!m_stats[i])
			continue;

		perf::Analyzer::FrameRange range;
		m_perf_analyzer->computeRange(range, m_stats[i]);

		vector<string> row;
		for(auto &point : points) {
			auto &stat = range.rows[point.exec_id];
			if(stat.num_instances == 0) {
				row.emplace_back();
				continue;
			}
			row.emplace_back(stdFormat("%d", (int)round(stat.gpu_min / 1024.0)));
		}

		if(allOf(row, ""))
			continue;

		row_names.emplace_back(m_setups[i]->name);
		rows.emplace_back(row);
	}

	if(rows) {
		print("Minimum GPU timings shown; unit: microseconds\n");
		print2DArray(column_names, row_names, rows);
	}
}

bool LucidApp::mainLoop(GlDevice &device, void *this_ptr) {
	return ((LucidApp *)this_ptr)->mainLoop(device);
}
