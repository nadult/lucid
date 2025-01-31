// Copyright (C) Krzysztof Jakubowski <nadult@fastmail.fm>
// This file is part of LucidRaster. See license.txt for details.

#include "pbr_renderer.h"

#include "scene.h"
#include "shader_structs.h"
#include "shading.h"
#include <fwk/gfx/camera.h>
#include <fwk/gfx/colored_triangle.h>
#include <fwk/gfx/shader_compiler.h>
#include <fwk/gfx/shader_defs.h>
#include <fwk/hash_map.h>
#include <fwk/vulkan/vulkan_buffer.h>
#include <fwk/vulkan/vulkan_device.h>
#include <fwk/vulkan/vulkan_image.h>
#include <fwk/vulkan/vulkan_pipeline.h>
#include <fwk/vulkan/vulkan_shader.h>
#include <fwk/vulkan/vulkan_swap_chain.h>

PbrRenderer::PbrRenderer() = default;
FWK_MOVABLE_CLASS_IMPL(PbrRenderer);

vector<Pair<string>> sharedShaderMacros(VulkanDevice &);

void PbrRenderer::addShaderDefs(VulkanDevice &device, ShaderCompiler &compiler,
								const ShaderConfig &shader_config) {
	vector<Pair<string>> vsh_macros = {{"VERTEX_SHADER", "1"}};
	vector<Pair<string>> fsh_macros = {{"FRAGMENT_SHADER", "1"}};
	auto shared_macros = shader_config.predefined_macros;
	insertBack(vsh_macros, shared_macros);
	insertBack(fsh_macros, shared_macros);

	compiler.add({"pbr_vert", VShaderStage::vertex, "pbr_material.glsl", vsh_macros});
	compiler.add({"pbr_frag", VShaderStage::fragment, "pbr_material.glsl", fsh_macros});
	compiler.add({"env_map_vert", VShaderStage::vertex, "env_map.glsl", vsh_macros});
	compiler.add({"env_map_frag", VShaderStage::fragment, "env_map.glsl", fsh_macros});
}

Ex<void> PbrRenderer::exConstruct(VulkanDevice &device, ShaderCompiler &compiler,
								  const IRect &viewport, VColorAttachment color_att) {
	auto depth_format = device.bestSupportedFormat(VDepthStencilFormat::d32f);
	auto depth_buffer =
		EX_PASS(VulkanImage::create(device, VImageSetup(depth_format, viewport.size())));
	m_depth_buffer = VulkanImageView::create(depth_buffer);
	// TODO: :we need to transition depth_buffer format too

	VDepthAttachment depth_att(depth_format, 1, defaultLayout(depth_format));
	color_att.sync =
		VColorSync(VLoadOp::load, VStoreOp::store, VImageLayout::general, VImageLayout::general);
	depth_att.sync = VDepthSync(VLoadOp::clear, VStoreOp::store, VImageLayout::undefined,
								defaultLayout(depth_format));
	m_render_pass = device.getRenderPass({color_att}, depth_att);

	m_viewport = viewport;

	auto frag_id = *compiler.find("pbr_frag");
	auto vert_id = *compiler.find("pbr_vert");
	auto env_frag_id = *compiler.find("env_map_frag");
	auto env_vert_id = *compiler.find("env_map_vert");
	m_shader_def_ids = {frag_id, vert_id, env_frag_id, env_vert_id};

	m_frag_module = EX_PASS(compiler.createShaderModule(device, frag_id));
	m_vert_module = EX_PASS(compiler.createShaderModule(device, vert_id));
	m_pipeline_layout = device.getPipelineLayout({m_frag_module, m_vert_module});

	auto env_frag_module = EX_PASS(compiler.createShaderModule(device, env_frag_id));
	auto env_vert_module = EX_PASS(compiler.createShaderModule(device, env_vert_id));

	VPipelineSetup setup;
	setup.pipeline_layout = device.getPipelineLayout({env_frag_module, env_vert_module});
	setup.render_pass = m_render_pass;
	setup.shader_modules = {{env_vert_module, env_frag_module}};
	setup.depth = VDepthSetup(VDepthFlag::test);
	setup.raster = VRasterSetup(VPrimitiveTopology::triangle_list, VPolygonMode::fill);
	setup.vertex_attribs = {{vertexAttrib<float2>(0, 0)}};
	setup.vertex_bindings = {{vertexBinding<float2>(0)}};
	m_env_pipeline = EX_PASS(VulkanPipeline::create(device, setup));

	auto quad_verts = Box<float2>({-1.0f, -1.0f}, {1.0f, 1.0f}).corners();
	array<float2, 6> quad_tris = {quad_verts[0], quad_verts[1], quad_verts[2],
								  quad_verts[0], quad_verts[2], quad_verts[3]};
	auto vb_usage =
		VBufferUsage::vertex_buffer | VBufferUsage::storage_buffer | VBufferUsage::transfer_dst;
	m_rect_vertices = EX_PASS(VulkanBuffer::createAndUpload(device, cspan(quad_tris), vb_usage));

	return {};
}

Ex<PVPipeline> PbrRenderer::getPipeline(VulkanDevice &device, const PipeConfig &config) {
	PERF_SCOPE();

	auto &ref = m_pipelines[config];
	if(!ref) {
		VPipelineSetup setup;
		setup.pipeline_layout = m_pipeline_layout;
		setup.render_pass = m_render_pass;
		setup.shader_modules = {{m_vert_module, m_frag_module}};
		setup.depth = VDepthSetup(VDepthFlag::test | VDepthFlag::write);
		VertexArray::getDefs(setup, true);

		setup.raster = VRasterSetup(VPrimitiveTopology::triangle_list,
									config.wireframe ? VPolygonMode::line : VPolygonMode::fill,
									mask(config.backface_culling, VCull::back));
		if(!config.opaque) {
			VBlendingMode additive_blend(VBlendFactor::src_alpha, VBlendFactor::one);
			VBlendingMode normal_blend(VBlendFactor::src_alpha, VBlendFactor::one_minus_src_alpha);
			setup.blending.attachments = {
				{config.additive_blending ? additive_blend : normal_blend}};
			setup.depth = VDepthSetup(VDepthFlag::test);
		}

		ref = EX_PASS(VulkanPipeline::create(device, setup));
	}

	return ref;
}

Ex<> PbrRenderer::renderPhase(const RenderContext &ctx, VBufferSpan<shader::PbrDrawCall> dc_buf,
							  bool opaque, bool wireframe) {
	auto &cmds = ctx.device.cmdQueue();
	PERF_GPU_SCOPE(cmds);

	cmds.bind(m_pipeline_layout);
	cmds.bindDS(0).set(0, VDescriptorType::uniform_buffer, m_lighting_buf);
	auto &verts = ctx.verts;
	cmds.bindVertices(0, verts.positions, verts.colors, verts.tex_coords, verts.normals,
					  verts.tangents);
	cmds.bindIndices(ctx.tris_ib);

	PipeConfig pipe_config{ctx.config.backface_culling, ctx.config.additive_blending, opaque,
						   wireframe, false};
	auto pipeline = EX_PASS(getPipeline(ctx.device, pipe_config));
	cmds.bind(pipeline);

	auto sampler = ctx.device.getSampler(ctx.config.sampler_setup);
	int prev_mat_id = -1;
	for(int dc : intRange(ctx.dcs)) {
		auto &draw_call = ctx.dcs[dc];
		auto &material = ctx.materials[draw_call.material_id];
		if(bool(draw_call.opts & DrawCallOpt::is_opaque) != opaque)
			continue;
		if(prev_mat_id != draw_call.material_id) {
			auto ds = cmds.bindDS(1);

			// TODO: uniform buffer alignment issue
			// https://vulkan.lunarg.com/doc/view/1.3.290.0/windows/1.3-extensions/vkspec.html#VUID-VkWriteDescriptorSet-descriptorType-00327
			// Switched from path-tracer to pbr renderer on conference
			ds.set(0, VDescriptorType::uniform_buffer, dc_buf.subSpan(dc, dc + 1));

			// TODO: different default textures for different map types
			auto &albedo_map = material.maps[SceneMapType::albedo];
			auto &normal_map = material.maps[SceneMapType::normal];
			auto &pbr_map = material.maps[SceneMapType::pbr];

			ds.set(1, {{sampler, albedo_map.vk_image},
					   {sampler, normal_map.vk_image},
					   {sampler, pbr_map.vk_image}});
			prev_mat_id = draw_call.material_id;
		}

		cmds.drawIndexed(draw_call.num_tris * 3, 1, draw_call.tri_offset * 3);
	}

	// TODO: what about ordering objects by material ?
	// TODO: add option to order objects in different ways ?
	// TODO: optional alpha test first for blended objects

	return {};
}

Ex<> PbrRenderer::renderEnvMap(const RenderContext &ctx) {
	auto &cmds = ctx.device.cmdQueue();
	PERF_GPU_SCOPE(cmds);
	DASSERT(ctx.lighting.env_map);

	auto ubo_usage = VBufferUsage::uniform_buffer | VBufferUsage::transfer_dst;
	shader::EnvMapDrawCall env_map_dc;
	float2 screen_size = float2(ctx.camera.params().viewport.size());
	env_map_dc.screen_size = screen_size;
	env_map_dc.inv_screen_size = vinv(screen_size);
	env_map_dc.inv_proj_view_matrix = inverseOrZero(ctx.camera.matrix());
	auto env_map_dc_buf = EX_PASS(VulkanBuffer::createAndUpload(ctx.device, cspan(&env_map_dc, 1),
																ubo_usage, VMemoryUsage::frame));

	cmds.bind(m_env_pipeline);
	auto ds = cmds.bindDS(0);
	ds.set(0, VDescriptorType::uniform_buffer, env_map_dc_buf.subSpan(0, 1));
	auto sampler = ctx.device.getSampler(ctx.config.sampler_setup);
	ds.set(1, {{sampler, ctx.lighting.env_map}});

	cmds.bindVertices(0, m_rect_vertices);
	cmds.draw(m_rect_vertices.size());
	return {};
}

Ex<> PbrRenderer::render(const RenderContext &ctx, bool wireframe) {
	auto &cmds = ctx.device.cmdQueue();
	PERF_GPU_SCOPE(cmds);

	// TODO: optimize this
	auto ubo_usage = VBufferUsage::uniform_buffer | VBufferUsage::transfer_dst;
	shader::Lighting lighting;
	lighting.ambient_color = ctx.lighting.ambient.color;
	lighting.ambient_power = ctx.lighting.ambient.power;
	lighting.sun_color = ctx.lighting.sun.color;
	lighting.sun_power = ctx.lighting.sun.power;
	lighting.sun_dir = ctx.lighting.sun.dir;
	m_lighting_buf = EX_PASS(VulkanBuffer::createAndUpload(ctx.device, cspan(&lighting, 1),
														   ubo_usage, VMemoryUsage::frame));

	int num_opaque = 0;

	// TODO: minimize it (do it only for different materials)
	vector<shader::PbrDrawCall> dcs;
	dcs.reserve(ctx.dcs.size());
	auto inv_proj_view_matrix = inverseOrZero(ctx.camera.matrix());

	for(const auto &draw_call : ctx.dcs) {
		auto &material = ctx.materials[draw_call.material_id];
		auto &simple_dc = dcs.emplace_back();
		if(draw_call.opts & DrawCallOpt::is_opaque)
			num_opaque++;
		simple_dc.world_camera_pos = ctx.camera.pos();
		simple_dc.proj_view_matrix = ctx.camera.matrix();
		simple_dc.inv_proj_view_matrix = inv_proj_view_matrix;
		simple_dc.material_color = float4(material.diffuse, material.opacity);
		simple_dc.draw_call_opts = uint(draw_call.opts.bits);
	}

	auto dc_buf =
		EX_PASS(VulkanBuffer::createAndUpload(ctx.device, dcs, ubo_usage, VMemoryUsage::frame));

	cmds.setViewport(m_viewport);
	cmds.setScissor(none);

	auto swap_chain = ctx.device.swapChain();
	auto swap_image = swap_chain->acquiredImage()->image();
	auto framebuffer = ctx.device.getFramebuffer({swap_chain->acquiredImage()}, m_depth_buffer);
	cmds.beginRenderPass(framebuffer, m_render_pass, none,
						 {FColor(ColorId::magneta), VClearDepthStencil(1.0)});

	if(num_opaque > 0)
		EXPECT(renderPhase(ctx, dc_buf, true, wireframe));
	if(ctx.lighting.env_map)
		EXPECT(renderEnvMap(ctx));
	if(num_opaque != ctx.dcs.size())
		EXPECT(renderPhase(ctx, dc_buf, false, wireframe));

	cmds.endRenderPass();
	m_lighting_buf = {};

	return {};
}
