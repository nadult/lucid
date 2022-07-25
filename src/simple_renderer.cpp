#include "simple_renderer.h"

#include "scene.h"
#include "shader_structs.h"
#include "shading.h"
#include <fwk/gfx/camera.h>
#include <fwk/gfx/colored_triangle.h>
#include <fwk/gfx/draw_call.h>
#include <fwk/gfx/shader_compiler.h>
#include <fwk/gfx/shader_defs.h>
#include <fwk/hash_map.h>
#include <fwk/vulkan/vulkan_buffer.h>
#include <fwk/vulkan/vulkan_device.h>
#include <fwk/vulkan/vulkan_image.h>
#include <fwk/vulkan/vulkan_pipeline.h>
#include <fwk/vulkan/vulkan_shader.h>
#include <fwk/vulkan/vulkan_swap_chain.h>

SimpleRenderer::SimpleRenderer() = default;
FWK_MOVABLE_CLASS_IMPL(SimpleRenderer);

void SimpleRenderer::addShaderDefs(ShaderCompiler &compiler) {
	vector<Pair<string>> vsh_macros = {{"VERTEX_SHADER", "1"}};
	vector<Pair<string>> fsh_macros = {{"FRAGMENT_SHADER", "1"}};

	compiler.add({"simple_vert", VShaderStage::vertex, "simple.glsl", vsh_macros});
	compiler.add({"simple_frag", VShaderStage::fragment, "simple.glsl", fsh_macros});
}

Ex<void> SimpleRenderer::exConstruct(VDeviceRef device, ShaderCompiler &compiler,
									 const IRect &viewport, VColorAttachment color_att) {
	auto depth_format = device->bestSupportedFormat(VDepthStencilFormat::d32f);
	auto depth_buffer =
		EX_PASS(VulkanImage::create(device, VImageSetup(depth_format, viewport.size())));
	m_depth_buffer = VulkanImageView::create(device, depth_buffer);
	// TODO: :we need to transition depth_buffer format too

	VDepthAttachment depth_att(depth_format, 1, defaultLayout(depth_format));
	color_att.sync = VColorSyncStd::clear;
	depth_att.sync = VDepthSync(VLoadOp::clear, VStoreOp::store, VImageLayout::undefined,
								defaultLayout(depth_format));
	m_render_pass = device->getRenderPass({color_att}, depth_att);

	ShaderDefs defs;
	m_viewport = viewport;
	auto fsh_bytecode = compiler.getSpirv("simple_frag");
	auto vsh_bytecode = compiler.getSpirv("simple_vert");

	m_frag_module = EX_PASS(VulkanShaderModule::create(device, fsh_bytecode));
	m_vert_module = EX_PASS(VulkanShaderModule::create(device, vsh_bytecode));
	m_pipeline_layout = device->getPipelineLayout({m_frag_module, m_vert_module});

	return {};
}

Ex<PVPipeline> SimpleRenderer::getPipeline(const RenderContext &ctx, bool opaque,
										   bool wireframe) const {
	PERF_SCOPE();

	VPipelineSetup setup;
	setup.pipeline_layout = m_pipeline_layout;
	setup.render_pass = m_render_pass;
	setup.shader_modules = {{m_vert_module, m_frag_module}};
	setup.depth = VDepthSetup(VDepthFlag::test | VDepthFlag::write);
	VertexArray::getDefs(setup);

	setup.raster = VRasterSetup(VPrimitiveTopology::triangle_list,
								wireframe ? VPolygonMode::line : VPolygonMode::fill,
								mask(ctx.config.backface_culling, VCull::back));
	if(!opaque) {
		VBlendingMode additive_blend(VBlendFactor::one, VBlendFactor::one);
		VBlendingMode normal_blend(VBlendFactor::src_alpha, VBlendFactor::one_minus_src_alpha);
		setup.blending.attachments = {
			{ctx.config.additive_blending ? additive_blend : normal_blend}};
	}

	// TODO: remove refs
	return VulkanPipeline::create(ctx.device.ref(), setup);
}

Matrix3 normalMatrix(const Matrix4 &affine) {
	Matrix3 out(affine[0].xyz(), affine[1].xyz(), affine[2].xyz());
	return transpose(inverse(out));
}

// TODO: add typed VSpan type
Ex<> SimpleRenderer::renderPhase(const RenderContext &ctx, PVBuffer simple_dc_buf, bool opaque,
								 bool wireframe) {
	// TODO: don't run if we don't have any drawcalls of given type

	auto &cmds = ctx.device.cmdQueue();
	PERF_GPU_SCOPE(cmds);

	auto pipeline = EX_PASS(getPipeline(ctx, opaque, wireframe));
	cmds.bind(pipeline);

	VSamplerSetup sam_setup;
	auto sampler = ctx.device.createSampler(sam_setup);

	int prev_mat_id = -1;
	for(int dc : intRange(ctx.dcs)) {
		auto &draw_call = ctx.dcs[dc];
		auto &material = ctx.materials[draw_call.material_id];
		if(bool(draw_call.opts & DrawCallOpt::is_opaque) != opaque)
			continue;
		if(prev_mat_id != draw_call.material_id) {
			cmds.bindDS(1)(0, span<shader::SimpleDrawCall>(simple_dc_buf, dc, 1),
						   VDescriptorType::uniform_buffer)(1, sampler,
															material.diffuse_tex.vk_image);
			prev_mat_id = draw_call.material_id;
		}

		cmds.drawIndexed(draw_call.num_tris * 3, 1, draw_call.tri_offset * 3);
	}

	// TODO: what about ordering objects by material ?
	// TODO: add option to order objects in different ways ?
	// TODO: optional alpha test first for blended objects

	return {};
}

Ex<> SimpleRenderer::render(const RenderContext &ctx, bool wireframe) {
	auto &cmds = ctx.device.cmdQueue();
	PERF_GPU_SCOPE(cmds);

	auto lighting_buf = EX_PASS(VulkanBuffer::create<shader::Lighting>(
		ctx.device.ref(), 1, VBufferUsage::uniform_buffer | VBufferUsage::transfer_dst,
		VMemoryUsage::frame));
	auto simple_dc_buf = EX_PASS(VulkanBuffer::create<shader::SimpleDrawCall>(
		ctx.device.ref(), ctx.dcs.size(), VBufferUsage::uniform_buffer | VBufferUsage::transfer_dst,
		VMemoryUsage::frame));

	shader::Lighting lighting;
	lighting.ambient_color = ctx.lighting.ambient.color;
	lighting.ambient_power = ctx.lighting.ambient.power;
	lighting.scene_color = ctx.lighting.scene.color;
	lighting.scene_power = ctx.lighting.scene.power;
	lighting.sun_color = ctx.lighting.sun.color;
	lighting.sun_power = ctx.lighting.sun.power;
	lighting.sun_dir = ctx.lighting.sun.dir;
	EXPECT(cmds.upload(lighting_buf, cspan(&lighting, 1)));

	int num_opaque = 0;

	// TODO: minimize it (do it only for different materials)
	vector<shader::SimpleDrawCall> simple_dcs;
	simple_dcs.reserve(ctx.dcs.size());
	for(const auto &draw_call : ctx.dcs) {
		auto &material = ctx.materials[draw_call.material_id];
		auto &simple_dc = simple_dcs.emplace_back();
		if(draw_call.opts & DrawCallOpt::is_opaque)
			num_opaque++;
		simple_dc.world_camera_pos = ctx.camera.pos();
		simple_dc.proj_view_matrix = ctx.camera.matrix();
		simple_dc.material_color = float4(material.diffuse, material.opacity);
		simple_dc.draw_call_opts = uint(draw_call.opts.bits);
		if(draw_call.opts & DrawCallOpt::has_uv_rect) {
			simple_dc.uv_rect_pos = draw_call.uv_rect.min();
			simple_dc.uv_rect_size = draw_call.uv_rect.size();
		}
		if(material.diffuse_tex)
			; // TODO
	}
	EXPECT(cmds.upload(simple_dc_buf, simple_dcs));
	cmds.bind(m_pipeline_layout);
	cmds.bindDS(0)(0, lighting_buf, VDescriptorType::uniform_buffer);
	cmds.setViewport(m_viewport);
	cmds.setScissor(none);

	auto verts = ctx.verts;
	// TODO: fix this properly
	if(!verts.col)
		verts.col = verts.pos;
	if(!verts.tex)
		verts.tex = verts.pos;
	if(!verts.nrm)
		verts.nrm = verts.pos;
	cmds.bindVertices({verts.pos, verts.col, verts.tex, verts.nrm});
	cmds.bindIndices(ctx.tris_ib);

	auto swap_chain = ctx.device.swapChain();
	auto framebuffer = ctx.device.getFramebuffer({swap_chain->acquiredImage()}, m_depth_buffer);
	cmds.beginRenderPass(framebuffer, m_render_pass, none,
						 {FColor(0.0, 0.2, 0.0), VClearDepthStencil(1.0)});

	if(num_opaque > 0)
		EXPECT(renderPhase(ctx, simple_dc_buf, true, wireframe));
	if(num_opaque != ctx.dcs.size())
		EXPECT(renderPhase(ctx, simple_dc_buf, false, wireframe));

	cmds.endRenderPass();

	return {};
}
