#include "simple_renderer.h"

#include "scene.h"
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
									 const IRect &viewport, VColorAttachment color_att,
									 VDepthAttachment depth_att) {
	vector<string> geom_locations{"in_pos", "in_color", "in_tex_coord", "in_normal"};
	ShaderDefs defs;
	m_viewport = viewport;
	//m_program = EX_PASS(Program::make("simple", defs + "SIMPLE_DRAW_CALL", geom_locations));

	auto fsh_bytecode = compiler.getSpirv("simple_frag");
	auto vsh_bytecode = compiler.getSpirv("simple_vert");

	auto fsh_module = EX_PASS(VulkanShaderModule::create(device, fsh_bytecode));
	auto vsh_module = EX_PASS(VulkanShaderModule::create(device, vsh_bytecode));

	VPipelineSetup setup;
	setup.render_pass = device->getRenderPass({color_att}, depth_att);
	setup.vertex_bindings = {{vertexBinding<float3>(0), vertexBinding<IColor>(1),
							  vertexBinding<float2>(2), vertexBinding<u32>(3)}};
	setup.vertex_attribs = {{vertexAttrib<float3>(0, 0), vertexAttrib<IColor>(1, 1),
							 vertexAttrib<float2>(2, 2), vertexAttrib<u32>(3, 3)}};
	setup.shader_modules = {{vsh_module, fsh_module}};
	setup.depth = VDepthSetup(VDepthFlag::test | VDepthFlag::write);

	// TODO: culling

	m_pipelines.resize(6);

	VBlendingMode additive_blend(VBlendFactor::one, VBlendFactor::one);
	VBlendingMode normal_blend(VBlendFactor::src_alpha, VBlendFactor::one_minus_src_alpha);
	for(int wireframe = 0; wireframe <= 1; wireframe++) {
		setup.raster = VRasterSetup(VPrimitiveTopology::triangle_list,
									wireframe ? VPolygonMode::line : VPolygonMode::fill);

		setup.blending.attachments = {};
		m_pipelines[pipelineId(true, wireframe, false)] =
			EX_PASS(VulkanPipeline::create(device, setup));

		setup.blending.attachments = {{normal_blend}};
		m_pipelines[pipelineId(false, wireframe, false)] =
			EX_PASS(VulkanPipeline::create(device, setup));

		setup.blending.attachments = {{additive_blend}};
		m_pipelines[pipelineId(false, wireframe, true)] =
			EX_PASS(VulkanPipeline::create(device, setup));
	}

	return {};
}

Matrix3 normalMatrix(const Matrix4 &affine) {
	Matrix3 out(affine[0].xyz(), affine[1].xyz(), affine[2].xyz());
	return transpose(inverse(out));
}

void SimpleRenderer::renderPhase(const RenderContext &ctx, bool opaque, bool wireframe) {
	auto &cmds = ctx.device.cmdQueue();
	PERF_GPU_SCOPE(cmds);

	auto pipeline = m_pipelines[pipelineId(opaque, wireframe, ctx.config.additive_blending)];

	auto swap_chain = ctx.device.swapChain();
	VColorAttachment color_att(swap_chain->format(), 1, VColorSyncStd::clear);
	VDepthAttachment depth_att(*ctx.depth_buffer->depthStencilFormat());
	auto render_pass = ctx.device.getRenderPass({{color_att}}, depth_att);
	//auto fb = ctx.device.getFramebuffer({swap_chain->acquiredImage()}, ctx.depth_buffer);

	/*
	ctx.lighting.setUniforms(m_program.glProgram());
	m_program.use();
	m_program["world_camera_pos"] = ctx.camera.pos();
	m_program["proj_view_matrix"] = ctx.camera.matrix();

	// TODO: what about ordering objects by material ?
	// TODO: add option to order objects in different ways ?
	// TODO: optional alpha test first for blended objects

	int prev_mat_id = -1;
	for(const auto &draw_call : ctx.dcs) {
		auto &material = ctx.materials[draw_call.material_id];
		if(bool(draw_call.opts & DrawCallOpt::is_opaque) != opaque)
			continue;
		if(prev_mat_id != draw_call.material_id) {
			m_program["material_color"] = float4(material.diffuse, material.opacity);
			if(material.diffuse_tex)
				material.diffuse_tex.gl_handle->bind(0);
			prev_mat_id = draw_call.material_id;
		}

		m_program["draw_call_opts"] = uint(draw_call.opts.bits);
		if(draw_call.opts & DrawCallOpt::has_uv_rect) {
			m_program["uv_rect_pos"] = draw_call.uv_rect.min();
			m_program["uv_rect_size"] = draw_call.uv_rect.size();
		}
		ctx.vao->draw(PrimitiveType::triangles, draw_call.num_tris * 3, draw_call.tri_offset * 3);
	}*/
}

void SimpleRenderer::render(const RenderContext &ctx, bool wireframe) {
	auto &cmds = ctx.device.cmdQueue();
	PERF_GPU_SCOPE(cmds);
	/*glViewport(m_viewport.x(), m_viewport.y(), m_viewport.width(), m_viewport.height());

	if(wireframe)
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	if(ctx.config.backface_culling)
		glEnable(GL_CULL_FACE);
	else
		glDisable(GL_CULL_FACE);

	renderPhase(ctx, true);
	renderPhase(ctx, false);

	if(wireframe)
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	GlTexture::unbind();*/
}
