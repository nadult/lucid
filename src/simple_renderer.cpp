#include "simple_renderer.h"

#include "scene.h"
#include "shading.h"
#include <fwk/gfx/camera.h>
#include <fwk/gfx/colored_triangle.h>
#include <fwk/gfx/draw_call.h>
#include <fwk/gfx/gl_buffer.h>
#include <fwk/gfx/gl_device.h>
#include <fwk/gfx/gl_program.h>
#include <fwk/gfx/gl_shader.h>
#include <fwk/gfx/gl_texture.h>
#include <fwk/gfx/gl_vertex_array.h>
#include <fwk/gfx/opengl.h>
#include <fwk/gfx/shader_defs.h>
#include <fwk/hash_map.h>
#include <fwk/vulkan/vulkan_device.h>

SimpleRenderer::SimpleRenderer() = default;
FWK_MOVABLE_CLASS_IMPL(SimpleRenderer);

Ex<void> SimpleRenderer::exConstruct(const IRect &viewport) {
	vector<string> geom_locations{"in_pos", "in_color", "in_tex_coord", "in_normal"};
	ShaderDefs defs;
	m_viewport = viewport;
	m_program = EX_PASS(Program::make("simple", defs + "SIMPLE_DRAW_CALL", geom_locations));
	return {};
}

Matrix3 normalMatrix(const Matrix4 &affine) {
	Matrix3 out(affine[0].xyz(), affine[1].xyz(), affine[2].xyz());
	return transpose(inverse(out));
}

void SimpleRenderer::renderPhase(const RenderContext &ctx, bool opaque) {
	auto &cmds = ctx.device.cmdQueue();
	PERF_GPU_SCOPE(cmds);

	if(opaque) {
		glDisable(GL_BLEND);
		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LESS);
		glDepthMask(1);
	} else {
		glEnable(GL_BLEND);
		auto dst_factor = ctx.config.additive_blending ? GL_ONE : GL_ONE_MINUS_SRC_ALPHA;
		glBlendFuncSeparate(GL_SRC_ALPHA, dst_factor, GL_ONE, dst_factor);
		glEnable(GL_DEPTH_TEST);
		glDepthMask(0);
	}

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
	}
}

void SimpleRenderer::render(const RenderContext &ctx, bool wireframe) {
	auto &cmds = ctx.device.cmdQueue();
	PERF_GPU_SCOPE(cmds);
	glViewport(m_viewport.x(), m_viewport.y(), m_viewport.width(), m_viewport.height());

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

	GlTexture::unbind();
}
