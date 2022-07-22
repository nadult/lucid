#pragma once

#include "program.h"
#include <fwk/gfx/color.h>
#include <fwk/gfx/material.h>
#include <fwk/gfx/matrix_stack.h>
#include <fwk/math/box.h>

class SimpleRenderer {
  public:
	SimpleRenderer();
	FWK_MOVABLE_CLASS(SimpleRenderer);

	static void addShaderDefs(ShaderCompiler &);

	Ex<void> exConstruct(VDeviceRef, ShaderCompiler &, const IRect &viewport, VColorAttachment);

	// TODO: wireframe to config
	Ex<> render(const RenderContext &, bool wireframe);

	const IRect &viewport() const { return m_viewport; }

  private:
	Ex<PVPipeline> getPipeline(const RenderContext &, bool opaque, bool wireframe) const;
	const SceneMaterial &bindMaterial(const RenderContext &, int mat_id);
	Ex<> renderPhase(const RenderContext &, PVBuffer, bool opaque, bool wireframe);

	PVShaderModule m_frag_module, m_vert_module;
	PVPipelineLayout m_pipeline_layout;
	PVImageView m_depth_buffer;
	PVRenderPass m_clear_rpass, m_draw_rpass;
	IRect m_viewport;
};
