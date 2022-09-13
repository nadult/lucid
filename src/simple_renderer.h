#pragma once

#include "lucid_base.h"
#include <fwk/gfx/color.h>
#include <fwk/math/box.h>

namespace shader {
struct SimpleDrawCall; // TODO: change name to diff from fwk
}

class SimpleRenderer {
  public:
	SimpleRenderer();
	FWK_MOVABLE_CLASS(SimpleRenderer);

	static void addShaderDefs(VulkanDevice &, ShaderCompiler &);

	Ex<> exConstruct(VDeviceRef, ShaderCompiler &, const IRect &viewport, VColorAttachment);

	// TODO: wireframe to config
	Ex<> render(const RenderContext &, bool wireframe);

	const IRect &viewport() const { return m_viewport; }

  private:
	Ex<PVPipeline> getPipeline(const RenderContext &, bool opaque, bool wireframe) const;
	const SceneMaterial &bindMaterial(const RenderContext &, int mat_id);
	Ex<> renderPhase(const RenderContext &, VBufferSpan<shader::SimpleDrawCall>, bool opaque,
					 bool wireframe);

	PVShaderModule m_frag_module, m_vert_module;
	PVPipelineLayout m_pipeline_layout;
	PVImageView m_depth_buffer;
	PVRenderPass m_render_pass;
	IRect m_viewport;
};
