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
	struct PipeConfig {
		FWK_ORDER_BY(PipeConfig, backface_culling, additive_blending, opaque, wireframe);

		bool backface_culling;
		bool additive_blending;
		bool opaque;
		bool wireframe;
	};

	Ex<PVPipeline> getPipeline(VulkanDevice &, const PipeConfig &);
	const SceneMaterial &bindMaterial(const RenderContext &, int mat_id);
	Ex<> renderPhase(const RenderContext &, VBufferSpan<shader::SimpleDrawCall>, bool opaque,
					 bool wireframe);

	HashMap<PipeConfig, PVPipeline> m_pipelines;
	PVShaderModule m_frag_module, m_vert_module;
	PVPipelineLayout m_pipeline_layout;
	PVImageView m_depth_buffer;
	PVRenderPass m_render_pass;
	IRect m_viewport;
};
