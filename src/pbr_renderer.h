// Copyright (C) Krzysztof Jakubowski <nadult@fastmail.fm>
// This file is part of LucidRaster. See license.txt for details.

#pragma once

#include "lucid_base.h"
#include <fwk/gfx/color.h>
#include <fwk/gfx/shader_compiler.h>
#include <fwk/math/box.h>

namespace shader {
struct PbrDrawCall;
struct Lighting;
}

class PbrRenderer {
  public:
	PbrRenderer();
	FWK_MOVABLE_CLASS(PbrRenderer);

	static void addShaderDefs(VulkanDevice &, ShaderCompiler &, const ShaderConfig &);
	CSpan<ShaderDefId> shaderDefIds() const { return m_shader_def_ids; }
	Ex<> exConstruct(VulkanDevice &, ShaderCompiler &, const IRect &viewport, VAttachment);

	// TODO: wireframe to config
	Ex<> render(const RenderContext &, bool wireframe);
	const IRect &viewport() const { return m_viewport; }

  private:
	struct PipeConfig {
		FWK_ORDER_BY(PipeConfig, backface_culling, additive_blending, opaque, wireframe, pbr);

		bool backface_culling;
		bool additive_blending;
		bool opaque;
		bool wireframe;
		bool pbr;
	};

	Ex<PVPipeline> getPipeline(VulkanDevice &, const PipeConfig &);
	Ex<> renderPhase(const RenderContext &, VBufferSpan<shader::PbrDrawCall>, bool opaque,
					 bool wireframe);
	Ex<> renderEnvMap(const RenderContext &);

	vector<ShaderDefId> m_shader_def_ids;
	HashMap<PipeConfig, PVPipeline> m_pipelines;
	PVShaderModule m_frag_module, m_vert_module;
	PVPipelineLayout m_pipeline_layout;
	PVPipeline m_env_pipeline;
	PVImageView m_depth_buffer;
	PVRenderPass m_render_pass;
	VBufferSpan<float2> m_rect_vertices;
	VBufferSpan<shader::Lighting> m_lighting_buf;
	IRect m_viewport;
};
