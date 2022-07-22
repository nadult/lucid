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

	Ex<void> exConstruct(VDeviceRef, ShaderCompiler &, const IRect &viewport, VColorAttachment,
						 VDepthAttachment);

	void render(const RenderContext &, bool wireframe);

	const IRect &viewport() const { return m_viewport; }

	static uint pipelineId(bool opaque, bool wireframe, bool additive) {
		return (opaque ? 0 : additive ? 2 : 1) + (wireframe ? 3 : 0);
	}

  private:
	const SceneMaterial &bindMaterial(const RenderContext &, int mat_id);
	void renderPhase(const RenderContext &, bool opaque, bool wireframe);

	vector<PVPipeline> m_pipelines;
	Program m_program;
	IRect m_viewport;
};
