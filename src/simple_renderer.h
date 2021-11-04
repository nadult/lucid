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
	Ex<void> exConstruct(const IRect &viewport);

	void render(const RenderContext &, bool wireframe);

	const IRect &viewport() const { return m_viewport; }

  private:
	const SceneMaterial &bindMaterial(const RenderContext &, int mat_id);
	void renderPhase(const RenderContext &, bool opaque);

	Program m_program;
	IRect m_viewport;
};
