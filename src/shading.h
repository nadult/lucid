#pragma once

#include "lucid_base.h"

#include <fwk/enum_map.h>

struct SunLight {
	float3 dir = {1, 0, 0}, color{1};
	float power = 2.0f;
};

struct SimpleLight {
	float3 color{1};
	float power = 0.5f;
};

struct SceneLighting {
	static SceneLighting makeDefault();
	void setUniforms(PProgram) const;

	SimpleLight ambient;
	SimpleLight scene;
	SunLight sun;
};
