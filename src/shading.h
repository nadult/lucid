#pragma once

#include "lucid_base.h"

#include <fwk/enum_map.h>

namespace shader {
struct Frustum;
struct Viewport;
struct Rect;
struct Lighting;
};

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
	operator shader::Lighting() const;

	SimpleLight ambient;
	SimpleLight scene;
	SunLight sun;
};

struct FrustumInfo {
	FrustumInfo() = default;
	FrustumInfo(const Camera &);
	operator shader::Frustum() const;

	array<float3, 4> origins;
	array<float3, 4> dirs;

	float3 dir0, origin0;
	float3 dirx, diry;
};

shader::Viewport makeViewport(const Camera &cam, int2 viewport_size);
shader::Rect makeRect(FRect rect);
