// Copyright (C) Krzysztof Jakubowski <nadult@fastmail.fm>
// This file is part of LucidRaster. See license.txt for details.

#include "shading.h"

#include <fwk/gfx/camera.h>
#include <fwk/io/stream.h>
#include <fwk/math/frustum.h>
#include <fwk/math/ray.h>

#include "shader_structs.h"

SceneLighting SceneLighting::makeDefault() {
	SceneLighting out;
	out.sun.dir = {0.842121, -0.300567, -0.447763};
	out.sun.color = {0.8, 0.8, 0.8};
	out.sun.power = 2.5;
	out.ambient.color = {0.8, 0.8, 0.6};
	out.ambient.power = 0.4f;
	return out;
}

SceneLighting::operator shader::Lighting() const {
	shader::Lighting out;
	out.ambient_color = float4(ambient.color, 1.0);
	out.sun_color = float4(sun.color, 1.0);
	out.sun_dir = float4(sun.dir, 0.0);
	out.ambient_power = ambient.power;
	out.sun_power = sun.power;
	return out;
}

FrustumInfo::FrustumInfo(const Camera &camera) {
	auto params = camera.params();
	auto iview = inverseOrZero(camera.viewMatrix());
	auto rays = fwk::Frustum(camera.projectionMatrix()).cornerRays();

	for(int i : intRange(dirs)) {
		origins[i] = rays[i].origin();
		dirs[i] = rays[i].dir();
		origins[i] = mulPoint(iview, origins[i]);
		dirs[i] = mulNormal(iview, dirs[i]);
	}
	origin0 = origins[0];
	dir0 = dirs[0];
	dirx = (dirs[3] - dirs[0]) * (1.0f / params.viewport.width());
	diry = (dirs[1] - dirs[0]) * (1.0f / params.viewport.height());
}

FrustumInfo::operator shader::Frustum() const {
	shader::Frustum out;
	for(int i : intRange(dirs)) {
		out.ws_dirs[i] = float4(dirs[i], 0.0);
		out.ws_origins[i] = float4(origins[i], 0.0);
	}
	out.ws_origin0 = float4(origin0, 1.0);
	out.ws_dir0 = float4(dir0, 0.0);
	out.ws_dirx = float4(dirx, 0.0);
	out.ws_diry = float4(diry, 0.0);
	return out;
}

shader::Viewport makeViewport(const Camera &cam, int2 viewport_size) {
	shader::Viewport out;
	// TODO: add view_matrix & view_proj_matrix ?
	out.proj_matrix = cam.projectionMatrix();
	out.near_plane = cam.params().depth.min;
	out.far_plane = cam.params().depth.max;
	out.inv_far_plane = 1.0f / cam.params().depth.max;
	out.size = float2(viewport_size);
	out.inv_size = vinv(float2(viewport_size));
	return out;
}

shader::Rect makeRect(FRect rect) {
	shader::Rect out;
	out.pos = rect.min();
	out.size = rect.size();
	out.min_uv = (rect.min() + float2(1.0f, 1.0f)) * 0.5f;
	out.max_uv = (rect.max() + float2(1.0f, 1.0f)) * 0.5f;
	return out;
}
