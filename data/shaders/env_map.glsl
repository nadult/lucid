// Copyright (C) Krzysztof Jakubowski <nadult@fastmail.fm>
// This file is part of LucidRaster. See license.txt for details.

#version 460

#include "shared/definitions.glsl"
#include "shared/funcs.glsl"

layout(set = 0, binding = 0) uniform ubo10 { EnvMapDrawCall env_map_dc; };
layout(set = 0, binding = 1) uniform sampler2D env_map;

#ifdef VERTEX_SHADER // -------------------------------------------------------

layout(location = 0) in vec2 in_pos;
layout(location = 0) out vec3 v_posWS;

void main() {
	gl_Position = vec4(in_pos, 0.999999, 1.0);
	v_posWS = (vec4(in_pos, 1.0, 1.0) * env_map_dc.inv_proj_view_matrix).xyz;
}

#elif defined(FRAGMENT_SHADER) // ---------------------------------------------

layout(location = 0) in vec3 v_posWS;
layout(location = 0) out vec4 f_color;

vec2 longLat(vec3 normal) {
	// convert normal to longitude and latitude
	float latitude = acos(normal.y) / PI;
	float longitude = (atan(normal.x, normal.z) + PI) / (2.0 * PI);
	return vec2(longitude, latitude);
}

vec2 screenPos() { return gl_FragCoord.xy * env_map_dc.inv_screen_size * 2.0 - 1.0; }

vec3 screenToWorld(vec2 pos, float z) {
	vec4 wpos = env_map_dc.inv_proj_view_matrix * vec4(pos, z, 1.0);
	return wpos.xyz * (1.0 / wpos.w);
}

void main() {
	vec2 screen_pos = screenPos();
	vec3 pos1 = screenToWorld(screen_pos, -1.0);
	vec3 pos2 = screenToWorld(screen_pos, 1.0);
	vec3 dir = -normalize(pos2 - pos1);

	vec2 tex_coord = longLat(dir);
	f_color.rgb = texture(env_map, tex_coord * vec2(1.0, -1.0)).xyz * 0.5;
	f_color.a = 1.0;
}

#endif
