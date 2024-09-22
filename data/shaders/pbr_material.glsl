// Copyright (C) Krzysztof Jakubowski <nadult@fastmail.fm>
// This file is part of LucidRaster. See license.txt for details.

#version 460

#include "shared/definitions.glsl"
#include "shared/funcs.glsl"

layout(set = 0, binding = 0) uniform ubo00 { Lighting lighting; };
layout(set = 1, binding = 0) uniform ubo10 { PbrDrawCall draw_call; };

layout(set = 1, binding = 1) uniform sampler2D albedo_tex;
layout(set = 1, binding = 2) uniform sampler2D normal_tex;
layout(set = 1, binding = 3) uniform sampler2D pbr_tex;

bool flagSet(uint flag) { return (draw_call.draw_call_opts & flag) != 0; }

#ifdef VERTEX_SHADER // -------------------------------------------------------

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec4 in_color;
layout(location = 2) in vec2 in_tex_coord;
layout(location = 3) in uint in_normal;
layout(location = 4) in uint in_tangent;

layout(location = 0) out vec2 v_tex_coord;
layout(location = 1) out vec4 v_color;
layout(location = 2) out vec3 v_posWS;
layout(location = 3) out vec3 v_normalWS;
layout(location = 4) out vec3 v_tangentWS;
layout(location = 5) out vec3 v_binormalWS;

void main() {
	vec3 posWS = vec4(in_pos, 1.0).xyz;
	gl_Position = draw_call.proj_view_matrix * vec4(posWS, 1.0);
	v_posWS = posWS;

	v_color = flagSet(INST_HAS_VERTEX_COLORS) ? in_color : vec4(1, 1, 1, 1);

	v_tex_coord = in_tex_coord;
	v_normalWS = decodeNormalUint(in_normal);
	v_tangentWS = decodeNormalUint(in_tangent);
	v_binormalWS = cross(v_tangentWS, v_normalWS);
}

#elif defined(FRAGMENT_SHADER) // ---------------------------------------------

layout(location = 0) in vec2 v_tex_coord;
layout(location = 1) in vec4 v_color;
layout(location = 2) in vec3 v_posWS;
layout(location = 3) in vec3 v_normalWS;
layout(location = 4) in vec3 v_tangentWS;
layout(location = 5) in vec3 v_binormalWS;

layout(location = 0) out vec4 f_color;

void main() {
	vec3 normal = v_normalWS;
	if(flagSet(INST_HAS_NORMAL_TEXTURE)) {
		vec3 normal_map = texture(normal_tex, v_tex_coord).rgb;
		normal_map = normal_map * 2.0 - 1.0;
		normal =
			normal_map.x * v_tangentWS + normal_map.y * v_binormalWS + normal_map.z * v_normalWS;
		normal = normalize(normal);
	}

	// roughness, metallic, ao
	vec3 pbr = vec3(1.0, 1.0, 1.0);
	if(flagSet(INST_HAS_PBR_TEXTURE)) {
		pbr = texture(pbr_tex, v_tex_coord).rgb;
	}

	vec4 color = draw_call.material_color * v_color;
	if(flagSet(INST_HAS_ALBEDO_TEXTURE)) {
		color *= texture(albedo_tex, v_tex_coord);
	}

	float light_value = max(0.0, dot(-lighting.sun_dir.xyz, normal) * 0.5 + 0.5) * pbr.z;
	f_color.rgb = finalShading(lighting, color.rgb, light_value);
	f_color.a = color.a;
}

#endif
