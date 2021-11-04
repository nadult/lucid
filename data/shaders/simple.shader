// $$include funcs lighting macros

uniform vec3 world_camera_pos;
uniform mat4 proj_view_matrix;

uniform uint draw_call_opts;
uniform vec4 material_color;

uniform bool use_uv_rect;
uniform vec2 uv_rect_pos;
uniform vec2 uv_rect_size;

#define FLAG_SET(flag) ((draw_call_opts & flag) != 0)

#ifdef VERTEX_SHADER // -------------------------------------------------------

in vec3 in_pos;
in vec4 in_color;
in vec2 in_tex_coord;
in uint in_normal;

out vec2 v_tex_coord;
out vec4 v_color;
out vec3 v_posWS;
out vec3 v_normalWS;

void main() {
	vec3 posWS = vec4(in_pos, 1.0).xyz;
	gl_Position = proj_view_matrix * vec4(posWS, 1.0);
	v_posWS = posWS;

	v_color = FLAG_SET(INST_HAS_VERTEX_COLORS)? in_color : vec4(1, 1, 1, 1);
	if(FLAG_SET(INST_HAS_VERTEX_TEX_COORDS))
		v_tex_coord = in_tex_coord;
	if(FLAG_SET(INST_HAS_VERTEX_NORMALS))
		v_normalWS = decodeNormalUint(in_normal);
}

#elif defined(FRAGMENT_SHADER) // ---------------------------------------------

in vec2 v_tex_coord;
in vec4 v_color;
in vec3 v_posWS;
in vec3 v_normalWS;

out vec4 f_color;

layout(binding = 0) uniform sampler2D color_tex;
layout(binding = 1) uniform sampler2D normal_tex;

void main() {
	vec3 normalWS;
	if(FLAG_SET(INST_HAS_VERTEX_NORMALS)) {
		normalWS = v_normalWS;
	}
	else {
		// Flat shading if no normal data is available
		normalWS = normalize(cross(dFdx(v_posWS), dFdy(v_posWS)));
	}

	vec4 color = material_color * v_color;
	if(FLAG_SET(INST_HAS_TEXTURE)) {
		vec2 tex_coord = v_tex_coord;
		vec2 tex_dx = dFdx(v_tex_coord);
		vec2 tex_dy = dFdy(v_tex_coord);
		if(FLAG_SET(INST_HAS_UV_RECT)) {
			// TODO: all textures need borders, even if POW2?
			tex_dx *= uv_rect_size;
			tex_dy *= uv_rect_size;
			tex_coord = uv_rect_pos + uv_rect_size * fract(tex_coord);
		}
		color *= textureGrad(color_tex, tex_coord, tex_dx, tex_dy);
	}

	float light_value = max(0.0, dot(-lighting.sun_dir, normalWS) * 0.7 + 0.3);
	f_color.rgb = finalShading(color.rgb, light_value);
	f_color.a = color.a;
}

#endif
