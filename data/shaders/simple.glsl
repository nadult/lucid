#include "funcs.glsl"
#include "macros.glsl"

layout(set = 0, binding = 0) uniform ubo00 { Lighting lighting; };

layout(set = 1, binding = 0) uniform ubo10 { SimpleDrawCall simple_dc; };
layout(set = 1, binding = 1) uniform sampler2D color_tex;

bool flagSet(uint flag) { return (simple_dc.draw_call_opts & flag) != 0; }

#ifdef VERTEX_SHADER // -------------------------------------------------------

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec4 in_color;
layout(location = 2) in vec2 in_tex_coord;
layout(location = 3) in uint in_normal;

layout(location = 0) out vec2 v_tex_coord;
layout(location = 1) out vec4 v_color;
layout(location = 2) out vec3 v_posWS;
layout(location = 3) out vec3 v_normalWS;

void main() {
	vec3 posWS = vec4(in_pos, 1.0).xyz;
	gl_Position = simple_dc.proj_view_matrix * vec4(posWS, 1.0);
	v_posWS = posWS;

	v_color = flagSet(INST_HAS_VERTEX_COLORS) ? in_color : vec4(1, 1, 1, 1);
	if(flagSet(INST_HAS_VERTEX_TEX_COORDS))
		v_tex_coord = in_tex_coord;
	if(flagSet(INST_HAS_VERTEX_NORMALS))
		v_normalWS = decodeNormalUint(in_normal);
}

#elif defined(FRAGMENT_SHADER) // ---------------------------------------------

layout(location = 0) in vec2 v_tex_coord;
layout(location = 1) in vec4 v_color;
layout(location = 2) in vec3 v_posWS;
layout(location = 3) in vec3 v_normalWS;

layout(location = 0) out vec4 f_color;

void main() {
	vec3 normalWS;
	if(flagSet(INST_HAS_VERTEX_NORMALS)) {
		normalWS = v_normalWS;
	} else {
		// Flat shading if no normal data is available
		normalWS = normalize(cross(dFdx(v_posWS), dFdy(v_posWS)));
	}

	vec4 color = simple_dc.material_color * v_color;
	if(flagSet(INST_HAS_TEXTURE)) {
		vec2 tex_coord = v_tex_coord;
		vec2 tex_dx = dFdx(v_tex_coord);
		vec2 tex_dy = dFdy(v_tex_coord);
		if(flagSet(INST_HAS_UV_RECT)) {
			// TODO: all textures need borders, even if POW2?
			tex_dx *= simple_dc.uv_rect_size;
			tex_dy *= simple_dc.uv_rect_size;
			tex_coord = simple_dc.uv_rect_pos + simple_dc.uv_rect_size * fract(tex_coord);
		}
		color *= textureGrad(color_tex, tex_coord, tex_dx, tex_dy);
	}

	float light_value = max(0.0, dot(-lighting.sun_dir, normalWS) * 0.7 + 0.3);
	f_color.rgb = finalShading(lighting, color.rgb, light_value);
	f_color.a = color.a;
}

#endif
