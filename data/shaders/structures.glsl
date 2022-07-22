#ifndef _STRUCTURES_GLSL_
#define _STRUCTURES_GLSL_

struct Lighting {
	vec3 ambient_color;
	vec3 scene_color;
	vec3 sun_color;
	vec3 sun_dir;
	float scene_power, sun_power, ambient_power;
};

struct Frustum {
	// World space rays;
	// Ray indices are compatible with rect vertex indices
	vec3 ws_origin[4];
	vec3 ws_dir[4];

	vec3 ws_dir0, ws_dirx, ws_diry;
	vec3 ws_shared_origin;

	// For computation from screen_pos
	vec2 vs_pos, vs_diff;
	vec3 ws_pos, ws_diff;
};

struct SimpleDrawCall {
	mat4 proj_view_matrix;
	vec4 material_color;
	vec2 uv_rect_pos;
	vec2 uv_rect_size;
	vec3 world_camera_pos;
	uint draw_call_opts;
};

struct Rect {
	vec2 pos, size;
	vec2 min_uv, max_uv;
};

#define LIGHTING_STRUCT_SIZE 15
#define SIMPLE_DRAW_CALL_STRUCT_SIZE 28

#endif