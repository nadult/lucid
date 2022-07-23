#ifndef _STRUCTURES_GLSL_
#define _STRUCTURES_GLSL_

struct Lighting {
	vec4 ambient_color;
	vec4 scene_color;
	vec4 sun_color;
	vec4 sun_dir;
	float scene_power, sun_power, ambient_power;
};

struct Frustum {
	// World space rays;
	// Ray indices are compatible with rect vertex indices
	vec4 ws_origin[4];
	vec4 ws_dir[4];

	vec4 ws_dir0, ws_dirx, ws_diry;
	vec4 ws_shared_origin;

	// For computation from screen_pos
	vec2 vs_pos, vs_diff;
	vec4 ws_pos, ws_diff;
};

struct SimpleDrawCall {
	mat4 proj_view_matrix;
	vec4 material_color;
	vec2 uv_rect_pos;
	vec2 uv_rect_size;
	uint draw_call_opts;
	vec4 world_camera_pos;
};

struct Rect {
	vec2 pos, size;
	vec2 min_uv, max_uv;
};

#endif