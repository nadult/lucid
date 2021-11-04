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

uniform Frustum frustum;

// TODO: optimization available in ortho mode
//       but shaders which use it would HAVE to be available in two versions...

// TODO: z is in what space?
vec3 frustumGetViewPos(vec2 screen_pos, float z) {
	return vec3((frustum.vs_pos + screen_pos * frustum.vs_diff) * z, z);
}

void frustumGetScreenRay(vec2 screen_pos, out vec3 origin, out vec3 dir) {
	// TODO: slow
	vec3 orig0 = mix(frustum.ws_origin[0], frustum.ws_origin[1], screen_pos.y);
	vec3 orig1 = mix(frustum.ws_origin[3], frustum.ws_origin[2], screen_pos.y);
	origin = mix(orig0, orig1, screen_pos.x);
	origin = frustum.ws_origin[0]; // TODO: won't work for orthogonal projection

	vec3 dir0 = mix(frustum.ws_dir[0], frustum.ws_dir[1], screen_pos.y);
	vec3 dir1 = mix(frustum.ws_dir[3], frustum.ws_dir[2], screen_pos.y);
	dir = normalize(mix(dir0, dir1, screen_pos.x));
}

vec3 frustumGetWorldPos(vec2 screen_pos, float z_view) {
	vec3 origin, dir;
	frustumGetScreenRay(screen_pos, origin, dir);
	return origin + dir * z_view;
}
