// $$include funcs

struct Viewport {
	mat4 proj_matrix;
	vec2 size, inv_size;
	float near_plane, far_plane;
	float inv_far_plane;
};


// Spaces:
//
//  View: (0, 0, 0) is camera position, z is in range -(near_plane, far_plane)
//        TODO: why is it negated?
//   NDC: (-1, -1, -1) - (1, 1, 1); near plane is at (0, 0, 1), far plane at (0, 0, -1)
//
// World * view_matrix -> View
//  View * proj_matrix -> Clip
//  Clip / w           -> NDC

uniform Viewport viewport;
	
float zndcToView(float zndc) {
	return (zndc * viewport.proj_matrix[3][3] - viewport.proj_matrix[3][2]) /
		   (zndc * viewport.proj_matrix[2][3] - viewport.proj_matrix[2][2]);
}

float decodeZView(vec2 xy) {
	return -decodeFloat2(xy) * viewport.far_plane;
}

vec2 encodeZView(float z) {
	return encodeFloat2(-z * viewport.inv_far_plane);
}

float depthToZView(float depth_value) {
	float zndc = 2.0 * depth_value - 1.0;
	return -zndcToView(zndc);
}

float zndcToDepth(float zndc) {
	return (-zndc + 1.0) * 0.5;
}
