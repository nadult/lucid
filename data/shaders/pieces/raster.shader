// $$include structures

#define LIX gl_LocalInvocationIndex
#define LID gl_LocalInvocationID

layout(std430, binding = 1) readonly buffer buf1_ { uvec4 g_tri_aabbs[]; };
layout(std430, binding = 2) readonly buffer buf2_ { uint g_quad_indices[]; };

layout(std430, binding = 3) readonly buffer buf3_ { float g_verts[]; };
layout(std430, binding = 4) readonly buffer buf4_ { vec2 g_tex_coords[]; };
layout(std430, binding = 5) readonly buffer buf5_ { uint g_colors[]; };
layout(std430, binding = 6) readonly buffer buf6_ { uint g_normals[]; };

layout(std430, binding = 8) buffer buf8_ { uint g_bin_quads[]; };

layout(std430, binding = 9) coherent buffer buf9_ { uint g_scratch_32[]; };
layout(std430, binding = 10) coherent buffer buf10_ { uvec2 g_scratch_64[]; };

layout(std430, binding = 11) readonly buffer buf11_ { InstanceData g_instances[]; };
layout(std430, binding = 12) readonly buffer buf12_ { vec4 g_uv_rects[]; };
layout(std430, binding = 13) writeonly buffer buf13_ { uint g_raster_image[]; };
layout(std430, binding = 14) readonly restrict buffer buf14_ { uvec2 g_tri_storage[]; };
layout(std430, binding = 15) readonly restrict buffer buf15_ { uvec4 g_quad_storage[]; };
layout(std430, binding = 16) readonly restrict buffer buf16_ { uvec4 g_scan_storage[]; };

layout(binding = 0) uniform sampler2D opaque_texture;
layout(binding = 1) uniform sampler2D transparent_texture;

// TODO: separate opaque and transparent objects, draw opaque objects first to texture
// then read it and use depth to optimize drawing
uniform uint background_color;

#ifdef ENABLE_TIMERS
shared uint s_timers[TIMERS_COUNT];
#define INIT_CLOCK() uint64_t clock0 = clockARB();
#define UPDATE_CLOCK(idx)                                                                          \
	if((LIX & 31) == 0) {                                                                          \
		uint64_t clock = clockARB();                                                               \
		atomicAdd(s_timers[idx], uint(clock - clock0) >> 4);                                       \
		clock0 = clock;                                                                            \
	}

void initTimers() {
	if(LIX < TIMERS_COUNT)
		s_timers[LIX] = 0;
}
void commitTimers() {
	if(LIX < TIMERS_COUNT)
		atomicAdd(g_info.timers[LIX], s_timers[LIX]);
}

#else
#define INIT_CLOCK()
#define UPDATE_CLOCK(idx)

void initTimers() {}
void commitTimers() {}
#endif

// Algorithm inspired by Nanite scanline rasterizer
void computeScanlineParams(vec3 tri0, vec3 tri1, vec3 tri2, vec2 start, out vec3 scan_min,
						   out vec3 scan_max, out vec3 scan_step) {
	vec3 nrm0 = cross(tri2, tri1 - tri2);
	vec3 nrm1 = cross(tri0, tri2 - tri0);
	vec3 nrm2 = cross(tri1, tri0 - tri1);
	float volume = dot(tri0, nrm0);
	if(volume < 0)
		nrm0 = -nrm0, nrm1 = -nrm1, nrm2 = -nrm2;

	vec3 edges[3] = {
		vec3(dot(nrm0, frustum.ws_dirx), dot(nrm0, frustum.ws_diry), dot(nrm0, frustum.ws_dir0)),
		vec3(dot(nrm1, frustum.ws_dirx), dot(nrm1, frustum.ws_diry), dot(nrm1, frustum.ws_dir0)),
		vec3(dot(nrm2, frustum.ws_dirx), dot(nrm2, frustum.ws_diry), dot(nrm2, frustum.ws_dir0)),
	};

	float inv_ex[3] = {1.0 / edges[0].x, 1.0 / edges[1].x, 1.0 / edges[2].x};
	vec3 scan_base = -vec3(edges[0].z * inv_ex[0], edges[1].z * inv_ex[1], edges[2].z * inv_ex[2]);
	scan_step = -vec3(edges[0].y * inv_ex[0], edges[1].y * inv_ex[1], edges[2].y * inv_ex[2]);

	bool sign0 = edges[0].x < 0.0;
	bool sign1 = edges[1].x < 0.0;
	bool sign2 = edges[2].x < 0.0;

	vec3 scan = scan_step * start.y + scan_base - vec3(start.x);
	scan_min = vec3(sign0 ? -1.0 / 0.0 : scan[0], sign1 ? -1.0 / 0.0 : scan[1],
					sign2 ? -1.0 / 0.0 : scan[2]);
	scan_max =
		vec3(sign0 ? scan[0] : 1.0 / 0.0, sign1 ? scan[1] : 1.0 / 0.0, sign2 ? scan[2] : 1.0 / 0.0);
}