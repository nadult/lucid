// $$include funcs frustum structures

// ~80% of time goes to data loading

#define LSIZE MAX_INSTANCE_QUADS
layout(local_size_x = LSIZE) in;

#define LID gl_LocalInvocationID
#define LIX gl_LocalInvocationIndex
#define WGID gl_WorkGroupID

#define MAX_PACKET_SIZE 4

uniform mat4 view_proj_matrix;
uniform bool enable_backface_culling;
uniform int u_num_instances;
uniform int u_packet_size;

// TODO: check if readonly/restrict makes a difference
layout(std430, binding = 1) readonly restrict buffer buf1_ { uint g_input_indices[]; };
layout(std430, binding = 2) readonly restrict buffer buf2_ { InstanceData g_instances[]; };
layout(std430, binding = 3) readonly restrict buffer buf3_ { float g_input_verts[]; };
layout(std430, binding = 4) readonly restrict buffer buf4_ { vec2 g_input_tex_coords[]; };
layout(std430, binding = 5) readonly restrict buffer buf5_ { uint g_input_colors[]; };
layout(std430, binding = 6) readonly restrict buffer buf6_ { uint g_input_normals[]; };

layout(std430, binding = 7) buffer buf7_ { uint g_quad_indices[]; };
layout(std430, binding = 8) buffer buf8_ { uint g_quad_aabbs[]; };
layout(std430, binding = 9) buffer buf9_ { uvec4 g_tri_aabbs[]; };
layout(std430, binding = 10) buffer buf10_ { uvec2 g_tri_storage[]; };

shared uint s_instance_id[MAX_PACKET_SIZE];

shared uint s_rejected_quads[REJECTION_TYPE_COUNT];

shared uint s_quad_aabbs[LSIZE];
shared uvec4 s_tri_aabbs[LSIZE];
shared uvec4 s_quad_indices[LSIZE];

shared int s_num_quads[MAX_PACKET_SIZE], s_index_offset[MAX_PACKET_SIZE];
shared int s_quad_offset[MAX_PACKET_SIZE], s_vertex_offset[MAX_PACKET_SIZE];

// For each instance we count two types of triangles: small and big
shared int s_num_visible[MAX_PACKET_SIZE * 2];
shared int s_out_offset[2];

shared vec3 s_ray_dir0;

// TODO: do something about divergence in input data
// TODO: muszę też mieć możliwość dodawania nowych wierzchołków

// TODO: muszę poprawnie zrobić clippowanie z Z-near; z innymi płaszczyznami wlasciwie tez powinienem
// (Z far pewnie mozna olac). OK ale jak już clipnę to powininem miec max 7 trojkątów. Co z nimi zrobić?
// dodać je na koniec listy trisów? globalnym atomiciem alokowac miejsce ? można tak zrobić
//
// TODO: problemy z wydajnością pojawiają się jak używam za dużo pamięci GPU

// TODO: Z NDC should be (0, 1) instead of (-1, 1)
// TODO: if we did this in compute shader we could first filter out invisible tris and then
// (in another sub-phase) generate data for visible tris; This way we could possibly increase
// occupancy within warps

// Each quad can be converted to 2 triangles: (v0, v1, v2), (v0, v2, v3)
//
// TODO: unikanie wielokrotnej rasteryzacji trójkątów, które spanują wiele bloków:
//       możemy podzielić tróəkąty na dwie grupy: małe i duże
//       duże obsługujemy tak jak do tej pory, małe (max 4x4, albo np. 6x6) rasteuzyjemy
//       nie w ramach bloku ale w ramach jego bboxa; następnie maski małych trójkątów
//       byłyby przycinane do konkretnych bloków

vec3 vertexLoad(uint vindex) {
	return vec3(g_input_verts[vindex * 3 + 0], g_input_verts[vindex * 3 + 1],
				g_input_verts[vindex * 3 + 2]);
}

uint vertexClipMask(vec4 pos) {
	return (pos.x < -pos.w ? 0x001 : 0) | (pos.x > pos.w ? 0x002 : 0) |
		   (pos.y < -pos.w ? 0x004 : 0) | (pos.y > pos.w ? 0x008 : 0) |
		   (pos.z < -pos.w ? 0x010 : 0) | (pos.z > pos.w ? 0x020 : 0);
}

// Computing AABBs for tris crossing Z-near clip plane is tricky
// Algorithm source: Calculating Screen Coverage (Jim Blinn's Corner, 1996)
// AABB: xmin, ymin, xmax, ymax
vec4 computeClippedAABB(vec4 v0, vec4 v1, vec4 v2, vec3 inv_w, uint clipmask) {
	vec4 aabb = vec4(1.0, 1.0, -1.0, -1.0);

	int any_vis = 0;
	vec4 vndc[3] = {v0, v1, v2};
	uint or_clipmask = clipmask | (clipmask >> 8) | (clipmask >> 16);

	for(int i = 0; i < 3; i++) {
		uint cur_clipmask = clipmask >> (i * 8);
		if((cur_clipmask & 0x3) == 0) {
			any_vis |= 0x1;
			if(vndc[i].x - aabb[0] * vndc[i].w < 0)
				aabb[0] = vndc[i].x * inv_w[i];
			if(vndc[i].x - aabb[2] * vndc[i].w > 0)
				aabb[2] = vndc[i].x * inv_w[i];
		}
		if((cur_clipmask & 0xc) == 0) {
			any_vis |= 0x10;
			if(vndc[i].y - aabb[1] * vndc[i].w < 0)
				aabb[1] = vndc[i].y * inv_w[i];
			if(vndc[i].y - aabb[3] * vndc[i].w > 0)
				aabb[3] = vndc[i].y * inv_w[i];
		}
	}
	if((any_vis & 0x0f) == 0) {
		aabb[0] = -1.0;
		aabb[2] = 1.0;
	} else if((or_clipmask & 0x3) != 0) {
		for(int i = 0; i < 3; i++) {
			uint cur_clipmask = clipmask >> (i * 8);
			if((cur_clipmask & 0x1) != 0 && vndc[i].x - aabb[0] * vndc[i].w < 0.0)
				aabb[0] = -1.0;
			if((cur_clipmask & 0x2) != 0 && vndc[i].x - aabb[2] * vndc[i].w > 0.0)
				aabb[2] = 1.0;
		}
	}

	if((any_vis & 0xf0) == 0) {
		aabb[1] = -1.0;
		aabb[3] = 1.0;
	} else if((or_clipmask & 0xc) != 0) {
		for(int i = 0; i < 3; i++) {
			uint cur_clipmask = clipmask >> (i * 8);
			if((cur_clipmask & 0x4) != 0 && vndc[i].y - aabb[1] * vndc[i].w < 0.0)
				aabb[1] = -1.0;
			if((cur_clipmask & 0x8) != 0 && vndc[i].y - aabb[3] * vndc[i].w > 0.0)
				aabb[3] = 1.0;
		}
	}

	return aabb;
}

// AABB: xmin, ymin, xmax, ymax
vec4 computeAABB(vec3 v0, vec3 v1, vec3 v2) {
	return vec4(min(min(v0.x, v1.x), v2.x), min(min(v0.y, v1.y), v2.y), max(max(v0.x, v1.x), v2.x),
				max(max(v0.y, v1.y), v2.y));
}

void processQuad(uint quad_id, uint v0, uint v1, uint v2, uint v3, uint local_instance_id) {
	// Clipping: https://www.casual-effects.com/research/McGuire2011Clipping/McGuire-Clipping.pdf
	// clipped triangle might be a polygon of up to 7 vertices...
	// TODO: we have to do culling, otherwise triangles behind camera can waste a lot of cycles
	// TODO: make sure that quads are convex?
	// in later phases; We have to cut them as early as possible!

	bool cull0 = v0 == v1 || v1 == v2 || v2 == v0;
	bool cull1 = v0 == v2 || v2 == v3 || v3 == v0;

	if(cull0 && cull1) {
		atomicAdd(s_rejected_quads[REJECTION_TYPE_OTHER], 1);
		return;
	}

	vec3 vws[4] = {vertexLoad(v0), vertexLoad(v1), vertexLoad(v2), vertexLoad(v3)};

	if(enable_backface_culling) {
		vec3 p0 = vws[0] - frustum.ws_origin[0];
		vec3 p1 = vws[1] - frustum.ws_origin[0];
		vec3 p2 = vws[2] - frustum.ws_origin[0];
		vec3 p3 = vws[3] - frustum.ws_origin[0];
		vec3 nrm0 = cross(p2, p1 - p2);
		vec3 nrm1 = cross(p3, p2 - p3);
		float volume0 = dot(p0, nrm0);
		float volume1 = dot(p0, nrm1);

		cull0 = cull0 || volume0 <= 0.0;
		cull1 = cull1 || volume1 <= 0.0;

		if(cull0 && cull1) {
			atomicAdd(s_rejected_quads[REJECTION_TYPE_BACKFACE], 1);
			return;
		}
	}
	int cull_flags = (cull0 ? 1 : 0) | (cull1 ? 2 : 0);

	vec4 vndc[4] = {
		view_proj_matrix * vec4(vws[0], 1.0),
		view_proj_matrix * vec4(vws[1], 1.0),
		view_proj_matrix * vec4(vws[2], 1.0),
		view_proj_matrix * vec4(vws[3], 1.0),
	};

	uint clipmask = vertexClipMask(vndc[0]) | (vertexClipMask(vndc[1]) << 8) |
					(vertexClipMask(vndc[2]) << 16) | (vertexClipMask(vndc[3]) << 24);
	uint and_clipmask = clipmask & (clipmask >> 8) & (clipmask >> 16) & (clipmask >> 24) & 0xff;
	uint or_clipmask = clipmask | (clipmask >> 8) | (clipmask >> 16) | (clipmask >> 24);

	// Culling triangles outside of one of clipping planes
	if(and_clipmask != 0) {
		atomicAdd(s_rejected_quads[REJECTION_TYPE_FRUSTUM], 1);
		return;
	}

	vec4 aabb0, aabb1;
	vec4 inv_w = vec4(1.0 / vndc[0].w, 1.0 / vndc[1].w, 1.0 / vndc[2].w, 1.0 / vndc[3].w);

	// Computing AABBs for tris crossing Z-near clip plane is tricky
	// Algorithm source: Calculating Screen Coverage (Jim Blinn's Corner, 1996)
	if((or_clipmask & 0x30) != 0) {
		aabb0 = computeClippedAABB(vndc[0], vndc[1], vndc[2], inv_w.xyz, clipmask);
		aabb1 = computeClippedAABB(vndc[0], vndc[2], vndc[3], inv_w.xzw,
								   (clipmask & 0xff) | ((clipmask & 0xffff0000) >> 8));
	}

	vndc[0].xyz *= inv_w[0];
	vndc[1].xyz *= inv_w[1];
	vndc[2].xyz *= inv_w[2];
	vndc[3].xyz *= inv_w[3];

	if((or_clipmask & 0x30) == 0) {
		// TODO: optimize
		aabb0 = computeAABB(vndc[0].xyz, vndc[1].xyz, vndc[2].xyz);
		aabb1 = computeAABB(vndc[0].xyz, vndc[2].xyz, vndc[3].xyz);
	}

	// Computing AABBs in screen coordinates
	const vec2 ndc_to_screen = vec2(float(VIEWPORT_SIZE_X) * 0.5, float(VIEWPORT_SIZE_Y) * 0.5);
	const ivec2 max_screen_pos = ivec2(VIEWPORT_SIZE_X - 1, VIEWPORT_SIZE_Y - 1);

	aabb0 = (aabb0 + vec4(1.0)) * vec4(ndc_to_screen, ndc_to_screen);
	aabb1 = (aabb1 + vec4(1.0)) * vec4(ndc_to_screen, ndc_to_screen);
	vec4 aabb = vec4(min(aabb0.xy, aabb1.xy), max(aabb0.zw, aabb1.zw));

	// Killing quads which fall between samples
	// TODO: smaller range in MSAA mode
	// TODO: why -0.5? are samples positioned incorrectly?
	if(ceil(aabb[0] - 0.5001) == floor(aabb[2] + 0.5001) ||
	   ceil(aabb[1] - 0.5001) == floor(aabb[3] + 0.5001)) {
		atomicAdd(s_rejected_quads[REJECTION_TYPE_BETWEEN_SAMPLES], 1);
		return;
	}

	// TODO: cull second degenerate triangle?

	// TODO: don't forget to change it in MSAA mode
	aabb0 = clamp(aabb0 + vec4(0.49, 0.49, -0.49, -0.49), vec4(0.0),
				  vec4(max_screen_pos, max_screen_pos));
	aabb1 = clamp(aabb1 + vec4(0.49, 0.49, -0.49, -0.49), vec4(0.0),
				  vec4(max_screen_pos, max_screen_pos));
	aabb = clamp(aabb + vec4(0.49, 0.49, -0.49, -0.49), vec4(0.0),
				 vec4(max_screen_pos, max_screen_pos));

	uvec4 bin_aabb = uvec4(aabb) >> BIN_SHIFT;
	uint enc_aabb = encodeAABB32(bin_aabb);

	uvec2 bin_size = uvec2(bin_aabb[2] - bin_aabb[0] + 1, bin_aabb[3] - bin_aabb[1] + 1);
	int size_type_idx = bin_size.x * bin_size.y <= 4 ? 0 : 1;
	uint out_idx = atomicAdd(s_num_visible[local_instance_id * 2 + size_type_idx], 1);
	if(size_type_idx == 1)
		out_idx = (LSIZE - 1) - out_idx;

	// TODO: for slivers try to encode in small amount of data, bins wihch for sure won't
	// intersect with the quad; For example: corner_id (2 bits), vertical coverage (7 bits, percentage),
	// horiz coverage (7 bits, percentage); This way we can encode 2 cropping lines in 32 bits (maybe)

	s_quad_aabbs[out_idx] = enc_aabb;
	s_tri_aabbs[out_idx] = uvec4(encodeAABB64(uvec4(aabb0)), encodeAABB64(uvec4(aabb1)));
	s_quad_indices[out_idx] = uvec4(v0, v1, v2, v3 | (cull_flags << 30));
}

#define TRI_SCRATCH(var_idx) g_tri_storage[scratch_tri_idx * 16 + var_idx]

void storeTri(uint scratch_tri_idx, uint instance_id, uint v0, uint v1, uint v2) {
	vec3 tri0 = vertexLoad(v0) - frustum.ws_shared_origin;
	vec3 tri1 = vertexLoad(v1) - frustum.ws_shared_origin;
	vec3 tri2 = vertexLoad(v2) - frustum.ws_shared_origin;

	vec3 normal = cross(tri0 - tri2, tri1 - tri0);
	float multiplier = 1.0 / length(normal);
	normal *= multiplier;
	uint unormal = encodeNormalUint(normal);

	vec3 edge0 = (tri0 - tri2) * multiplier;
	vec3 edge1 = (tri1 - tri0) * multiplier;

	float plane_dist = dot(normal, tri0);
	vec3 nrm_tri0 = cross(tri0, normal);
	float param0 = dot(edge0, nrm_tri0);
	float param1 = dot(edge1, nrm_tri0);

	uint instance_flags = g_instances[instance_id].flags;
	uint instance_color = g_instances[instance_id].color;

	// Nice optimization for barycentric computations:
	// dot(cross(edge, dir), normal) == dot(dir, cross(normal, edge))
	edge0 = cross(normal, edge0);
	edge1 = cross(normal, edge1);

	edge0 = vec3(dot(edge0, frustum.ws_dirx), dot(edge0, frustum.ws_diry), dot(edge0, s_ray_dir0));
	edge1 = vec3(dot(edge1, frustum.ws_dirx), dot(edge1, frustum.ws_diry), dot(edge1, s_ray_dir0));

	vec3 pnormal = normal * (1.0 / plane_dist);
	vec3 depth_eq = vec3(dot(pnormal, frustum.ws_dirx), dot(pnormal, frustum.ws_diry),
						 dot(pnormal, s_ray_dir0));

	TRI_SCRATCH(0) = uvec2(floatBitsToUint(depth_eq.x), floatBitsToUint(depth_eq.y));
	TRI_SCRATCH(1) = uvec2(floatBitsToUint(depth_eq.z), instance_flags | (instance_id << 16));
	TRI_SCRATCH(2) = uvec2(floatBitsToUint(param0), floatBitsToUint(param1));
	TRI_SCRATCH(3) = uvec2(floatBitsToUint(edge0.x), floatBitsToUint(edge0.y));
	TRI_SCRATCH(4) = uvec2(floatBitsToUint(edge0.z), floatBitsToUint(edge1.x));
	TRI_SCRATCH(5) = uvec2(floatBitsToUint(edge1.y), floatBitsToUint(edge1.z));
	TRI_SCRATCH(6) = uvec2(unormal, instance_color);

	uint vcolor2 = 0;
	if((instance_flags & INST_HAS_VERTEX_COLORS) != 0) {
		TRI_SCRATCH(7) = uvec2(g_input_colors[v0], g_input_colors[v1]);
		vcolor2 = g_input_colors[v2];
	}
	uint vnormal2 = 0;
	if((instance_flags & INST_HAS_VERTEX_NORMALS) != 0) {
		TRI_SCRATCH(9) = uvec2(g_input_normals[v0], g_input_normals[v1]);
		vnormal2 = g_input_normals[v2];
	}

	if((instance_flags & (INST_HAS_VERTEX_COLORS | INST_HAS_VERTEX_NORMALS)) != 0) {
		TRI_SCRATCH(8) = uvec2(vcolor2, vnormal2);
	}
	if((instance_flags & INST_HAS_TEXTURE) != 0) {
		vec2 tex0 = g_input_tex_coords[v0];
		vec2 tex1 = g_input_tex_coords[v1];
		vec2 tex2 = g_input_tex_coords[v2];
		tex1 -= tex0;
		tex2 -= tex0;
		TRI_SCRATCH(10) = floatBitsToUint(tex0);
		TRI_SCRATCH(11) = floatBitsToUint(tex1);
		TRI_SCRATCH(12) = floatBitsToUint(tex2);
	}

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
	vec3 scan_step = -vec3(edges[0].y * inv_ex[0], edges[1].y * inv_ex[1], edges[2].y * inv_ex[2]);

	bool sign0 = edges[0].x < 0.0;
	bool sign1 = edges[1].x < 0.0;
	bool sign2 = edges[2].x < 0.0;

	vec2 start = vec2(-0.5, 0.5);
	vec3 scan = scan_step * start.y + scan_base - vec3(start.x);
	uvec3 uscan = floatBitsToUint(scan);
	uscan = (uscan & ~1) | uvec3(sign0 ? 1 : 0, sign1 ? 1 : 0, sign2 ? 1 : 0);

	TRI_SCRATCH(13) = uscan.xy;
	TRI_SCRATCH(14) = uvec2(uscan.z, floatBitsToUint(scan_step.x));
	TRI_SCRATCH(15) = floatBitsToUint(scan_step.yz);
}

void addVisibleQuad(uint idx, uint local_instance_id) {
	uint instance_id = s_instance_id[local_instance_id];

	// Encode AABB+instance in 64-bit ?
	// Max verts: 2^26, max_instances: 2^18
	uint v0 = (s_quad_indices[LIX].x & 0x03ffffff) | ((instance_id & 0x3f) << 26);
	uint v1 = (s_quad_indices[LIX].y & 0x03ffffff) | ((instance_id & 0xfc0) << 20);
	uint v2 = (s_quad_indices[LIX].z & 0x03ffffff) | ((instance_id & 0x3f000) << 14);
	uint v3 = s_quad_indices[LIX].w; // includes cull_flags

	g_quad_indices[idx * 4 + 0] = v0;
	g_quad_indices[idx * 4 + 1] = v1;
	g_quad_indices[idx * 4 + 2] = v2;
	g_quad_indices[idx * 4 + 3] = v3;

	g_quad_aabbs[idx] = s_quad_aabbs[LIX];
	g_tri_aabbs[idx] = s_tri_aabbs[LIX];

	uint cull_flags = v3 >> 30;
	v0 &= 0x03ffffff, v1 &= 0x03ffffff, v2 &= 0x03ffffff, v3 &= 0x03ffffff;

	// TODO: shading data per-quad, not per triangle
	if((cull_flags & 1) == 0)
		storeTri(idx * 2 + 0, instance_id, v0, v1, v2);
	if((cull_flags & 2) == 0)
		storeTri(idx * 2 + 1, instance_id, v0, v2, v3);
}

void main() {
	// TODO: Could we just use 1?
	int packet_size = u_packet_size;

	if(LIX < packet_size) {
		int instance_id = int(WGID.x * packet_size + LIX);
		if(instance_id < u_num_instances) {
			int num_quads = g_instances[instance_id].num_quads;
			s_num_quads[LIX] = num_quads;
			s_vertex_offset[LIX] = g_instances[instance_id].vertex_offset;
			s_index_offset[LIX] = g_instances[instance_id].index_offset;
			s_quad_offset[LIX] = atomicAdd(g_info.num_input_quads, num_quads);
			s_instance_id[LIX] = instance_id;
		} else {
			s_num_quads[LIX] = 0;
		}
	}
	if(LIX < packet_size * 2)
		s_num_visible[LIX] = 0;
	if(LIX < REJECTION_TYPE_COUNT)
		s_rejected_quads[LIX] = 0;
	if(LIX == 0)
		s_ray_dir0 = frustum.ws_dir0 + (frustum.ws_dirx + frustum.ws_diry) * 0.5;
	barrier();
	for(int i = 0; i < packet_size; i++) {
		if(LIX < s_num_quads[i]) {
			int vertex_offset = s_vertex_offset[i];
			int index_offset = s_index_offset[i] + int(LIX) * 4;

			// Note: loading indices to SMEM first is a bit slower
			uint v0 = g_input_indices[index_offset + 0] + vertex_offset;
			uint v1 = g_input_indices[index_offset + 1] + vertex_offset;
			uint v2 = g_input_indices[index_offset + 2] + vertex_offset;
			uint v3 = g_input_indices[index_offset + 3] + vertex_offset;

			processQuad(s_quad_offset[i] + int(LIX), v0, v1, v2, v3, i);
		}
		barrier();
		if(LIX < 2)
			s_out_offset[LIX] =
				atomicAdd(g_info.num_visible_quads[LIX], s_num_visible[i * 2 + LIX]);
		// TODO: properly handle overflow
		barrier();

		uint num_small = s_num_visible[i * 2 + 0];
		uint num_large = s_num_visible[i * 2 + 1];

		{ // Storing basic quad info
			int out_offset = -1;
			if(LIX < num_small)
				out_offset = s_out_offset[0] + int(LIX);
			// TODO: hole in the middle makes it a bit less effective
			else if(LIX >= LSIZE - num_large) {
				int idx = s_out_offset[1] + int((LSIZE - 1) - LIX);
				out_offset = (MAX_VISIBLE_QUADS - 1) - idx;
			}
			if(out_offset != -1)
				addVisibleQuad(out_offset, i);
		}
	}
	barrier();
	if(LIX < REJECTION_TYPE_COUNT)
		atomicAdd(g_info.num_rejected_quads[LIX], s_rejected_quads[LIX]);

	// Last group computes number of dispatches for binning phase
	if(LIX == 0) {
		uint num_finished = atomicAdd(g_info.a_setup_work_groups, 1);
		if(num_finished == gl_NumWorkGroups.x - 1) {
			groupMemoryBarrier();
			int num_quads = g_info.num_visible_quads[0] + g_info.num_visible_quads[1];
			int batch_size = BIN_DISPATCHER_BATCH_SIZE;
			int num_dispatches = (num_quads + (batch_size - 1)) / batch_size;
			g_info.num_binning_dispatches[0] = uint(clamp(num_dispatches, 4, MAX_DISPATCHES));
			g_info.num_binning_dispatches[1] = 1;
			g_info.num_binning_dispatches[2] = 1;
		}
	}
}
