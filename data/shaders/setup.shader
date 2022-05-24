// $$include funcs frustum declarations

// ~80% of time goes to data loading

#define LSIZE MAX_LSIZE
layout(local_size_x = LSIZE) in;

#define LID gl_LocalInvocationID
#define LIX gl_LocalInvocationIndex
#define WGID gl_WorkGroupID

#define MAX_INSTANCES 4

uniform mat4 view_proj_matrix;
uniform int enable_backface_culling;
uniform int num_instances;

// TODO: check if readonly/restrict makes a differencr
layout(std430, binding = 0) readonly restrict buffer buf0_ { uint g_input_indices[]; };
layout(std430, binding = 1) readonly restrict buffer buf1_ { InstanceData g_instances[]; };
layout(std430, binding = 2) readonly restrict buffer buf2_ { float g_input_verts[]; };

BIN_COUNTERS_BUFFER(3);
layout(std430, binding = 4) buffer buf4_ { uint g_quad_indices[]; };
layout(std430, binding = 5) buffer buf5_ { uint g_quad_aabbs[]; };
layout(std430, binding = 6) buffer buf6_ { uvec4 g_tri_aabbs[]; };

shared uint s_instance_id[MAX_INSTANCES];

shared uint s_rejected_quads[REJECTED_TYPE_COUNT];

shared uint s_quad_aabbs[LSIZE];
shared uvec4 s_tri_aabbs[LSIZE];
shared uvec4 s_quad_indices[LSIZE];

shared int s_num_quads[MAX_INSTANCES], s_index_offset[MAX_INSTANCES];
shared int s_quad_offset[MAX_INSTANCES], s_vertex_offset[MAX_INSTANCES];

// For each instance we count two types of triangles: small and big
shared int s_num_visible[MAX_INSTANCES * 2];
shared int s_out_offset[2];

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

	if(v0 == v2 || ((v0 == v1 || v1 == v2) && (v2 == v3 || (v3 == v0)))) {
		atomicAdd(s_rejected_quads[REJECTED_OTHER], 1);
		return;
	}

	vec3 vws[4] = {vertexLoad(v0), vertexLoad(v1), vertexLoad(v2), vertexLoad(v3)};

	// TODO: on conference a piece of chair disappears
	if(enable_backface_culling != 0) {
		vec3 edge10 = vws[1] - vws[0];
		vec3 edge20 = vws[2] - vws[0];
		vec3 edge30 = vws[3] - vws[0];
		vec3 nrm1 = cross(edge10, edge20);
		vec3 nrm2 = cross(edge20, edge30);
		vec3 point = frustum.ws_origin[0] - vws[0];

		// TODO: what about orthogonal projection?
		// TODO: is this really a good way to back-face cull?
		if(dot(nrm1, point) < 0.0 && ((v3 == v2) || dot(nrm2, point) <= 0.0)) {
			atomicAdd(s_rejected_quads[REJECTED_BACKFACE], 1);
			return;
		}
	}

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
		atomicAdd(s_rejected_quads[REJECTED_FRUSTUM], 1);
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
		atomicAdd(s_rejected_quads[REJECTED_BETWEEN_SAMPLES], 1);
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

	uvec4 aabb_ui = uvec4(aabb);
	uint enc_aabb = encodeAABB32(aabb_ui >> (BIN_SIZE == 64 ? TILE_SHIFT : BIN_SHIFT));
	uvec4 bin_aabb = aabb_ui >> BIN_SHIFT;

	uvec2 bin_size = uvec2(bin_aabb[2] - bin_aabb[0] + 1, bin_aabb[3] - bin_aabb[1] + 1);
	int size_type_idx = bin_size.x * bin_size.y <= 3 ? 0 : 1;
	uint out_idx = atomicAdd(s_num_visible[local_instance_id * 2 + size_type_idx], 1);
	if(size_type_idx == 1)
		out_idx = (LSIZE - 1) - out_idx;

	s_quad_aabbs[out_idx] = enc_aabb;
	s_tri_aabbs[out_idx] = uvec4(encodeAABB64(uvec4(aabb0)), encodeAABB64(uvec4(aabb1)));
	s_quad_indices[out_idx] = uvec4(v0, v1, v2, v3);
}

void addVisibleQuad(uint idx, uint local_instance_id) {
	uint instance_id = s_instance_id[local_instance_id];

	// Encode AABB+instance in 64-bit ?
	// Max verts: 2^26, max_instances: 2^18
	uint v0 = (s_quad_indices[LIX].x & 0x03ffffff) | ((instance_id & 0x3f) << 26);
	uint v1 = (s_quad_indices[LIX].y & 0x03ffffff) | ((instance_id & 0xfc0) << 20);
	uint v2 = (s_quad_indices[LIX].z & 0x03ffffff) | ((instance_id & 0x3f000) << 14);
	uint v3 = s_quad_indices[LIX].w & 0x03ffffff;
	g_quad_indices[idx * 4 + 0] = v0;
	g_quad_indices[idx * 4 + 1] = v1;
	g_quad_indices[idx * 4 + 2] = v2;
	g_quad_indices[idx * 4 + 3] = v3;

	g_quad_aabbs[idx] = s_quad_aabbs[LIX];
	g_tri_aabbs[idx] = s_tri_aabbs[LIX];
}

void main() {
	// TODO: drop MAX_INSTANCES ? just use 1 ?
	if(LIX < MAX_INSTANCES) {
		int instance_id = int(WGID.x * MAX_INSTANCES + LIX);
		if(instance_id < num_instances) {
			int num_quads = g_instances[instance_id].num_quads;
			s_num_quads[LIX] = num_quads;
			s_vertex_offset[LIX] = g_instances[instance_id].vertex_offset;
			s_index_offset[LIX] = g_instances[instance_id].index_offset;
			s_quad_offset[LIX] = atomicAdd(g_bins.num_input_quads, num_quads);
			s_instance_id[LIX] = instance_id;
		} else {
			s_num_quads[LIX] = 0;
		}
	}
	if(LIX < MAX_INSTANCES * 2)
		s_num_visible[LIX] = 0;
	if(LIX < REJECTED_TYPE_COUNT)
		s_rejected_quads[LIX] = 0;
	barrier();
	for(int i = 0; i < MAX_INSTANCES; i++) {
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
				atomicAdd(g_bins.num_visible_quads[LIX], s_num_visible[i * 2 + LIX]);
		barrier();

		int out_offset = -1;
		if(LIX < s_num_visible[i * 2 + 0])
			out_offset = s_out_offset[0] + int(LIX);
		// TODO: hole in the middle makes it a bit less effective
		else if(LIX >= LSIZE - s_num_visible[i * 2 + 1]) {
			int idx = s_out_offset[1] + int((LSIZE - 1) - LIX);
			out_offset = (MAX_QUADS - 1) - idx;
		}

		if(out_offset != -1)
			addVisibleQuad(out_offset, i);
	}
	barrier();
	if(LIX < REJECTED_TYPE_COUNT)
		atomicAdd(g_bins.num_rejected_quads[LIX], s_rejected_quads[LIX]);

	// Last group computes number of dispatches for binning phase
	if(LIX == 0) {
		uint num_finished = atomicAdd(g_bins.num_finished_setup_groups, 1);
		if(num_finished == gl_NumWorkGroups.x - 1) {
			groupMemoryBarrier();
			int num_quads = g_bins.num_visible_quads[0] + g_bins.num_visible_quads[1];
			int batch_size = BINNING_LSIZE;
			int num_dispatches = (num_quads + (batch_size - 1)) / batch_size;
			g_bins.num_binning_dispatches[0] = uint(clamp(num_dispatches, 4, MAX_DISPATCHES));
			g_bins.num_binning_dispatches[1] = 1;
			g_bins.num_binning_dispatches[2] = 1;
		}
	}
}
