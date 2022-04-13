// $$include funcs lighting frustum viewport data

// TODO: add synthetic test: 256 planes one after another
// TODO: cleanup in the beginning (group definitions)

// NOTE: converting integer multiplications to shifts does not increase perf

#define LIX gl_LocalInvocationIndex
#define LID gl_LocalInvocationID

#define LSIZE 256
#define LSHIFT 8

#define BUFFER_SIZE (LSIZE * 9)

#define MAX_BLOCK_ROW_TRIS 1024 // TODO: detect overflow
#define MAX_BLOCK_TRIS 256
#define MAX_BLOCK_TRIS_SHIFT 8

#define MAX_SCRATCH_TRIS 2048
#define MAX_SCRATCH_TRIS_SHIFT 11

#define SEGMENT_SIZE 128
#define SEGMENT_SHIFT 7

#define MAX_SEGMENTS 32
#define MAX_SEGMENTS_SHIFT 5

#undef BLOCK_SIZE
#undef BLOCK_SHIFT

#define BLOCK_WIDTH 8
#define BLOCK_HEIGHT 4

#define NUM_BLOCK_COLS 8
#define NUM_BLOCK_ROWS 16

#define BLOCK_STEP (LSIZE / NUM_BLOCK_COLS)

layout(local_size_x = LSIZE) in;

layout(std430, binding = 0) buffer buf0_ { uvec4 g_tri_aabbs[]; };
layout(std430, binding = 1) buffer buf1_ { uint g_quad_indices[]; };

layout(std430, binding = 2) readonly buffer buf2_ { float g_verts[]; };
layout(std430, binding = 3) readonly buffer buf3_ { vec2 g_tex_coords[]; };
layout(std430, binding = 4) readonly buffer buf4_ { uint g_colors[]; };
layout(std430, binding = 5) readonly buffer buf5_ { uint g_normals[]; };

layout(std430, binding = 6) buffer buf6_ { BinCounters g_bins; };
layout(std430, binding = 8) buffer buf8_ { uint g_bin_quads[]; };

layout(std430, binding = 9) coherent buffer buf9_ { uint g_scratch_32[]; };
layout(std430, binding = 10) coherent buffer buf10_ { uvec2 g_scratch_64[]; };

layout(std430, binding = 11) readonly buffer buf11_ { InstanceData g_instances[]; };
layout(std430, binding = 12) readonly buffer buf12_ { vec4 g_uv_rects[]; };
layout(std430, binding = 13) writeonly buffer buf13_ { uint g_raster_image[]; };

layout(binding = 0) uniform sampler2D opaque_texture;
layout(binding = 1) uniform sampler2D transparent_texture;

// TODO: separate opaque and transparent objects, draw opaque objects first to texture
// then read it and use depth to optimize drawing
uniform uint background_color;

#define SATURATE(val) clamp(val, 0.0, 1.0)

#define WORKGROUP_32_SCRATCH_SIZE (32 * 1024)
#define WORKGROUP_32_SCRATCH_SHIFT 15

#define WORKGROUP_64_SCRATCH_SIZE (64 * 1024)
#define WORKGROUP_64_SCRATCH_SHIFT 16

#define TRI_SCRATCH(var_idx) g_scratch_64[scratch_tri_offset + (var_idx << MAX_SCRATCH_TRIS_SHIFT)]

uint scratch32BlockRowTrisOffset(uint by) {
	return (gl_WorkGroupID.x << WORKGROUP_32_SCRATCH_SHIFT) + by * MAX_BLOCK_ROW_TRIS;
}

uint scratch32BlockTrisOffset(uint bx) {
	return (gl_WorkGroupID.x << WORKGROUP_32_SCRATCH_SHIFT) + 16 * 1024 + bx * MAX_BLOCK_TRIS;
}

uint scratch64TriOffset(uint tri_idx) {
	return (gl_WorkGroupID.x << WORKGROUP_64_SCRATCH_SHIFT) + tri_idx;
}

uint scratch64BlockRowTrisOffset(uint by) {
	return (gl_WorkGroupID.x << WORKGROUP_64_SCRATCH_SHIFT) + 32 * 1024 + by * MAX_BLOCK_ROW_TRIS;
}

uint scratch64BlockTrisOffset(uint bx) {
	return (gl_WorkGroupID.x << WORKGROUP_64_SCRATCH_SHIFT) + 48 * 1024 + bx * MAX_BLOCK_TRIS;
}

shared int s_num_bins, s_bin_id, s_bin_raster_offset;
shared ivec2 s_bin_pos;
shared vec3 s_bin_ray_dir0;

shared uint s_block_row_tri_count[NUM_BLOCK_ROWS];
shared uint s_block_tri_count[NUM_BLOCK_COLS];
shared uint s_block_frag_count[NUM_BLOCK_COLS];

shared uint s_bin_quad_count, s_bin_quad_offset;

shared uint s_buffer[BUFFER_SIZE + 1];
shared uint s_mini_buffer[LSIZE];
shared uint s_segments[NUM_BLOCK_COLS][MAX_SEGMENTS];
shared int s_raster_error;

#define REDUCE_GROUP_COUNT 8
shared uint s_reduce_groups[REDUCE_GROUP_COUNT];

void outputPixel(ivec2 pixel_pos, uint color) {
	g_raster_image[s_bin_raster_offset + pixel_pos.x + (pixel_pos.y << BIN_SHIFT)] = color;
}

// Note: UPDATE_CLOCK should be called after a barrier
// TODO: make it more accurate
#ifdef ENABLE_TIMINGS
#define MAX_TIMERS 8
shared uint64_t s_timings[MAX_TIMERS];
#define INIT_CLOCK() uint64_t clock0 = clockARB();
#define UPDATE_CLOCK(idx)                                                                          \
	if(LIX == 0) {                                                                                 \
		uint64_t clock = clockARB();                                                               \
		s_timings[idx] += clock - clock0;                                                          \
		clock0 = clock;                                                                            \
	}

void initTimers() {
	if(LIX < MAX_TIMERS)
		s_timings[LIX] = 0;
}
void commitTimers() {
	if(LIX < MAX_TIMERS)
		atomicAdd(g_bins.timings[LIX], uint(s_timings[LIX]));
}

#else
#define INIT_CLOCK()
#define UPDATE_CLOCK(idx)

void initTimers() {}
void commitTimers() {}
#endif

#ifdef VENDOR_NVIDIA
uint swap(uint x, int mask, uint dir) {
	uint y = shuffleXorNV(x, mask, 32);
	return uint(x < y) == dir ? y : x;
}
uint bitExtract(uint value, int boffset) { return (value >> boffset) & 1; }
uint xorBits(uint value, int bit0, int bit1) { return ((value >> bit0) ^ (value >> bit1)) & 1; }
#endif

shared uint s_sort_max_block_rcount;

void sortTileTris() {
	if(LIX < NUM_BLOCK_COLS) {
		uint count = s_block_tri_count[LIX];
		// rcount: count rounded up to next power of 2
		uint rcount = max(32, (count & (count - 1)) == 0 ? count : (2 << findMSB(count)));
		if(LIX == 0)
			s_sort_max_block_rcount = 0;
		atomicMax(s_sort_max_block_rcount, rcount);
	}
	barrier();

	uint gid = LIX >> (LSHIFT - 3);
	uint lid = LIX & (BLOCK_STEP - 1);
	uint goffset = gid << MAX_BLOCK_TRIS_SHIFT;
	uint count = s_block_tri_count[gid];
	// TODO: max_rcount is only needed for barriers, computations should be performed up to rcount
	// But it seems, that using rcount directly is actually a bit slower... (Sponza)
	uint max_rcount = s_sort_max_block_rcount;
	for(uint i = lid + count; i < max_rcount; i += BLOCK_STEP)
		s_buffer[goffset + i] = 0xffffffff;
	barrier();

#ifdef VENDOR_NVIDIA
	for(uint i = lid; i < max_rcount; i += BLOCK_STEP) {
		uint value = s_buffer[goffset + i];
		// TODO: register sort could be faster
		value = swap(value, 0x01, xorBits(lid, 1, 0)); // K = 2
		value = swap(value, 0x02, xorBits(lid, 2, 1)); // K = 4
		value = swap(value, 0x01, xorBits(lid, 2, 0));
		value = swap(value, 0x04, xorBits(lid, 3, 2)); // K = 8
		value = swap(value, 0x02, xorBits(lid, 3, 1));
		value = swap(value, 0x01, xorBits(lid, 3, 0));
		value = swap(value, 0x08, xorBits(lid, 4, 3)); // K = 16
		value = swap(value, 0x04, xorBits(lid, 4, 2));
		value = swap(value, 0x02, xorBits(lid, 4, 1));
		value = swap(value, 0x01, xorBits(lid, 4, 0));
		//value = swap(value, 0x10, xorBits(lid, 5, 4)); // K = 32
		//value = swap(value, 0x08, xorBits(lid, 5, 3));
		//value = swap(value, 0x04, xorBits(lid, 5, 2));
		//value = swap(value, 0x02, xorBits(lid, 5, 1));
		//value = swap(value, 0x01, xorBits(lid, 5, 0));
		s_buffer[goffset + i] = value;
	}
	barrier();
	int start_k = 32, end_j = 32;
#else
	int start_k = 2, end_j = 1;
#endif
	for(uint k = start_k; k <= max_rcount; k = 2 * k) {
		for(uint j = k >> 1; j >= end_j; j = j >> 1) {
			for(uint i = lid; i < max_rcount; i += BLOCK_STEP * 2) {
				uint idx = (i & j) != 0 ? i + BLOCK_STEP - j : i;
				uint lvalue = s_buffer[goffset + idx];
				uint rvalue = s_buffer[goffset + idx + j];
				if(((idx & k) != 0) == (lvalue.x < rvalue.x)) {
					s_buffer[goffset + idx] = rvalue;
					s_buffer[goffset + idx + j] = lvalue;
				}
			}
			barrier();
		}
#ifdef VENDOR_NVIDIA
		for(uint i = lid; i < max_rcount; i += BLOCK_STEP) {
			uint bit = (i & k) == 0 ? 0 : 1;
			uint value = s_buffer[goffset + i];
			value = swap(value, 0x10, bit ^ bitExtract(lid, 4));
			value = swap(value, 0x08, bit ^ bitExtract(lid, 3));
			value = swap(value, 0x04, bit ^ bitExtract(lid, 2));
			value = swap(value, 0x02, bit ^ bitExtract(lid, 1));
			value = swap(value, 0x01, bit ^ bitExtract(lid, 0));
			s_buffer[goffset + i] = value;
		}
		barrier();
#endif
	}
}

// TODO: don't store triangles which generate very small number of samples in scratch,
// instead precompute them directly when sampling; We would have to somehow group those triangles together
//
// TODO: use scratch based on uints, not uvec2, maybe it will be a bit faster?

void storeTriangle(uint tri_idx, vec3 tri0, vec3 tri1, vec3 tri2, uint v0, uint v1, uint v2,
				   uint instance_id) {
	uint scratch_tri_offset = scratch64TriOffset(tri_idx);
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

	edge0 =
		vec3(dot(edge0, frustum.ws_dirx), dot(edge0, frustum.ws_diry), dot(edge0, s_bin_ray_dir0));
	edge1 =
		vec3(dot(edge1, frustum.ws_dirx), dot(edge1, frustum.ws_diry), dot(edge1, s_bin_ray_dir0));

	vec3 pnormal = normal * (1.0 / plane_dist);
	vec3 depth_eq = vec3(dot(pnormal, frustum.ws_dirx), dot(pnormal, frustum.ws_diry),
						 dot(pnormal, s_bin_ray_dir0));

	TRI_SCRATCH(0) = uvec2(floatBitsToUint(depth_eq.x), floatBitsToUint(depth_eq.y));
	TRI_SCRATCH(1) = uvec2(floatBitsToUint(depth_eq.z), instance_flags | (instance_id << 16));
	TRI_SCRATCH(2) = uvec2(floatBitsToUint(param0), floatBitsToUint(param1));
	TRI_SCRATCH(3) = uvec2(floatBitsToUint(edge0.x), floatBitsToUint(edge0.y));
	TRI_SCRATCH(4) = uvec2(floatBitsToUint(edge0.z), floatBitsToUint(edge1.x));
	TRI_SCRATCH(5) = uvec2(floatBitsToUint(edge1.y), floatBitsToUint(edge1.z));

	TRI_SCRATCH(6) = uvec2(unormal, instance_color);

	uint vcolor2 = 0;
	if((instance_flags & INST_HAS_VERTEX_COLORS) != 0) {
		TRI_SCRATCH(7) = uvec2(g_colors[v0], g_colors[v1]);
		vcolor2 = g_colors[v2];
	}
	uint vnormal2 = 0;
	if((instance_flags & INST_HAS_VERTEX_NORMALS) != 0) {
		TRI_SCRATCH(9) = uvec2(g_normals[v0], g_normals[v1]);
		vnormal2 = g_normals[v2];
	}

	if((instance_flags & (INST_HAS_VERTEX_COLORS | INST_HAS_VERTEX_NORMALS)) != 0) {
		TRI_SCRATCH(8) = uvec2(vcolor2, vnormal2);
	}
	if((instance_flags & INST_HAS_TEXTURE) != 0) {
		vec2 tex0 = g_tex_coords[v0];
		vec2 tex1 = g_tex_coords[v1];
		vec2 tex2 = g_tex_coords[v2];
		tex1 -= tex0;
		tex2 -= tex0;
		TRI_SCRATCH(10) = floatBitsToUint(tex0);
		TRI_SCRATCH(11) = floatBitsToUint(tex1);
		TRI_SCRATCH(12) = floatBitsToUint(tex2);
	}
}

void getTriangleParams(uint scratch_tri_offset, out vec3 depth_eq, out vec2 bary_params,
					   out vec3 edge0, out vec3 edge1, out uint instance_id,
					   out uint instance_flags) {
	{
		uvec2 val0 = TRI_SCRATCH(0), val1 = TRI_SCRATCH(1), val2 = TRI_SCRATCH(2);
		depth_eq =
			vec3(uintBitsToFloat(val0[0]), uintBitsToFloat(val0[1]), uintBitsToFloat(val1[0]));
		bary_params = vec2(uintBitsToFloat(val2[0]), uintBitsToFloat(val2[1]));
		instance_flags = val1[1] & 0xffff;
		instance_id = val1[1] >> 16;
	}
	{
		uvec2 val0 = TRI_SCRATCH(3), val1 = TRI_SCRATCH(4), val2 = TRI_SCRATCH(5);
		edge0 = vec3(uintBitsToFloat(val0[0]), uintBitsToFloat(val0[1]), uintBitsToFloat(val1[0]));
		edge1 = vec3(uintBitsToFloat(val1[1]), uintBitsToFloat(val2[0]), uintBitsToFloat(val2[1]));
	}
}

void getTriangleSecondaryParams(uint scratch_tri_offset, out uint unormal,
								out uint instance_color) {
	uvec2 val0 = TRI_SCRATCH(6);
	unormal = val0.x;
	instance_color = val0.y;
}

void getTriangleVertexColors(uint scratch_tri_offset, out vec4 color0, out vec4 color1,
							 out vec4 color2) {
	uvec2 val0 = TRI_SCRATCH(7);
	uvec2 val1 = TRI_SCRATCH(8);
	color0 = decodeRGBA8(val0[0]);
	color1 = decodeRGBA8(val0[1]);
	color2 = decodeRGBA8(val1[0]);
}

void getTriangleVertexNormals(uint scratch_tri_offset, out vec3 normal0, out vec3 normal1,
							  out vec3 normal2) {
	uvec2 val0 = TRI_SCRATCH(9);
	uvec2 val1 = TRI_SCRATCH(8);
	normal0 = decodeNormalUint(val0[0]);
	normal1 = decodeNormalUint(val0[1]);
	normal2 = decodeNormalUint(val1[1]);
}

void getTriangleVertexTexCoords(uint scratch_tri_offset, out vec2 tex0, out vec2 tex1,
								out vec2 tex2) {
	uvec2 val0 = TRI_SCRATCH(10);
	uvec2 val1 = TRI_SCRATCH(11);
	uvec2 val2 = TRI_SCRATCH(12);
	tex0 = uintBitsToFloat(val0);
	tex1 = uintBitsToFloat(val1);
	tex2 = uintBitsToFloat(val2);
}

uint computeScanlineParams(vec3 tri0, vec3 tri1, vec3 tri2, out vec3 scan_base,
						   out vec3 scan_step) {
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
	scan_base = -vec3(edges[0].z * inv_ex[0], edges[1].z * inv_ex[1], edges[2].z * inv_ex[2]);
	scan_step = -vec3(edges[0].y * inv_ex[0], edges[1].y * inv_ex[1], edges[2].y * inv_ex[2]);
	return (edges[0].x < 0.0 ? 1 : 0) | (edges[1].x < 0.0 ? 2 : 0) | (edges[2].x < 0.0 ? 4 : 0);
}

void generateRowTris(uint tri_idx, vec3 tri0, vec3 tri1, vec3 tri2, int min_by, int max_by) {
	// Inspired by Nanite scanline rasterizer
	vec3 scan_min, scan_max, scan_step;

	{
		float sx = s_bin_pos.x - 0.5f;
		float sy = s_bin_pos.y + (float(min_by) * 4.0 + 0.5f);

		vec3 scan_base;
		uint sign_mask = computeScanlineParams(tri0, tri1, tri2, scan_base, scan_step);

		vec3 scan = scan_step * sy + scan_base - vec3(sx, sx, sx);
		scan_min = vec3((sign_mask & 1) == 0 ? scan[0] : -1.0 / 0.0,
						(sign_mask & 2) == 0 ? scan[1] : -1.0 / 0.0,
						(sign_mask & 4) == 0 ? scan[2] : -1.0 / 0.0);
		scan_max = vec3((sign_mask & 1) != 0 ? scan[0] : 1.0 / 0.0,
						(sign_mask & 2) != 0 ? scan[1] : 1.0 / 0.0,
						(sign_mask & 4) != 0 ? scan[2] : 1.0 / 0.0);
	}

	uint dst_offset_64 = scratch64BlockRowTrisOffset(0);
	uint dst_offset_32 = scratch32BlockRowTrisOffset(0);

	for(int by = min_by; by <= max_by; by++) {
		float xmin0 = max(max(scan_min[0], scan_min[1]), max(scan_min[2], 0.0));
		float xmax0 = min(min(scan_max[0], scan_max[1]), min(scan_max[2], BIN_SIZE));
		scan_min += scan_step, scan_max += scan_step;
		float xmin1 = max(max(scan_min[0], scan_min[1]), max(scan_min[2], 0.0));
		float xmax1 = min(min(scan_max[0], scan_max[1]), min(scan_max[2], BIN_SIZE));
		scan_min += scan_step, scan_max += scan_step;
		float xmin2 = max(max(scan_min[0], scan_min[1]), max(scan_min[2], 0.0));
		float xmax2 = min(min(scan_max[0], scan_max[1]), min(scan_max[2], BIN_SIZE));
		scan_min += scan_step, scan_max += scan_step;
		float xmin3 = max(max(scan_min[0], scan_min[1]), max(scan_min[2], 0.0));
		float xmax3 = min(min(scan_max[0], scan_max[1]), min(scan_max[2], BIN_SIZE));
		scan_min += scan_step, scan_max += scan_step;

		int imin0 = int(xmin0), imax0 = int(xmax0) - 1;
		int imin1 = int(xmin1), imax1 = int(xmax1) - 1;
		int imin2 = int(xmin2), imax2 = int(xmax2) - 1;
		int imin3 = int(xmin3), imax3 = int(xmax3) - 1;

		if(imin0 > imax0)
			imin0 = 63, imax0 = 0;
		if(imin1 > imax1)
			imin1 = 63, imax1 = 0;
		if(imin2 > imax2)
			imin2 = 63, imax2 = 0;
		if(imin3 > imax3)
			imin3 = 63, imax3 = 0;

		// TODO: bx mask
		uint bx_mask = ((0xff << (imin0 >> 3)) & (0xff >> (7 - (imax0 >> 3)))) |
					   ((0xff << (imin1 >> 3)) & (0xff >> (7 - (imax1 >> 3)))) |
					   ((0xff << (imin2 >> 3)) & (0xff >> (7 - (imax2 >> 3)))) |
					   ((0xff << (imin3 >> 3)) & (0xff >> (7 - (imax3 >> 3))));
		if(bx_mask == 0)
			continue;

		uint roffset = atomicAdd(s_block_row_tri_count[by], 1) + by * MAX_BLOCK_ROW_TRIS;
		g_scratch_64[dst_offset_64 + roffset] =
			uvec2((imin0 << 0) | (imin1 << 6) | (imin2 << 12) | (imin3 << 18),
				  (imax0 << 0) | (imax1 << 6) | (imax2 << 12) | (imax3 << 18));
		g_scratch_32[dst_offset_32 + roffset] = tri_idx | (bx_mask << 16);
	}
}

void processQuads() {
	// TODO: optimization: in many cases all rows may very well fit in SMEM,
	// maybe it would be worth it not to use scratch at all then?
	// TODO: this loop is slooooow
	// TODO: divide big tris across different threads
	for(uint i = LIX >> 1; i < s_bin_quad_count; i += LSIZE / 2) {
		uint second_tri = LIX & 1;
		uint quad_idx = g_bin_quads[s_bin_quad_offset + i] & 0xffffff;

		uvec4 aabb = g_tri_aabbs[quad_idx];
		aabb = decodeAABB(second_tri != 0 ? aabb.zw : aabb.xy);
		int min_by = clamp(int(aabb[1]) - s_bin_pos.y, 0, 63) >> 2;
		int max_by = clamp(int(aabb[3]) - s_bin_pos.y, 0, 63) >> 2;

		uvec4 verts = uvec4(g_quad_indices[quad_idx * 4 + 0], g_quad_indices[quad_idx * 4 + 1],
							g_quad_indices[quad_idx * 4 + 2], g_quad_indices[quad_idx * 4 + 3]);
		uint instance_id =
			(verts[0] >> 26) | ((verts[1] >> 20) & 0xfc0) | ((verts[2] >> 14) & 0x3f000);
		uint v0 = verts[0] & 0x03ffffff;
		uint v1 = verts[1 + second_tri] & 0x03ffffff;
		uint v2 = verts[2 + second_tri] & 0x03ffffff;

		vec3 tri0 = vec3(g_verts[v0 * 3 + 0], g_verts[v0 * 3 + 1], g_verts[v0 * 3 + 2]) -
					frustum.ws_shared_origin;
		vec3 tri1 = vec3(g_verts[v1 * 3 + 0], g_verts[v1 * 3 + 1], g_verts[v1 * 3 + 2]) -
					frustum.ws_shared_origin;
		vec3 tri2 = vec3(g_verts[v2 * 3 + 0], g_verts[v2 * 3 + 1], g_verts[v2 * 3 + 2]) -
					frustum.ws_shared_origin;

		// TODO: store only if samples were generated
		// TODO: do triangle storing later
		uint tri_idx = i * 2 + (LIX & 1);
		storeTriangle(tri_idx, tri0, tri1, tri2, v0, v1, v2, instance_id);
		generateRowTris(tri_idx, tri0, tri1, tri2, min_by, max_by);
	}
}

// Nowy pomysł:
// - Zamiast przetwarzania na raz 1 kafla 16x16 operujemy na 8 blokach 8x4
// - przetwarzamy te bloki linia po linii (16 iteracji w sumie)
// - porządkowanie sampli nie będzie potrzebne, bo sample będą od razu uporządkowane w grupy 32-elementowe
// - dla każdego bloku wyznaczamy niezależne segmenty po 128 sampli;
// - w każdym bloku 8x4 może być max 256 trókątów: w segmentach można używać 8-bitowych indeksów;
//   będzie trzeba trochę więcej miejsca na segmenty (2x albo 4x tyle)
// - początkowa generacja powinna rozrzucać tile-row-trisy do 16 różnych grup (nie trzeba zapisywać qy)
// - troszkę trzeba będzie zmodyfikować sortowanie
//
// - load, shade & reduce operują w pętli biorą po jednym segmencie z każdego bloku; bariery nie powinny być potrzebne,
//   muszę tylko zapewnić, że wszystkie warpy we wszystkich fazach działają na tych samych danych
//
// - problem: przy 16x16 miałem na koniec jeden niepełny segment, teraz będę miał 8;
//   Może to jest do rowiązania na potem ?
// - lokalność w czasie samplowania będzie trochę gorsza, ale jeśli mam jeden materiał to nie jest to problem
//
// - na początek muszę powyłączać końcowe fazy i przerobić wszytko od początku jeszcze raz

void generateBlocks(uint by) {
	uint src_offset_32 = scratch32BlockRowTrisOffset(by);
	uint src_offset_64 = scratch64BlockRowTrisOffset(by);
	uint tri_count = s_block_row_tri_count[by];

	if(LIX < MAX_SEGMENTS * NUM_BLOCK_COLS) {
		s_segments[LIX >> MAX_SEGMENTS_SHIFT][LIX & (MAX_SEGMENTS - 1)] = ~0u;
		if(LIX < NUM_BLOCK_COLS)
			s_block_tri_count[LIX] = 0;
	}
	barrier(); // TODO: could it be removed?

	{
		uint bx = LIX & 7, bx_bit = 0x10000 << bx;
		uint buf_offset = bx << MAX_BLOCK_TRIS_SHIFT;
		// TODO: optimize this loop? iterate over bits for atomicAdd?
		for(uint i = LIX >> 3; i < tri_count; i += LSIZE / 8) {
			if((g_scratch_32[src_offset_32 + i] & bx_bit) == 0)
				continue;
			uint tri_offset = atomicAdd(s_block_tri_count[bx], 1);
			if(tri_offset < MAX_BLOCK_TRIS)
				s_buffer[buf_offset + tri_offset] = i;
			else
				atomicOr(s_raster_error, 0x100 << bx);
		}
		barrier();
		if(s_raster_error != 0)
			return;
	}

	uint bx = LIX >> (LSHIFT - 3);
	uint buf_offset = bx << MAX_BLOCK_TRIS_SHIFT;
	uint dst_offset_64 = scratch64BlockTrisOffset(bx);
	tri_count = s_block_tri_count[bx];
	int startx = int(bx << 3);

	for(uint i = LIX & (BLOCK_STEP - 1); i < tri_count; i += BLOCK_STEP) {
		uint row_idx = s_buffer[buf_offset + i];

		// TODO: load these together with shuffles?
		uint tri_info = g_scratch_32[src_offset_32 + row_idx];
		uvec2 tri_rows = g_scratch_64[src_offset_64 + row_idx];
		uint tri_idx = tri_info & 0xffff;

		uint min_bits, count_bits, num_frags;
		vec2 cpos;

		{
			int minx0 = max(int((tri_rows.x >> 0) & 0x3f) - startx, 0);
			int minx1 = max(int((tri_rows.x >> 6) & 0x3f) - startx, 0);
			int minx2 = max(int((tri_rows.x >> 12) & 0x3f) - startx, 0);
			int minx3 = max(int((tri_rows.x >> 18) & 0x3f) - startx, 0);

			int maxx0 = min(int((tri_rows.y >> 0) & 0x3f) - startx, 7);
			int maxx1 = min(int((tri_rows.y >> 6) & 0x3f) - startx, 7);
			int maxx2 = min(int((tri_rows.y >> 12) & 0x3f) - startx, 7);
			int maxx3 = min(int((tri_rows.y >> 18) & 0x3f) - startx, 7);

			int count0 = max(maxx0 - minx0 + 1, 0);
			int count1 = max(maxx1 - minx1 + 1, 0);
			int count2 = max(maxx2 - minx2 + 1, 0);
			int count3 = max(maxx3 - minx3 + 1, 0);

			cpos = vec2(float(maxx0 + minx0 + 1), 1.0) * count0;
			cpos += vec2(float(maxx1 + minx1 + 1), 3.0) * count1;
			cpos += vec2(float(maxx2 + minx2 + 1), 5.0) * count2;
			cpos += vec2(float(maxx3 + minx3 + 1), 7.0) * count3;

			// TODO: minimize bits
			min_bits =
				(minx0 & 15) | ((minx1 & 15) << 4) | ((minx2 & 15) << 8) | ((minx3 & 15) << 12);
			count_bits = count0 | (count1 << 5) | (count2 << 10) | (count3 << 15);
			num_frags = count0 + count1 + count2 + count3;
		}
		cpos *= 0.5 / float(num_frags);
		cpos += vec2(bx << 3, (by << 2));

		uint scratch_tri_offset = scratch64TriOffset(tri_idx);
		vec2 val0 = uintBitsToFloat(TRI_SCRATCH(0));
		vec2 val1 = uintBitsToFloat(TRI_SCRATCH(1));
		vec3 depth_eq = vec3(val0.x, val0.y, val1.x);
		float ray_pos = depth_eq.x * cpos.x + (depth_eq.y * cpos.y + depth_eq.z);
		float depth = 0x3fffe * SATURATE(1.0 - inversesqrt(ray_pos + 1)); // 18 bits

		if(num_frags == 0) // This means that bx_mask is invalid
			RECORD(0, 0, 0, 0);
		g_scratch_64[dst_offset_64 + i] = uvec2(min_bits | (tri_idx << 16), count_bits);

		// 8 bitów na index, 6 bitów na ilość sampli, 18 bitów na głębię
		s_buffer[buf_offset + i] = i | (num_frags << 8) | (uint(depth) << 14);
	}
	barrier();
	sortTileTris();
	barrier();

#define PREFIX_SUM_STEP(value, step)                                                               \
	{                                                                                              \
		uint temp = shuffleUpNV(value, step, 32);                                                  \
		if((LIX & 31) >= step)                                                                     \
			value += temp;                                                                         \
	}

	for(uint i = LIX & (BLOCK_STEP - 1); i < tri_count; i += BLOCK_STEP) {
		uint idx = s_buffer[buf_offset + i];
		uint num_frags = (idx >> 8) & 0x3f;
		idx &= 0xff;

		// Computing triangle-ordered sample offsets within each block
		PREFIX_SUM_STEP(num_frags, 1);
		PREFIX_SUM_STEP(num_frags, 2);
		PREFIX_SUM_STEP(num_frags, 4);
		PREFIX_SUM_STEP(num_frags, 8);
		PREFIX_SUM_STEP(num_frags, 16);
		s_buffer[buf_offset + i] = num_frags | (idx << 16);
	}
	barrier();

	// Computing prefix sum across whole blocks (at most 8 * 32 elements)
	if(LIX < 64) {
		uint bx = LIX >> 3, warp_idx = LIX & 7, warp_offset = warp_idx << 5;
		uint value = s_buffer[(bx << MAX_BLOCK_TRIS_SHIFT) + warp_offset + 31] & 0xffff, temp;
		temp = shuffleUpNV(value, 1, 8), value += warp_idx >= 1 ? temp : 0;
		temp = shuffleUpNV(value, 2, 8), value += warp_idx >= 2 ? temp : 0;
		temp = shuffleUpNV(value, 4, 8), value += warp_idx >= 4 ? temp : 0;
		s_mini_buffer[LIX] = value;
	}
	barrier();

	for(uint i = 32 + (LIX & (BLOCK_STEP - 1)); i < tri_count; i += BLOCK_STEP)
		s_buffer[buf_offset + i] += s_mini_buffer[bx * 8 + (i >> 5) - 1];
	barrier();

	// Storing offsets to scratch mem
	// Also finding first triangle for each segment
	uint dst_offset_32 = scratch32BlockTrisOffset(bx);
	for(uint i = LIX & (BLOCK_STEP - 1); i < tri_count; i += BLOCK_STEP) {
		uint tri_offset = i == 0 ? 0 : s_buffer[buf_offset + i - 1] & 0xffff;
		uint tri_value = s_buffer[buf_offset + i];
		uint tile_tri_idx = tri_value & 0xffff0000;
		tri_value = (tri_value & 0xffff) - tri_offset;

		uint seg_id = tri_offset >> SEGMENT_SHIFT;
		if(seg_id < MAX_SEGMENTS) { // TODO: MAX_SEGMENTS-1?
			uint seg_offset = tri_offset & (SEGMENT_SIZE - 1);
			if(seg_offset == 0)
				s_segments[bx][seg_id] = i;
			else if(seg_offset + tri_value > SEGMENT_SIZE)
				s_segments[bx][seg_id + 1] = i;
		}

		g_scratch_32[dst_offset_32 + i] = tri_offset | tile_tri_idx;
	}
	barrier();

	if(LIX < MAX_SEGMENTS * NUM_BLOCK_COLS) {
		uint seg_id = LIX & (MAX_SEGMENTS - 1);
		uint bx = LIX >> MAX_SEGMENTS_SHIFT;
		uint tri_count = s_block_tri_count[bx];

		uint cur_value = s_segments[bx][seg_id];
		uint next_value = seg_id + 1 == MAX_SEGMENTS ? ~0u : s_segments[bx][seg_id + 1];
		next_value = next_value == ~0u ? tri_count : min(tri_count, next_value + 1);
		cur_value = cur_value == ~0u ? 0 : cur_value | ((next_value - cur_value) << 16);
		s_segments[bx][seg_id] = cur_value;
	}

#ifdef SHADER_DEBUG
	for(uint i = LIX & (BLOCK_STEP - 1); i < tri_count; i += BLOCK_STEP) {
		uint value = s_buffer[buf_offset + i] & 0xffff;
		uint prev_value = i == 0 ? 0 : s_buffer[buf_offset + i - 1] & 0xffff;
		if(value <= prev_value)
			RECORD(i, tri_count, prev_value, value);
	}
#endif

	if(LIX < NUM_BLOCK_COLS) {
		uint bx = LIX;
		uint num_tris = s_block_tri_count[bx];
		uint frag_count = num_tris == 0 ? 0 : s_buffer[bx * MAX_BLOCK_TRIS + num_tris - 1] & 0xffff;
		//uint segment_count = (frag_count + SEGMENT_SIZE - 1) >> SEGMENT_SHIFT;
		s_block_frag_count[bx] = frag_count;
		if(frag_count > MAX_SEGMENTS * SEGMENT_SIZE)
			atomicOr(s_raster_error, 1 << bx);
	}
}

void loadSamples(int bx, int by, int segment_id, int frag_count) {
	uint first_tri = s_segments[bx][segment_id];
	uint tri_count = first_tri >> 16;
	first_tri &= 0xffff;

	uint src_offset_32 = scratch32BlockTrisOffset(bx) + first_tri;
	uint src_offset_64 = scratch64BlockTrisOffset(bx);
	uint buf_offset = bx << SEGMENT_SHIFT;
	int first_offset = segment_id << SEGMENT_SHIFT;

	int y = int(LIX & 3);
	uint min_shift = y << 2, count_shift = min_shift + y;

	// TODO: group differently for better memory accesses (and measure)
	for(uint i = (LIX & (BLOCK_STEP - 1)) >> 2; i < tri_count; i += BLOCK_STEP / 4) {
		uint tri_info = g_scratch_32[src_offset_32 + i];
		uvec2 tri_data = g_scratch_64[src_offset_64 + (tri_info >> 16)];
		int tri_offset = int(tri_info & 0xffff) - first_offset;
		int minx = int((tri_data.x >> min_shift) & 15); // TODO: too many bits
		int count_bits = int(tri_data.y & ((1 << 20) - 1));

		// TODO: precompute it somehow?
		for(uint j = 0; j < y; j++) {
			tri_offset += count_bits & 31;
			count_bits >>= 5;
		}
		int countx = min(count_bits & 31, SEGMENT_SIZE - tri_offset);
		if(countx <= 0)
			continue;

		uint scratch_tri_offset = scratch64TriOffset(tri_data.x >> 16);

		uint pixel_id = (y << 3) | minx;
		uint value = pixel_id | (scratch_tri_offset << 8);

		for(int j = 0; j < countx; j++) {
			if(tri_offset >= 0) // TODO: remove this check
				s_buffer[buf_offset + tri_offset] = value;
			tri_offset++;
			value++;
		}
	}
}

// TODO: Can we improve speed of loading vertex data?
uint shadeSample(ivec2 bin_pixel_pos, uint scratch_tri_offset, out float out_depth) {
	float px = float(bin_pixel_pos.x), py = float(bin_pixel_pos.y);

	vec3 depth_eq, edge0_eq, edge1_eq;
	uint instance_id, instance_flags;
	vec2 bary_params;
	getTriangleParams(scratch_tri_offset, depth_eq, bary_params, edge0_eq, edge1_eq, instance_id,
					  instance_flags);
	uint instance_color, unormal;
	getTriangleSecondaryParams(scratch_tri_offset, unormal, instance_color);

	float inv_ray_pos = depth_eq.x * px + (depth_eq.y * py + depth_eq.z);
	out_depth = inv_ray_pos;
	float ray_pos = 1.0 / inv_ray_pos;

	float e0 = edge0_eq.x * px + (edge0_eq.y * py + edge0_eq.z);
	float e1 = edge1_eq.x * px + (edge1_eq.y * py + edge1_eq.z);
	vec2 bary = vec2(e0, e1) * ray_pos;

	vec2 bary_dx, bary_dy;
	if((instance_flags & INST_HAS_TEXTURE) != 0) {
		float ray_posx = 1.0 / (inv_ray_pos + depth_eq.x);
		float ray_posy = 1.0 / (inv_ray_pos + depth_eq.y);

		bary_dx = vec2(e0 + edge0_eq.x, e1 + edge1_eq.x) * ray_posx - bary;
		bary_dy = vec2(e0 + edge0_eq.y, e1 + edge1_eq.y) * ray_posy - bary;
	}
	bary -= bary_params;

	vec4 color = decodeRGBA8(instance_color);
	if((instance_flags & INST_HAS_TEXTURE) != 0) {
		vec2 tex0, tex1, tex2;
		getTriangleVertexTexCoords(scratch_tri_offset, tex0, tex1, tex2);

		vec2 tex_coord = bary[0] * tex1 + (bary[1] * tex2 + tex0);
		vec2 tex_dx = bary_dx[0] * tex1 + bary_dx[1] * tex2;
		vec2 tex_dy = bary_dy[0] * tex1 + bary_dy[1] * tex2;

		if((instance_flags & INST_HAS_UV_RECT) != 0) {
			vec4 uv_rect = g_uv_rects[instance_id];
			tex_coord = uv_rect.zw * fract(tex_coord) + uv_rect.xy;
			tex_dx *= uv_rect.zw, tex_dy *= uv_rect.zw;
		}

		vec4 tex_col;
		if((instance_flags & INST_TEX_OPAQUE) != 0)
			tex_col = vec4(textureGrad(opaque_texture, tex_coord, tex_dx, tex_dy).xyz, 1.0);
		else
			tex_col = textureGrad(transparent_texture, tex_coord, tex_dx, tex_dy);
		color *= tex_col;
	}

	if((instance_flags & INST_HAS_VERTEX_COLORS) != 0) {
		vec4 col0, col1, col2;
		getTriangleVertexColors(scratch_tri_offset, col0, col1, col2);
		color *= (1.0 - bary[0] - bary[1]) * col0 + (bary[0] * col1 + bary[1] * col2);
	}

	if(color.a == 0.0)
		return 0;

	vec3 normal;
	if((instance_flags & INST_HAS_VERTEX_NORMALS) != 0) {
		vec3 nrm0, nrm1, nrm2;
		getTriangleVertexNormals(scratch_tri_offset, nrm0, nrm1, nrm2);
		nrm1 -= nrm0;
		nrm2 -= nrm0;
		normal = bary[0] * nrm1 + (bary[1] * nrm2 + nrm0);
	} else {
		normal = decodeNormalUint(unormal);
	}

	float light_value = max(0.0, dot(-lighting.sun_dir, normal) * 0.7 + 0.3);
	color.rgb = SATURATE(finalShading(color.rgb, light_value));
	return encodeRGBA8(color);
}

void shadeSamples(uint bx, uint by, uint sample_count) {
	// TODO: what's the best way to fix broken pixels?
	// full sort ? recreate full depth values and sort pairs?

	uint buf_offset = bx << SEGMENT_SHIFT;
	for(uint i = LIX & 31; i < sample_count; i += 32) {
		uint value = s_buffer[buf_offset + i];
		uint pixel_id = value & 31;
		uint scratch_tri_offset = value >> 8;
		ivec2 pix_pos = ivec2((pixel_id & 7) + (bx << 3), (pixel_id >> 3) + (by << 2));
		float depth;
		s_buffer[buf_offset + i] = shadeSample(pix_pos, scratch_tri_offset, depth);
		s_buffer[buf_offset + i + SEGMENT_SIZE * 8] = (floatBitsToUint(depth) & ~31) | pixel_id;
	}
}

struct ReductionContext {
#ifdef VISUALIZE_ERRORS
	vec4 prev_depths;
#else
	vec3 prev_depths;
#endif
	uvec3 prev_colors;
	vec3 out_color;
};

void initReduceSamples(out ReductionContext ctx) {
#ifdef VISUALIZE_ERRORS
	ctx.prev_depths = vec4(-1.0);
#else
	ctx.prev_depths = vec3(-1.0);
#endif
	ctx.prev_colors = uvec3(0);
	ctx.out_color = decodeRGB8(background_color); // TODO
}

#define POS_OFFSET (SEGMENT_SIZE * 2)
#define BITS_OFFSET (SEGMENT_SIZE * 2 + LSIZE)

// TODO: optimize
void reduceSamples(uint bx, uint sample_count, in out ReductionContext ctx) {
	uint buf_offset = bx << SEGMENT_SHIFT;
	uint mini_offset = LIX & ~31;
	uint pixel_bit = 1u << (LIX & 31);
	uint pixel_id = LIX & 31;

	// Możemy robić od razu grupować piksele w warpach tak jak chcę: 8x4 a nie 16x2
	for(uint i = 0; i < sample_count; i += 32) {
		uint sample_offset = i + (LIX & 31);
		uint sample_pixel_id = s_buffer[buf_offset + sample_offset + SEGMENT_SIZE * 8] & 31;

		s_mini_buffer[LIX] = 0;
		if(sample_offset < sample_count)
			atomicOr(s_mini_buffer[mini_offset + sample_pixel_id], pixel_bit);

		uint bitmask = s_mini_buffer[mini_offset + pixel_id];
		int j = findLSB(bitmask);
		while(j != -1) {
			// TODO: pass through regs?
			uint value = s_buffer[buf_offset + i + j];
			float depth = uintBitsToFloat(s_buffer[buf_offset + i + j + SEGMENT_SIZE * 8]);

			bitmask &= ~(1 << j);
			j = findLSB(bitmask);

			if(depth < ctx.prev_depths[0]) {
				SWAP_UINT(value, ctx.prev_colors[0]);
				SWAP_FLOAT(depth, ctx.prev_depths[0]);
				if(ctx.prev_depths[0] < ctx.prev_depths[1]) {
					SWAP_UINT(ctx.prev_colors[1], ctx.prev_colors[0]);
					SWAP_FLOAT(ctx.prev_depths[1], ctx.prev_depths[0]);
					if(ctx.prev_depths[1] < ctx.prev_depths[2]) {
						SWAP_UINT(ctx.prev_colors[2], ctx.prev_colors[1]);
						SWAP_FLOAT(ctx.prev_depths[2], ctx.prev_depths[1]);

#ifdef VISUALIZE_ERRORS
						if(ctx.prev_depths[2] < ctx.prev_depths[3]) {
							ctx.prev_colors[0] = 0xff0000ff;
							i = sample_count;
							break;
						}
#endif
					}
				}
			}

#ifdef VISUALIZE_ERRORS
			ctx.prev_depths[3] = ctx.prev_depths[2];
#endif
			ctx.prev_depths[2] = ctx.prev_depths[1];
			ctx.prev_depths[1] = ctx.prev_depths[0];
			ctx.prev_depths[0] = depth;

			if(ctx.prev_colors[2] != 0) {
				vec4 cur_color = decodeRGBA8(ctx.prev_colors[2]);
#ifdef ADDITIVE_BLENDING
				ctx.out_color += cur_color.rgb * cur_color.a;
#else
				float cur_trans = 1.0 - cur_color.a;
				ctx.out_color = cur_color.rgb * cur_color.a + ctx.out_color * cur_trans;
#endif
			}

			ctx.prev_colors[2] = ctx.prev_colors[1];
			ctx.prev_colors[1] = ctx.prev_colors[0];
			ctx.prev_colors[0] = value;
		}
	}
}

void finishReduceSamples(int bx, int by, ReductionContext ctx) {
	for(int i = 2; i >= 0; i--)
		if(ctx.prev_colors[i] != 0) {
			vec4 cur_color = decodeRGBA8(ctx.prev_colors[i]);
			float cur_transparency = 1.0 - cur_color.a;
#ifdef ADDITIVE_BLENDING
			ctx.out_color += cur_color.rgb * cur_color.a;
#else
			float cur_trans = 1.0 - cur_color.a;
			ctx.out_color = cur_color.rgb * cur_color.a + ctx.out_color * cur_trans;
#endif
		}

	uint enc_color = encodeRGB8(SATURATE(ctx.out_color));
	outputPixel(ivec2((LIX & 7) + (bx << 3), ((LIX >> 3) & 3) + (by << 2)), enc_color);
}

shared uint s_vis_pixels[BIN_SIZE * BLOCK_HEIGHT];

void initVisualizeSamples() {
	s_vis_pixels[LIX] = 0;
	barrier();
}

void visualizeSamples(int bx, uint sample_count) {
	int buf_offset = bx << SEGMENT_SHIFT;
	for(uint i = LIX & 31; i < sample_count; i += 32) {
		uint value = s_buffer[buf_offset + i];
		uint x = (value & 7) + (bx << 3), y = (value >> 3) & 3;
		atomicAdd(s_vis_pixels[x | (y << 6)], 1);
	}
}

void finishVisualizeSamples(uint by) {
	vec3 color = vec3(s_vis_pixels[LIX]) / 32.0;
	uint enc_col = encodeRGBA8(vec4(SATURATE(color), 1.0));
	outputPixel(ivec2(LIX & 63, (LIX >> 6) + (by << 2)), enc_col);
}

void rasterFragmentCounts(int by) {
	barrier();
	for(uint i = LIX; i < BLOCK_HEIGHT * BIN_SIZE; i += LSIZE) {
		ivec2 pixel_pos = ivec2(i & (BIN_SIZE - 1), by * BLOCK_HEIGHT + (i >> BIN_SHIFT));
		uint bx = pixel_pos.x / BLOCK_WIDTH, by = pixel_pos.y / BLOCK_HEIGHT;
		uint count0 = s_block_frag_count[bx] & 0xffff;
		//count0 = s_block_tri_count[bx] * 8;

		vec3 color = vec3(count0, count0, count0) / 512;
		uint enc_col = encodeRGBA8(vec4(SATURATE(color), 1.0));
		outputPixel(pixel_pos, enc_col);
	}
	barrier();
}

void rasterBin(int bin_id) {
	INIT_CLOCK();

	if(LIX < NUM_BLOCK_ROWS) {
		if(LIX == 0) {
			ivec2 bin_pos = ivec2(bin_id % BIN_COUNT_X, bin_id / BIN_COUNT_X) * BIN_SIZE;
			s_bin_pos = bin_pos;
			s_bin_quad_count = g_bins.bin_quad_counts[bin_id];
			s_bin_quad_offset = g_bins.bin_quad_offsets[bin_id];
			s_bin_ray_dir0 = frustum.ws_dir0 + frustum.ws_dirx * (bin_pos.x + 0.5) +
							 frustum.ws_diry * (bin_pos.y + 0.5);
			s_raster_error = 0;
		}

		s_block_row_tri_count[LIX] = 0;
	}
	barrier();
	processQuads();
	groupMemoryBarrier();
	barrier();
	UPDATE_CLOCK(0);

	for(int by = 0; by < NUM_BLOCK_ROWS; by++) {
		barrier();
		generateBlocks(by);
		groupMemoryBarrier();
		barrier();
		UPDATE_CLOCK(1);

		if(s_raster_error != 0) {
			ivec2 pixel_pos = ivec2(LIX & (BIN_SIZE - 1), (LIX >> BIN_SHIFT) + (by << 2));
			uint bx = pixel_pos.x >> 3;
			uint color = 0xff000032;
			if((s_raster_error & (1 << bx)) != 0)
				color += 0x32;
			if((s_raster_error & (0x100 << bx)) != 0)
				color += 0x64;
			outputPixel(pixel_pos, color);
			barrier();
			if(LIX == 0)
				s_raster_error = 0;
			continue;
		}

		int segment_id = 0;
		int bx = int(LIX >> 5);
		int frag_count = int(s_block_frag_count[bx]);
		// TODO: handle empty block?

		ReductionContext context;
		initReduceSamples(context);
		//initVisualizeSamples();

		while(frag_count > 0) {
			int cur_frag_count = min(frag_count, SEGMENT_SIZE);
			frag_count -= SEGMENT_SIZE;

			loadSamples(bx, by, segment_id++, cur_frag_count);
			UPDATE_CLOCK(2);

			//visualizeSamples(bx, cur_frag_count);
			shadeSamples(bx, by, cur_frag_count);
			UPDATE_CLOCK(3);

			reduceSamples(bx, cur_frag_count, context);
			UPDATE_CLOCK(4);
		}
		barrier();

		finishReduceSamples(bx, by, context);
		//finishVisualizeSamples(by);
		//rasterFragmentCounts(by);
	}
}

int loadNextBin() {
	if(LIX == 0) {
		uint bin_idx = atomicAdd(g_bins.small_bin_counter, 1);
		s_bin_id = bin_idx < s_num_bins ? g_bins.small_bins[bin_idx] : -1;
		s_bin_raster_offset = s_bin_id << (BIN_SHIFT * 2);
	}
	barrier();
	return s_bin_id;
}

void main() {
	initTimers();
	if(LIX == 0)
		s_num_bins = g_bins.num_small_bins;

	int bin_id = loadNextBin();
	while(bin_id != -1) {
		barrier();
		rasterBin(bin_id);
		bin_id = loadNextBin();
	}
	commitTimers();
}
