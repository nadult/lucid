// $$include funcs lighting frustum viewport data

#define LIX gl_LocalInvocationIndex
#define LID gl_LocalInvocationID

#define LSIZE 512
#define LSHIFT 9

#define MAX_ROWS 8

#define BLOCK_ROW_SIZE (64 * 8)

layout(local_size_x = LSIZE) in;

layout(std430, binding = 0) buffer buf0_ { uvec4 g_tri_aabbs[]; };
layout(std430, binding = 1) buffer buf1_ { uint g_quad_indices[]; };

layout(std430, binding = 2) readonly buffer buf2_ { float g_verts[]; };
layout(std430, binding = 3) readonly buffer buf3_ { vec2 g_tex_coords[]; };
layout(std430, binding = 4) readonly buffer buf4_ { uint g_colors[]; };
layout(std430, binding = 5) readonly buffer buf5_ { uint g_normals[]; };

layout(std430, binding = 6) buffer buf6_ { BinCounters g_bins; };

layout(std430, binding = 8) buffer buf8_ { uint g_bin_quads[]; };
layout(std430, binding = 9) coherent buffer buf9_ { uvec2 g_scratch[]; };

layout(std430, binding = 10) readonly buffer buf10_ { InstanceData g_instances[]; };
layout(std430, binding = 11) readonly buffer buf11_ { vec4 g_uv_rects[]; };
layout(std430, binding = 12) writeonly buffer buf12_ { uint g_raster_image[]; };

layout(binding = 0) uniform sampler2D opaque_texture;
layout(binding = 1) uniform sampler2D transparent_texture;

uniform bool additive_blending;

#define WORKGROUP_SCRATCH_SIZE (256 * 1024)
#define WORKGROUP_SCRATCH_SHIFT 18

// TODO: does that mean that occupancy is so low?
#define SAMPLES_PER_THREAD 8
#define MAX_SAMPLES (LSIZE * SAMPLES_PER_THREAD)

#define BLOCK_COUNT 8

#define MAX_BLOCK_ROW_TRIS 2048
#define MAX_BLOCK_TRIS (MAX_SAMPLES / BLOCK_COUNT)
#define MAX_SCRATCH_TRIS 2048
#define MAX_SCRATCH_TRIS_SHIFT 11

#define TRI_SCRATCH(var_idx) g_scratch[scratch_tri_offset + (var_idx << MAX_SCRATCH_TRIS_SHIFT)]

uint scratchBlockRowTrisOffset(uint by) {
	return (gl_WorkGroupID.x << WORKGROUP_SCRATCH_SHIFT) + by * MAX_BLOCK_ROW_TRIS * 2;
}

uint scratchBlockTrisOffset(uint bx) {
	return (gl_WorkGroupID.x << WORKGROUP_SCRATCH_SHIFT) + 32 * 1024 + bx * MAX_BLOCK_TRIS * 2;
}

uint scratchTriOffset(uint tri_idx) {
	return (gl_WorkGroupID.x << WORKGROUP_SCRATCH_SHIFT) + 64 * 1024 + tri_idx;
}

uint scratchBlockTrisDepthsOffset(uint bx) {
	return (gl_WorkGroupID.x << WORKGROUP_SCRATCH_SHIFT) + 128 * 1024 + bx * MAX_BLOCK_TRIS;
}

// Note: in the context of this shader, block size is 8x8, not 4x4
shared int s_num_bins, s_bin_id, s_bin_raster_offset;
shared ivec2 s_bin_pos;
shared vec3 s_bin_ray_dir0;

shared uint s_block_row_tri_counts[BLOCK_COUNT];
shared uint s_curblock_counters[BLOCK_COUNT * 4];

void outputPixel(ivec2 pixel_pos, uint color) {
	g_raster_image[s_bin_raster_offset + pixel_pos.x + (pixel_pos.y << BIN_SHIFT)] = color;
}

//#define ENABLE_TIMINGS

// Note: UPDATE_CLOCK should be called after a barrier
#ifdef ENABLE_TIMINGS
shared uint64_t s_timings[8];
#define INIT_CLOCK() uint64_t clock0 = clockARB();
#define UPDATE_CLOCK(idx)                                                                          \
	if(LIX == 0) {                                                                                 \
		uint64_t clock = clockARB();                                                               \
		s_timings[idx] += clock - clock0;                                                          \
		clock0 = clock;                                                                            \
	}

void initTimers() {
	if(LIX < 8)
		s_timings[LIX] = 0;
}
void commitTimers() {
	if(LIX < 8)
		atomicAdd(g_bins.timings[LIX], uint(s_timings[LIX]));
}

#else
#define INIT_CLOCK()
#define UPDATE_CLOCK(idx)

void initTimers() {}
void commitTimers() {}
#endif

void resetBlockCounters() {
	if(LIX < BLOCK_COUNT)
		s_curblock_counters[LIX] = 0;
}

// These functions work only within current block row
#define BLOCK_TRI_COUNT(bx) s_curblock_counters[bx]

// TODO: more info
#define BLOCK_FRAG_COUNT(bx) s_curblock_counters[bx + BLOCK_COUNT]
#define BLOCK_FRAG_OFFSET(bx) s_curblock_counters[bx + BLOCK_COUNT * 2]
#define BLOCK_GROUP_FRAG_COUNT(bx) s_curblock_counters[bx + BLOCK_COUNT * 3]

// How many blocks in a row can we rasterize in single step (log2)
// -1: invalid (not enough space for samples)
//  0: 1 block, 1: 2 blocks, 2: 4 blocks, 3: 8 blocks (best option)
shared int s_max_raster_blocks;

// TODO: add protection from too big number of samples:
// maximum per row for raster_bin = min(4 * LSIZE, 32768) ?
// we have to somehow estimate max# of samples during categorization?
// max 512 tris per block -> max samples = 512 * 64: fits in 15 bits

shared uint s_bin_quad_count, s_bin_quad_offset;

shared uint s_buffer[MAX_SAMPLES + 1];
shared uint s_mini_buffer[16 * BLOCK_COUNT];

#ifdef VENDOR_NVIDIA
uint swap(uint x, int mask, uint dir) {
	uint y = shuffleXorNV(x, mask, 32);
	return uint(x < y) == dir ? y : x;
}

uint bitExtract(uint value, int boffset) { return (value >> boffset) & 1; }

uint xorBits(uint value, int bit0, int bit1) { return ((value >> bit0) ^ (value >> bit1)) & 1; }
#endif

shared uint s_sort_max_block_rcount;

void sortBlockTris() {
	if(LIX < 8) {
		uint count = BLOCK_TRI_COUNT(LIX);
		// rcount: count rounded up to next power of 2
		uint rcount = max(32, (count & (count - 1)) == 0 ? count : (2 << findMSB(count)));
		if(LIX == 0)
			s_sort_max_block_rcount = 0;
		atomicMax(s_sort_max_block_rcount, rcount);
	}
	barrier();

	uint gid = LIX >> (LSHIFT - 3);
	uint lid = LIX & (LSIZE / 8 - 1);
	uint goffset = gid * MAX_BLOCK_TRIS;
	uint count = BLOCK_TRI_COUNT(gid);
	// TODO: max_rcount is only needed for barriers, computations should be performed up to rcount
	// But it seems, that using rcount directly is actually a bit slower... (Sponza)
	uint max_rcount = s_sort_max_block_rcount;
	for(uint i = lid + count; i < max_rcount; i += LSIZE / 8)
		s_buffer[goffset + i] = 0xffffffff;
	barrier();

#ifdef VENDOR_NVIDIA
	for(uint i = lid; i < max_rcount; i += LSIZE / 8) {
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
		value = swap(value, 0x10, xorBits(lid, 5, 4)); // K = 32
		value = swap(value, 0x08, xorBits(lid, 5, 3));
		value = swap(value, 0x04, xorBits(lid, 5, 2));
		value = swap(value, 0x02, xorBits(lid, 5, 1));
		value = swap(value, 0x01, xorBits(lid, 5, 0));
		s_buffer[goffset + i] = value;
	}
	barrier();
	int start_k = 64, end_j = 32;
#else
	int start_k = 2, end_j = 1;
#endif
	for(uint k = start_k; k <= max_rcount; k = 2 * k) {
		for(uint j = k >> 1; j >= end_j; j = j >> 1) {
			for(uint i = lid; i < max_rcount; i += (LSIZE / 8) * 2) {
				uint idx = (i & j) != 0 ? i + (LSIZE / 8) - j : i;
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
		for(uint i = lid; i < max_rcount; i += LSIZE / 8) {
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

// TODO: add synthetic test: 256 planes one after another

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

// TODO: można by tutaj użyć algorytmu bazującego na liniach
void generateTriGroups(uint tri_idx, vec3 tri0, vec3 tri1, vec3 tri2, int min_by, int max_by) {
	// Inspired by Nanite scanline rasterizer
	vec3 scan_min, scan_max, scan_step;
	{
		float sx = s_bin_pos.x - 0.5f;
		float sy = s_bin_pos.y + min_by * 8 + 0.5f;

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

	uint soffset = scratchBlockRowTrisOffset(min_by);
	for(int by = min_by; by <= max_by; by++) {
		uint row_ranges[4] = {0, 0, 0, 0};
		uint bmask = 0;

		for(int y = 0; y < 8; y++) {
			float xmin = max(max(scan_min[0], scan_min[1]), max(scan_min[2], 0.0));
			float xmax = min(min(scan_max[0], scan_max[1]), min(scan_max[2], BIN_SIZE));

			scan_min += scan_step;
			scan_max += scan_step;

			// TODO: use floor/ceil?
			int imin = int(xmin), imax = int(xmax) - 1;
			if(imin <= imax) {
				uint bmin = imin >> 3, bmax = imax >> 3;
				bmask |= (0xff << bmin) & (0xff >> (7 - bmax));
			}
			row_ranges[y >> 1] |= (imin <= imax ? (uint(imin) | (uint(imax) << 6)) : 0x3f)
								  << ((y & 1) * 12);
		}

		if(bmask != 0) {
			uint roffset = atomicAdd(s_block_row_tri_counts[by], 1) * 2;
			g_scratch[soffset + roffset] =
				uvec2(row_ranges[0] | (tri_idx << 24), row_ranges[1] | ((tri_idx & 0xff00) << 16));
			g_scratch[soffset + roffset + 1] = uvec2(row_ranges[2] | (bmask << 24), row_ranges[3]);
		}
		soffset += MAX_BLOCK_ROW_TRIS * 2;
	}
}

// TODO: don't store triangles which generate very small number of samples in scratch,
// instead precompute them directly when sampling; We would have to somehow group those triangles together
//
// TODO: use scratch based on uints, not uvec2, maybe it will be a bit faster?

void storeTriangle(uint tri_idx, vec3 tri0, vec3 tri1, vec3 tri2, uint v0, uint v1, uint v2,
				   uint instance_id) {
	uint scratch_tri_offset = scratchTriOffset(tri_idx);
	vec3 normal = cross(tri0 - tri2, tri1 - tri0);
	float multiplier = 1.0 / length(normal);
	normal *= multiplier;

	vec3 edge0 = (tri0 - tri2) * multiplier;
	vec3 edge1 = (tri1 - tri0) * multiplier;

	float plane_dist = dot(normal, tri0);
	float param0 = dot(cross(edge0, tri0), normal);
	float param1 = dot(cross(edge1, tri0), normal);

	// Nice optimization for barycentric computations:
	// dot(cross(edge, dir), normal) == dot(dir, cross(normal, edge))
	edge0 = cross(normal, edge0);
	edge1 = cross(normal, edge1);

	TRI_SCRATCH(0) = uvec2(floatBitsToUint(normal.x), floatBitsToUint(normal.y));
	TRI_SCRATCH(1) = uvec2(floatBitsToUint(normal.z), floatBitsToUint(plane_dist));
	TRI_SCRATCH(2) = uvec2(floatBitsToUint(param0), floatBitsToUint(param1));
	TRI_SCRATCH(3) = uvec2(floatBitsToUint(edge0.x), floatBitsToUint(edge0.y));
	TRI_SCRATCH(4) = uvec2(floatBitsToUint(edge0.z), floatBitsToUint(edge1.x));
	TRI_SCRATCH(5) = uvec2(floatBitsToUint(edge1.y), floatBitsToUint(edge1.z));

	vec3 pnormal = normal * (1.0 / plane_dist);
	vec3 depth_eq = vec3(dot(pnormal, s_bin_ray_dir0), dot(pnormal, frustum.ws_dirx),
						 dot(pnormal, frustum.ws_diry));
	TRI_SCRATCH(14) = uvec2(floatBitsToUint(depth_eq.x), floatBitsToUint(depth_eq.y));
	TRI_SCRATCH(15) = uvec2(floatBitsToUint(depth_eq.z), 0);

	uint instance_flags = g_instances[instance_id].flags;
	uint instance_color = g_instances[instance_id].color;

	TRI_SCRATCH(6) = uvec2(instance_flags, instance_color);
	TRI_SCRATCH(7) = uvec2(instance_id, 0);

	uint vcolor2 = 0;
	if((instance_flags & INST_HAS_VERTEX_COLORS) != 0) {
		TRI_SCRATCH(8) = uvec2(g_colors[v0], g_colors[v1]);
		vcolor2 = g_colors[v2];
	}
	uint vnormal2 = 0;
	if((instance_flags & INST_HAS_VERTEX_NORMALS) != 0) {
		TRI_SCRATCH(10) = uvec2(g_normals[v0], g_normals[v1]);
		vnormal2 = g_normals[v2];
	}

	if((instance_flags & (INST_HAS_VERTEX_COLORS | INST_HAS_VERTEX_NORMALS)) != 0) {
		TRI_SCRATCH(9) = uvec2(vcolor2, vnormal2);
	}
	if((instance_flags & INST_HAS_TEXTURE) != 0) {
		vec2 tex0 = g_tex_coords[v0];
		vec2 tex1 = g_tex_coords[v1];
		vec2 tex2 = g_tex_coords[v2];
		tex1 -= tex0;
		tex2 -= tex0;
		TRI_SCRATCH(11) = floatBitsToUint(tex0);
		TRI_SCRATCH(12) = floatBitsToUint(tex1);
		TRI_SCRATCH(13) = floatBitsToUint(tex2);
	}
}

void getTriangleParams(uint scratch_tri_offset, out vec3 normal, out vec3 params, out vec3 edge0,
					   out vec3 edge1, out uint instance_id, out uint instance_flags,
					   out uint instance_color) {
	{
		uvec2 val0 = TRI_SCRATCH(0), val1 = TRI_SCRATCH(1), val2 = TRI_SCRATCH(2);
		normal = vec3(uintBitsToFloat(val0[0]), uintBitsToFloat(val0[1]), uintBitsToFloat(val1[0]));
		params = vec3(uintBitsToFloat(val1[1]), uintBitsToFloat(val2[0]), uintBitsToFloat(val2[1]));
	}
	{
		uvec2 val0 = TRI_SCRATCH(3), val1 = TRI_SCRATCH(4), val2 = TRI_SCRATCH(5);
		edge0 = vec3(uintBitsToFloat(val0[0]), uintBitsToFloat(val0[1]), uintBitsToFloat(val1[0]));
		edge1 = vec3(uintBitsToFloat(val1[1]), uintBitsToFloat(val2[0]), uintBitsToFloat(val2[1]));
	}

	{
		uvec2 val0 = TRI_SCRATCH(6);
		instance_flags = val0.x;
		instance_color = val0.y;
		instance_id = TRI_SCRATCH(7).x;
	}
}

void getTriangleVertexColors(uint scratch_tri_offset, out vec4 color0, out vec4 color1,
							 out vec4 color2) {
	uvec2 val0 = TRI_SCRATCH(8);
	uvec2 val1 = TRI_SCRATCH(9);
	color0 = decodeRGBA8(val0[0]);
	color1 = decodeRGBA8(val0[1]);
	color2 = decodeRGBA8(val1[0]);
}

void getTriangleVertexNormals(uint scratch_tri_offset, out vec3 normal0, out vec3 normal1,
							  out vec3 normal2) {
	uvec2 val0 = TRI_SCRATCH(10);
	uvec2 val1 = TRI_SCRATCH(9);
	normal0 = decodeNormalUint(val0[0]);
	normal1 = decodeNormalUint(val0[1]);
	normal2 = decodeNormalUint(val1[1]);
}

void getTriangleVertexTexCoords(uint scratch_tri_offset, out vec2 tex0, out vec2 tex1,
								out vec2 tex2) {
	uvec2 val0 = TRI_SCRATCH(11);
	uvec2 val1 = TRI_SCRATCH(12);
	uvec2 val2 = TRI_SCRATCH(13);
	tex0 = uintBitsToFloat(val0);
	tex1 = uintBitsToFloat(val1);
	tex2 = uintBitsToFloat(val2);
}

void generateRows() {
	// TODO: optimization: in many cases all rows may very well fit in SMEM,
	// maybe it would be worth it not to use scratch at all then?
	// TODO: this loop is slooooow
	// TODO: divide big tris across different threads
	for(uint i = LIX >> 1; i < s_bin_quad_count; i += LSIZE / 2) {
		uint second_tri = LIX & 1;
		uint quad_idx = g_bin_quads[s_bin_quad_offset + i] & 0xffffff;

		uvec4 aabb = g_tri_aabbs[quad_idx];
		aabb = decodeAABB(second_tri != 0 ? aabb.zw : aabb.xy);
		int min_by = clamp(int(aabb[1]) - s_bin_pos.y, 0, 63) >> 3;
		int max_by = clamp(int(aabb[3]) - s_bin_pos.y, 0, 63) >> 3;

		uint verts[4] = {g_quad_indices[quad_idx * 4 + 0], g_quad_indices[quad_idx * 4 + 1],
						 g_quad_indices[quad_idx * 4 + 2], g_quad_indices[quad_idx * 4 + 3]};
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
		generateTriGroups(tri_idx, tri0, tri1, tri2, min_by, max_by);
	}
}

void generateBlocks1(uint by) {
	// TODO: is this really the best order?
	uint bx = LIX & 7;
	uint src_offset = scratchBlockRowTrisOffset(by);
	uint buf_offset = bx * MAX_BLOCK_TRIS;
	uint tri_count = s_block_row_tri_counts[by];
	uint y = LIX & 7, shift = (y & 1) * 12;
	resetBlockCounters();
	barrier();

	int min_bx = int(bx * 8);
	for(uint i = LIX >> 3; i < tri_count; i += LSIZE / 8) {
		// TODO: load these together with shuffles?
		uint full_rows[4] = {
			g_scratch[src_offset + i * 2 + 0].x, g_scratch[src_offset + i * 2 + 0].y,
			g_scratch[src_offset + i * 2 + 1].x, g_scratch[src_offset + i * 2 + 1].y};

		// TODO: load range data in groups
		uint bmask = full_rows[2] >> 24;
		if((bmask & (1 << bx)) == 0)
			continue;

		// TODO: keep tri_idx & bmask in one place
		uint tri_idx = (full_rows[0] >> 24) | ((full_rows[1] >> 16) & 0xff00);

		uint scratch_tri_offset = scratchTriOffset(tri_idx);
		vec2 val0 = uintBitsToFloat(TRI_SCRATCH(14));
		vec2 val1 = uintBitsToFloat(TRI_SCRATCH(15));
		vec3 depth_eq = vec3(val0.x, val0.y, val1.x);
		depth_eq.x += depth_eq.y * (bx << 3) + depth_eq.z * (by << 3);

		float min_depth = 999999999.0;
		// TODO: optimize based on sign of depth_eq.y & depth_eq.z
		for(int j = 0; j < 4; j++) {
			int row0 = int(full_rows[j] & 0xfff), row1 = int((full_rows[j] >> 12) & 0xfff);
			// TODO: these are computed twice
			int minx0 = max((row0 & 0x3f) - min_bx, 0),
				maxx0 = min(((row0 >> 6) & 0x3f) - min_bx, 7);
			int minx1 = max((row1 & 0x3f) - min_bx, 0),
				maxx1 = min(((row1 >> 6) & 0x3f) - min_bx, 7);

			float row_depth = depth_eq.x + depth_eq.z * (j * 2);
			if(maxx0 >= minx0) {
				float depth = row_depth + depth_eq.y * (depth_eq.y < 0.0 ? minx0 : maxx0);
				min_depth = min(min_depth, depth);
			}
			if(maxx1 >= minx1) {
				float depth =
					row_depth + depth_eq.y * (depth_eq.y < 0.0 ? minx1 : maxx1) + depth_eq.z;
				min_depth = min(min_depth, depth);
			}
		}

		float depth = float(0xfffff) / (1.0 + max(0.0, 1.0 / min_depth)); // 20 bits is enough

		uint idx = atomicAdd(BLOCK_TRI_COUNT(bx), 1);
		if(idx < MAX_BLOCK_TRIS)
			s_buffer[buf_offset + idx] = i | (uint(depth) << 12);
		else
			s_max_raster_blocks = -1;
	}

	barrier();
	if(s_max_raster_blocks == -1)
		return;
	sortBlockTris();
	barrier();

	bx = LIX >> (LSHIFT - 3);
	min_bx = int(bx * 8);
	buf_offset = bx * MAX_BLOCK_TRIS;
	tri_count = BLOCK_TRI_COUNT(bx);
	uint dst_soffset = scratchBlockTrisOffset(bx);
	uint depth_soffset = scratchBlockTrisDepthsOffset(bx);

#define PREFIX_SUM_STEP(value, step)                                                               \
	{                                                                                              \
		uint temp = shuffleUpNV(value, step, 32);                                                  \
		if((LIX & 31) >= step)                                                                     \
			value += temp;                                                                         \
	}

	for(uint i = LIX & (LSIZE / 8 - 1); i < tri_count; i += LSIZE / 8) {
		uint idx = s_buffer[buf_offset + i] & 0xfff;

		// TODO: load range data in groups
		uint full_rows[4] = {
			g_scratch[src_offset + idx * 2 + 0].x, g_scratch[src_offset + idx * 2 + 0].y,
			g_scratch[src_offset + idx * 2 + 1].x, g_scratch[src_offset + idx * 2 + 1].y};
		uint tri_idx = (full_rows[0] >> 24) | ((full_rows[1] >> 16) & 0xff00);
		uint bits[2];

		uint scratch_tri_offset = scratchTriOffset(tri_idx);
		vec2 val0 = uintBitsToFloat(TRI_SCRATCH(14));
		vec2 val1 = uintBitsToFloat(TRI_SCRATCH(15));
		vec3 depth_eq = vec3(val0.x, val0.y, val1.x);
		depth_eq.x += depth_eq.y * (bx << 3) + depth_eq.z * (by << 3);

		// TODO: how to compute these accurately ? sample in the middle of pixel?
		float min_depth = 999999999.0, max_depth = -999999999.0;

		for(int j = 0; j < 2; j++) {
			int rows01 = int(full_rows[j * 2 + 0]), rows23 = int(full_rows[j * 2 + 1]);
			int minx0 = max(((rows01 >> 0) & 0x3f) - min_bx, 0),
				maxx0 = min(((rows01 >> 6) & 0x3f) - min_bx, 7);
			int minx1 = max(((rows01 >> 12) & 0x3f) - min_bx, 0),
				maxx1 = min(((rows01 >> 18) & 0x3f) - min_bx, 7);
			int minx2 = max(((rows23 >> 0) & 0x3f) - min_bx, 0),
				maxx2 = min(((rows23 >> 6) & 0x3f) - min_bx, 7);
			int minx3 = max(((rows23 >> 12) & 0x3f) - min_bx, 0),
				maxx3 = min(((rows23 >> 18) & 0x3f) - min_bx, 7);

			uint mask0 = (minx0 <= maxx0 ? ~0u : 0);
			uint mask1 = (minx1 <= maxx1 ? ~0u : 0);
			uint mask2 = (minx2 <= maxx2 ? ~0u : 0);
			uint mask3 = (minx3 <= maxx3 ? ~0u : 0);

			float row_depth = depth_eq.x + depth_eq.z * (j * 4);
#define COMPUTE_ROW_DEPTH(rmin, rmax)                                                              \
	if(rmax >= rmin) {                                                                             \
		float depth0 = row_depth + depth_eq.y * rmin;                                              \
		float depth1 = row_depth + depth_eq.y * (rmax + 1);                                        \
		min_depth = min(min_depth, min(depth0, depth1));                                           \
		max_depth = max(max_depth, max(depth0, depth1));                                           \
	}                                                                                              \
	row_depth += depth_eq.z;

			COMPUTE_ROW_DEPTH(minx0, maxx0)
			COMPUTE_ROW_DEPTH(minx1, maxx1)
			COMPUTE_ROW_DEPTH(minx2, maxx2)
			COMPUTE_ROW_DEPTH(minx3, maxx3)

			uint bits0 = mask0 & (0x000000ffu << minx0) & (0x000000ffu >> (7 - maxx0));
			uint bits1 = mask1 & (0x0000ff00u << minx1) & (0x0000ff00u >> (7 - maxx1));
			uint bits2 = mask2 & (0x00ff0000u << minx2) & (0x00ff0000u >> (7 - maxx2));
			uint bits3 = mask3 & (0xff000000u << minx3) & (0xff000000u >> (7 - maxx3));

			bits[j] = bits0 | bits1 | bits2 | bits3;
		}

		g_scratch[depth_soffset + i] =
			uvec2(floatBitsToUint(min_depth), floatBitsToUint(max_depth));

		uint num_frags0123 = bitCount(bits[0]);
		uint num_frags4567 = bitCount(bits[1]);
		uint num_frags = num_frags0123 + num_frags4567;
		if(num_frags == 0) // This means that bmasks are invalid
			RECORD(0, 0, 0, 0);

		g_scratch[dst_soffset + i] = uvec2(bits[0], bits[1]);
		g_scratch[dst_soffset + i + MAX_BLOCK_TRIS].x =
			tri_idx | (num_frags << 16) | (num_frags0123 << 24);
	}
	barrier();
}

void generateBlocks2(uint by) {
	uint bx = LIX >> (LSHIFT - 3);
	uint min_bx = int(bx * 8);
	uint buf_offset = bx * MAX_BLOCK_TRIS;
	uint tri_count = BLOCK_TRI_COUNT(bx);
	uint dst_soffset = scratchBlockTrisOffset(bx);

	for(uint i = LIX & (LSIZE / 8 - 1); i < tri_count; i += LSIZE / 8) {
		uint value = g_scratch[dst_soffset + i + MAX_BLOCK_TRIS].x;
		uint num_frags = (value >> 16) & 0xff;

		// Computing triangle-ordered sample offsets within each block
		PREFIX_SUM_STEP(num_frags, 1);
		PREFIX_SUM_STEP(num_frags, 2);
		PREFIX_SUM_STEP(num_frags, 4);
		PREFIX_SUM_STEP(num_frags, 8);
		PREFIX_SUM_STEP(num_frags, 16);
		s_buffer[buf_offset + i] = num_frags;
	}

	barrier();

	// Computing offsets for each triangle within block
	uint idx32 = (LIX & (LSIZE / 8 - 1)) * 32;
	// Note: here we expect that idx32 < 32 * 16
	if(idx32 < tri_count) {
		uint value = s_buffer[buf_offset + idx32 + 31];
		PREFIX_SUM_STEP(value, 1);
		PREFIX_SUM_STEP(value, 2);
		PREFIX_SUM_STEP(value, 4);
		PREFIX_SUM_STEP(value, 8);
		s_mini_buffer[bx * 16 + (idx32 >> 5)] = value;
	}
	barrier();

	for(uint i = 32 + (LIX & (LSIZE / 8 - 1)); i < tri_count; i += LSIZE / 8)
		s_buffer[buf_offset + i] += s_mini_buffer[bx * 16 + (i >> 5) - 1];
	barrier();

	// Storing offsets to scratch mem
	for(uint i = LIX & (LSIZE / 8 - 1); i < tri_count; i += LSIZE / 8) {
		uint value = i == 0 ? 0 : s_buffer[buf_offset + i - 1];
		g_scratch[dst_soffset + i + MAX_BLOCK_TRIS].y = value;
	}

#ifdef SHADER_DEBUG
	for(uint i = LIX & (LSIZE / 8 - 1); i < tri_count; i += LSIZE / 8) {
		uint value = s_buffer[buf_offset + i] & 0xffff;
		uint prev_value = i == 0 ? 0 : s_buffer[buf_offset + i - 1] & 0xffff;
		if(value <= prev_value)
			RECORD(i, tri_count, prev_value, value);
	}
#endif

	if(LIX < BLOCK_COUNT) {
		uint bx = LIX;
		uint num_tris = BLOCK_TRI_COUNT(bx);
		uint frag_count1 = num_tris == 0 ? 0 : s_buffer[bx * MAX_BLOCK_TRIS + num_tris - 1];
		BLOCK_FRAG_COUNT(bx) = frag_count1;

		uint frag_count2 = frag_count1 + shuffleXorNV(frag_count1, 1, 8);
		uint frag_count4 = frag_count2 + shuffleXorNV(frag_count2, 2, 8);
		uint frag_count8 = frag_count4 + shuffleXorNV(frag_count4, 4, 8);

		int max_raster_blocks = 3;
		// TODO: compute this on first warp only
		if(frag_count8 > MAX_SAMPLES) {
			bool fail4 = anyInvocationARB(frag_count4 > MAX_SAMPLES);
			bool fail2 = anyInvocationARB(frag_count2 > MAX_SAMPLES);
			bool fail1 = anyInvocationARB(frag_count1 > MAX_SAMPLES);
			max_raster_blocks = fail1 ? -1 : fail2 ? 0 : fail4 ? 1 : 2;
		}
		s_max_raster_blocks = max_raster_blocks;

		if(max_raster_blocks >= 0) {
			uint frag_offset = 0;
			uint temp1 = shuffleUpNV(frag_count1, 1, 8);
			uint temp2 = shuffleUpNV(frag_count2, 2, 8);
			uint temp4 = shuffleUpNV(frag_count4, 4, 8);
			if(max_raster_blocks >= 1 && (bx & 1) != 0)
				frag_offset += temp1;
			if(max_raster_blocks >= 2 && (bx & 2) != 0)
				frag_offset += temp2;
			if(max_raster_blocks >= 3 && (bx & 4) != 0)
				frag_offset += temp4;

			BLOCK_FRAG_OFFSET(bx) = frag_offset;

			uint block_count = 1 << max_raster_blocks;
			uint first_bx = bx & ~(block_count - 1), last_bx = first_bx + block_count - 1;
			uint group_count =
				shuffleNV(frag_count1, last_bx, 8) + shuffleNV(frag_offset, last_bx, 8);
			BLOCK_GROUP_FRAG_COUNT(bx) = group_count;
			BLOCK_FRAG_OFFSET(bx) += group_count >> 16;
		}
	}
	barrier();
}

void splitTris(int bx, int by) {
	uint tri_count = BLOCK_TRI_COUNT(bx);
	uint dst_soffset = scratchBlockTrisOffset(bx);
	uint depth_soffset = scratchBlockTrisDepthsOffset(bx);

#define BUFFER(i, j) s_buffer[(i) + MAX_BLOCK_TRIS * (j)]
	for(uint i = 0; i < 8; i++)
		BUFFER(LIX, i) = 0;
	barrier();

	// TODO: compute min/max depth in this loop?
	for(uint i = LIX; i < tri_count; i += LSIZE) {
		uvec2 bits = g_scratch[dst_soffset + i];
		uvec2 depths = g_scratch[depth_soffset + i];
		uint info = g_scratch[dst_soffset + i + MAX_BLOCK_TRIS].x;

		BUFFER(i, 0) = bits.x;
		BUFFER(i, 1) = bits.y;
		BUFFER(i, 2) = info;
		BUFFER(i, 3) = depths.x;
		BUFFER(i, 4) = depths.y;
		BUFFER(i, 6) = 1;
	}

	barrier();

	// Identifying groups with overlapping tris
	for(uint i = LIX; i < tri_count; i += LSIZE) {
		uvec2 bits = uvec2(BUFFER(i, 0), BUFFER(i, 1));
		float min_depth = uintBitsToFloat(BUFFER(i, 3));
		float max_depth = uintBitsToFloat(BUFFER(i, 4));
		uint tri_idx = BUFFER(i, 2) & 0xffff;

		uint scratch_tri_offset = scratchTriOffset(tri_idx);
		vec2 val0 = uintBitsToFloat(TRI_SCRATCH(14));
		vec2 val1 = uintBitsToFloat(TRI_SCRATCH(15));
		vec3 depth_eq = vec3(val0.x, val0.y, val1.x);
		depth_eq.x += depth_eq.y * (bx << 3) + depth_eq.z * (by << 3);

		bool hit = false;

		for(uint j = i + 1; j < tri_count; j++) {
			float min_depth_j = uintBitsToFloat(BUFFER(j, 3));
			if(min_depth_j > max_depth && min_depth_j >= min_depth) // TODO: second check needed?
				break;

			uvec2 bits_j = uvec2(BUFFER(j, 0), BUFFER(j, 1));
			if((bits.x & bits_j.x) == 0 && (bits.y & bits_j.y) == 0)
				continue;

			uint tri_idx = BUFFER(j, 2) & 0xffff;
			uint scratch_tri_offset = scratchTriOffset(tri_idx);
			vec2 val0 = uintBitsToFloat(TRI_SCRATCH(14));
			vec2 val1 = uintBitsToFloat(TRI_SCRATCH(15));
			vec3 depth_eq_j = vec3(val0.x, val0.y, val1.x);
			depth_eq_j.x += depth_eq_j.y * (bx << 3) + depth_eq_j.z * (by << 3);

			bool cur_hit = false;
			for(int y = 0; y < 8; y++) {
				uint bits_i = ((y < 4 ? bits.x : bits.y) >> ((y & 3) << 3)) & 0xff;
				uint bits_j = ((y < 4 ? bits_j.x : bits_j.y) >> ((y & 3) << 3)) & 0xff;
				uint bits = bits_i & bits_j;

				float depth_row = depth_eq.x + depth_eq.z * y;
				float depth_row_j = depth_eq_j.x + depth_eq_j.z * y;

				if(bits != 0) {
					int min_x = findLSB(bits), max_x = findMSB(bits) + 1;
					if(depth_row + depth_eq.y * min_x >= depth_row_j + depth_eq_j.y * min_x ||
					   depth_row + depth_eq.y * max_x >= depth_row_j + depth_eq_j.y * max_x)
						cur_hit = true;
				}
			}

			if(cur_hit) {
				BUFFER(j, 6) = 0;
				hit = true;
			}
		}

		if(hit)
			BUFFER(i, 6) = 0;
	}
	barrier();
	for(uint i = LIX, count = (tri_count + 31) & ~31; i < count; i += LSIZE) {
		uint cur_offset = BUFFER(i, 6);
		PREFIX_SUM_STEP(cur_offset, 1);
		PREFIX_SUM_STEP(cur_offset, 2);
		PREFIX_SUM_STEP(cur_offset, 4);
		PREFIX_SUM_STEP(cur_offset, 8);
		PREFIX_SUM_STEP(cur_offset, 16);
		BUFFER(i, 7) = cur_offset;
	}
	barrier();

	// Computing offsets for each triangle within block
	uint idx32 = LIX * 32;
	// Note: here we expect that idx32 < 32 * 16
	if(idx32 < tri_count) {
		uint value = BUFFER(idx32 + 31, 7);
		PREFIX_SUM_STEP(value, 1);
		PREFIX_SUM_STEP(value, 2);
		PREFIX_SUM_STEP(value, 4);
		PREFIX_SUM_STEP(value, 8);
		s_mini_buffer[idx32 >> 5] = value;
	}
	barrier();
	for(uint i = LIX; i < tri_count; i += LSIZE) {
		uint warp_idx = i >> 5;
		if(warp_idx > 0)
			BUFFER(i, 7) += s_mini_buffer[warp_idx - 1];
	}
	barrier();

	// TODO: compute min/max depth in this loop?
	for(uint i = LIX; i < tri_count; i += LSIZE) {
		uvec2 bits = uvec2(BUFFER(i, 0), BUFFER(i, 1));
		uint info = BUFFER(i, 2);

		uint cur_value = BUFFER(i, 6);
		uint cur_offset = BUFFER(i, 7);
		cur_offset -= cur_value;

		if(cur_value != 0) {
			g_scratch[dst_soffset + cur_offset] = bits;
			g_scratch[dst_soffset + cur_offset + MAX_BLOCK_TRIS].x = info;
		}
	}
	if(LIX == 0)
		BLOCK_TRI_COUNT(bx) = BUFFER(tri_count - 1, 7);

	barrier();

	for(uint i = 0; i < 8; i++)
		BUFFER(LIX, i) = 0;

	barrier();

#undef BUFFER
}

//#define BLOCK_DEPTH_OVERLAPS

#ifdef BLOCK_DEPTH_OVERLAPS
shared uint s_overlaps[BLOCK_COUNT * BLOCK_COUNT];

void computeDepthRanges(int bx, int by, int bx_step) {
	bx += int(LIX >> (LSHIFT - bx_step));

	uint tri_count = BLOCK_TRI_COUNT(bx);
	uint soffset = scratchBlockTrisOffset(bx);
	uint buf_offset = (bx & ((1 << bx_step) - 1)) * MAX_BLOCK_TRIS * 2;
	vec3 ray_dir_base = s_bin_ray_dir0 + (bx * 8) * frustum.ws_dirx + (by * 8) * frustum.ws_diry;

	s_overlaps[bx + by * 8] = 0;
	barrier();

	// This takes way more time than just the comparisons?
	for(uint i = LIX & ((LSIZE >> bx_step) - 1); i < tri_count; i += (LSIZE >> bx_step)) {
		uvec2 tri_bitmask = g_scratch[soffset + i];
		uvec2 tri_info = g_scratch[soffset + i + MAX_BLOCK_TRIS];

		uint tri_idx = tri_info.x & 0xffff;
		uint scratch_tri_offset = scratchTriOffset(tri_idx);
		vec2 val0 = uintBitsToFloat(TRI_SCRATCH(14));
		vec2 val1 = uintBitsToFloat(TRI_SCRATCH(15));
		vec3 depth_eq = vec3(val0.x, val0.y, val1.x);
		//float ray_pos = depth_eq.x + depth_eq.y * cpos.x + depth_eq.z * cpos.y;

		float min_depth = 999999999.0, max_depth = -999999999.0;
		depth_eq.x += depth_eq.y * (bx << 3) + depth_eq.z * (by << 3);

		for(uint y = 0; y < 8; y++) {
			uint bits = y < 4 ? tri_bitmask.x : tri_bitmask.y;
			bits = (bits >> (y & 3) * 8) & 0xff;
			uint min_x = findLSB(bits), max_x = findMSB(bits);

			float row_depth = depth_eq.x + depth_eq.z * y;
			float ray_pos = row_depth + depth_eq.y * min_x;
			min_depth = min(min_depth, ray_pos);
			max_depth = max(max_depth, ray_pos);

			ray_pos = row_depth + depth_eq.y * max_x;
			min_depth = min(min_depth, ray_pos);
			max_depth = max(max_depth, ray_pos);
		}

		s_buffer[buf_offset + i * 2 + 0] = floatBitsToUint(min_depth);
		s_buffer[buf_offset + i * 2 + 1] = floatBitsToUint(max_depth);
	}
	barrier();

	uint num_overlaps = 0;

	for(uint i = LIX & ((LSIZE >> bx_step) - 1); i < tri_count; i += (LSIZE >> bx_step)) {
		float min_depth = uintBitsToFloat(s_buffer[buf_offset + i * 2 + 0]);
		float max_depth = uintBitsToFloat(s_buffer[buf_offset + i * 2 + 1]);
		for(uint j = 0; j < i; j++) {
			float min_depth_j = uintBitsToFloat(s_buffer[buf_offset + j * 2 + 0]);
			float max_depth_j = uintBitsToFloat(s_buffer[buf_offset + j * 2 + 1]);

			if(min_depth_j < max_depth && max_depth_j > min_depth)
				num_overlaps++;
		}
	}

	atomicAdd(s_overlaps[bx + by * BLOCK_COUNT], num_overlaps);
	barrier();
}

void rasterDepthOverlaps(int by) {
	barrier();
	for(uint i = LIX; i < BIN_SIZE * MAX_ROWS; i += LSIZE) {
		ivec2 pixel_pos = ivec2(i & (BIN_SIZE - 1), by * 8 + (i >> BIN_SHIFT));
		uint tri_count = BLOCK_TRI_COUNT(pixel_pos.x / 8);
		float mul = tri_count == 0 ? 0.0 : 1.0 / float(tri_count);
		float count0 = float(s_overlaps[pixel_pos.x / 8 + by * 8]) * mul;

		vec3 color = vec3(count0, count0, count0) / 64.0;
		if(count0 <= 4)
			color = vec3(0, 1, 0);
		else if(count0 <= 8)
			color = vec3(0.5, 1, 0);
		else if(count0 <= 16)
			color = vec3(1, 1, 0);
		else if(count0 <= 32)
			color = vec3(0.8, 0.4, 0);
		else
			color = vec3(1, 0, 0);

		uint enc_col = encodeRGBA8(vec4(min(color, vec3(1.0)), 0.5));
		outputPixel(pixel_pos, enc_col);
	}
	barrier();
}
#endif

void loadSamples(int bx, int by, int max_raster_blocks) {
	bx += int(LIX >> (LSHIFT - max_raster_blocks));

	uint soffset = scratchBlockTrisOffset(bx);
	uint tri_count = BLOCK_TRI_COUNT(bx);

	uint block_offset = BLOCK_FRAG_OFFSET(bx) & 0xffff;

	int y = int(LIX & 7);
	uint y_shift = (y & 3) * 8;

	// TODO: load tri data similarily as in loadSamples and use shuffles to extract
	uint istep = (LSIZE / 8) >> max_raster_blocks;
	for(uint i = (LIX & ((LSIZE >> max_raster_blocks) - 1)) >> 3; i < tri_count; i += istep) {
		uint tri_bitmask = g_scratch[soffset + i][y < 4 ? 0 : 1];
		uvec2 info = g_scratch[soffset + i + MAX_BLOCK_TRIS];
		uint tri_idx = info.x & 0xffff;
		uint tri_offset = block_offset + info.y + (y >= 4 ? info.x >> 24 : 0);
		tri_offset += bitCount(tri_bitmask & ((1 << y_shift) - 1));
		tri_bitmask = (tri_bitmask >> y_shift) & 0xff;

		if(tri_bitmask == 0)
			continue;
		int count = bitCount(tri_bitmask);
		uint pixel_id = (y << 6) | (bx * 8 + findLSB(tri_bitmask));
		uint value = (pixel_id << 23) | tri_idx;
		for(uint i = 0; i < count; i++) {
			//if(tri_offset >= MAX_SAMPLES)
			//RECORD(x, y, tri_offset, BLOCK_FRAG_COUNT(bx));
			s_buffer[tri_offset++] = value;
			value += 1 << 23;
		}
	}
}

void reduceSamples(int bx, int by, int max_raster_blocks) {
	if(LIX >= (BIN_SIZE << max_raster_blocks))
		return;

	bx += int(LIX >> 6);
	int x = int(LIX & 7) + bx * 8, y = int((LIX >> 3) & 7);

	uint soffset = scratchBlockTrisOffset(bx);
	uint tri_count = BLOCK_TRI_COUNT(bx);
	uint block_offset = BLOCK_FRAG_OFFSET(bx) & 0xffff;

	// TODO: share pixels between threads for max_raster_blocks <= 4?
	// TODO: WARP_SIZE?

	uint pixel_bit = 1u << ((y & 3) * 8 + (x & 7));
	vec3 out_color = vec3(0);
	float out_transparency = 1.0;

	for(uint i = 0; i < tri_count; i += 32) {
		uint sub_count = min(32, tri_count - i);
		uint sel_tri_offset = 0, sel_tri_bitmask, tris_bitmask;
		{
			bool in_range = false;
			if((LIX & 31) < sub_count) {
				uvec2 bits = g_scratch[soffset + i + (LIX & 31)];
				uvec2 info = g_scratch[soffset + i + (LIX & 31) + MAX_BLOCK_TRIS];
				sel_tri_bitmask = y < 4 ? bits.x : bits.y;
				sel_tri_offset = block_offset + info.y + (y >= 4 ? info.x >> 24 : 0);
				in_range = sel_tri_bitmask != 0;
			}
			tris_bitmask = uint(ballotARB(in_range));
		}

		int j = findLSB(tris_bitmask);
		while(j != -1) {
			uint tri_offset = shuffleNV(sel_tri_offset, j, 32);
			uint tri_bitmask = shuffleNV(sel_tri_bitmask, j, 32);
			tris_bitmask &= ~(1 << j);
			j = findLSB(tris_bitmask);
			if((tri_bitmask & pixel_bit) == 0)
				continue;

			tri_offset += bitCount(tri_bitmask & (pixel_bit - 1));
			uint value = s_buffer[tri_offset];
			if(value != 0) {
				vec4 cur_color = decodeRGBA8(value);
				float cur_transparency = 1.0 - cur_color.a;
				out_color = (additive_blending ? out_color : out_color * cur_transparency) +
							cur_color.rgb * cur_color.a;
				out_transparency *= cur_transparency;
			}
		}
	}

	out_color = min(out_color, vec3(1.0));
	uint enc_color = encodeRGBA8(vec4(out_color, 1.0 - out_transparency));
	outputPixel(ivec2(x, by * 8 + y), enc_color);
}

void reduceSamplesWithCheck(int bx, int by, int max_raster_blocks) {
	if(LIX >= (BIN_SIZE << max_raster_blocks))
		return;

	bx += int(LIX >> 6);
	int x = int(LIX & 7) + bx * 8, y = int((LIX >> 3) & 7);

	uint soffset = scratchBlockTrisOffset(bx);
	uint tri_count = BLOCK_TRI_COUNT(bx);

	uint block_offset = BLOCK_FRAG_OFFSET(bx) & 0xffff;

	uint pixel_bit = 1u << ((y & 3) * 8 + (x & 7));
	float prev_depth = -1.0;
	vec3 out_color = vec3(0);
	float out_transparency = 1.0;

	for(uint i = 0; i < tri_count; i += 32) {
		uint sub_count = min(32, tri_count - i);
		uint sel_tri_offset = 0, sel_tri_bitmask, tris_bitmask;
		vec3 sel_depth_eq;
		{
			bool in_range = false;
			if((LIX & 31) < sub_count) {
				uvec2 bits = g_scratch[soffset + i + (LIX & 31)];
				uvec2 info = g_scratch[soffset + i + (LIX & 31) + MAX_BLOCK_TRIS];

				sel_tri_bitmask = y < 4 ? bits.x : bits.y;
				sel_tri_offset = block_offset + info.y + (y >= 4 ? info.x >> 24 : 0);

				uint tri_idx = info.x & 0xffff;
				uint scratch_tri_offset = scratchTriOffset(tri_idx);
				vec2 val0 = uintBitsToFloat(TRI_SCRATCH(14));
				vec2 val1 = uintBitsToFloat(TRI_SCRATCH(15));
				sel_depth_eq = vec3(val0.x, val0.y, val1.x);
				in_range = sel_tri_bitmask != 0;
			}
			tris_bitmask = uint(ballotARB(in_range));
		}

		int j = findLSB(tris_bitmask);
		while(j != -1) {
			uint tri_offset = shuffleNV(sel_tri_offset, j, 32);
			uint tri_bitmask = shuffleNV(sel_tri_bitmask, j, 32);
			vec3 depth_eq = shuffleNV(sel_depth_eq, j, 32);
			tris_bitmask &= ~(1 << j);
			j = findLSB(tris_bitmask);
			if((tri_bitmask & pixel_bit) == 0)
				continue;

			tri_offset += bitCount(tri_bitmask & (pixel_bit - 1));
			uint value = s_buffer[tri_offset];
			if(value == 0)
				continue;

			float depth = depth_eq.x + depth_eq.y * x + depth_eq.z * (y + (by << 3));
			if(depth < prev_depth) {
				out_color = vec3(1.0, 0.0, 0.0);
				out_transparency = 0.0;
				pixel_bit = 0;
				continue;
			}

			prev_depth = depth;
			vec4 cur_color = decodeRGBA8(value);
			float cur_transparency = 1.0 - cur_color.a;
			out_color = (additive_blending ? out_color : out_color * cur_transparency) +
						cur_color.rgb * cur_color.a;
			out_transparency *= cur_transparency;
		}
	}

	out_color = min(out_color, vec3(1.0));
	uint enc_color = encodeRGBA8(vec4(out_color, 1.0 - out_transparency));
	outputPixel(ivec2(x, by * 8 + y), enc_color);
}

// TODO: Can we improve speed of loading vertex data?
uint shadeSample(ivec2 bin_pixel_pos, uint scratch_tri_offset) {
	vec3 ray_dir =
		s_bin_ray_dir0 + frustum.ws_dirx * bin_pixel_pos.x + frustum.ws_diry * bin_pixel_pos.y;

	vec3 normal, params, edge0, edge1;
	uint instance_id, instance_flags, instance_color;
	getTriangleParams(scratch_tri_offset, normal, params, edge0, edge1, instance_id, instance_flags,
					  instance_color);

	float ray_pos = params[0] / dot(normal, ray_dir);
	vec2 bary = vec2(dot(edge0, ray_dir), dot(edge1, ray_dir)) * ray_pos;

	vec2 bary_dx, bary_dy;
	if((instance_flags & INST_HAS_TEXTURE) != 0) {
		vec3 ray_dirx = ray_dir + frustum.ws_dirx;
		vec3 ray_diry = ray_dir + frustum.ws_diry;

		float ray_posx = params[0] / dot(normal, ray_dirx);
		float ray_posy = params[0] / dot(normal, ray_diry);

		bary_dx = vec2(dot(edge0, ray_dirx), dot(edge1, ray_dirx)) * ray_posx - bary;
		bary_dy = vec2(dot(edge0, ray_diry), dot(edge1, ray_diry)) * ray_posy - bary;
	}
	bary -= vec2(params[1], params[2]);
	// params, edge0 & edge1 no longer needed!

	vec4 color = decodeRGBA8(instance_color);
	if((instance_flags & INST_HAS_TEXTURE) != 0) {
		vec2 tex0, tex1, tex2;
		getTriangleVertexTexCoords(scratch_tri_offset, tex0, tex1, tex2);

		vec2 tex_coord = tex0 + bary[0] * tex1 + bary[1] * tex2;
		vec2 tex_dx = bary_dx[0] * tex1 + bary_dx[1] * tex2;
		vec2 tex_dy = bary_dy[0] * tex1 + bary_dy[1] * tex2;

		if((instance_flags & INST_HAS_UV_RECT) != 0) {
			vec4 uv_rect = g_uv_rects[instance_id];
			tex_coord = uv_rect.xy + uv_rect.zw * fract(tex_coord);
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
		color *= (1.0 - bary[0] - bary[1]) * col0 + bary[0] * col1 + bary[1] * col2;
	}

	if(color.a == 0.0)
		return 0;

	if((instance_flags & INST_HAS_VERTEX_NORMALS) != 0) {
		vec3 nrm0, nrm1, nrm2;
		getTriangleVertexNormals(scratch_tri_offset, nrm0, nrm1, nrm2);
		nrm1 -= nrm0;
		nrm2 -= nrm0;
		normal = nrm0 + bary[0] * nrm1 + bary[1] * nrm2;
	}

	float light_value = max(0.0, dot(-lighting.sun_dir, normal) * 0.7 + 0.3);
	color.rgb = min(finalShading(color.rgb, light_value), vec3(1.0));
	return encodeRGBA8(color);
}

void shadeSamples(uint bx, uint by) {
	// TODO: what's the best way to fix broken pixels?
	// full sort ? recreate full depth values and sort pairs?

	// Shading samples grouped by triangles
	// TODO: how can we make sure that tris which generate >= 32 samples are all handles by single warp?

	uint sample_count = BLOCK_GROUP_FRAG_COUNT(bx) & 0xffff;

	for(uint i = LIX; i < sample_count; i += LSIZE) {
		uint value = s_buffer[i];
		uint pixel_id = value >> 23;
		uint scratch_tri_offset = scratchTriOffset(value & ((1 << 23) - 1));
		ivec2 bin_pixel_pos = ivec2(pixel_id & (BIN_SIZE - 1), (by * 8) + (pixel_id >> 6));
		s_buffer[i] = shadeSample(bin_pixel_pos, scratch_tri_offset);
	}
}

void rasterInvalidBlockRow(int by, vec3 color) {
	for(uint i = LIX; i < BIN_SIZE * MAX_ROWS; i += LSIZE) {
		ivec2 pixel_pos = ivec2(i & (BIN_SIZE - 1), by * 8 + (i >> BIN_SHIFT));
		vec4 color = vec4(color, 1.0);
		uint enc_col = encodeRGBA8(color);
		outputPixel(pixel_pos, enc_col);
	}
}

void rasterFragmentCounts(int by) {
	barrier();
	for(uint i = LIX; i < BIN_SIZE * MAX_ROWS; i += LSIZE) {
		ivec2 pixel_pos = ivec2(i & (BIN_SIZE - 1), by * 8 + (i >> BIN_SHIFT));
		uint count0 = BLOCK_FRAG_COUNT(pixel_pos.x / 8) & 0xffff;
		uint count1 = BLOCK_FRAG_COUNT(pixel_pos.x / 8) >> 16;
		//count0 = BLOCK_FRAG_OFFSET(pixel_pos.x / 8);
		//count1 = BLOCK_GROUP_FRAG_COUNT(pixel_pos.x / 8);

		vec3 color = vec3(count0, count0, count0) / 4096.0;
		uint enc_col = encodeRGBA8(vec4(min(color, vec3(1.0)), 1.0));
		outputPixel(pixel_pos, enc_col);
	}
	barrier();
}

void rasterBin(int bin_id) {
	INIT_CLOCK();

	if(LIX < BIN_SIZE) {
		if(LIX < BLOCK_COUNT) {
			if(LIX == 0) {
				ivec2 bin_pos = ivec2(bin_id % BIN_COUNT_X, bin_id / BIN_COUNT_X) * BIN_SIZE;
				s_bin_pos = bin_pos;
				s_bin_quad_count = g_bins.bin_quad_counts[bin_id];
				s_bin_quad_offset = g_bins.bin_quad_offsets[bin_id];
				s_bin_ray_dir0 = frustum.ws_dir0 + frustum.ws_dirx * (bin_pos.x + 0.5) +
								 frustum.ws_diry * (bin_pos.y + 0.5);
				s_max_raster_blocks = 0;
			}

			s_block_row_tri_counts[LIX] = 0;
		}
	}
	barrier();
	generateRows();
	groupMemoryBarrier();
	barrier();
	UPDATE_CLOCK(0);

	for(int by = 0; by < BLOCK_COUNT; by++) {
		barrier();
		generateBlocks1(by);
		groupMemoryBarrier();
		for(int bx = 0; bx < BLOCK_COUNT; bx++)
			splitTris(bx, by);
		groupMemoryBarrier();
		generateBlocks2(by);
		groupMemoryBarrier();
		barrier();
		UPDATE_CLOCK(1);

#ifdef BLOCK_DEPTH_OVERLAPS
		// 0.6 ms na sponzie? To chyba nie dużo?
		// OK ale jeszcze trzeba podzielić maski!
		for(int bx = 0; bx < BLOCK_COUNT; bx += 4)
			computeDepthRanges(bx, by, 2);
#endif

		// How many rows can we rasterize in single step?
		int max_raster_blocks = s_max_raster_blocks;
		if(max_raster_blocks == -1) {
			float value = max_raster_blocks == -1 ? 0.2 :
						  max_raster_blocks == 0  ? 0.5 :
						  max_raster_blocks == 1  ? 0.8 :
													  1.0;
			rasterInvalidBlockRow(by, vec3(value, 0.0, 0.0));
			continue;
		}

		for(int bx = 0; bx < BLOCK_COUNT; bx += (1 << max_raster_blocks)) {
			loadSamples(bx, by, max_raster_blocks);
			barrier();
			UPDATE_CLOCK(2);

			shadeSamples(bx, by);
			barrier();
			UPDATE_CLOCK(3);

			reduceSamplesWithCheck(bx, by, max_raster_blocks);
			barrier();
			UPDATE_CLOCK(4);
		}

#ifdef BLOCK_DEPTH_OVERLAPS
		rasterDepthOverlaps(by);
#else
		//rasterFragmentCounts(by);
#endif
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
