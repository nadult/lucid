// $$include funcs lighting frustum viewport data

#define LIX gl_LocalInvocationIndex
#define LID gl_LocalInvocationID

#define LSIZE 256
#define LSHIFT 8

#define TILE_STEP (LSIZE / XTILES_PER_BIN)

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

#define WORKGROUP_SCRATCH_SIZE (128 * 1024)
#define WORKGROUP_SCRATCH_SHIFT 17

// TODO: for some reason, enabling timings makes whole shader work faster
// it started after optimising UV coordinates computation (added edge equations)
// Problem is caused by different limit on used registers:
// enabling timings increases register limit from 48 to 64, probably allowing for
// better optimisations...

// TODO: does that mean that occupancy is so low?
#define SAMPLES_PER_THREAD 8
#define MAX_SAMPLES (LSIZE * SAMPLES_PER_THREAD)

#define MAX_TILE_ROW_TRIS 2048
#define MAX_TILE_TRIS 512

#define MAX_SCRATCH_TRIS 2048
#define MAX_SCRATCH_TRIS_SHIFT 11

#define TRI_SCRATCH(var_idx) g_scratch[scratch_tri_offset + (var_idx << MAX_SCRATCH_TRIS_SHIFT)]

#define SATURATE(val) clamp(val, 0.0, 1.0)

uint scratchTileRowTrisOffset(uint ty) {
	return (gl_WorkGroupID.x << WORKGROUP_SCRATCH_SHIFT) + ty * (MAX_TILE_ROW_TRIS * 4);
}

uint scratchTileTrisOffset(uint tx) {
	return (gl_WorkGroupID.x << WORKGROUP_SCRATCH_SHIFT) + 32 * 1024 + tx * (MAX_TILE_TRIS * 4);
}

uint scratchTriOffset(uint tri_idx) {
	return (gl_WorkGroupID.x << WORKGROUP_SCRATCH_SHIFT) + 64 * 1024 + tri_idx;
}

shared int s_num_bins, s_bin_id, s_bin_raster_offset;
shared ivec2 s_bin_pos;
shared vec3 s_bin_ray_dir0;

shared uint s_block_row_tri_counts[XTILES_PER_BIN];
shared uint s_curblock_counters[XTILES_PER_BIN * 4];

void outputPixel(ivec2 pixel_pos, uint color) {
	g_raster_image[s_bin_raster_offset + pixel_pos.x + (pixel_pos.y << BIN_SHIFT)] = color;
}

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
	if(LIX < XTILES_PER_BIN)
		s_curblock_counters[LIX] = 0;
}

// These functions work only within current block row
#define TILE_TRI_COUNT(tx) s_curblock_counters[tx]

// TODO: more info
#define TILE_FRAG_COUNT(tx) s_curblock_counters[tx + XTILES_PER_BIN]
#define TILE_FRAG_OFFSET(tx) s_curblock_counters[tx + XTILES_PER_BIN * 2]

shared int s_raster_error;

// TODO: add protection from too big number of samples:
// maximum per row for raster_bin = min(4 * LSIZE, 32768) ?
// we have to somehow estimate max# of samples during categorization?
// max 512 tris per block -> max samples = 512 * 64: fits in 15 bits

shared uint s_bin_quad_count, s_bin_quad_offset;

shared uint s_buffer[MAX_SAMPLES + 1];
shared uint s_mini_buffer[32 * XTILES_PER_BIN];

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
	if(LIX < XTILES_PER_BIN) {
		uint count = TILE_TRI_COUNT(LIX);
		// rcount: count rounded up to next power of 2
		uint rcount = max(32, (count & (count - 1)) == 0 ? count : (2 << findMSB(count)));
		if(LIX == 0)
			s_sort_max_block_rcount = 0;
		atomicMax(s_sort_max_block_rcount, rcount);
	}
	barrier();

	uint gid = LIX >> (LSHIFT - 2);
	uint lid = LIX & (LSIZE / 4 - 1);
	uint goffset = gid * MAX_TILE_TRIS;
	uint count = TILE_TRI_COUNT(gid);
	// TODO: max_rcount is only needed for barriers, computations should be performed up to rcount
	// But it seems, that using rcount directly is actually a bit slower... (Sponza)
	uint max_rcount = s_sort_max_block_rcount;
	for(uint i = lid + count; i < max_rcount; i += LSIZE / 4)
		s_buffer[goffset + i] = 0xffffffff;
	barrier();

#ifdef VENDOR_NVIDIA
	for(uint i = lid; i < max_rcount; i += LSIZE / 4) {
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
		//	value = swap(value, 0x10, xorBits(lid, 5, 4)); // K = 32
		//	value = swap(value, 0x08, xorBits(lid, 5, 3));
		//	value = swap(value, 0x04, xorBits(lid, 5, 2));
		//	value = swap(value, 0x02, xorBits(lid, 5, 1));
		//	value = swap(value, 0x01, xorBits(lid, 5, 0));
		s_buffer[goffset + i] = value;
	}
	barrier();
	int start_k = 32, end_j = 32;
#else
	int start_k = 2, end_j = 1;
#endif
	for(uint k = start_k; k <= max_rcount; k = 2 * k) {
		for(uint j = k >> 1; j >= end_j; j = j >> 1) {
			for(uint i = lid; i < max_rcount; i += (LSIZE / 4) * 2) {
				uint idx = (i & j) != 0 ? i + (LSIZE / 4) - j : i;
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
		for(uint i = lid; i < max_rcount; i += LSIZE / 4) {
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
void generateTriGroups(uint tri_idx, vec3 tri0, vec3 tri1, vec3 tri2, int min_ty, int max_ty) {
	// Inspired by Nanite scanline rasterizer
	vec3 scan_min, scan_max, scan_step;

	{
		float sx = s_bin_pos.x - 0.5f;
		float sy = s_bin_pos.y + min_ty * 16 + 0.5f;

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

	uint soffset = scratchTileRowTrisOffset(min_ty);
	for(int ty = min_ty; ty <= max_ty; ty++) {
		uint row_ranges[8] = {0, 0, 0, 0, 0, 0, 0, 0};
		uint tmask = 0;

		for(int y = 0; y < 16; y++) {
			float xmin = max(max(scan_min[0], scan_min[1]), max(scan_min[2], 0.0));
			float xmax = min(min(scan_max[0], scan_max[1]), min(scan_max[2], BIN_SIZE));

			scan_min += scan_step;
			scan_max += scan_step;

			// TODO: use floor/ceil?
			int imin = int(xmin), imax = int(xmax) - 1;
			if(imin <= imax) {
				uint tmin = imin >> 4, tmax = imax >> 4;
				tmask |= (0xf << tmin) & (0xf >> (3 - tmax));
			}

			// TODO: instead of min+max, save min+count, it should fit?
			row_ranges[y >> 1] |= (imin <= imax ? (uint(imin) | (uint(imax) << 6)) : 0x3f)
								  << ((y & 1) * 12);
		}

		if(tmask != 0) {
			uint roffset = atomicAdd(s_block_row_tri_counts[ty], 1) * 4;
			g_scratch[soffset + roffset] =
				uvec2(row_ranges[0] | (tri_idx << 24), row_ranges[1] | ((tri_idx & 0xff00) << 16));
			g_scratch[soffset + roffset + 1] = uvec2(row_ranges[2] | (tmask << 24), row_ranges[3]);
			g_scratch[soffset + roffset + 2] = uvec2(row_ranges[4], row_ranges[5]);
			g_scratch[soffset + roffset + 3] = uvec2(row_ranges[6], row_ranges[7]);
		}
		soffset += MAX_TILE_ROW_TRIS * 4;
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
		int min_ty = clamp(int(aabb[1]) - s_bin_pos.y, 0, 63) >> 4;
		int max_ty = clamp(int(aabb[3]) - s_bin_pos.y, 0, 63) >> 4;

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
		generateTriGroups(tri_idx, tri0, tri1, tri2, min_ty, max_ty);
	}
}

void generateBlocks(uint ty) {
	// TODO: is this really the best order?
	uint tx = LIX & (XTILES_PER_BIN - 1);
	uint src_offset = scratchTileRowTrisOffset(ty);
	uint buf_offset = tx * MAX_TILE_TRIS;
	uint tri_count = s_block_row_tri_counts[ty];
	resetBlockCounters();
	barrier();

	int min_tx = int(tx << 4);
	for(uint i = LIX >> 2; i < tri_count; i += TILE_STEP) {
		// TODO: load these together with shuffles?
		uint full_rows[8] = {
			g_scratch[src_offset + i * 4 + 0].x, g_scratch[src_offset + i * 4 + 0].y,
			g_scratch[src_offset + i * 4 + 1].x, g_scratch[src_offset + i * 4 + 1].y,
			g_scratch[src_offset + i * 4 + 2].x, g_scratch[src_offset + i * 4 + 2].y,
			g_scratch[src_offset + i * 4 + 3].x, g_scratch[src_offset + i * 4 + 3].y};

		// TODO: load range data in groups
		uint tmask = full_rows[2] >> 24;
		if((tmask & (1 << tx)) == 0)
			continue;

		// TODO: keep tri_idx & bmask in one place
		uint tri_idx = (full_rows[0] >> 24) | ((full_rows[1] >> 16) & 0xff00);

		vec2 cpos = vec2(0, 0);
		float weight = 0.0;

		for(int j = 0; j < 8; j++) {
			int row0 = int(full_rows[j] & 0xfff), row1 = int((full_rows[j] >> 12) & 0xfff);

			// TODO: handling of empty rows?
			// TODO: these are computed twice
			int minx0 = max((row0 & 0x3f) - min_tx, 0),
				maxx0 = min(((row0 >> 6) & 0x3f) - min_tx, 15);
			int minx1 = max((row1 & 0x3f) - min_tx, 0),
				maxx1 = min(((row1 >> 6) & 0x3f) - min_tx, 15);

			int count0 = max(0, maxx0 - minx0 + 1);
			int count1 = max(0, maxx1 - minx1 + 1);
			cpos += vec2(float(maxx0 + minx0 + 1) * 0.5, j * 2 + 0 + 0.5) * count0;
			cpos += vec2(float(maxx1 + minx1 + 1) * 0.5, j * 2 + 1 + 0.5) * count1;
			weight += count0 + count1;
		}
		cpos /= weight;
		cpos += vec2(tx << 4, ty << 4);

		uint scratch_tri_offset = scratchTriOffset(tri_idx);
		vec2 val0 = uintBitsToFloat(TRI_SCRATCH(0));
		vec2 val1 = uintBitsToFloat(TRI_SCRATCH(1));
		vec3 depth_eq = vec3(val0.x, val0.y, val1.x);
		float ray_pos = depth_eq.x * cpos.x + (depth_eq.y * cpos.y + depth_eq.z);
		float depth = float(0x7ffff) / max(0.5, 4.0 - ray_pos); // 19 bits is enough

		uint idx = atomicAdd(TILE_TRI_COUNT(tx), 1);
		if(idx < MAX_TILE_TRIS)
			s_buffer[buf_offset + idx] = i | (uint(depth) << 13);
		else
			s_raster_error = 1;
	}

	barrier();
	if(s_raster_error == 1)
		return;
	sortTileTris();
	barrier();

	tx = LIX >> (LSHIFT - 2);
	min_tx = int(tx << 4); // TODO: bad name
	buf_offset = tx * MAX_TILE_TRIS;
	tri_count = TILE_TRI_COUNT(tx);
	uint dst_offset = scratchTileTrisOffset(tx);

#define PREFIX_SUM_STEP(value, step)                                                               \
	{                                                                                              \
		uint temp = shuffleUpNV(value, step, 32);                                                  \
		if((LIX & 31) >= step)                                                                     \
			value += temp;                                                                         \
	}

	for(uint i = LIX & (TILE_STEP - 1); i < tri_count; i += TILE_STEP) {
		uint idx = s_buffer[buf_offset + i] & 0x1fff;

		// TODO: load range data in groups
		uint full_rows[8] = {
			g_scratch[src_offset + idx * 4 + 0].x, g_scratch[src_offset + idx * 4 + 0].y,
			g_scratch[src_offset + idx * 4 + 1].x, g_scratch[src_offset + idx * 4 + 1].y,
			g_scratch[src_offset + idx * 4 + 2].x, g_scratch[src_offset + idx * 4 + 2].y,
			g_scratch[src_offset + idx * 4 + 3].x, g_scratch[src_offset + idx * 4 + 3].y};

		uint tri_idx = (full_rows[0] >> 24) | ((full_rows[1] >> 16) & 0xff00);
		uint bits[4];

		uint num_frags_[4];
		uint enable_bits = 0;

		for(int j = 0; j < 4; j++) {
			int rows01 = int(full_rows[j * 2 + 0]), rows23 = int(full_rows[j * 2 + 1]);
			int minx0 = max(((rows01 >> 0) & 0x3f) - min_tx, 0),
				maxx0 = min(((rows01 >> 6) & 0x3f) - min_tx, 15);
			int minx1 = max(((rows01 >> 12) & 0x3f) - min_tx, 0),
				maxx1 = min(((rows01 >> 18) & 0x3f) - min_tx, 15);
			int minx2 = max(((rows23 >> 0) & 0x3f) - min_tx, 0),
				maxx2 = min(((rows23 >> 6) & 0x3f) - min_tx, 15);
			int minx3 = max(((rows23 >> 12) & 0x3f) - min_tx, 0),
				maxx3 = min(((rows23 >> 18) & 0x3f) - min_tx, 15);

			uint cur_enable_bits = ((minx0 <= maxx0 ? 1 : 0) | (minx1 <= maxx1 ? 2 : 0) |
									(minx2 <= maxx2 ? 4 : 0) | (minx3 <= maxx3 ? 8 : 0));

			int count0 = max(maxx0 - minx0, 0);
			int count1 = max(maxx1 - minx1, 0);
			int count2 = max(maxx2 - minx2, 0);
			int count3 = max(maxx3 - minx3, 0);

			minx0 = min(minx0, 15);
			minx1 = min(minx1, 15);
			minx2 = min(minx2, 15);
			minx3 = min(minx3, 15);

			bits[j] = (minx0 << 0) | (minx1 << 4) | (minx2 << 8) | (minx3 << 12) | (count0 << 16) |
					  (count1 << 20) | (count2 << 24) | (count3 << 28);
			num_frags_[j] = count0 + count1 + count2 + count3 + bitCount(cur_enable_bits);
			enable_bits |= cur_enable_bits << (j << 2);
		}

		num_frags_[1] += num_frags_[0];
		num_frags_[2] += num_frags_[1];
		uint num_frags = num_frags_[2] + num_frags_[3];

		uint frag_offsets = (num_frags_[0] << 8) | (num_frags_[1] << 16) | (num_frags_[2] << 24);

		if(num_frags == 0) // This means that bmasks are invalid
			RECORD(0, 0, 0, 0);

		uint cur_offset = dst_offset + i * 4;
		g_scratch[cur_offset + 0] = uvec2(bits[0], bits[1]);
		g_scratch[cur_offset + 1] = uvec2(bits[2], bits[3]);
		g_scratch[cur_offset + 2] = uvec2(tri_idx | (enable_bits << 16), frag_offsets);

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
	uint idx32 = (LIX & (TILE_STEP - 1)) * 32;
	// Note: here we expect that idx32 < 32 * 16
	if(idx32 < tri_count) {
		uint value = s_buffer[buf_offset + idx32 + 31];
		PREFIX_SUM_STEP(value, 1);
		PREFIX_SUM_STEP(value, 2);
		PREFIX_SUM_STEP(value, 4);
		PREFIX_SUM_STEP(value, 8);
		s_mini_buffer[tx * 16 + (idx32 >> 5)] = value;
	}
	barrier();

	for(uint i = 32 + (LIX & (TILE_STEP - 1)); i < tri_count; i += TILE_STEP)
		s_buffer[buf_offset + i] += s_mini_buffer[tx * 16 + (i >> 5) - 1];
	barrier();

	// Storing offsets to scratch mem
	for(uint i = LIX & (TILE_STEP - 1); i < tri_count; i += TILE_STEP) {
		uint value = i == 0 ? 0 : s_buffer[buf_offset + i - 1];
		g_scratch[dst_offset + i * 4 + 3].x = value;
	}

#ifdef SHADER_DEBUG
	for(uint i = LIX & (TILE_STEP - 1); i < tri_count; i += TILE_STEP) {
		uint value = s_buffer[buf_offset + i] & 0xffff;
		uint prev_value = i == 0 ? 0 : s_buffer[buf_offset + i - 1] & 0xffff;
		if(value <= prev_value)
			RECORD(i, tri_count, prev_value, value);
	}
#endif

	if(LIX < XTILES_PER_BIN) {
		uint tx = LIX;
		uint num_tris = TILE_TRI_COUNT(tx);
		uint frag_count = num_tris == 0 ? 0 : s_buffer[tx * MAX_TILE_TRIS + num_tris - 1];
		TILE_FRAG_COUNT(tx) = frag_count;
		TILE_FRAG_OFFSET(tx) = 0;
		// TODO: offsets
	}
}

void loadSamples(int tx, int ty) {
	uint soffset = scratchTileTrisOffset(tx);
	uint tri_count = TILE_TRI_COUNT(tx);
	uint tile_offset = TILE_FRAG_OFFSET(tx) & 0xffff;

	int yblock = int(LIX & 3);
	int sub_offset_shift = yblock << 3;

	// TODO: load tri data similarily as in loadSamples and use shuffles to extract
	for(uint i = LIX >> 2; i < tri_count; i += LSIZE / 4) {
		uint src_idx = soffset + i * 4;

		uvec2 info = g_scratch[src_idx + 2];
		uint tri_bits = g_scratch[src_idx + (yblock >> 1)][yblock & 1];
		uint tri_idx = info.x & 0xffff;
		uint enable_bits = ((info.x >> 16) >> (yblock << 2)) & 0xf;
		uint tri_offset = g_scratch[src_idx + 3].x + ((info.y >> sub_offset_shift) & 0xff);

		if(enable_bits == 0)
			continue;

		for(uint y = 0; y < 4; y++) {
			if((enable_bits & (1 << y)) == 0) {
				tri_bits >>= 4;
				continue;
			}

			uint minx = tri_bits & 15, countx = ((tri_bits >> 16) & 15) + 1;
			tri_bits >>= 4;

			uint pixel_id = ((y << 4) + (yblock << 6)) | minx;
			uint value = (pixel_id << 16) | tri_idx;
			for(uint i = 0; i < countx; i++) {
				s_buffer[tri_offset++] = value;
				value += 1 << 16;
			}
		}
	}
}

// TODO: optimize
void reduceSamples(int tx, int ty) {
	int x = int(LIX & 15);
	int y = int((LIX >> 4) & 15);

	uint soffset = scratchTileTrisOffset(tx);
	uint tri_count = TILE_TRI_COUNT(tx);
	uint tile_offset = TILE_FRAG_OFFSET(tx) & 0xffff;

	// TODO: share pixels between threads for max_raster_blocks <= 4?
	// TODO: WARP_SIZE?

	uint pixel_bit = 1u << (((y & 1) << 4) + x);

	float prev_depths[4] = {-1.0, -1.0, -1.0, -1.0};
	uint prev_colors[3] = {0, 0, 0};

	vec3 out_color = vec3(0);
	float out_transparency = 1.0;

	for(uint i = 0; i < tri_count; i += 32) {
		uint sub_count = min(32, tri_count - i);
		uint sel_tri_offset = 0, sel_tri_bitmask, tris_bitmask;
		vec3 sel_depth_eq;
		{
			bool in_range = false;
			if((LIX & 31) < sub_count) {
				int yblock = y >> 2;
				int sub_offset_shift = yblock << 3;

				uint idx = soffset + (i + (LIX & 31)) * 4;
				uvec2 info = g_scratch[idx + 2];
				uint tri_bits = g_scratch[idx + (yblock >> 1)][yblock & 1];
				uint tri_idx = info.x & 0xffff;
				uint enable_bits = ((info.x >> 16) >> (yblock << 2)) & 0xf;
				uint tri_offset = g_scratch[idx + 3].x + ((info.y >> sub_offset_shift) & 0xff);

				if((y & 2) != 0) {
					uint count0 = (enable_bits & 1) == 1 ? ((tri_bits >> 16) & 0xf) + 1 : 0;
					uint count1 = (enable_bits & 2) == 2 ? ((tri_bits >> 20) & 0xf) + 1 : 0;
					tri_offset += count0 + count1;
					tri_bits >>= 8;
					enable_bits >>= 2;
				}

				uint minx0 = (tri_bits >> 0) & 15;
				uint minx1 = (tri_bits >> 4) & 15;
				uint countx0 = (enable_bits & 1) != 0 ? ((tri_bits >> 16) & 15) + 1 : 0;
				uint countx1 = (enable_bits & 2) != 0 ? ((tri_bits >> 20) & 15) + 1 : 0;
				uint bits0 = ((1 << countx0) - 1) << (minx0 + 0);
				uint bits1 = ((1 << countx1) - 1) << (minx1 + 16);

				sel_tri_bitmask = bits0 | bits1;
				sel_tri_offset = tri_offset;

				uint scratch_tri_offset = scratchTriOffset(tri_idx);
				vec2 val0 = uintBitsToFloat(TRI_SCRATCH(0));
				vec2 val1 = uintBitsToFloat(TRI_SCRATCH(1));
				sel_depth_eq = vec3(val0.x, val0.y, val1.x);
				sel_depth_eq.z += sel_depth_eq.x * (tx << 4) + sel_depth_eq.y * (ty << 4);
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

			float depth = depth_eq.x * x + (depth_eq.y * y + depth_eq.z);
			if(depth < prev_depths[0]) {
				SWAP_UINT(value, prev_colors[0]);
				SWAP_FLOAT(depth, prev_depths[0]);
				if(prev_depths[0] < prev_depths[1]) {
					SWAP_UINT(prev_colors[1], prev_colors[0]);
					SWAP_FLOAT(prev_depths[1], prev_depths[0]);
					if(prev_depths[1] < prev_depths[2]) {
						SWAP_UINT(prev_colors[2], prev_colors[1]);
						SWAP_FLOAT(prev_depths[2], prev_depths[1]);
						if(prev_depths[2] < prev_depths[3]) {
							prev_colors[0] = 0xff0000ff;
							pixel_bit = 0;
							continue;
						}
					}
				}
			}

			prev_depths[3] = prev_depths[2];
			prev_depths[2] = prev_depths[1];
			prev_depths[1] = prev_depths[0];
			prev_depths[0] = depth;

			if(prev_colors[2] != 0) {
				vec4 cur_color = decodeRGBA8(prev_colors[2]);
				float cur_transparency = 1.0 - cur_color.a;
				out_color = cur_color.rgb * cur_color.a +
							(additive_blending ? out_color : out_color * cur_transparency);
				out_transparency *= cur_transparency;
			}

			prev_colors[2] = prev_colors[1];
			prev_colors[1] = prev_colors[0];
			prev_colors[0] = value;
		}
	}

	for(int i = 2; i >= 0; i--)
		if(prev_colors[i] != 0) {
			vec4 cur_color = decodeRGBA8(prev_colors[i]);
			float cur_transparency = 1.0 - cur_color.a;
			out_color = cur_color.rgb * cur_color.a +
						(additive_blending ? out_color : out_color * cur_transparency);
			out_transparency *= cur_transparency;
		}

	uint enc_color = encodeRGBA8(vec4(SATURATE(out_color), 1.0 - out_transparency));
	outputPixel(ivec2((tx << 4) + x, (ty << 4) + y), enc_color);
}

// TODO: Can we improve speed of loading vertex data?
uint shadeSample(ivec2 bin_pixel_pos, uint scratch_tri_offset) {
	float px = float(bin_pixel_pos.x), py = float(bin_pixel_pos.y);

	vec3 depth_eq, edge0_eq, edge1_eq;
	uint instance_id, instance_flags;
	vec2 bary_params;
	getTriangleParams(scratch_tri_offset, depth_eq, bary_params, edge0_eq, edge1_eq, instance_id,
					  instance_flags);
	uint instance_color, unormal;
	getTriangleSecondaryParams(scratch_tri_offset, unormal, instance_color);

	float inv_ray_pos = depth_eq.x * px + (depth_eq.y * py + depth_eq.z);
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

void shadeSamples(uint tx, uint ty) {
	// TODO: what's the best way to fix broken pixels?
	// full sort ? recreate full depth values and sort pairs?

	// Shading samples grouped by triangles
	// TODO: how can we make sure that tris which generate >= 32 samples are all handles by single warp?

	uint sample_count = TILE_FRAG_COUNT(tx) & 0xffff;
	for(uint i = LIX; i < sample_count; i += LSIZE) {
		uint value = s_buffer[i];
		uint pixel_id = value >> 16;
		uint tri_idx = value & 0xffff;
		uint scratch_tri_offset = scratchTriOffset(tri_idx);
		ivec2 pix_pos = ivec2((pixel_id & 15) + (tx << 4), (pixel_id >> 4) + (ty << 4));
		s_buffer[i] = shadeSample(pix_pos, scratch_tri_offset);
	}
}

shared uint s_pixels[TILE_SIZE * TILE_SIZE];

void visualizeSamples(uint tx, uint ty) {
	s_pixels[LIX] = 0;
	barrier();
	// TODO: what's the best way to fix broken pixels?
	// full sort ? recreate full depth values and sort pairs?

	// Shading samples grouped by triangles
	// TODO: how can we make sure that tris which generate >= 32 samples are all handles by single warp?

	uint sample_count = TILE_FRAG_COUNT(tx) & 0xffff;
	for(uint i = LIX; i < sample_count; i += LSIZE) {
		uint value = s_buffer[i];
		atomicAdd(s_pixels[value >> 16], 1);
	}
	barrier();

	ivec2 pixel_pos = ivec2(LIX & (TILE_SIZE - 1), (LIX >> TILE_SHIFT));
	pixel_pos += ivec2(tx << TILE_SHIFT, ty << TILE_SHIFT);
	uint value = s_pixels[LIX];
	vec3 color = vec3(value, value, value) / 32.0;
	uint enc_col = encodeRGBA8(vec4(SATURATE(color), 1.0));
	outputPixel(pixel_pos, enc_col);
}

void rasterInvalidTile(int tx, int ty, vec3 color) {
	uint enc_col = encodeRGBA8(vec4(color, 1.0));

	for(uint i = LIX; i < TILE_SIZE * TILE_SIZE; i += LSIZE) {
		ivec2 pixel_pos = ivec2(i & (TILE_SIZE - 1), i >> TILE_SHIFT);
		outputPixel(pixel_pos + ivec2(tx, ty) * TILE_SIZE, enc_col);
	}
}

void rasterFragmentCounts(int ty) {
	barrier();
	for(uint i = LIX; i < BIN_SIZE * TILE_SIZE; i += LSIZE) {
		ivec2 pixel_pos = ivec2(i & (BIN_SIZE - 1), ty * TILE_SIZE + (i >> BIN_SHIFT));
		uint count0 = TILE_FRAG_COUNT(pixel_pos.x / TILE_SIZE) & 0xffff;
		//count0 = TILE_TRI_COUNT(pixel_pos.x / TILE_SIZE);
		//count0 = TILE_FRAG_OFFSET(pixel_pos.x / 8);

		vec3 color = vec3(count0, count0, count0) / 4096.0;
		if(count0 > MAX_SAMPLES)
			color = vec3(1.0, 0.0, 0.0);
		uint enc_col = encodeRGBA8(vec4(SATURATE(color), 1.0));
		outputPixel(pixel_pos, enc_col);
	}
	barrier();
}

void rasterBin(int bin_id) {
	INIT_CLOCK();

	if(LIX < XTILES_PER_BIN) {
		if(LIX == 0) {
			ivec2 bin_pos = ivec2(bin_id % BIN_COUNT_X, bin_id / BIN_COUNT_X) * BIN_SIZE;
			s_bin_pos = bin_pos;
			s_bin_quad_count = g_bins.bin_quad_counts[bin_id];
			s_bin_quad_offset = g_bins.bin_quad_offsets[bin_id];
			s_bin_ray_dir0 = frustum.ws_dir0 + frustum.ws_dirx * (bin_pos.x + 0.5) +
							 frustum.ws_diry * (bin_pos.y + 0.5);
			s_raster_error = 0;
		}

		s_block_row_tri_counts[LIX] = 0;
	}
	barrier();
	generateRows();
	groupMemoryBarrier();
	barrier();
	UPDATE_CLOCK(0);

	for(int ty = 0; ty < XTILES_PER_BIN; ty++) {
		barrier();
		generateBlocks(ty);
		groupMemoryBarrier();
		barrier();
		UPDATE_CLOCK(1);

		if(s_raster_error == 1) {
			for(int tx = 0; tx < XTILES_PER_BIN; tx++)
				rasterInvalidTile(tx, ty, vec3(0.2, 0.0, 0.0));
			if(LIX == 0)
				s_raster_error = 0;
			continue;
		}

		for(int tx = 0; tx < XTILES_PER_BIN; tx++) {
			if(TILE_FRAG_COUNT(tx) > MAX_SAMPLES) {
				rasterInvalidTile(tx, ty, vec3(0.5, 0.0, 0.0));
				continue;
			}

			loadSamples(tx, ty);
			barrier();
			UPDATE_CLOCK(2);

			//visualizeSamples(tx, ty);
			shadeSamples(tx, ty);
			barrier();
			UPDATE_CLOCK(3);

			reduceSamples(tx, ty);
			barrier();
			UPDATE_CLOCK(4);
		}

		//rasterFragmentCounts(ty);
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
