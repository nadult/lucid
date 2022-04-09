// $$include funcs lighting frustum viewport data

// TODO: add synthetic test: 256 planes one after another
// TODO: cleanup in the beginning (group definitions)

// NOTE: converting integer multiplications to shifts does not increase perf

#define LIX gl_LocalInvocationIndex
#define LID gl_LocalInvocationID

#define LSIZE 256
#define LSHIFT 8

#define BUFFER_SIZE (LSIZE * 10)

#define MAX_TILE_ROW_TRIS 2048
#define MAX_TILE_TRIS 512

#define MAX_SCRATCH_TRIS 2048
#define MAX_SCRATCH_TRIS_SHIFT 11

#define SEGMENT_SIZE (LSIZE * 8)
#define SEGMENT_SHIFT (LSHIFT + 3)

#define MAX_SEGMENTS 32
#define MAX_SEGMENTS_SHIFT 5

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

#define WORKGROUP_32_SCRATCH_SIZE (16 * 1024)
#define WORKGROUP_32_SCRATCH_SHIFT 14

#define WORKGROUP_64_SCRATCH_SIZE (64 * 1024)
#define WORKGROUP_64_SCRATCH_SHIFT 16

#define TRI_SCRATCH(var_idx) g_scratch_64[scratch_tri_offset + (var_idx << MAX_SCRATCH_TRIS_SHIFT)]

uint scratch32TileRowTrisOffset(uint ty) {
	return (gl_WorkGroupID.x << WORKGROUP_32_SCRATCH_SHIFT) + ty * MAX_TILE_ROW_TRIS;
}

uint scratch32TileTrisOffset(uint tx) {
	return (gl_WorkGroupID.x << WORKGROUP_32_SCRATCH_SHIFT) + 8 * 1024 + tx * MAX_TILE_TRIS;
}

uint scratch64TriOffset(uint tri_idx) {
	return (gl_WorkGroupID.x << WORKGROUP_64_SCRATCH_SHIFT) + tri_idx;
}

uint scratch64TileRowTrisOffset(uint ty) {
	return (gl_WorkGroupID.x << WORKGROUP_64_SCRATCH_SHIFT) + 32 * 1024 + ty * MAX_TILE_ROW_TRIS;
}

uint scratch64TileTrisOffset(uint tx) {
	return (gl_WorkGroupID.x << WORKGROUP_64_SCRATCH_SHIFT) + 40 * 1024 + tx * MAX_TILE_TRIS;
}

shared int s_num_bins, s_bin_id, s_bin_raster_offset;
shared ivec2 s_bin_pos;
shared vec3 s_bin_ray_dir0;

shared uint s_tile_row_tri_counts[XTILES_PER_BIN];
shared uint s_tile_tri_count[XTILES_PER_BIN];
shared uint s_tile_frag_count[XTILES_PER_BIN];

shared uint s_bin_quad_count, s_bin_quad_offset;

shared uint s_buffer[BUFFER_SIZE + 1];
shared uint s_mini_buffer[32 * XTILES_PER_BIN];
shared uint s_segments[XTILES_PER_BIN][MAX_SEGMENTS];
shared int s_raster_error;

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
		uint count = s_tile_tri_count[LIX];
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
	uint count = s_tile_tri_count[gid];
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

// TODO: można by tutaj użyć algorytmu bazującego na liniach
void generateTileRowTris(uint tri_idx, vec3 tri0, vec3 tri1, vec3 tri2, int min_qy, int max_qy) {
	// Inspired by Nanite scanline rasterizer
	vec3 scan_min, scan_max, scan_step;

	{
		float sx = s_bin_pos.x - 0.5f;
		float sy = s_bin_pos.y + (float(min_qy) * 4.0 + 0.5f);

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

	uint dst_offset_64 = scratch64TileRowTrisOffset(0);
	uint dst_offset_32 = scratch32TileRowTrisOffset(0);

	for(int qy = min_qy; qy <= max_qy; qy++) {
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

		uint tx_mask = ((0xf << (imin0 >> 4)) & (0xf >> (3 - (imax0 >> 4)))) |
					   ((0xf << (imin1 >> 4)) & (0xf >> (3 - (imax1 >> 4)))) |
					   ((0xf << (imin2 >> 4)) & (0xf >> (3 - (imax2 >> 4)))) |
					   ((0xf << (imin3 >> 4)) & (0xf >> (3 - (imax3 >> 4))));
		if(tx_mask == 0)
			continue;

		uint ty = qy >> 2;
		uint roffset = atomicAdd(s_tile_row_tri_counts[ty], 1) + ty * MAX_TILE_ROW_TRIS;

		g_scratch_64[dst_offset_64 + roffset] =
			uvec2((imin0 << 0) | (imin1 << 6) | (imin2 << 12) | (imin3 << 18),
				  (imax0 << 0) | (imax1 << 6) | (imax2 << 12) | (imax3 << 18));
		g_scratch_32[dst_offset_32 + roffset] = tri_idx | (tx_mask << 16) | ((qy & 3) << 24);
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
		int min_qy = clamp(int(aabb[1]) - s_bin_pos.y, 0, 63) >> 2;
		int max_qy = clamp(int(aabb[3]) - s_bin_pos.y, 0, 63) >> 2;

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
		generateTileRowTris(tri_idx, tri0, tri1, tri2, min_qy, max_qy);
	}
}

void generateTiles(uint ty) {
	uint src_offset_32 = scratch32TileRowTrisOffset(ty);
	uint src_offset_64 = scratch64TileRowTrisOffset(ty);
	uint tri_count = s_tile_row_tri_counts[ty];

	if(LIX < MAX_SEGMENTS * XTILES_PER_BIN) {
		s_segments[LIX >> MAX_SEGMENTS_SHIFT][LIX & (MAX_SEGMENTS - 1)] = ~0u;
		if(LIX < XTILES_PER_BIN)
			s_tile_tri_count[LIX] = 0;
	}
	barrier();

	for(uint tx = LIX & 3, tx_bit = 0x10000 << tx, buf_offset = tx * MAX_TILE_TRIS, i = LIX >> 2;
		i < tri_count; i += LSIZE / 4) {
		if((g_scratch_32[src_offset_32 + i] & tx_bit) == 0)
			continue;
		uint tri_offset = atomicAdd(s_tile_tri_count[tx], 1);
		if(tri_offset < MAX_TILE_TRIS)
			s_buffer[buf_offset + tri_offset] = i;
		else
			atomicOr(s_raster_error, 0x10 << tx);
	}
	barrier();
	if(s_raster_error != 0)
		return;

	uint tx = LIX >> (LSHIFT - 2);
	uint buf_offset = tx * MAX_TILE_TRIS;
	tri_count = s_tile_tri_count[tx];
	int startx = int(tx << 4); // TODO: bad name

	for(uint i = LIX & (TILE_STEP - 1); i < tri_count; i += TILE_STEP) {
		uint idx = s_buffer[buf_offset + i];

		// TODO: load these together with shuffles?
		uint tri_info = g_scratch_32[src_offset_32 + idx];
		uvec2 tri_rows = g_scratch_64[src_offset_64 + idx];
		uint tri_idx = tri_info & 0xffff;
		uint qy = tri_info >> 24;

		vec2 cpos = vec2(0, 0);
		float weight = 0.0;

		{
			// TODO: this is counted twice...
			int minx0 = max(int((tri_rows.x >> 0) & 0x3f) - startx, 0);
			int minx1 = max(int((tri_rows.x >> 6) & 0x3f) - startx, 0);
			int minx2 = max(int((tri_rows.x >> 12) & 0x3f) - startx, 0);
			int minx3 = max(int((tri_rows.x >> 18) & 0x3f) - startx, 0);

			int maxx0 = min(int((tri_rows.y >> 0) & 0x3f) - startx, 15);
			int maxx1 = min(int((tri_rows.y >> 6) & 0x3f) - startx, 15);
			int maxx2 = min(int((tri_rows.y >> 12) & 0x3f) - startx, 15);
			int maxx3 = min(int((tri_rows.y >> 18) & 0x3f) - startx, 15);

			int count0 = max(maxx0 - minx0 + 1, 0);
			int count1 = max(maxx1 - minx1 + 1, 0);
			int count2 = max(maxx2 - minx2 + 1, 0);
			int count3 = max(maxx3 - minx3 + 1, 0);

			cpos += vec2(float(maxx0 + minx0 + 1) * 0.5, 0.5) * count0;
			cpos += vec2(float(maxx1 + minx1 + 1) * 0.5, 1.5) * count1;
			cpos += vec2(float(maxx2 + minx2 + 1) * 0.5, 2.5) * count2;
			cpos += vec2(float(maxx3 + minx3 + 1) * 0.5, 3.5) * count3;

			weight += count0 + count1 + count2 + count3;
		}
		cpos /= weight;
		cpos += vec2(tx << 4, (ty << 4) + (qy << 2));

		uint scratch_tri_offset = scratch64TriOffset(tri_idx);
		vec2 val0 = uintBitsToFloat(TRI_SCRATCH(0));
		vec2 val1 = uintBitsToFloat(TRI_SCRATCH(1));
		vec3 depth_eq = vec3(val0.x, val0.y, val1.x);
		float ray_pos = depth_eq.x * cpos.x + (depth_eq.y * cpos.y + depth_eq.z);
		float depth = (0x1ffff * 0.99f) * SATURATE(1.0 - inversesqrt(ray_pos + 1)); // 17 bits

		s_buffer[buf_offset + i] = idx | (uint(depth) << 15);
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

	uint dst_offset_32 = scratch32TileTrisOffset(tx);
	uint dst_offset_64 = scratch64TileTrisOffset(tx);

	for(uint i = LIX & (TILE_STEP - 1); i < tri_count; i += TILE_STEP) {
		uint idx = s_buffer[buf_offset + i] & 0x1fff;

		// TODO: load range data in groups
		uint tri_info = g_scratch_32[src_offset_32 + idx];
		uvec2 tri_rows = g_scratch_64[src_offset_64 + idx];
		uint tri_idx = tri_info & 0xffff;
		uint qy = tri_info >> 24;
		uint min_bits, count_bits;
		uint num_frags;
		uint row_mask = 0;

		{
			int minx0 = max(int((tri_rows.x >> 0) & 0x3f) - startx, 0);
			int minx1 = max(int((tri_rows.x >> 6) & 0x3f) - startx, 0);
			int minx2 = max(int((tri_rows.x >> 12) & 0x3f) - startx, 0);
			int minx3 = max(int((tri_rows.x >> 18) & 0x3f) - startx, 0);

			int maxx0 = min(int((tri_rows.y >> 0) & 0x3f) - startx, 15);
			int maxx1 = min(int((tri_rows.y >> 6) & 0x3f) - startx, 15);
			int maxx2 = min(int((tri_rows.y >> 12) & 0x3f) - startx, 15);
			int maxx3 = min(int((tri_rows.y >> 18) & 0x3f) - startx, 15);

			int count0 = max(maxx0 - minx0 + 1, 0);
			int count1 = max(maxx1 - minx1 + 1, 0);
			int count2 = max(maxx2 - minx2 + 1, 0);
			int count3 = max(maxx3 - minx3 + 1, 0);

			row_mask = (count0 > 0 ? 1 : 0) | (count1 > 0 ? 2 : 0) | (count2 > 0 ? 4 : 0) |
					   (count3 > 0 ? 8 : 0);

			min_bits =
				(minx0 & 15) | ((minx1 & 15) << 4) | ((minx2 & 15) << 8) | ((minx3 & 15) << 12);
			count_bits = count0 | (count1 << 5) | (count2 << 10) | (count3 << 15);
			num_frags = count0 + count1 + count2 + count3;
		}
		row_mask <<= qy << 2;

		if(num_frags == 0) // This means that tx_masks are invalid
			RECORD(0, 0, 0, 0);

		g_scratch_64[dst_offset_64 + i] =
			uvec2(min_bits | (tri_idx << 16), count_bits | (qy << 20));

		// Computing triangle-ordered sample offsets within each block
		PREFIX_SUM_STEP(num_frags, 1);
		PREFIX_SUM_STEP(num_frags, 2);
		PREFIX_SUM_STEP(num_frags, 4);
		PREFIX_SUM_STEP(num_frags, 8);
		PREFIX_SUM_STEP(num_frags, 16);
		s_buffer[buf_offset + i] = num_frags | (row_mask << 16);
	}
	barrier();

	// Computing offsets for each triangle within block
	// Note: here we expect that idx32 < 32 * 16
	uint idx32 = (LIX & (TILE_STEP - 1)) << 5;
	if(idx32 < tri_count) {
		uint value = s_buffer[buf_offset + idx32 + 31] & 0xffff;
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
	// Also finding first triangle for each segment
	for(uint i = LIX & (TILE_STEP - 1); i < tri_count; i += TILE_STEP) {
		uint tri_offset = i == 0 ? 0 : s_buffer[buf_offset + i - 1] & 0xffff;
		uint tri_value = s_buffer[buf_offset + i];
		uint row_mask = tri_value & 0xffff0000;
		tri_value = (tri_value & 0xffff) - tri_offset;

		uint seg_id = tri_offset >> SEGMENT_SHIFT;
		if(seg_id < MAX_SEGMENTS) {
			uint seg_offset = tri_offset & (SEGMENT_SIZE - 1);
			if(seg_offset == 0)
				s_segments[tx][seg_id] = i;
			else if(seg_offset + tri_value > SEGMENT_SIZE)
				s_segments[tx][seg_id + 1] = i;
		}

		g_scratch_32[dst_offset_32 + i] = tri_offset | row_mask;
	}
	barrier();
	if(LIX < MAX_SEGMENTS * XTILES_PER_BIN) {
		uint seg_id = LIX & (MAX_SEGMENTS - 1);
		uint tx = LIX >> MAX_SEGMENTS_SHIFT;
		uint tri_count = s_tile_tri_count[tx];

		uint cur_value = s_segments[tx][seg_id];
		uint next_value = seg_id + 1 == MAX_SEGMENTS ? ~0u : s_segments[tx][seg_id + 1];
		next_value = next_value == ~0u ? tri_count : min(tri_count, next_value + 1);
		cur_value = cur_value == ~0u ? 0 : cur_value | ((next_value - cur_value) << 16);
		s_segments[tx][seg_id] = cur_value;
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
		uint num_tris = s_tile_tri_count[tx];
		uint frag_count = num_tris == 0 ? 0 : s_buffer[tx * MAX_TILE_TRIS + num_tris - 1] & 0xffff;
		//uint segment_count = (frag_count + SEGMENT_SIZE - 1) >> SEGMENT_SHIFT;
		s_tile_frag_count[tx] = frag_count;
		if(frag_count > MAX_SEGMENTS * SEGMENT_SIZE)
			atomicOr(s_raster_error, 1 << tx);
	}
}

shared uint s_src_offset_32, s_src_offset_64, s_tri_count;
shared int s_first_offset;

void initSegment(int tx, int ty, int segment_id) {
	if(LIX == 0) {
		s_first_offset = segment_id << SEGMENT_SHIFT;
		uint first_tri = s_segments[tx][segment_id];
		s_tri_count = first_tri >> 16;
		first_tri &= 0xffff;

		s_src_offset_32 = scratch32TileTrisOffset(tx) + first_tri;
		s_src_offset_64 = scratch64TileTrisOffset(tx) + first_tri;
	}
}

// TODO:
// - przygotować dane tak, żeby były łatwiej przetwarzalne podczas ładowania i redukcji
//   np. zamiast 4 rzędów 2; problem: wydajne matchowanie tych rzędów w redukcji
//   (tak, żeby wszystkie grupy wątków nie musiały przelatywać po wszystkich rzędach)
// - możliwość obniżenia ilości sampli do 4, tak, żeby można było przechowywać też głębię i
//   żeby nie trzeba było re-compute-ować jej;

void loadSamples(int tx, int ty, int frag_count) {
	int y = int(LIX & 3);
	uint min_shift = y << 2, count_shift = min_shift + y;

	// TODO: group differently for better memory accesses (and measure)
	int first_offset = s_first_offset;
	uint src_offset_32 = s_src_offset_32, src_offset_64 = s_src_offset_64;
	for(uint tri_count = s_tri_count, i = (LIX >> 2); i < tri_count; i += LSIZE / 4) {
		uvec2 tri_info = g_scratch_64[src_offset_64 + i];
		int tri_offset = int(g_scratch_32[src_offset_32 + i] & 0xffff) - first_offset;
		int minx = int((tri_info.x >> min_shift) & 15);
		int count_bits = int(tri_info.y & ((1 << 20) - 1));

		// TODO: precompute it somehow?
		for(uint j = 0; j < y; j++) {
			tri_offset += count_bits & 31;
			count_bits >>= 5;
		}
		int countx = min(count_bits & 31, SEGMENT_SIZE - tri_offset);
		if(countx <= 0)
			continue;

		uint scratch_tri_offset = scratch64TriOffset(tri_info.x >> 16);
		uint qy = tri_info.y >> 20;

		uint pixel_id = ((y << 4) + (qy << 6)) | minx;
		uint value = pixel_id | (scratch_tri_offset << 8);
		for(int j = 0; j < countx; j++) {
			if(tri_offset >= 0) // TODO: remove this check
				s_buffer[tri_offset] = value;
			tri_offset++;
			value++;
		}
	}
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

void shadeSamples(uint tx, uint ty, uint sample_count) {
	// TODO: what's the best way to fix broken pixels?
	// full sort ? recreate full depth values and sort pairs?

	for(uint i = LIX; i < sample_count; i += LSIZE) {
		uint value = s_buffer[i];
		uint pixel_id = value & 0xff;
		uint scratch_tri_offset = value >> 8;
		ivec2 pix_pos = ivec2((pixel_id & 15) + (tx << 4), (pixel_id >> 4) + (ty << 4));
		s_buffer[i] = shadeSample(pix_pos, scratch_tri_offset);
	}
}

void initReduceSamples(out vec4 prev_depths, out uvec4 prev_colors) {
	prev_depths = vec4(-1.0);
	prev_colors = uvec4(0, 0, 0, background_color);
}

// TODO: optimize
void reduceSamples(int tx, int ty, uint frag_count, in out vec4 prev_depths,
				   in out uvec4 prev_colors, bool finish) {
	int x = int(LIX & 15);
	int y = int((LIX >> 4) & 15);

	uint pixel_bit = 1u << (((y & 1) << 4) + x);

	vec4 out_color = decodeRGBA8(prev_colors[3]);

	// Każdy warp robi listę trisów; Problem: 8 bitów nie wystarczy..., chyba że pogrupujemy po 256 trisów
	// wtedy mamy max 64 uinty wymagane do przechowania listy trisów

	uint tri_count = s_tri_count;
	int first_offset = s_first_offset;
	uint buf_offset = SEGMENT_SIZE + (LIX >> 5) * 64;
	uint warp_row_mask = 0x30000 << (y & ~1);

	for(uint gi = 0; gi < tri_count; gi += 256) {
		uint cur_tri_count = min(tri_count - gi, 256);
		uint warp_id = LIX & 31;

		uint filtered_count = 0;
		s_buffer[buf_offset + warp_id] = 0;
		s_buffer[buf_offset + 32 + warp_id] = 0;

		for(uint i = 0; i < cur_tri_count; i += 32) {
			uint tile_tri_idx = i + warp_id;
			bool in_range = false;
			if(tile_tri_idx < cur_tri_count) {
				uint row_mask = g_scratch_32[s_src_offset_32 + gi + tile_tri_idx];
				in_range = (warp_row_mask & row_mask) != 0;
			}
			uint in_range_mask = uint(ballotARB(in_range));

			// TODO: better (faster) way to store/retrieve indices?
			if(in_range) {
				uint prev_mask = in_range_mask & ((1 << warp_id) - 1);
				uint cur_offset = filtered_count + bitCount(prev_mask);
				atomicOr(s_buffer[buf_offset + (cur_offset >> 2)],
						 tile_tri_idx << ((cur_offset & 3) << 3));
			}
			filtered_count += bitCount(in_range_mask);
		}

		for(uint i = 0; i < filtered_count; i += 32) {
			uint sub_count = min(32, filtered_count - i);
			int sel_tri_offset = 0;
			uint sel_tri_bitmask, tris_bitmask;
			vec3 sel_depth_eq;

			if(warp_id < sub_count) {
				uint warp_idx = i + warp_id;
				uint tile_tri_idx =
					gi + ((s_buffer[buf_offset + (warp_idx >> 2)] >> ((warp_idx & 3) << 3)) & 0xff);

				uvec2 tri_info = g_scratch_64[s_src_offset_64 + tile_tri_idx];
				sel_tri_offset =
					int(g_scratch_32[s_src_offset_32 + tile_tri_idx] & 0xffff) - first_offset;

				// TODO: quicker filtering of tris with invalid QY?
				int min_bits = int(tri_info.x & 0xffff);
				int count_bits = int(tri_info.y & 0xfffff);

				// TODO: mask pixels out of segment here
				if((y & 2) != 0) {
					sel_tri_offset += (count_bits & 31) + ((count_bits >> 5) & 31);
					count_bits >>= 10;
					min_bits >>= 8;
				}

				int minx0 = min_bits & 15, minx1 = (min_bits >> 4) & 15;
				int countx0 = count_bits & 31, countx1 = (count_bits >> 5) & 31;

				uint bits0 = ((1 << countx0) - 1) << (minx0 + 0);
				uint bits1 = ((1 << countx1) - 1) << (minx1 + 16);
				sel_tri_bitmask = bits0 | bits1;

				uint tri_idx = tri_info.x >> 16;
				uint scratch_tri_offset = scratch64TriOffset(tri_idx);
				vec2 val0 = uintBitsToFloat(TRI_SCRATCH(0));
				vec2 val1 = uintBitsToFloat(TRI_SCRATCH(1));
				sel_depth_eq = vec3(val0.x, val0.y, val1.x);
				sel_depth_eq.z += sel_depth_eq.x * (tx << 4) + sel_depth_eq.y * (ty << 4);
			}

			for(int j = 0; j < sub_count; j++) {
				int tri_offset = shuffleNV(sel_tri_offset, j, 32);
				uint tri_bitmask = shuffleNV(sel_tri_bitmask, j, 32);
				vec3 depth_eq = shuffleNV(sel_depth_eq, j, 32);
				tri_offset += bitCount(tri_bitmask & (pixel_bit - 1));

				// TODO: tri_offset check shouldn't be needed?
				if((tri_bitmask & pixel_bit) == 0 ||
				   (tri_offset & (SEGMENT_SIZE - 1)) != tri_offset)
					continue;

				uint value = s_buffer[tri_offset];
				if(value == 0)
					continue;

				// It's actually faster to recompute depth in reduce than reuse depth
				// computed during sampling, because we can put 2x as many samples in SMEM
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

#ifdef VISUALIZE_ERRORS
							if(prev_depths[2] < prev_depths[3]) {
								prev_colors[0] = 0xff0000ff;
								i = filtered_count;
								break;
							}
#endif
						}
					}
				}

#ifdef VISUALIZE_ERRORS
				prev_depths[3] = prev_depths[2];
#endif
				prev_depths[2] = prev_depths[1];
				prev_depths[1] = prev_depths[0];
				prev_depths[0] = depth;

				if(prev_colors[2] != 0) {
					vec4 cur_color = decodeRGBA8(prev_colors[2]);
#ifdef ADDITIVE_BLENDING
					out_color.rgb += cur_color.rgb * cur_color.a;
					out_color.a += cur_color.a;
#else
					float cur_trans = 1.0 - cur_color.a;
					out_color.rgb = cur_color.rgb * cur_color.a + out_color.rgb * cur_trans;
					out_color.a = cur_color.a + out_color.a * cur_trans;
#endif
				}

				prev_colors[2] = prev_colors[1];
				prev_colors[1] = prev_colors[0];
				prev_colors[0] = value;
			}
		}
	}

	if(finish) {
		for(int i = 2; i >= 0; i--)
			if(prev_colors[i] != 0) {
				vec4 cur_color = decodeRGBA8(prev_colors[i]);
				float cur_transparency = 1.0 - cur_color.a;
#ifdef ADDITIVE_BLENDING
				out_color.rgb += cur_color.rgb * cur_color.a;
				out_color.a += cur_color.a;
#else
				float cur_trans = 1.0 - cur_color.a;
				out_color.rgb = cur_color.rgb * cur_color.a + out_color.rgb * cur_trans;
				out_color.a = cur_color.a + out_color.a * cur_trans;
#endif
			}

		uint enc_color = encodeRGBA8(SATURATE(out_color));
		outputPixel(ivec2((tx << 4) + x, (ty << 4) + y), enc_color);

	} else {
		prev_colors[3] = encodeRGBA8(SATURATE(out_color));
	}
}

shared uint s_pixels[TILE_SIZE * TILE_SIZE];

void resetVisualizeSamples() {
	s_pixels[LIX] = 0;
	barrier();
}

void visualizeSamples(uint sample_count) {
	for(uint i = LIX; i < sample_count; i += LSIZE) {
		uint value = s_buffer[i];
		atomicAdd(s_pixels[value & 0xff], 1);
	}
}

void finishVisualizeSamples(uint tx, uint ty) {
	ivec2 pixel_pos = ivec2(LIX & (TILE_SIZE - 1), (LIX >> TILE_SHIFT));
	pixel_pos += ivec2(tx << TILE_SHIFT, ty << TILE_SHIFT);
	uint value = s_pixels[LIX];
	vec3 color = vec3(value, value, value) / 32.0;
	uint enc_col = encodeRGBA8(vec4(SATURATE(color), 1.0));
	outputPixel(pixel_pos, enc_col);
}

void fillTile(int tx, int ty, uint enc_color) {
	ivec2 pixel_pos = ivec2((tx << TILE_SHIFT) + (LIX & 15), (ty << TILE_SHIFT) + (LIX >> 4));
	outputPixel(pixel_pos, enc_color);
}

void rasterFragmentCounts(int ty) {
	barrier();
	for(uint i = LIX; i < BIN_SIZE * TILE_SIZE; i += LSIZE) {
		ivec2 pixel_pos = ivec2(i & (BIN_SIZE - 1), ty * TILE_SIZE + (i >> BIN_SHIFT));
		uint count0 = s_tile_frag_count[pixel_pos.x / TILE_SIZE] & 0xffff;
		//count0 = s_tile_tri_count[pixel_pos.x / TILE_SIZE] * 8;

		vec3 color = vec3(count0, count0, count0) / 4096;
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

		s_tile_row_tri_counts[LIX] = 0;
	}
	barrier();
	processQuads();
	groupMemoryBarrier();
	barrier();
	UPDATE_CLOCK(0);

	for(int ty = 0; ty < XTILES_PER_BIN; ty++) {
		barrier();
		generateTiles(ty);
		groupMemoryBarrier();
		barrier();
		UPDATE_CLOCK(1);

		if(s_raster_error != 0) {
			for(int tx = 0; tx < XTILES_PER_BIN; tx++) {
				float value = 0.2;
				if((s_raster_error & (1 << tx)) != 0)
					value += 0.2;
				if((s_raster_error & (0x10 << tx)) != 0)
					value += 0.4;
				vec4 color = vec4(value, 0.0, 0.0, 1.0);
				fillTile(tx, ty, encodeRGBA8(color));
			}
			if(LIX == 0)
				s_raster_error = 0;
			continue;
		}

		for(int tx = 0; tx < XTILES_PER_BIN; tx++) {
			int frag_count = int(s_tile_frag_count[tx]);
			int segment_id = 0;

			if(frag_count == 0) {
				fillTile(tx, ty, background_color);
				continue;
			}

			vec4 prev_depths;
			uvec4 prev_colors;

			//resetVisualizeSamples();
			initReduceSamples(prev_depths, prev_colors);

			while(frag_count > 0) {
				int cur_frag_count = min(frag_count, SEGMENT_SIZE);

				initSegment(tx, ty, segment_id++);
				frag_count -= SEGMENT_SIZE;
				barrier();

				loadSamples(tx, ty, cur_frag_count);
				barrier();
				UPDATE_CLOCK(2);

				//visualizeSamples(cur_frag_count);
				shadeSamples(tx, ty, cur_frag_count);
				barrier();
				UPDATE_CLOCK(3);

				reduceSamples(tx, ty, cur_frag_count, prev_depths, prev_colors, frag_count <= 0);
				barrier();
				UPDATE_CLOCK(4);
			}

			//finishVisualizeSamples(tx, ty);
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
