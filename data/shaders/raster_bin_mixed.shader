// $$include funcs lighting frustum viewport data

#define LIX gl_LocalInvocationIndex
#define LID gl_LocalInvocationID

#define LSIZE 256
#define LSHIFT 8

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

#define BLOCK_WIDTH 8
#define BLOCK_HEIGHT 4
#define BLOCK_COUNTX 8
#define BLOCK_COUNTY 16
#define BLOCK_COUNT_SQ (BLOCK_COUNTX * BLOCK_COUNTY)

#define BLOCK_ROW_COUNT 8

#define MAX_BLOCK_ROW_TRIS 2048
#define MAX_BLOCK_TRIS 256
#define MAX_SCRATCH_TRIS 2048
#define MAX_SCRATCH_TRIS_SHIFT 11

#define BUFFER_SIZE (MAX_BLOCK_TRIS * BLOCK_COUNTX)

#define TRI_SCRATCH(var_idx) g_scratch[scratch_tri_offset + (var_idx << MAX_SCRATCH_TRIS_SHIFT)]

// TODO: use shifts, it makes a difference
uint scratchBlockRowTrisOffset(uint by) {
	return (gl_WorkGroupID.x << WORKGROUP_SCRATCH_SHIFT) + by * MAX_BLOCK_ROW_TRIS * 2;
}

uint scratchBlockTrisOffset(uint bx, uint by) {
	return (gl_WorkGroupID.x << WORKGROUP_SCRATCH_SHIFT) + 32 * 1024 +
		   (bx + (by << 3)) * MAX_BLOCK_TRIS * 2;
}

uint scratchBlockSegmentOffset(uint bx, uint by) {
	return (gl_WorkGroupID.x << WORKGROUP_SCRATCH_SHIFT) + 96 * 1024 +
		   (bx + (by << 3)) * MAX_BLOCK_TRIS;
}

uint scratchBlockSegmentRowsOffset(uint bx, uint by) {
	return (gl_WorkGroupID.x << WORKGROUP_SCRATCH_SHIFT) + 128 * 1024 +
		   (bx + (by << 3)) * MAX_BLOCK_TRIS;
}

uint scratchTriOffset(uint tri_idx) {
	return (gl_WorkGroupID.x << WORKGROUP_SCRATCH_SHIFT) + 160 * 1024 + tri_idx;
}

uint sumU16x2(uint pair) { return (pair & 0xffff) + (pair >> 16); }

// Note: in the context of this shader, block size is 8x8, not 4x4

shared int s_num_bins, s_bin_id, s_bin_raster_offset;
shared ivec2 s_bin_pos;
shared vec3 s_bin_ray_dir0;

shared uint s_block_row_tri_counts[BLOCK_ROW_COUNT];
shared uint s_block_counters[BLOCK_COUNT_SQ * 2];

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
	if(LIX < BLOCK_COUNT_SQ)
		s_block_counters[LIX] = 0;
}

#define BLOCK_TRI_COUNT(bx, by) s_block_counters[(bx) + ((by) << 3) + BLOCK_COUNT_SQ * 0]
#define BLOCK_FRAG_COUNT(bx, by) s_block_counters[(bx) + ((by) << 3) + BLOCK_COUNT_SQ * 1]

shared bool s_invalid_bin;

shared uint s_bin_quad_count, s_bin_quad_offset;

shared uint s_buffer[BUFFER_SIZE + 7];
shared uint s_mini_buffer[16 * BLOCK_COUNTY];

#ifdef VENDOR_NVIDIA
uint swap(uint x, int mask, uint dir) {
	uint y = shuffleXorNV(x, mask, 32);
	return uint(x < y) == dir ? y : x;
}

uint bitExtract(uint value, int boffset) { return (value >> boffset) & 1; }

uint xorBits(uint value, int bit0, int bit1) { return ((value >> bit0) ^ (value >> bit1)) & 1; }
#endif

shared uint s_sort_max_block_rcount;

void sortBlockTris(uint by) {
	if(LIX < 8) {
		uint count = BLOCK_TRI_COUNT(LIX, by);
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
	uint count = BLOCK_TRI_COUNT(gid, by);
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

		// These steps won't work if we have 32 threads/ group; Why?
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

			if(y == 3)
				bmask <<= 8;
		}

		if(bmask != 0) {
			uint roffset = atomicAdd(s_block_row_tri_counts[by], 1) * 2;
			g_scratch[soffset + roffset] =
				uvec2(row_ranges[0] | (tri_idx << 24), row_ranges[1] | ((tri_idx & 0xff00) << 16));
			g_scratch[soffset + roffset + 1] =
				uvec2(row_ranges[2] | ((bmask & 0xff00) << 16), row_ranges[3] | (bmask << 24));
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

void generateBlocks(uint by) {
	// TODO: is this really the best order?
	uint bx = LIX & 7;
	uint src_offset = scratchBlockRowTrisOffset(by >> 1);
	uint buf_offset = bx * MAX_BLOCK_TRIS;
	uint tri_count = s_block_row_tri_counts[by >> 1];

	int min_bx = int(bx * 8);
	vec3 ray_dir_base = s_bin_ray_dir0 + (bx * BLOCK_WIDTH) * frustum.ws_dirx +
						(by * BLOCK_HEIGHT) * frustum.ws_diry;

	for(uint i = LIX >> 3; i < tri_count; i += LSIZE / 8) {
		// TODO: load these together with shuffles?
		uint full_rows[4] = {
			g_scratch[src_offset + i * 2 + 0].x, g_scratch[src_offset + i * 2 + 0].y,
			g_scratch[src_offset + i * 2 + 1].x, g_scratch[src_offset + i * 2 + 1].y};

		// TODO: load range data in groups
		uint bmask = full_rows[2 + (by & 1)] >> 24;
		if((bmask & (1 << bx)) == 0)
			continue;

		// TODO: keep tri_idx & bmask in one place
		uint tri_idx = (full_rows[0] >> 24) | ((full_rows[1] >> 16) & 0xff00);

		vec2 cpos = vec2(0, 0);
		float weight = 0.0;
		for(int j = 0; j < 2; j++) {
			int row0 = int(full_rows[j + (by & 1) * 2] & 0xfff);
			int row1 = int((full_rows[j + (by & 1) * 2] >> 12) & 0xfff);

			// TODO: these are computed twice
			int minx0 = max((row0 & 0x3f) - min_bx, 0),
				maxx0 = min(((row0 >> 6) & 0x3f) - min_bx, 7);
			int minx1 = max((row1 & 0x3f) - min_bx, 0),
				maxx1 = min(((row1 >> 6) & 0x3f) - min_bx, 7);

			int count0 = max(0, maxx0 - minx0 + 1);
			int count1 = max(0, maxx1 - minx1 + 1);
			cpos += vec2(float(maxx0 + minx0 + 1) * 0.5, j * 2 + 0 + 0.5) * count0;
			cpos += vec2(float(maxx1 + minx1 + 1) * 0.5, j * 2 + 1 + 0.5) * count1;
			weight += count0 + count1;
		}
		cpos /= weight;
		vec3 ray_dir = ray_dir_base + cpos.x * frustum.ws_dirx + cpos.y * frustum.ws_diry;

		uint scratch_tri_offset = scratchTriOffset(tri_idx);
		vec2 val0 = uintBitsToFloat(TRI_SCRATCH(0));
		vec2 val1 = uintBitsToFloat(TRI_SCRATCH(1));
		vec3 normal = vec3(val0.x, val0.y, val1.x);
		float param0 = val1.y; //TODO: premul by normal
		float ray_pos = param0 / dot(normal, ray_dir);
		float depth = float(0x7ffff) / (1.0 + max(0.0, ray_pos)); // 19 bits is enough

		uint idx = atomicAdd(BLOCK_TRI_COUNT(bx, by), 1);
		if(idx < MAX_BLOCK_TRIS)
			s_buffer[buf_offset + idx] = i | (uint(depth) << 13);
		else
			s_invalid_bin = true;
	}

	barrier();
	if(s_invalid_bin)
		return;
	sortBlockTris(by);
	barrier();

	bx = LIX >> (LSHIFT - 3);
	min_bx = int(bx * 8);
	buf_offset = bx * MAX_BLOCK_TRIS;
	tri_count = BLOCK_TRI_COUNT(bx, by);
	uint dst_offset = scratchBlockTrisOffset(bx, by);

#ifdef SHADER_DEBUG
	for(uint i = LIX & (LSIZE / 8 - 1); i < tri_count; i += LSIZE / 8) {
		uint value = s_buffer[buf_offset + i];
		uint prev_value = i == 0 ? 0 : s_buffer[buf_offset + i - 1];
		if(value <= prev_value)
			RECORD(i, tri_count, prev_value, value);
	}
#endif

	// każdy wątek zapisuje sobie rzędy dla swoich trójkątów
	uint seg_tri_rows[8];
	uint num_seg_tri_rows = 0, local_tri_idx = 0;

	for(uint i = LIX & (LSIZE / 8 - 1); i < tri_count; i += LSIZE / 8) {
		uint idx = s_buffer[buf_offset + i] & 0x1fff;

		// TODO: jak zaznaczyć pusty rząd ? po prostu zapisać ilość rzędów i zapisywać jeden po drugim?

		// TODO: load range data in groups
		uint full_rows[4] = {
			g_scratch[src_offset + idx * 2 + 0].x, g_scratch[src_offset + idx * 2 + 0].y,
			g_scratch[src_offset + idx * 2 + 1].x, g_scratch[src_offset + idx * 2 + 1].y};
		uint tri_idx = (full_rows[0] >> 24) | ((full_rows[1] >> 16) & 0xff00);

		int rows01 = int(full_rows[(by & 1) * 2]), rows23 = int(full_rows[1 + (by & 1) * 2]);
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

		uint bits0 = mask0 & (0x000000ffu << minx0) & (0x000000ffu >> (7 - maxx0));
		uint bits1 = mask1 & (0x0000ff00u << minx1) & (0x0000ff00u >> (7 - maxx1));
		uint bits2 = mask2 & (0x00ff0000u << minx2) & (0x00ff0000u >> (7 - maxx2));
		uint bits3 = mask3 & (0xff000000u << minx3) & (0xff000000u >> (7 - maxx3));

		uint bits = bits0 | bits1 | bits2 | bits3;
		uint num_rows = (bits0 == 0 ? 0 : 1) + (bits1 == 0 ? 0 : 1) + (bits2 == 0 ? 0 : 1) +
						(bits3 == 0 ? 0 : 1);

		num_seg_tri_rows |= num_rows << (local_tri_idx << 2);
		seg_tri_rows[local_tri_idx] = bits;
		local_tri_idx++;

		uint num_frags = bitCount(bits);
		if(num_frags == 0) // This means that bmasks are invalid
			RECORD(0, 0, 0, 0);

		g_scratch[dst_offset + i] = uvec2(bits, tri_idx);

		// Computing triangle-ordered sample offsets within each block
		uint temp, value = num_frags | (num_rows << 16);
		temp = shuffleUpNV(value, 1, 32);
		if((LIX & 31) >= 1)
			value += temp;
		temp = shuffleUpNV(value, 2, 32);
		if((LIX & 31) >= 2)
			value += temp;
		temp = shuffleUpNV(value, 4, 32);
		if((LIX & 31) >= 4)
			value += temp;
		temp = shuffleUpNV(value, 8, 32);
		if((LIX & 31) >= 8)
			value += temp;
		temp = shuffleUpNV(value, 16, 32);
		if((LIX & 31) >= 16)
			value += temp;
		s_buffer[buf_offset + i] = value;
	}
	barrier();

	// Computing offsets for each triangle within block
	uint idx32 = (LIX & (LSIZE / 8 - 1)) * 32;
	// Note: here we expect that idx32 < 32 * 16
	if(idx32 < tri_count) {
		uint value = s_buffer[buf_offset + idx32 + 31], temp;
		temp = shuffleUpNV(value, 1, 32);
		if((LIX & 31) >= 1)
			value += temp;
		temp = shuffleUpNV(value, 2, 32);
		if((LIX & 31) >= 2)
			value += temp;
		temp = shuffleUpNV(value, 4, 32);
		if((LIX & 31) >= 4)
			value += temp;
		temp = shuffleUpNV(value, 8, 32);
		if((LIX & 31) >= 8)
			value += temp;
		s_mini_buffer[bx * 16 + (idx32 >> 5)] = value;
	}
	barrier();

	for(uint i = 32 + (LIX & (LSIZE / 8 - 1)); i < tri_count; i += LSIZE / 8)
		s_buffer[buf_offset + i] += s_mini_buffer[bx * 16 + (i >> 5) - 1];
	barrier();

	// Storing offsets to scratch mem
	/*for(uint i = LIX & (LSIZE / 8 - 1); i < tri_count; i += LSIZE / 8) {
		uint value = i == 0? 0 : s_buffer[buf_offset + i - 1];
		g_scratch[dst_offset + i + MAX_BLOCK_TRIS].y = value;
	}*/

#ifdef SHADER_DEBUG
	for(uint i = LIX & (LSIZE / 8 - 1); i < tri_count; i += LSIZE / 8) {
		uint value = s_buffer[buf_offset + i] & 0xffff;
		uint prev_value = i == 0 ? 0 : s_buffer[buf_offset + i - 1] & 0xffff;
		if(value <= prev_value)
			RECORD(i, tri_count, prev_value, value);
	}
#endif

	if(LIX < BLOCK_COUNTX) {
		uint bx = LIX;
		uint num_tris = BLOCK_TRI_COUNT(bx, by);
		BLOCK_FRAG_COUNT(bx, by) = num_tris == 0 ? 0 : s_buffer[bx * MAX_BLOCK_TRIS + num_tris - 1];
	}

	barrier();

	// TODO: last segment can have empty samples
	// TODO: limited number of segments: max 256

	// generating segments
	uint first_idx = LIX & (LSIZE / 8 - 1);
	uint idx_count = (tri_count - first_idx + (LSIZE / 8 - 1)) >> (LSHIFT - 3);
	uint offsets[8];
	// TODO: is using arrays a good idea? maybe let's try to unroll it instead?
	// it's using lmem0 array in asm

	for(int k = 0; k < idx_count; k++) {
		uint buf_idx = buf_offset + first_idx + (k << (LSHIFT - 3));
		if(buf_idx >= buf_offset + tri_count)
			RECORD(k, buf_idx - buf_offset, tri_count, 0);
		offsets[k] = buf_idx == buf_offset ? 0 : s_buffer[buf_idx - 1];
	}
	barrier();
	// TODO: we have to clear only odd slots
	for(uint i = LIX; i < BUFFER_SIZE; i += LSIZE)
		s_buffer[i] = 0;
	barrier();

	// Segment header:
	// - first tri idx (10 bits)
	// - first row idx (10 bits)
	// - first row offset (10 bits)
	// wszystko generowane przez wątek odpowiadający za 1 trójkąt
	for(int k = 0; k < idx_count; k++) {
		uint tri_sample_offset = offsets[k] & 0xffff;
		uint tri_row_offset = offsets[k] >> 16;
		uint tri_idx = first_idx + (k << (LSHIFT - 3));

		uint row_data = seg_tri_rows[k];
		uint tri_sample_count = bitCount(row_data);

		uint sample_offset = tri_sample_offset;
		uint row_id = tri_row_offset;
		for(uint y = 0; y < 4; y++) {
			uint row_bits = (row_data >> (y << 3)) & 0xff;
			if(row_bits == 0)
				continue;
			uint num_row_samples = bitCount(row_bits);

			uint seg_id = sample_offset >> 5, seg_offset = sample_offset & 31;
			if(seg_offset != 0)
				atomicOr(s_buffer[buf_offset + seg_id * 2 + 1], 1u << seg_offset);
			if(seg_offset == 0)
				s_buffer[buf_offset + seg_id * 2 + 0] = tri_idx | (row_id << 14);
			if(seg_offset + num_row_samples > 32)
				s_buffer[buf_offset + seg_id * 2 + 2] =
					tri_idx | (row_id << 14) | ((32 - seg_offset) << 26);
			sample_offset += num_row_samples;
			row_id++;
		}
	}
	barrier();

	uint seg_count = ((BLOCK_FRAG_COUNT(bx, by) & 0xffff) + 31) >> 5;
	dst_offset = scratchBlockSegmentOffset(bx, by);
	for(uint i = LIX & (LSIZE / 8 - 1); i < seg_count; i += LSIZE / 8)
		g_scratch[dst_offset + i] =
			uvec2(s_buffer[buf_offset + i * 2 + 0], s_buffer[buf_offset + i * 2 + 1]);

	barrier();
	for(uint i = LIX; i < BUFFER_SIZE; i += LSIZE)
		s_buffer[i] = 0;
	barrier();

	for(int k = 0; k < idx_count; k++) {
		uint tri_sample_offset = offsets[k] & 0xffff;
		uint tri_row_offset = offsets[k] >> 16;
		uint tri_idx = first_idx + (k << (LSHIFT - 3));

		uint row_data = seg_tri_rows[k];
		uint sample_offset = 0, row_id = tri_row_offset;
		for(uint y = 0; y < 4; y++) {
			uint row_bits = (row_data >> (y << 3)) & 0xff;
			if(row_bits == 0)
				continue;
			uint x = findLSB(row_bits);
			uint row_value = x | (y << 3) | (tri_idx << 6);
			atomicOr(s_buffer[buf_offset + (row_id >> 1)], (row_value & 0xffff)
															   << ((row_id & 1) << 4));
			row_id++;
		}
	}
	barrier();
	uint row2_count = ((BLOCK_FRAG_COUNT(bx, by) >> 16) + 1) >> 1;
	dst_offset = scratchBlockSegmentRowsOffset(bx, by);
	for(uint i = LIX & (LSIZE / 8 - 1); i < row2_count; i += LSIZE / 8)
		g_scratch[dst_offset + i] =
			uvec2(s_buffer[buf_offset + i * 2 + 0], s_buffer[buf_offset + i * 2 + 1]);
}

uint shadeSample(ivec2 bin_pixel_pos, uint scratch_tri_offset, out float out_depth);

// Aktualne statsy (3050 full):
//      bunny: 770    68     49    5
// conference: 1990   61     106   5
//     sponza: 5335   61     121   6
//     teapot: 723    26     45    5
void shadeAndReduceSamples(int by) {
	int bx = int(LIX >> (LSHIFT - 3));

	uint soffset = scratchBlockTrisOffset(bx, by);
	uint seg_soffset = scratchBlockSegmentOffset(bx, by);
	uint row_soffset = scratchBlockSegmentRowsOffset(bx, by);

	uint block_frag_count = BLOCK_FRAG_COUNT(bx, by) & 0xffff;
	uint block_seg_count = (block_frag_count + 31) >> 5;
	uint buf_offset = bx * MAX_BLOCK_TRIS;

	// TODO: problem: muszę synchronizować wszystkie wątki i tak... bo muszę robić bariery
	// Może później będę mógł po zakończeniu przetwarzania danego bloku od razu się przełączyć
	// na kolejny?

	// TODO: more ?
	float prev_depths[4] = {-1.0, -1.0, -1.0, -1.0};
	uint prev_colors[3] = {0, 0, 0};
	vec3 out_color = vec3(0);
	float out_transparency = 1.0;

	for(uint seg_id = 0; seg_id < block_seg_count; seg_id++) {
		uint i = (seg_id << 5) + (LIX & 31);
		s_buffer[buf_offset + (LIX & 31)] = 0;

		uint sample_color = 0;
		float sample_depth;

		if(i < block_frag_count) {
			uint seg_offset = i & 31;
			uvec2 seg_info = g_scratch[seg_soffset + seg_id];

			uint seg_mask = seg_offset == 31 ? ~0u : (1u << (seg_offset + 1)) - 1;
			uint masked_bits = seg_mask & seg_info.y;

			// TODO: more bits for rows
			uint row_idx = ((seg_info.x >> 14) & 0xfff) + bitCount(masked_bits);
			uint row_offset =
				seg_offset + (masked_bits == 0 ? seg_info.x >> 26 : -findMSB(masked_bits));
			uint row_data = (g_scratch[row_soffset + (row_idx >> 2)][(row_idx & 2) >> 1] >>
							 ((row_idx & 1) * 16)) &
							0xffff;

			uint rx = (row_data & 0x7) + row_offset, ry = (row_data >> 3) & 3;
			atomicOr(s_buffer[buf_offset + rx], 1 << (LIX & 31));
			atomicOr(s_buffer[buf_offset + 8 + ry], 1 << (LIX & 31));

			uint tri_idx = g_scratch[soffset + (row_data >> 6)].y & 0xffff;

			sample_color = shadeSample(ivec2(bx * BLOCK_WIDTH + rx, by * BLOCK_HEIGHT + ry),
									   scratchTriOffset(tri_idx), sample_depth);
			sample_depth = 1.0 / (1.0 + sample_depth);
		}

		uint rx = LIX & 7, ry = (LIX >> 3) & 3;
		uint bitmask = s_buffer[buf_offset + rx] & s_buffer[buf_offset + 8 + ry];

		int j = findLSB(bitmask);
		while(anyInvocationARB(j != -1)) {
			// TODO: pass through regs
			uint color = shuffleNV(sample_color, j, 32);
			float depth = shuffleNV(sample_depth, j, 32);
			if(j == -1)
				continue;

			bitmask &= ~(1 << j);
			j = findLSB(bitmask);

			if(color == 0)
				continue;

			if(depth < prev_depths[0]) {
				SWAP_UINT(color, prev_colors[0]);
				SWAP_FLOAT(depth, prev_depths[0]);
				if(prev_depths[0] < prev_depths[1]) {
					SWAP_UINT(prev_colors[1], prev_colors[0]);
					SWAP_FLOAT(prev_depths[1], prev_depths[0]);
					if(prev_depths[1] < prev_depths[2]) {
						SWAP_UINT(prev_colors[2], prev_colors[1]);
						SWAP_FLOAT(prev_depths[2], prev_depths[1]);
						if(prev_depths[2] < prev_depths[3]) {
							prev_colors[0] = 0xff0000ff;
							prev_depths[0] = 999999.0;
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
				out_color = (additive_blending ? out_color : out_color * cur_transparency) +
							cur_color.rgb * cur_color.a;
				out_transparency *= cur_transparency;
			}

			prev_colors[2] = prev_colors[1];
			prev_colors[1] = prev_colors[0];
			prev_colors[0] = color;
		}
	}

	for(int i = 2; i >= 0; i--)
		if(prev_colors[i] != 0) {
			vec4 cur_color = decodeRGBA8(prev_colors[i]);
			float cur_transparency = 1.0 - cur_color.a;
			out_color = (additive_blending ? out_color : out_color * cur_transparency) +
						cur_color.rgb * cur_color.a;
			out_transparency *= cur_transparency;
		}

	out_color = min(out_color, vec3(1.0));
	uint enc_color = encodeRGBA8(vec4(out_color, 1.0 - out_transparency));
	outputPixel(ivec2(bx * BLOCK_WIDTH + (LIX & 7), by * BLOCK_HEIGHT + ((LIX >> 3) & 3)),
				enc_color);
}

// Shading 2 samples at once didn't help:
// - decreased computation cost is not worth it because of increased register pressure
// - it seems that it does not help at all with loading vertex attribs; it makes sense:
//   if they are in the cache then it's not a problem...
// Can we improve speed of loading vertex data?
uint shadeSample(ivec2 bin_pixel_pos, uint scratch_tri_offset, out float out_depth) {
	vec3 ray_dir =
		s_bin_ray_dir0 + frustum.ws_dirx * bin_pixel_pos.x + frustum.ws_diry * bin_pixel_pos.y;

	vec3 normal, params, edge0, edge1;
	uint instance_id, instance_flags, instance_color;
	getTriangleParams(scratch_tri_offset, normal, params, edge0, edge1, instance_id, instance_flags,
					  instance_color);

	float ray_pos = params[0] / dot(normal, ray_dir);
	out_depth = ray_pos;
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

void rasterInvalidBlockRow(int by, vec3 color) {
	for(uint i = LIX; i < BIN_SIZE * BLOCK_HEIGHT; i += LSIZE) {
		ivec2 pixel_pos = ivec2(i & (BIN_SIZE - 1), by * BLOCK_HEIGHT + (i >> BIN_SHIFT));
		vec4 color = vec4(color, 1.0);
		uint enc_col = encodeRGBA8(color);
		outputPixel(pixel_pos, enc_col);
	}
}

void rasterBlockTriCounts(int by) {
	barrier();
	for(uint i = LIX; i < BIN_SIZE * BLOCK_HEIGHT; i += LSIZE) {
		ivec2 pixel_pos = ivec2(i & (BIN_SIZE - 1), by * BLOCK_HEIGHT + (i >> BIN_SHIFT));
		uint count = BLOCK_TRI_COUNT(pixel_pos.x / BLOCK_WIDTH, by);
		vec3 color = count > MAX_BLOCK_TRIS ?
						   vec3(1.0, 0.0, 0.0) :
						   vec3(0.5, 0.0, 0.0) + vec3(count, count, count) / float(MAX_BLOCK_TRIS);
		outputPixel(pixel_pos, encodeRGBA8(vec4(min(color, vec3(1.0)), 1.0)));
	}
}

void rasterCounts(int by) {
	barrier();
	for(uint i = LIX; i < BIN_SIZE * BLOCK_HEIGHT; i += LSIZE) {
		ivec2 pixel_pos = ivec2(i & (BIN_SIZE - 1), by * BLOCK_HEIGHT + (i >> BIN_SHIFT));
		uint count0 = BLOCK_FRAG_COUNT(pixel_pos.x / BLOCK_WIDTH, by) & 0xffff;
		//uint count0 = BLOCK_FRAG_COUNT(pixel_pos.x / BLOCK_WIDTH, by) >> 16;
		vec3 color = vec3(count0, count0, count0) / 512.0;
		uint enc_col = encodeRGBA8(vec4(min(color, vec3(1.0)), 1.0));
		outputPixel(pixel_pos, enc_col);
	}
}

void rasterBin(int bin_id) {
	INIT_CLOCK();

	if(LIX < BIN_SIZE) {
		if(LIX == 0) {
			ivec2 bin_pos = ivec2(bin_id % BIN_COUNT_X, bin_id / BIN_COUNT_X) * BIN_SIZE;
			s_bin_pos = bin_pos;
			s_bin_quad_count = g_bins.bin_quad_counts[bin_id];
			s_bin_quad_offset = g_bins.bin_quad_offsets[bin_id];
			s_bin_ray_dir0 = frustum.ws_dir0 + frustum.ws_dirx * (bin_pos.x + 0.5) +
							 frustum.ws_diry * (bin_pos.y + 0.5);
			s_invalid_bin = false;
		}
		if(LIX < BLOCK_ROW_COUNT)
			s_block_row_tri_counts[LIX] = 0;
	}
	resetBlockCounters();
	barrier();
	generateRows();
	groupMemoryBarrier();
	barrier();
	UPDATE_CLOCK(0);

	for(int by = 0; by < BLOCK_COUNTY; by++) {
		generateBlocks(by);
		barrier();
	}

	if(s_invalid_bin) {
		//for(int by = 0; by < BLOCK_COUNTY; by++)
		//	rasterInvalidBlockRow(by, vec3(0.2, 0.0, 0.0));
		for(int by = 0; by < BLOCK_COUNTY; by++)
			rasterBlockTriCounts(by);
		return;
	}

	groupMemoryBarrier();
	barrier();
	UPDATE_CLOCK(1);

	for(int by = 0; by < BLOCK_COUNTY; by++) {
		shadeAndReduceSamples(by);
		barrier();
	}
	UPDATE_CLOCK(2);
	//for(int by = 0; by < BLOCK_COUNTY; by++)
	//rasterCounts(by);
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
