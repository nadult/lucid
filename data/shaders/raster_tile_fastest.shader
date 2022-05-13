// $$include funcs lighting frustum viewport data raster

// In this rasterizer we're processing half-blocks (8x4); We have 8 hblocks in a tile

// TODO: use more threads per bin (better data coherency);
// at least 512, 1024 if possible? process 2-4 tiles at once
//
// TODO: we should limit scratch memory size as much as possible; in case of large number of SMs
// we would need a lot memory just for scratch; and it would be mostly unused;
//
// Maybe we could allocate this memory dynamically somehow?

#define LSIZE 256
#define LSHIFT 8

#define NUM_WARPS (LSIZE / 32)
#define NUM_WARPS_SHIFT (LSHIFT - 5)

#define WARP_SIZE 32
#define WARP_STEP 32
#define WARP_MASK 31

#define BUFFER_SIZE (LSIZE * 8)

#define MAX_HBLOCK_TRIS 1024

#define MAX_SCRATCH_TRIS 4096
#define MAX_SCRATCH_TRIS_SHIFT 12

#define SEGMENT_SIZE 128
#define SEGMENT_SHIFT 7

#define HBLOCKS_PER_TILE 8

layout(local_size_x = LSIZE) in;

layout(std430, binding = 8) buffer buf8_ { uint g_tile_tris[]; };

#define WORKGROUP_32_SCRATCH_SIZE (32 * 1024)
#define WORKGROUP_32_SCRATCH_SHIFT 15

#define WORKGROUP_64_SCRATCH_SIZE (64 * 1024)
#define WORKGROUP_64_SCRATCH_SHIFT 16

#define TRI_SCRATCH(var_idx) g_scratch_64[scratch_tri_offset + (var_idx << MAX_SCRATCH_TRIS_SHIFT)]

uint scratch64TriOffset(uint tri_idx) {
	return (gl_WorkGroupID.x << WORKGROUP_64_SCRATCH_SHIFT) + tri_idx;
}

uint scratch64InitialHBlockOffset(uint bid) {
	return (gl_WorkGroupID.x << WORKGROUP_64_SCRATCH_SHIFT) + 32 * 1024 + bid * MAX_HBLOCK_TRIS;
}

uint scratch64SortedHBlockTrisOffset(uint bid) {
	return (gl_WorkGroupID.x << WORKGROUP_64_SCRATCH_SHIFT) + 48 * 1024 + bid * MAX_HBLOCK_TRIS;
}

shared int s_num_bins, s_bin_id, s_bin_raster_offset;
shared int s_tile_tri_count, s_tile_tri_offset, s_tile_raster_offset;
shared int s_non_empty_tile_tri_count;
shared ivec2 s_bin_pos, s_tile_pos;
shared int s_tile_tri_counts[TILES_PER_BIN];
shared int s_tile_tri_offsets[TILES_PER_BIN];
shared vec3 s_tile_ray_dirs0[TILES_PER_BIN], s_tile_ray_dir0;

shared uint s_group_size, s_group_shift, s_group_step, s_group_max_shift;

// Low 16-bits: triangle count, High 16-bits: fragment count
shared uint s_hblock_counts[HBLOCKS_PER_TILE];

shared uint s_buffer[BUFFER_SIZE + 1];
shared uint s_mini_buffer[LSIZE];

// TODO: how many segments? Should we recreate them every 256 tris?
shared uint s_segments[HBLOCKS_PER_TILE * WARP_SIZE];
shared int s_raster_error;

// Only used when debugging
shared uint s_vis_pixels[LSIZE];

void outputPixel(ivec2 pixel_pos, uint color) {
	//color = tintColor(color, vec3(0, 1, 0), 0.2);
	g_raster_image[s_tile_raster_offset + pixel_pos.x + (pixel_pos.y << BIN_SHIFT)] = color;
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
	// TODO: flag for instance color?

	// Nice optimization for barycentric computations:
	// dot(cross(edge, dir), normal) == dot(dir, cross(normal, edge))
	edge0 = cross(normal, edge0);
	edge1 = cross(normal, edge1);

	edge0 =
		vec3(dot(edge0, frustum.ws_dirx), dot(edge0, frustum.ws_diry), dot(edge0, s_tile_ray_dir0));
	edge1 =
		vec3(dot(edge1, frustum.ws_dirx), dot(edge1, frustum.ws_diry), dot(edge1, s_tile_ray_dir0));

	vec3 pnormal = normal * (1.0 / plane_dist);
	vec3 depth_eq = vec3(dot(pnormal, frustum.ws_dirx), dot(pnormal, frustum.ws_diry),
						 dot(pnormal, s_tile_ray_dir0));

	TRI_SCRATCH(0) = uvec2(floatBitsToUint(depth_eq.x), floatBitsToUint(depth_eq.y));
	TRI_SCRATCH(1) = uvec2(floatBitsToUint(depth_eq.z), instance_flags | (instance_id << 16));
	TRI_SCRATCH(2) = uvec2(floatBitsToUint(param0), floatBitsToUint(param1));
	TRI_SCRATCH(3) = uvec2(floatBitsToUint(edge0.x), floatBitsToUint(edge0.y));
	TRI_SCRATCH(4) = uvec2(floatBitsToUint(edge0.z), floatBitsToUint(edge1.x));
	TRI_SCRATCH(5) = uvec2(floatBitsToUint(edge1.y), floatBitsToUint(edge1.z));

	// TODO: instance color...
	TRI_SCRATCH(6) = uvec2(unormal, v0);
	TRI_SCRATCH(7) = uvec2(v1, v2);
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

void getTriangleSecondaryParams(uint scratch_tri_offset, out uint unormal, out uint v0, out uint v1,
								out uint v2) {
	uvec2 val0 = TRI_SCRATCH(6);
	uvec2 val1 = TRI_SCRATCH(7);
	unormal = val0.x;
	v0 = val0.y;
	v1 = val1.x;
	v2 = val1.y;
}

int generateHBlockTris(uint tri_idx, vec3 tri0, vec3 tri1, vec3 tri2, int min_by, int max_by) {
	vec2 scan_start = vec2(s_tile_pos) + vec2(-0.5f, float(min_by) * 4.0 + 0.5f);
	vec3 scan_min, scan_max, scan_step;
	computeScanlineParams(tri0, tri1, tri2, scan_start, scan_min, scan_max, scan_step);

	uint dst_offset_64 = scratch64InitialHBlockOffset(0);
	int tile_tri_idx = -1;

	// TODO: is it worth it to make this loop more work-efficient?
	for(int by = min_by; by <= max_by; by++) {
#define SCAN_STEP(id)                                                                              \
	int min##id = int(max(max(scan_min[0], scan_min[1]), max(scan_min[2], 0.0)));                  \
	int max##id = int(min(min(scan_max[0], scan_max[1]), min(scan_max[2], TILE_SIZE))) - 1;        \
	if(min##id > max##id)                                                                          \
		min##id = TILE_SIZE - 1, max##id = 0;                                                      \
	scan_min += scan_step, scan_max += scan_step;
		SCAN_STEP(0);
		SCAN_STEP(1);
		SCAN_STEP(2);
		SCAN_STEP(3);
#undef SCAN_STEP

		ivec4 qmin0 = ivec4(min0, min1, min2, min3);
		ivec4 qmax0 = min(ivec4(max0, max1, max2, max3), 7);
		ivec4 qcount0 = max(qmax0 - qmin0 + 1, 0);
		uint count0 = qcount0[0] + qcount0[1] + qcount0[2] + qcount0[3];

		ivec4 qmin1 = max(ivec4(min0, min1, min2, min3) - 8, 0);
		ivec4 qmax1 = min(ivec4(max0, max1, max2, max3) - 8, 7);
		ivec4 qcount1 = max(qmax1 - qmin1 + 1, 0);
		qmin1 &= 7;
		qmin0 &= 7;
		uint count1 = qcount1[0] + qcount1[1] + qcount1[2] + qcount1[3];

		if(count0 + count1 == 0)
			continue;

		if(tile_tri_idx == -1)
			tile_tri_idx = atomicAdd(s_non_empty_tile_tri_count, 1);

		if(count0 != 0) {
			uint bid = (by << 1) + 0;
			uint boffset = atomicAdd(s_hblock_counts[bid], 1 + (count0 << 16)) & 0xffff;
			uint min_bits = qmin0.x | (qmin0.y << 4) | (qmin0.z << 8) | (qmin0.w << 12);
			uint count_bits =
				(qcount0.x << 16) | (qcount0.y << 20) | (qcount0.z << 24) | (qcount0.w << 28);
			uvec2 value = uvec2(min_bits | count_bits, tile_tri_idx | (count0 << 16));
			if(boffset < MAX_HBLOCK_TRIS)
				g_scratch_64[dst_offset_64 + boffset + bid * MAX_HBLOCK_TRIS] = value;
			else
				atomicOr(s_raster_error, 1 << bid);
		}
		if(count1 != 0) {
			uint bid = (by << 1) + 1;
			uint boffset = atomicAdd(s_hblock_counts[bid], 1 + (count1 << 16)) & 0xffff;
			uint min_bits = qmin1.x | (qmin1.y << 4) | (qmin1.z << 8) | (qmin1.w << 12);
			uint count_bits =
				(qcount1.x << 16) | (qcount1.y << 20) | (qcount1.z << 24) | (qcount1.w << 28);
			uvec2 value = uvec2(min_bits | count_bits, tile_tri_idx | (count1 << 16));
			if(boffset < MAX_HBLOCK_TRIS)
				g_scratch_64[dst_offset_64 + boffset + bid * MAX_HBLOCK_TRIS] = value;
			else
				atomicOr(s_raster_error, 1 << bid);
		}
	}
	return tile_tri_idx;
}

void processInputTris() {
	for(uint i = LIX; i < s_tile_tri_count; i += LSIZE) {
		uint tri_idx = g_tile_tris[s_tile_tri_offset + i];
		uint second_tri = tri_idx >> 31;
		uint quad_idx = tri_idx & 0x7fffffff;

		uvec4 aabb = g_tri_aabbs[quad_idx];
		aabb = decodeAABB(second_tri != 0 ? aabb.zw : aabb.xy);
		int min_by = clamp(int(aabb[1]) - s_tile_pos.y, 0, 15) >> 2;
		int max_by = clamp(int(aabb[3]) - s_tile_pos.y, 0, 15) >> 2;

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

		int tile_tri_idx = generateHBlockTris(i, tri0, tri1, tri2, min_by, max_by);
		// TODO: separate this ?
		if(tile_tri_idx >= MAX_SCRATCH_TRIS)
			atomicOr(s_raster_error, 0x80000000);
		else if(tile_tri_idx != -1)
			storeTriangle(tile_tri_idx, tri0, tri1, tri2, v0, v1, v2, instance_id);
	}
}

shared uint s_sort_rcount[HBLOCKS_PER_TILE];
shared uint s_sort_max_rcount;

void prepareSortTris() {
	// TODO: optimize: take groups into account
	if(LIX < HBLOCKS_PER_TILE) {
		uint count = s_hblock_counts[LIX] & 0xffff;
		// rcount: count rounded up to next power of 2
		uint rcount = max(32, (count & (count - 1)) == 0 ? count : (2 << findMSB(count)));
		s_sort_rcount[LIX] = rcount;
		uint max_rcount = rcount;
		max_rcount = max(max_rcount, shuffleXorNV(max_rcount, 1, 8));
		max_rcount = max(max_rcount, shuffleXorNV(max_rcount, 2, 8));
		max_rcount = max(max_rcount, shuffleXorNV(max_rcount, 4, 8));
		if(LIX == 0)
			s_sort_max_rcount = max_rcount;
	}
}

#ifdef VENDOR_NVIDIA
uint swap(uint x, int mask, uint dir) {
	uint y = shuffleXorNV(x, mask, 32);
	return uint(x < y) == dir ? y : x;
}
uint bitExtract(uint value, int boffset) { return (value >> boffset) & 1; }
uint xorBits(uint value, int bit0, int bit1) { return ((value >> bit0) ^ (value >> bit1)) & 1; }
#endif

void sortTris(uint count, uint buf_offset) {
	barrier();

	uint max_rcount = s_sort_max_rcount;
	uint group_step = WARP_STEP << s_group_shift;
	uint lid = LIX & (group_step - 1);

	for(uint i = lid + count; i < max_rcount; i += group_step)
		s_buffer[buf_offset + i] = 0xffffffff;
	barrier();
#ifdef VENDOR_NVIDIA
	for(uint i = lid; i < max_rcount; i += group_step) {
		uint value = s_buffer[buf_offset + i];
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
		s_buffer[buf_offset + i] = value;
	}
	barrier();

	int start_k = 32, end_j = 32;
#else
	int start_k = 2, end_j = 1;
#endif

	for(uint k = start_k; k <= max_rcount; k = 2 * k) {
		for(uint j = k >> 1; j >= end_j; j = j >> 1) {
			for(uint i = lid; i < max_rcount; i += group_step * 2) {
				uint idx = (i & j) != 0 ? i + group_step - j : i;
				uint lvalue = s_buffer[buf_offset + idx];
				uint rvalue = s_buffer[buf_offset + idx + j];
				if(((idx & k) != 0) == (lvalue.x < rvalue.x)) {
					s_buffer[buf_offset + idx] = rvalue;
					s_buffer[buf_offset + idx + j] = lvalue;
				}
			}
			barrier();
		}
#ifdef VENDOR_NVIDIA
		for(uint i = lid; i < max_rcount; i += group_step) {
			uint bit = (i & k) == 0 ? 0 : 1;
			uint value = s_buffer[buf_offset + i];
			value = swap(value, 0x10, bit ^ bitExtract(lid, 4));
			value = swap(value, 0x08, bit ^ bitExtract(lid, 3));
			value = swap(value, 0x04, bit ^ bitExtract(lid, 2));
			value = swap(value, 0x02, bit ^ bitExtract(lid, 1));
			value = swap(value, 0x01, bit ^ bitExtract(lid, 0));
			s_buffer[buf_offset + i] = value;
		}
		barrier();
#endif
	}
}

// TODO: process groups of 1, 2, 4 or 8 blocks (in case of large number of tris/block)
void generateBlocks(uint bid) {
	uint by = bid >> 1, bx = bid & 1;

	uint src_offset_64 = scratch64InitialHBlockOffset(bid);
	uint dst_offset_64 = scratch64SortedHBlockTrisOffset(bid);
	uint tri_count = s_hblock_counts[bid] & 0xffff;
	uint buf_offset = (bid & (s_group_step - 1)) << s_group_max_shift;
	uint group_step = WARP_STEP << s_group_shift;
	uint group_mask = group_step - 1;

	for(uint i = LIX & group_mask; i < tri_count; i += group_step) {
		uvec2 tri_data = g_scratch_64[src_offset_64 + i];
		uint tri_idx = tri_data.y & 0xffff;
		uint num_frags = tri_data.y >> 16;

		ivec4 xmin = (ivec4(tri_data.x) >> ivec4(0, 4, 8, 12)) & 7;
		ivec4 xcount = (ivec4(tri_data.x) >> ivec4(16, 20, 24, 28)) & 15;
		vec4 cpx = vec4(xmin * 2 + xcount) * xcount;
		vec4 cpy = vec4(1.0, 3.0, 5.0, 7.0) * xcount;
		vec2 cpos = vec2(cpx[0] + cpx[1] + cpx[2] + cpx[3], cpy[0] + cpy[1] + cpy[2] + cpy[3]);
		cpos = cpos * (0.5 / float(num_frags)) + vec2(bx << 3, by << 2);

		uint scratch_tri_offset = scratch64TriOffset(tri_idx);
		vec2 val0 = uintBitsToFloat(TRI_SCRATCH(0));
		vec2 val1 = uintBitsToFloat(TRI_SCRATCH(1));
		vec3 depth_eq = vec3(val0.x, val0.y, val1.x);
		float ray_pos = depth_eq.x * cpos.x + (depth_eq.y * cpos.y + depth_eq.z);
		float depth = 0xffffe * SATURATE(inversesqrt(ray_pos + 1)); // 20 bits

		// 12 bits for block-tri index, 20 bits for depth
		s_buffer[buf_offset + i] = i | (uint(depth) << 12);
	}

	barrier();
	sortTris(tri_count, buf_offset);
	barrier();

#ifdef SHADER_DEBUG
	for(uint i = LIX & group_mask; i < tri_count; i += group_step) {
		uint value = s_buffer[buf_offset + i];
		uint prev_value = i == 0 ? 0 : s_buffer[buf_offset + i - 1];
		if(value <= prev_value)
			RECORD(i, tri_count, prev_value, value);
	}
#endif

	for(uint i = LIX & group_mask; i < tri_count; i += group_step) {
		uint block_tri_idx = s_buffer[buf_offset + i] & 0xfff;
		uint value = g_scratch_64[src_offset_64 + block_tri_idx].y >> 16;

		// Computing triangle-ordered sample offsets within each block
		uint warp_id = LIX & 31, temp;
		temp = shuffleUpNV(value, 1, 32), value += warp_id >= 1 ? temp : 0;
		temp = shuffleUpNV(value, 2, 32), value += warp_id >= 2 ? temp : 0;
		temp = shuffleUpNV(value, 4, 32), value += warp_id >= 4 ? temp : 0;
		temp = shuffleUpNV(value, 8, 32), value += warp_id >= 8 ? temp : 0;
		temp = shuffleUpNV(value, 16, 32), value += warp_id >= 16 ? temp : 0;
		s_buffer[buf_offset + i] = block_tri_idx | (value << 16);
	}
	barrier();

	// TODO: increase max elems to 1024 (2048 is too much?)
	// Computing prefix sum across whole blocks (at most 8 * 32 elements)
	// TODO: make it work for all group sizes
	if(LIX < 64) {
		uint group_warp_shift = 3 + s_group_shift;
		uint group_warp_mask = (1 << group_warp_shift) - 1;
		uint fbid = bid & ~(s_group_step - 1);
		uint bid = LIX >> group_warp_shift;
		uint warp_idx = LIX & group_warp_mask, warp_offset = warp_idx << 5;
		uint buf_offset = bid << s_group_max_shift;
		uint tri_count = s_hblock_counts[fbid + bid] & 0xffff;
		uint value = 0;

		if(warp_offset < tri_count) {
			uint tri_idx = min(warp_offset + 31, tri_count - 1);
			value = s_buffer[buf_offset + tri_idx] >> 16;
		}
		uint sum = value, temp;
		temp = shuffleUpNV(sum, 1, 32), sum += warp_idx >= 1 ? temp : 0;
		temp = shuffleUpNV(sum, 2, 32), sum += warp_idx >= 2 ? temp : 0;
		temp = shuffleUpNV(sum, 4, 32), sum += warp_idx >= 4 ? temp : 0;
		temp = shuffleUpNV(sum, 8, 32), sum += warp_idx >= 8 ? temp : 0;
		temp = shuffleUpNV(sum, 16, 32), sum += warp_idx >= 16 ? temp : 0;
		s_mini_buffer[LIX] = sum - value;
	}
	barrier();

	// Storing triangle fragment offsets to scratch mem
	uint mini_offset = buf_offset >> 5;
	for(uint i = LIX & group_mask; i < tri_count; i += group_step) {
		uint tri_offset = 0;
		if(i > 0) {
			uint prev = i - 1;
			tri_offset = s_buffer[buf_offset + prev] >> 16;
			tri_offset += s_mini_buffer[mini_offset + (prev >> 5)];
		}

		uint block_tri_idx = s_buffer[buf_offset + i] & 0xffff;
		uvec2 tri_data = g_scratch_64[src_offset_64 + block_tri_idx];
		tri_data.y = (tri_data.y & 0xffff) | (tri_offset << 16); // TODO: opt
		g_scratch_64[dst_offset_64 + i] = tri_data;
	}
	barrier();
}

// Jak chcemy generowaæ segmenty ?
// Max 32 segmenty i po prostu mamy jakiœ startowy tris?
void findSegments(uint bid, uint first_segment_id) {
	uint segment_id = first_segment_id + (LIX & 31);
	uint target_frag_offset = segment_id * SEGMENT_SIZE;
	uint src_offset_64 = scratch64SortedHBlockTrisOffset(bid);

	// TODO: this can be greatly optimized
	uint tri_count = s_hblock_counts[bid] & 0xffff;
	int min_tri_id = 0, max_tri_id = int(tri_count) - 1;
	while(min_tri_id + 1 < max_tri_id) {
		int mid = (min_tri_id + max_tri_id) >> 1;
		uint value = g_scratch_64[src_offset_64 + mid].y >> 16;
		if(value > target_frag_offset)
			max_tri_id = target_frag_offset + 32 <= value ? mid - 1 : mid;
		else
			min_tri_id = value + 32 <= target_frag_offset ? mid + 1 : mid;
	}

	int tri_id = min_tri_id;
	while(tri_id < tri_count) {
		uvec2 tri_data = g_scratch_64[src_offset_64 + tri_id];
		// Maybe it's cheaper to just compare with next value?
		uvec4 qcount = (uvec4(tri_data.x) >> uvec4(16, 20, 24, 28)) & 15;
		uint cur_count = qcount.x + qcount.y + qcount.z + qcount.w;
		uint cur_offset = tri_data.y >> 16;
		if(target_frag_offset < cur_offset + cur_count)
			break;
		tri_id++;
	}

	s_segments[bid * WARP_SIZE + (LIX & 31)] = tri_id;
}

void loadSamples(uint bid, uint segment_id) {
	uint first_tri = s_segments[bid * WARP_SIZE + segment_id];
	uint tri_count = min((s_hblock_counts[bid] & 0xffff) - first_tri, SEGMENT_SIZE);

	uint src_offset_64 = scratch64SortedHBlockTrisOffset(bid) + first_tri;
	uint buf_offset = bid << SEGMENT_SHIFT;
	int first_offset = int(segment_id << SEGMENT_SHIFT);

	int y = int(LIX & 3);
	uint count_shift = 16 + (y << 2), min_shift = (y << 2);
	int mask1 = y >= 1 ? ~0 : 0, mask2 = y >= 2 ? ~0 : 0;

	// TODO: group differently for better memory accesses (and measure)
	for(uint i = (LIX & WARP_MASK) >> 2; i < tri_count; i += WARP_STEP / 4) {
		uvec2 tri_data = g_scratch_64[src_offset_64 + i];
		uint tri_idx = tri_data.y & 0xffff;
		int tri_offset = int(tri_data.y >> 16) - first_offset;
		if(tri_offset > SEGMENT_SIZE)
			break;

		int minx = int((tri_data.x >> min_shift) & 7);
		int countx = int((tri_data.x >> count_shift) & 15);
		int prevx = countx + (shuffleUpNV(countx, 1, 4) & mask1);
		prevx += shuffleUpNV(prevx, 2, 4) & mask2;
		tri_offset += prevx - countx;

		countx = min(countx, SEGMENT_SIZE - tri_offset);
		if(tri_offset < 0) {
			countx += tri_offset;
			minx -= tri_offset;
			tri_offset = 0;
		}
		if(countx <= 0)
			continue;

		uint scratch_tri_offset = scratch64TriOffset(tri_idx);
		uint pixel_id = (y << 3) | minx;
		uint value = pixel_id | (scratch_tri_offset << 8);
		for(int j = 0; j < countx; j++) {
			s_buffer[buf_offset + tri_offset] = value;
			tri_offset++;
			value++;
		}
	}
}

uint shadeSample(ivec2 tile_pixel_pos, uint scratch_tri_offset, out float out_depth) {
	float px = float(tile_pixel_pos.x), py = float(tile_pixel_pos.y);

	vec3 depth_eq, edge0_eq, edge1_eq;
	uint instance_id, instance_flags;
	vec2 bary_params;
	getTriangleParams(scratch_tri_offset, depth_eq, bary_params, edge0_eq, edge1_eq, instance_id,
					  instance_flags);

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

	uint unormal, v0, v1, v2;
	getTriangleSecondaryParams(scratch_tri_offset, unormal, v0, v1, v2);

	uint instance_color = g_instances[instance_id].color;
	vec4 color = decodeRGBA8(instance_color);
	if((instance_flags & INST_HAS_TEXTURE) != 0) {
		vec2 tex0 = g_tex_coords[v0];
		vec2 tex1 = g_tex_coords[v1] - tex0;
		vec2 tex2 = g_tex_coords[v2] - tex0;

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
		vec4 col0 = decodeRGBA8(g_colors[v0]);
		vec4 col1 = decodeRGBA8(g_colors[v1]);
		vec4 col2 = decodeRGBA8(g_colors[v2]);
		color *= (1.0 - bary[0] - bary[1]) * col0 + (bary[0] * col1 + bary[1] * col2);
	}

	if(color.a == 0.0)
		return 0;

	vec3 normal;
	if((instance_flags & INST_HAS_VERTEX_NORMALS) != 0) {
		vec3 nrm0 = decodeNormalUint(g_normals[v0]);
		vec3 nrm1 = decodeNormalUint(g_normals[v1]) - nrm0;
		vec3 nrm2 = decodeNormalUint(g_normals[v2]) - nrm0;
		normal = bary[0] * nrm1 + (bary[1] * nrm2 + nrm0);
	} else {
		normal = decodeNormalUint(unormal);
	}

	float light_value = max(0.0, dot(-lighting.sun_dir, normal) * 0.7 + 0.3);
	color.rgb = SATURATE(finalShading(color.rgb, light_value));
	return encodeRGBA8(color);
}

void shadeSamples(uint bid, uint sample_count) {
	// TODO: what's the best way to fix broken pixels?
	// full sort ? recreate full depth values and sort pairs?

	uint buf_offset = bid << SEGMENT_SHIFT;
	ivec2 half_block_pos = ivec2((bid & 1) << 3, (bid >> 1) << 2);

	for(uint i = LIX & WARP_MASK; i < sample_count; i += WARP_STEP) {
		uint value = s_buffer[buf_offset + i];
		uint pixel_id = value & 31;
		uint scratch_tri_offset = value >> 8;
		ivec2 pix_pos = half_block_pos + ivec2(pixel_id & 7, pixel_id >> 3);
		float depth;
		s_buffer[buf_offset + i] = shadeSample(pix_pos, scratch_tri_offset, depth);
		s_buffer[buf_offset + i + SEGMENT_SIZE * NUM_WARPS] =
			(floatBitsToUint(depth) & ~31) | pixel_id;
	}
}

struct ReductionContext {
#ifdef VISUALIZE_ERRORS
	vec4 prev_depths;
#else
	vec3 prev_depths;
#endif
	uvec3 prev_colors;
	float out_trans;
	uint out_color;
};

void initReduction(out ReductionContext ctx) {
#ifdef VISUALIZE_ERRORS
	ctx.prev_depths = vec4(999999999.0);
#else
	ctx.prev_depths = vec3(999999999.0);
#endif
	ctx.prev_colors = uvec3(0);
	ctx.out_color = 0;
	ctx.out_trans = 1.0;
}

void reduceSamples(uint bid, uint sample_count, in out ReductionContext ctx) {
	uint buf_offset = bid << SEGMENT_SHIFT;
	uint mini_offset = LIX & ~31;
	uint pixel_bit = 1u << (LIX & 31);
	uint pixel_id = LIX & 31;
	vec3 out_color = decodeRGB10(ctx.out_color);

	for(uint i = 0; i < sample_count; i += 32) {
		uint sample_offset = i + (LIX & 31);
		uint sample_pixel_id = s_buffer[buf_offset + sample_offset + SEGMENT_SIZE * NUM_WARPS] & 31;

		s_mini_buffer[LIX] = 0;
		if(sample_offset < sample_count)
			atomicOr(s_mini_buffer[mini_offset + sample_pixel_id], pixel_bit);

		uint bitmask = s_mini_buffer[mini_offset + pixel_id];
		int j = findLSB(bitmask);
		while(j != -1) {
			// TODO: pass through regs?
			uint value = s_buffer[buf_offset + i + j];
			float depth = uintBitsToFloat(s_buffer[buf_offset + i + j + SEGMENT_SIZE * NUM_WARPS]);

			bitmask &= ~(1 << j);
			j = findLSB(bitmask);

			if(depth > ctx.prev_depths[0]) {
				SWAP_UINT(value, ctx.prev_colors[0]);
				SWAP_FLOAT(depth, ctx.prev_depths[0]);
				if(ctx.prev_depths[0] > ctx.prev_depths[1]) {
					SWAP_UINT(ctx.prev_colors[1], ctx.prev_colors[0]);
					SWAP_FLOAT(ctx.prev_depths[1], ctx.prev_depths[0]);
					if(ctx.prev_depths[1] > ctx.prev_depths[2]) {
						SWAP_UINT(ctx.prev_colors[2], ctx.prev_colors[1]);
						SWAP_FLOAT(ctx.prev_depths[2], ctx.prev_depths[1]);

#ifdef VISUALIZE_ERRORS
						if(ctx.prev_depths[2] > ctx.prev_depths[3]) {
							ctx.prev_colors[2] = 0xff0000ff;
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
				out_color += cur_color.rgb * cur_color.a;
#else
				out_color += cur_color.rgb * cur_color.a * ctx.out_trans;
				ctx.out_trans *= 1.0 - cur_color.a;

#ifdef ALPHA_THRESHOLD
				if(allInvocationsARB(ctx.out_trans < 1.0 / 255.0)) {
					i = sample_count;
					break;
				}
#endif
#endif
			}

			ctx.prev_colors[2] = ctx.prev_colors[1];
			ctx.prev_colors[1] = ctx.prev_colors[0];
			ctx.prev_colors[0] = value;
		}
	}

	// TODO: check if encode+decode for out_color is really needed (to save 2 regs)
	ctx.out_color = encodeRGB10(out_color);
}

void finishReduction(ivec2 pixel_pos, ReductionContext ctx) {
	vec3 out_color = decodeRGB10(ctx.out_color);

	for(int i = 2; i >= 0; i--)
		if(ctx.prev_colors[i] != 0) {
			vec4 cur_color = decodeRGBA8(ctx.prev_colors[i]);
			float cur_transparency = 1.0 - cur_color.a;
#ifdef ADDITIVE_BLENDING
			out_color += cur_color.rgb * cur_color.a;
#else
			out_color += cur_color.rgb * cur_color.a * ctx.out_trans;
			ctx.out_trans *= 1.0 - cur_color.a;
#endif
		}

	out_color += ctx.out_trans * decodeRGB8(background_color);
	uint enc_color = encodeRGB8(SATURATE(out_color)); // TODO: 10 bit
	outputPixel(pixel_pos, enc_color);
}

void initVisualizeSamples() { s_vis_pixels[LIX] = 0; }

void visualizeSamples(uint bid, uint sample_count) {
	uint buf_offset = bid << SEGMENT_SHIFT;
	for(uint i = LIX & 31; i < sample_count; i += 32) {
		uint pixel_id = s_buffer[buf_offset + i] & 31;
		atomicAdd(s_vis_pixels[(LIX & ~31) + pixel_id], 1);
	}
}

void finishVisualizeSamples(uint bid, ivec2 pixel_pos) {
	uint pixel_id = (pixel_pos.x & 7) + ((pixel_pos.y & 3) << 3);
	vec3 color = vec3(s_vis_pixels[(LIX & ~31) + pixel_id]) / 32.0;
	uint enc_col = encodeRGBA8(vec4(SATURATE(color), 1.0));
	outputPixel(pixel_pos, enc_col);
}

vec3 colorizeValue(uint value, uvec4 steps) {
	vec3 color;
	if(value < steps[0])
		color = vec3(value) / steps[0];
	else if(value < steps[1]) {
		float t = float(value - steps[0]) / (steps[1] - steps[0]);
		color = vec3(1.0 - t * 0.25, 1.0 - t * 0.25, 1.0 - t);
	} else if(value < steps[2]) {
		float t = float(value - steps[1]) / (steps[2] - steps[1]);
		color = vec3(0.75 + 0.25 * t, 0.75 - 0.75 * t, t);
	} else {
		float t = min(1.0, float(value - steps[2]) / (steps[3] - steps[2]));
		color = vec3(1.0, 0.0, 1.0 - t);
	}
	return color;
}

void visualizeHBlockTriangleCounts(uint bid, ivec2 pixel_pos) {
	uint count = s_hblock_counts[bid] & 0xffff;
	//count = s_tile_tri_count;
	vec3 color = colorizeValue(count, uvec4(256, 512, 1024, 2048));
	if(s_tile_tri_count > MAX_SCRATCH_TRIS)
		color = vec3(1, 0, 0);
	outputPixel(pixel_pos, encodeRGBA8(vec4(SATURATE(color), 1.0)));
}

void visualizeHBlockFragmentCounts(uint bid, ivec2 pixel_pos) {
	uint count = s_hblock_counts[bid] >> 16;
	vec3 color = colorizeValue(count, uvec4(256, 1024, 2048, 4096));
	if(s_tile_tri_count > MAX_SCRATCH_TRIS)
		color = vec3(1, 0, 0);
	outputPixel(pixel_pos, encodeRGBA8(vec4(SATURATE(color), 1.0)));
}

void visualizeErrors() {
	uint bid = LIX >> 5, bx = bid & 1, by = bid >> 1;
	ivec2 pixel_pos = ivec2((LIX & 7) + bx * 8, ((LIX >> 3) & 3) + by * 4);
	uint color = 0xff000031;
	if((s_raster_error & (1 << bid)) != 0)
		color += 0x32;
	if((s_raster_error & 0x80000000) != 0)
		color += 0x80;
	outputPixel(pixel_pos, color);
}

#define NUM_BIN_STEPS (64 / NUM_WARPS)

void rasterBin(int bin_id) {
	INIT_CLOCK();

	if(LIX < TILES_PER_BIN) {
		s_tile_tri_counts[LIX] = int(g_tiles.tile_tri_counts[bin_id][LIX]);
		s_tile_tri_offsets[LIX] = int(g_tiles.tile_tri_offsets[bin_id][LIX]);
		ivec2 bin_pos = ivec2(bin_id % BIN_COUNT_X, bin_id / BIN_COUNT_X) * BIN_SIZE;
		ivec2 tile_pos = bin_pos + ivec2(LIX & 3, LIX >> 2) * TILE_SIZE;
		s_tile_ray_dirs0[LIX] = frustum.ws_dir0 + frustum.ws_dirx * (tile_pos.x + 0.5) +
								frustum.ws_diry * (tile_pos.y + 0.5);
		if(LIX == 0) {
			s_raster_error = 0;
			s_bin_pos = bin_pos;
		}
	}
	barrier();

	for(int tile_id = 0; tile_id < TILES_PER_BIN; tile_id++) {
		barrier();
		if(LIX < HBLOCKS_PER_TILE) {
			s_hblock_counts[LIX] = 0;
			if(LIX == 0) {
				s_tile_tri_count = s_tile_tri_counts[tile_id];
				s_non_empty_tile_tri_count = 0;
				s_tile_tri_offset = s_tile_tri_offsets[tile_id];
				ivec2 tile_pos = ivec2(tile_id & 3, tile_id >> 2) * TILE_SIZE;
				s_tile_raster_offset =
					(bin_id << (BIN_SHIFT * 2)) + tile_pos.x + (tile_pos.y << BIN_SHIFT);
				s_tile_pos = tile_pos + s_bin_pos;
				s_tile_ray_dir0 = s_tile_ray_dirs0[tile_id];
			}
		}
		barrier();
		processInputTris();
		groupMemoryBarrier();
		barrier();
		if(LIX < HBLOCKS_PER_TILE) {
			uint max_num_tris = s_hblock_counts[LIX] & 0xffff;
			max_num_tris = max(max_num_tris, shuffleXorNV(max_num_tris, 1, 8));
			max_num_tris = max(max_num_tris, shuffleXorNV(max_num_tris, 2, 8));
			max_num_tris = max(max_num_tris, shuffleXorNV(max_num_tris, 4, 8));

			if(LIX == 0) {
				s_group_size = max_num_tris <= 256 ? 1 : max_num_tris <= 512 ? 2 : 4;
				uint group_shift = max_num_tris <= 256 ? 0 : max_num_tris <= 512 ? 1 : 2;
				s_group_shift = group_shift;
				s_group_step = 8 >> group_shift;
				s_group_max_shift = 8 + group_shift;
			}
		}

		// TODO: merge it with preceeding if block
		prepareSortTris();
		barrier();
		UPDATE_CLOCK(0);

		// TODO: handle tri_count == 0

		if(s_raster_error != 0) {
			visualizeErrors();
			barrier();
			if(LIX == 0)
				s_raster_error = 0;
			continue;
		}

		for(uint bid = LIX >> (5 + s_group_shift); bid < HBLOCKS_PER_TILE; bid += s_group_step)
			generateBlocks(bid);
		barrier();
		groupMemoryBarrier();
		UPDATE_CLOCK(1);

		uint bid = LIX >> 5;
		uint num_frags = s_hblock_counts[bid] >> 16;
		uint num_segments = (num_frags + SEGMENT_SIZE - 1) >> SEGMENT_SHIFT;

		ReductionContext context;
		initReduction(context);
		//initVisualizeSamples();

		for(uint segment_id = 0; segment_id < num_segments; segment_id++) {
			if((segment_id & 31) == 0)
				findSegments(bid, segment_id);
			uint cur_samples = min(SEGMENT_SIZE, num_frags - segment_id * SEGMENT_SIZE);
			loadSamples(bid, segment_id);
			UPDATE_CLOCK(2);
			//visualizeSamples(bid, cur_samples);
			shadeSamples(bid, cur_samples);
			UPDATE_CLOCK(3);
			reduceSamples(bid, cur_samples, context);
			UPDATE_CLOCK(4);

#ifdef ALPHA_THRESHOLD
			if(allInvocationsARB(context.out_trans < 1.0 / 255.0))
				break;
#endif
		}

		ivec2 pixel_pos = ivec2((LIX & 7) + ((bid & 1) << 3), ((LIX >> 3) & 3) + ((bid >> 1) << 2));
		finishReduction(pixel_pos, context);
		UPDATE_CLOCK(6);

		//finishVisualizeSamples(bid, pixel_pos);
		//visualizeHBlockTriangleCounts(bid, pixel_pos);
		//visualizeHBlockFragmentCounts(bid, pixel_pos);
	}
	barrier();
}

int loadNextBin() {
	if(LIX == 0) {
		uint bin_idx = atomicAdd(g_tiles.medium_bin_counter, 1);
		s_bin_id = bin_idx < s_num_bins ? g_bins.medium_bins[bin_idx] : -1;
		s_bin_raster_offset = s_bin_id << (BIN_SHIFT * 2);
	}
	barrier();
	return s_bin_id;
}

void main() {
	initTimers();
	if(LIX == 0)
		s_num_bins = g_bins.num_medium_bins;

	int bin_id = loadNextBin();
	while(bin_id != -1) {
		barrier();
		rasterBin(bin_id);
		bin_id = loadNextBin();
	}
	commitTimers();
}
