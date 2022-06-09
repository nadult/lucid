// $$include funcs lighting frustum viewport raster

#define LSIZE 256
#define LSHIFT 8

// TODO: add synthetic test: 256 planes one after another
// TODO: cleanup in the beginning (group definitions)

// NOTE: converting integer multiplications to shifts does not increase perf

#define NUM_WARPS (LSIZE / 32)

#define WARP_STEP 32
#define WARP_MASK 31

#define BUFFER_SIZE (LSIZE * 8)

#define MAX_BLOCK_ROW_TRIS 1024 // TODO: detect overflow
#define MAX_BLOCK_TRIS 256
#define MAX_BLOCK_TRIS_SHIFT 8

#define MAX_SCRATCH_TRIS 1024
#define MAX_SCRATCH_TRIS_SHIFT 10

#define SEGMENT_SIZE 256
#define SEGMENT_SHIFT 8

#define INVALID_SEGMENT 0xffff

#define MAX_SEGMENTS_SHIFT 5
#define MAX_SEGMENTS WARP_STEP

#undef BLOCK_SIZE
#undef BLOCK_SHIFT

// In this shader, we're using 8x8 blocks and 8x4 half blocks
#define BLOCK_SIZE 8
#define BLOCK_SHIFT 3

#define BLOCK_ROWS (BIN_SIZE / BLOCK_SIZE)
#define BLOCK_ROWS_SHIFT (BIN_SHIFT - BLOCK_SHIFT)
#define BLOCK_ROWS_MASK (BLOCK_ROWS - 1)

#define BIN_MASK (BIN_SIZE - 1)

layout(local_size_x = LSIZE) in;

// TODO: too much mem used
#define WORKGROUP_64_SCRATCH_SIZE (64 * 1024)
#define WORKGROUP_64_SCRATCH_SHIFT 16

// TODO: names
#define TRI_SCRATCH(var_idx) g_tri_storage[scratch_tri_idx * 8 + var_idx]
#define QUAD_SCRATCH(var_idx) g_quad_storage[scratch_quad_idx + var_idx * MAX_VISIBLE_QUADS]
#define QUAD_TEX_SCRATCH(var_idx)                                                                  \
	g_quad_storage[scratch_quad_idx * 2 + MAX_VISIBLE_QUADS * 2 + var_idx]
#define SCAN_SCRATCH(var_idx) g_scan_storage[scratch_tri_idx * 2 + var_idx]

uint scratch64BlockRowTrisOffset(uint by) {
	return (gl_WorkGroupID.x << WORKGROUP_64_SCRATCH_SHIFT) + by * (MAX_BLOCK_ROW_TRIS * 2);
}

uint scratch64BlockTrisOffset(uint bid) {
	return (gl_WorkGroupID.x << WORKGROUP_64_SCRATCH_SHIFT) + 32 * 1024 +
		   bid * (MAX_BLOCK_TRIS * 2);
}

uint scratch64HalfBlockTrisOffset(uint hbid) {
	return (gl_WorkGroupID.x << WORKGROUP_64_SCRATCH_SHIFT) + 48 * 1024 + hbid * MAX_BLOCK_TRIS;
}

shared int s_num_bins, s_bin_id, s_bin_raster_offset;
shared uint s_bin_quad_count, s_bin_quad_offset;
shared ivec2 s_bin_pos;
shared vec3 s_ray_dir0;

shared uint s_block_row_tri_count[BLOCK_ROWS];
shared uint s_block_tri_count[NUM_WARPS];
shared uint s_hblock_counts[NUM_WARPS * 2];

shared uint s_buffer[BUFFER_SIZE + 1];
shared uint s_mini_buffer[LSIZE];
shared uint s_segments[LSIZE * 2];

shared int s_raster_error;
shared int s_medium_bin_count;

// Only used when debugging
shared uint s_vis_pixels[LSIZE];

void outputPixel(ivec2 pixel_pos, uint color) {
	//color = tintColor(color, vec3(0.2, 0.3, 0.4), 0.8);
	g_raster_image[s_bin_raster_offset + pixel_pos.x + (pixel_pos.y << BIN_SHIFT)] = color;
}

void getTriangleParams(uint scratch_tri_idx, out vec3 depth_eq, out vec2 bary_params,
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

void getTriangleSecondaryParams(uint scratch_tri_idx, out uint unormal, out uint instance_color) {
	uvec2 val0 = TRI_SCRATCH(6);
	unormal = val0.x;
	instance_color = val0.y;
}

void getTriangleVertexColors(uint scratch_tri_idx, out vec4 color0, out vec4 color1,
							 out vec4 color2) {
	uint scratch_quad_idx = scratch_tri_idx >> 1;
	uint second_tri = scratch_tri_idx & 1;
	uvec4 colors = QUAD_SCRATCH(0);
	color0 = decodeRGBA8(colors[0]);
	color1 = decodeRGBA8(colors[1 + second_tri]);
	color2 = decodeRGBA8(colors[2 + second_tri]);
}

void getTriangleVertexNormals(uint scratch_tri_idx, out vec3 normal0, out vec3 normal1,
							  out vec3 normal2) {
	uint scratch_quad_idx = scratch_tri_idx >> 1;
	uint second_tri = scratch_tri_idx & 1;
	uvec4 normals = QUAD_SCRATCH(1);
	normal0 = decodeNormalUint(normals[0]);
	normal1 = decodeNormalUint(normals[1 + second_tri]);
	normal2 = decodeNormalUint(normals[2 + second_tri]);
}

void getTriangleVertexTexCoords(uint scratch_tri_idx, out vec2 tex0, out vec2 tex1, out vec2 tex2) {
	uint scratch_quad_idx = scratch_tri_idx >> 1;
	uint second_tri = scratch_tri_idx & 1;
	uvec4 tex_coords0 = QUAD_TEX_SCRATCH(0);
	uvec4 tex_coords1 = QUAD_TEX_SCRATCH(1);
	tex0 = uintBitsToFloat(tex_coords0.xy);
	tex1 = uintBitsToFloat(second_tri == 0 ? tex_coords0.zw : tex_coords1.xy);
	tex2 = uintBitsToFloat(second_tri == 0 ? tex_coords1.xy : tex_coords1.zw);
}

void loadScanlineParams(uint scratch_tri_idx, out vec3 scan_min, out vec3 scan_max,
						out vec3 scan_step, out uint y_aabb) {
	uvec4 val0 = SCAN_SCRATCH(0);
	uvec4 val1 = SCAN_SCRATCH(1);
	bool xsign0 = (val1.w & 1) == 1;
	bool xsign1 = (val1.w & 2) == 2;
	bool xsign2 = (val1.w & 4) == 4;
	vec3 scan = uintBitsToFloat(val0.xyz);
	scan_step = uintBitsToFloat(val1.xyz);
	y_aabb = val0.w;

	scan_min = vec3(xsign0 ? -1.0 / 0.0 : scan[0], xsign1 ? -1.0 / 0.0 : scan[1],
					xsign2 ? -1.0 / 0.0 : scan[2]);
	scan_max = vec3(xsign0 ? scan[0] : 1.0 / 0.0, xsign1 ? scan[1] : 1.0 / 0.0,
					xsign2 ? scan[2] : 1.0 / 0.0);
}

void generateRowTris(uint tri_idx) {
	uint dst_offset_64 = scratch64BlockRowTrisOffset(0);

	vec3 scan_min, scan_max, scan_step;
	uint y_aabb;
	loadScanlineParams(tri_idx, scan_min, scan_max, scan_step, y_aabb);
	int min_by = clamp(int(y_aabb & 0xffff) - s_bin_pos.y, 0, BIN_MASK) >> BLOCK_SHIFT;
	int max_by = clamp(int(y_aabb >> 16) - s_bin_pos.y, 0, BIN_MASK) >> BLOCK_SHIFT;

	// TODO: add start to loadScanline
	vec3 start_offset = scan_step * float(min_by * 8 + s_bin_pos.y) - vec3(s_bin_pos.x);
	scan_min += start_offset;
	scan_max += start_offset;

	// TODO: is it worth it to make this loop more work-efficient?
	for(int by = min_by; by <= max_by; by++) {
#define SCAN_STEP(id)                                                                              \
	int min##id = int(max(max(scan_min[0], scan_min[1]), max(scan_min[2], 0.0)));                  \
	int max##id = int(min(min(scan_max[0], scan_max[1]), min(scan_max[2], BIN_SIZE))) - 1;         \
	if(min##id > max##id)                                                                          \
		min##id = BIN_MASK, max##id = 0;                                                           \
	scan_min += scan_step, scan_max += scan_step;

		uint bx_mask;
		uvec2 min_bits, max_bits;

#define BX_MASK_ROW(id)                                                                            \
	((((1 << BLOCK_ROWS) - 1) << (min##id >> BLOCK_SHIFT)) &                                       \
	 (((1 << BLOCK_ROWS) - 1) >> (BLOCK_ROWS_MASK - (max##id >> BLOCK_SHIFT))))
		{
			SCAN_STEP(0);
			SCAN_STEP(1);
			SCAN_STEP(2);
			SCAN_STEP(3);

			bx_mask = BX_MASK_ROW(0) | BX_MASK_ROW(1) | BX_MASK_ROW(2) | BX_MASK_ROW(3);
			min_bits.x = (min0 << 0) | (min1 << 6) | (min2 << 12) | (min3 << 18);
			max_bits.x = (max0 << 0) | (max1 << 6) | (max2 << 12) | (max3 << 18);
		}
		{
			SCAN_STEP(0);
			SCAN_STEP(1);
			SCAN_STEP(2);
			SCAN_STEP(3);

			bx_mask |= BX_MASK_ROW(0) | BX_MASK_ROW(1) | BX_MASK_ROW(2) | BX_MASK_ROW(3);
			min_bits.y = (min0 << 0) | (min1 << 6) | (min2 << 12) | (min3 << 18);
			max_bits.y = (max0 << 0) | (max1 << 6) | (max2 << 12) | (max3 << 18);
		}

#undef BX_MASK_ROW
#undef SCAN_STEP

		if(bx_mask == 0)
			continue;

		uint roffset = atomicAdd(s_block_row_tri_count[by], 1) + by * (MAX_BLOCK_ROW_TRIS * 2);
		g_scratch_64[dst_offset_64 + roffset] =
			uvec2(min_bits.x | (bx_mask << 24), min_bits.y | ((tri_idx & 0xff0000) << 8));
		g_scratch_64[dst_offset_64 + roffset + MAX_BLOCK_ROW_TRIS] =
			uvec2(max_bits.x | ((tri_idx & 0xff) << 24), max_bits.y | ((tri_idx & 0xff00) << 16));
	}
}

void processQuads() {
	// TODO: optimization: in many cases all rows may very well fit in SMEM,
	// maybe it would be worth it not to use scratch at all then?
	// TODO: this loop is slooooow
	// TODO: divide big tris across different threads
	for(uint i = LIX >> 1; i < s_bin_quad_count; i += LSIZE / 2) {
		uint second_tri = LIX & 1;
		uint bin_quad_idx = g_bin_quads[s_bin_quad_offset + i];
		uint quad_idx = bin_quad_idx & 0xfffffff;
		uint cull_flag = (bin_quad_idx >> (30 + second_tri)) & 1;
		if(cull_flag == 1)
			continue;

#ifdef SHADER_DEBUG
		if(quad_idx >= MAX_VISIBLE_QUADS ||
		   (quad_idx >= g_info.num_visible_quads[0] &&
			quad_idx < (MAX_VISIBLE_QUADS - 1 - g_info.num_visible_quads[1])))
			atomicOr(s_raster_error, ~0);
#endif

		// TODO: scratch_tri_idx -> storage_
		uint scratch_tri_idx = quad_idx * 2 + second_tri;
		generateRowTris(scratch_tri_idx);
	}
}

shared uint s_sort_rcount[NUM_WARPS];

void prepareSortTris() {
	if(LIX < NUM_WARPS) {
		uint count = s_block_tri_count[LIX];
		// rcount: count rounded up to next power of 2, minimum: 32
		s_sort_rcount[LIX] = max(32, (count & (count - 1)) == 0 ? count : (2 << findMSB(count)));
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

void sortTris(uint lbid, uint count, uint buf_offset) {
	uint lid = LIX & WARP_MASK;
	uint rcount = s_sort_rcount[lbid];
	for(uint i = lid + count; i < rcount; i += WARP_STEP)
		s_buffer[buf_offset + i] = 0xffffffff;

#ifdef VENDOR_NVIDIA
	for(uint i = lid; i < rcount; i += WARP_STEP) {
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
	int start_k = 32, end_j = 32;
#else
	int start_k = 2, end_j = 1;
#endif
	for(uint k = start_k; k <= rcount; k = 2 * k) {
		for(uint j = k >> 1; j >= end_j; j = j >> 1) {
			for(uint i = lid; i < rcount; i += WARP_STEP * 2) {
				uint idx = (i & j) != 0 ? i + WARP_STEP - j : i;
				uint lvalue = s_buffer[buf_offset + idx];
				uint rvalue = s_buffer[buf_offset + idx + j];
				if(((idx & k) != 0) == (lvalue.x < rvalue.x)) {
					s_buffer[buf_offset + idx] = rvalue;
					s_buffer[buf_offset + idx + j] = lvalue;
				}
			}
		}
#ifdef VENDOR_NVIDIA
		for(uint i = lid; i < rcount; i += WARP_STEP) {
			uint bit = (i & k) == 0 ? 0 : 1;
			uint value = s_buffer[buf_offset + i];
			value = swap(value, 0x10, bit ^ bitExtract(lid, 4));
			value = swap(value, 0x08, bit ^ bitExtract(lid, 3));
			value = swap(value, 0x04, bit ^ bitExtract(lid, 2));
			value = swap(value, 0x02, bit ^ bitExtract(lid, 1));
			value = swap(value, 0x01, bit ^ bitExtract(lid, 0));
			s_buffer[buf_offset + i] = value;
		}
#endif
	}
}

// TODO: maybe process smaller amount of blocks at the same time?
// smaller chance that it will leave cache
void generateBlocks(uint bid) {
	int lbid = int(LIX >> 5);
	bid += lbid;
	uint by = bid >> BLOCK_ROWS_SHIFT, bx = bid & BLOCK_ROWS_MASK;

	uint src_offset_64 = scratch64BlockRowTrisOffset(by);
	uint tri_count = s_block_row_tri_count[by];
	uint buf_offset = lbid << MAX_BLOCK_TRIS_SHIFT;

	s_segments[LIX] = INVALID_SEGMENT;
	s_segments[LIX + LSIZE] = INVALID_SEGMENT;

	{
		uint bx_bits_shift = 24 + bx;
		uint block_tri_count = 0;
		uint thread_bit_mask = ~(0xffffffffu << (LIX & 31));

		for(uint i = LIX & WARP_MASK; i < tri_count; i += WARP_STEP) {
			uint bx_bit = (g_scratch_64[src_offset_64 + i].x >> bx_bits_shift) & 1;
			uint bit_mask = uint(ballotARB(bx_bit != 0));
			if(bit_mask == 0)
				continue;

			uint warp_offset = bitCount(bit_mask & thread_bit_mask);
			if(bx_bit != 0) {
				uint tri_offset =
					(block_tri_count + warp_offset) & ((1 << MAX_BLOCK_TRIS_SHIFT) - 1);
				if(tri_offset < MAX_BLOCK_TRIS)
					s_buffer[buf_offset + tri_offset] = i;
			}
			block_tri_count += bitCount(bit_mask);
		}

		if((LIX & 31) == 0) {
			s_block_tri_count[lbid] = block_tri_count;
			if(block_tri_count > MAX_BLOCK_TRIS)
				atomicOr(s_raster_error, 1 << lbid);
		}
		barrier();
		if(s_raster_error != 0)
			return;
	}
	prepareSortTris();

	uint dst_offset_64 = scratch64BlockTrisOffset(lbid);
	tri_count = s_block_tri_count[lbid];
	int startx = int(bx << 3);
	vec2 block_pos = vec2(s_bin_pos + ivec2(bx << 3, by << 3));

	for(uint i = LIX & WARP_MASK; i < tri_count; i += WARP_STEP) {
		uint row_idx = s_buffer[buf_offset + i];

		uvec2 tri_mins = g_scratch_64[src_offset_64 + row_idx];
		uvec2 tri_maxs = g_scratch_64[src_offset_64 + row_idx + MAX_BLOCK_ROW_TRIS];
		uint scratch_tri_idx =
			(tri_maxs.x >> 24) | ((tri_maxs.y >> 16) & 0xff00) | ((tri_mins.y >> 8) & 0xff0000);

		uvec2 bits;
		uvec2 num_frags;
		vec2 cpos = vec2(0);

#define PROCESS_4_ROWS(e)                                                                          \
	{                                                                                              \
		ivec4 xmin = max(((ivec4(tri_mins.e) >> ivec4(0, 6, 12, 18)) & BIN_MASK) - startx, 0);     \
		ivec4 xmax = min(((ivec4(tri_maxs.e) >> ivec4(0, 6, 12, 18)) & BIN_MASK) - startx, 7);     \
		ivec4 count = max(xmax - xmin + 1, 0);                                                     \
		vec4 cpx = vec4(xmin * 2 + count) * count;                                                 \
		vec4 cpy = vec4(1.0, 3.0, 5.0, 7.0) * count;                                               \
		cpos += vec2(cpx[0] + cpx[1] + cpx[2] + cpx[3], cpy[0] + cpy[1] + cpy[2] + cpy[3]);        \
		num_frags.e = count[0] + count[1] + count[2] + count[3];                                   \
		uint min_bits =                                                                            \
			(xmin[0] & 7) | ((xmin[1] & 7) << 3) | ((xmin[2] & 7) << 6) | ((xmin[3] & 7) << 9);    \
		uint count_bits =                                                                          \
			(count[0] << 16) | (count[1] << 20) | (count[2] << 24) | (count[3] << 28);             \
		bits.e = min_bits | count_bits;                                                            \
	}

		PROCESS_4_ROWS(x);
		PROCESS_4_ROWS(y);

		uint num_all_frags = num_frags.x + num_frags.y;
		cpos = cpos * (0.5 / float(num_all_frags)) + block_pos;

		vec2 val0 = uintBitsToFloat(TRI_SCRATCH(0));
		vec2 val1 = uintBitsToFloat(TRI_SCRATCH(1));
		vec3 depth_eq = vec3(val0.x, val0.y, val1.x);
		float ray_pos = depth_eq.x * cpos.x + (depth_eq.y * cpos.y + depth_eq.z);
		float depth = 0xffffe * SATURATE(inversesqrt(ray_pos + 1)); // 20 bits

		if(num_all_frags == 0) // This means that bx_mask is invalid
			RECORD(0, 0, 0, 0);
		uint frag_bits = num_frags.x | (num_frags.y << 6);
		// TODO: change order
		g_scratch_64[dst_offset_64 + i] = bits;
		g_scratch_64[dst_offset_64 + i + MAX_BLOCK_TRIS] = uvec2(scratch_tri_idx, frag_bits);

		// 12 bits for tile-tri index, 20 bits for depth
		s_buffer[buf_offset + i] = i | (uint(depth) << 12);
	}
	barrier();

	// For 3 tris and less depth filter is enough, there is no need to sort
	if(tri_count > 3)
		sortTris(lbid, tri_count, buf_offset);

	barrier();
	groupMemoryBarrier();

#ifdef SHADER_DEBUG
	// Making sure that tris are properly ordered
	if(tri_count > 3)
		for(uint i = LIX & WARP_MASK; i < tri_count; i += WARP_STEP) {
			uint value = s_buffer[buf_offset + i];
			uint prev_value = i == 0 ? 0 : s_buffer[buf_offset + i - 1];
			if(value <= prev_value)
				RECORD(i, tri_count, prev_value, value);
		}
#endif

#define PREFIX_SUM_STEP(value, step)                                                               \
	{                                                                                              \
		uint temp = shuffleUpNV(value, step, 32);                                                  \
		if((LIX & 31) >= step)                                                                     \
			value += temp;                                                                         \
	}

	for(uint i = LIX & WARP_MASK; i < tri_count; i += WARP_STEP) {
		uint idx = s_buffer[buf_offset + i] & 0xff;
		uint counts = g_scratch_64[dst_offset_64 + idx + MAX_BLOCK_TRIS].y;
		uint num_frags1 = counts & 63, num_frags2 = (counts >> 6) & 63;
		uint num_frags = num_frags1 | (num_frags2 << 12);

		// Computing triangle-ordered sample offsets within each block
		PREFIX_SUM_STEP(num_frags, 1);
		PREFIX_SUM_STEP(num_frags, 2);
		PREFIX_SUM_STEP(num_frags, 4);
		PREFIX_SUM_STEP(num_frags, 8);
		PREFIX_SUM_STEP(num_frags, 16);
		s_buffer[buf_offset + i] = num_frags | (idx << 24);
	}
	barrier();

	// Computing prefix sum across whole blocks (at most 8 * 32 elements)
	if(LIX < 8 * NUM_WARPS) {
		uint lbid = LIX >> 3, warp_idx = LIX & 7, warp_offset = warp_idx << 5;
		uint buf_offset = lbid << MAX_BLOCK_TRIS_SHIFT;
		uint tri_count = s_block_tri_count[lbid];
		uint value = 0;

		if(warp_offset < tri_count) {
			uint block_tri_idx = min(warp_offset + 31, tri_count - 1);
			value = s_buffer[buf_offset + block_tri_idx];
		}
		value = (value & 0xfff) | ((value & 0xfff000) << 4);
		uint sum = value, temp;
		temp = shuffleUpNV(sum, 1, 8), sum += warp_idx >= 1 ? temp : 0;
		temp = shuffleUpNV(sum, 2, 8), sum += warp_idx >= 2 ? temp : 0;
		temp = shuffleUpNV(sum, 4, 8), sum += warp_idx >= 4 ? temp : 0;

		if(warp_idx == 7) {
			s_hblock_counts[lbid * 2 + 0] = ((sum & 0xffff) << 16) | tri_count;
			s_hblock_counts[lbid * 2 + 1] = (sum & 0xffff0000) | tri_count;
		}

		s_mini_buffer[LIX] = sum - value;
	}
	barrier();
	if(s_raster_error != 0)
		return;

	// Storing triangle fragment offsets to scratch mem
	// Also finding first triangle for each segment
	src_offset_64 = dst_offset_64;
	dst_offset_64 = scratch64HalfBlockTrisOffset(lbid << 1);

	uint seg_block1_offset = lbid << (MAX_SEGMENTS_SHIFT + 1);
	uint seg_block2_offset = seg_block1_offset + MAX_SEGMENTS;
	for(uint i = LIX & WARP_MASK; i < tri_count; i += WARP_STEP) {
		uint tri_offset = 0;
		if(i > 0) {
			uint prev = i - 1;
			tri_offset = s_buffer[buf_offset + prev] & 0xffffff;
			tri_offset = (tri_offset & 0xfff) | ((tri_offset & 0xfff000) << 4);
			tri_offset += s_mini_buffer[(lbid << 3) + (prev >> 5)];
		}

		uint tri_value = s_buffer[buf_offset + i];
		uint block_tri_idx = tri_value >> 24;
		tri_value = (tri_value & 0xfff) | ((tri_value & 0xfff000) << 4);
		tri_value = (tri_value + s_mini_buffer[(lbid << 3) + (i >> 5)]) - tri_offset;

		uint tri_offset0 = tri_offset & 0xffff, tri_offset1 = tri_offset >> 16;
		uint tri_value0 = tri_value & 0xffff, tri_value1 = tri_value >> 16;

		uint seg_offset0 = tri_offset0 & (SEGMENT_SIZE - 1);
		uint seg_offset1 = tri_offset1 & (SEGMENT_SIZE - 1);
		uint seg_id0 = tri_offset0 >> SEGMENT_SHIFT;
		uint seg_id1 = tri_offset1 >> SEGMENT_SHIFT;
		bool first_seg0 = seg_offset0 == 0;
		bool first_seg1 = seg_offset1 == 0;
		if(seg_offset0 + tri_value0 > SEGMENT_SIZE)
			seg_id0++, first_seg0 = true;
		if(seg_offset1 + tri_value1 > SEGMENT_SIZE)
			seg_id1++, first_seg1 = true;
		seg_offset0 <<= 24;
		seg_offset1 <<= 24;

		if(first_seg0 && tri_value0 > 0)
			s_segments[seg_block1_offset + seg_id0] = i | seg_offset0;
		if(first_seg1 && tri_value1 > 0)
			s_segments[seg_block2_offset + seg_id1] = i | seg_offset1;

		uint scratch_tri_idx = g_scratch_64[src_offset_64 + block_tri_idx + MAX_BLOCK_TRIS].x;
		uvec2 tri_data = g_scratch_64[src_offset_64 + block_tri_idx];
		g_scratch_64[dst_offset_64 + i] = uvec2(scratch_tri_idx | seg_offset0, tri_data.x);
		g_scratch_64[dst_offset_64 + i + MAX_BLOCK_TRIS] =
			uvec2(scratch_tri_idx | seg_offset1, tri_data.y);
	}
}

void finalizeSegments() {
	uint hbid = LIX >> 4, seg_group_offset = hbid << MAX_SEGMENTS_SHIFT;
	uint counts = s_hblock_counts[hbid];
	uint num_segments = ((counts >> 16) + SEGMENT_SIZE - 1) >> SEGMENT_SHIFT;
	uint tri_count = counts & 0xffff;

	for(uint seg_id = LIX & 15; seg_id < num_segments; seg_id += 16) {
		uint cur_value = s_segments[seg_group_offset + seg_id];
		cur_value &= 0xffffff;
		uint next_value =
			seg_id + 1 < num_segments ? s_segments[seg_group_offset + seg_id + 1] : tri_count;
		bool next_tri_overlaps = next_value > 0xffffff;
		next_value &= 0xffffff;
		uint seg_tri_count = next_value - cur_value + (next_tri_overlaps ? 1 : 0);
		s_segments[seg_group_offset + seg_id] = cur_value | (seg_tri_count << 16);
	}
}

void loadSamples(uint hbid, int segment_id) {
	uint seg_group_offset = (hbid & (NUM_WARPS * 2 - 1)) << MAX_SEGMENTS_SHIFT;
	uint segment_data = s_segments[seg_group_offset + segment_id];
	uint tri_count = segment_data >> 16, first_tri = segment_data & 0xffff;
	uint src_offset_64 = scratch64HalfBlockTrisOffset(hbid & (NUM_WARPS * 2 - 1)) + first_tri;

	int y = int(LIX & 3);
	uint count_shift = 16 + (y << 2), min_shift = (y << 1) + y;
	int mask1 = y >= 1 ? ~0 : 0, mask2 = y >= 2 ? ~0 : 0;
	uint buf_offset = (LIX >> 5) << SEGMENT_SHIFT;

	// TODO: group differently for better memory accesses (and measure)
	for(uint i = (LIX & WARP_MASK) >> 2; i < tri_count; i += WARP_STEP / 4) {
		uvec2 tri_data = g_scratch_64[src_offset_64 + i];
		int tri_offset = int(tri_data.x >> 24);
		if(i == 0 && tri_offset != 0)
			tri_offset -= SEGMENT_SIZE;
		uint scratch_tri_idx = tri_data.x & 0xffffff;

		int minx = int((tri_data.y >> min_shift) & 7);
		int countx = int((tri_data.y >> count_shift) & 15);
		int prevx = countx + (shuffleUpNV(countx, 1, 4) & mask1);
		prevx += (shuffleUpNV(prevx, 2, 4) & mask2);
		tri_offset += prevx - countx;

		countx = min(countx, SEGMENT_SIZE - tri_offset);
		if(tri_offset < 0) {
			countx += tri_offset;
			minx -= tri_offset;
			tri_offset = 0;
		}
		if(countx <= 0)
			continue;

		uint pixel_id = (y << 3) | minx;
		uint value = pixel_id | (scratch_tri_idx << 5);

		for(int j = 0; j < countx; j++) {
			s_buffer[buf_offset + tri_offset] = value;
			tri_offset++;
			value++;
		}
	}
}

uint shadeSample(ivec2 pixel_pos, uint scratch_tri_idx, out float out_depth) {
	float px = float(pixel_pos.x), py = float(pixel_pos.y);

	vec3 depth_eq, edge0_eq, edge1_eq;
	uint instance_id, instance_flags;
	vec2 bary_params;
	getTriangleParams(scratch_tri_idx, depth_eq, bary_params, edge0_eq, edge1_eq, instance_id,
					  instance_flags);
	uint instance_color, unormal;
	getTriangleSecondaryParams(scratch_tri_idx, unormal, instance_color);

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
		getTriangleVertexTexCoords(scratch_tri_idx, tex0, tex1, tex2);

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
		getTriangleVertexColors(scratch_tri_idx, col0, col1, col2);
		color *= (1.0 - bary[0] - bary[1]) * col0 + (bary[0] * col1 + bary[1] * col2);
	}

	if(color.a == 0.0)
		return 0;

	vec3 normal;
	if((instance_flags & INST_HAS_VERTEX_NORMALS) != 0) {
		vec3 nrm0, nrm1, nrm2;
		getTriangleVertexNormals(scratch_tri_idx, nrm0, nrm1, nrm2);
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

void initReduceSamples(out ReductionContext ctx) {
#ifdef VISUALIZE_ERRORS
	ctx.prev_depths = vec4(999999999.0);
#else
	ctx.prev_depths = vec3(999999999.0);
#endif
	ctx.prev_colors = uvec3(0);
	ctx.out_color = 0;
	ctx.out_trans = 1.0;
}

void shadeAndReduceSamples(uint hbid, uint sample_count, in out ReductionContext ctx) {
	uint buf_offset = (LIX >> 5) << SEGMENT_SHIFT;
	uint bx = (hbid >> 1) & BLOCK_ROWS_MASK;
	uint hby = (hbid & 1) + ((hbid >> (BLOCK_ROWS_SHIFT + 1)) << 1);
	uint mini_offset = LIX & ~31;
	uint reduce_pixel_bit = 1u << (LIX & 31);
	ivec2 half_block_pos = ivec2(bx << 3, hby << 2) + s_bin_pos;
	vec3 out_color = decodeRGB10(ctx.out_color);

	for(uint i = 0; i < sample_count; i += WARP_STEP) {
		// TODO: we don't need s_mini_buffer here, we can use s_buffer, thus decreasing mini_buffer size
		s_mini_buffer[LIX] = 0;
		uvec2 sample_s;
		uint sample_id = i + (LIX & 31);
		if(sample_id < sample_count) {
			uint value = s_buffer[buf_offset + sample_id];
			uint sample_pixel_id = value & 31;
			uint scratch_tri_idx = value >> 5;
			ivec2 pix_pos = half_block_pos + ivec2(sample_pixel_id & 7, sample_pixel_id >> 3);
			float sample_depth;
			uint sample_color = shadeSample(pix_pos, scratch_tri_idx, sample_depth);
			sample_s = uvec2(sample_color, floatBitsToUint(sample_depth));
			atomicOr(s_mini_buffer[mini_offset + sample_pixel_id], reduce_pixel_bit);
		}

		uint bitmask = s_mini_buffer[LIX];
		int j = findLSB(bitmask);
		while(anyInvocationARB(j != -1)) {
			uvec2 value = shuffleNV(sample_s, j, 32);
			uint color = value.x;
			float depth = uintBitsToFloat(value.y);

			if(j == -1)
				continue;
			bitmask &= ~(1 << j);
			j = findLSB(bitmask);

			if(depth > ctx.prev_depths[0]) {
				SWAP_UINT(color, ctx.prev_colors[0]);
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
			ctx.prev_colors[0] = color;
		}
	}

	// TODO: check if encode+decode for out_color is really needed (to save 2 regs)
	ctx.out_color = encodeRGB10(SATURATE(out_color));
}

void finishReduceSamples(ivec2 pixel_pos, ReductionContext ctx) {
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

void visualizeSamples(uint sample_count) {
	uint buf_offset = (LIX >> 5) << SEGMENT_SHIFT;
	for(uint i = LIX & 31; i < sample_count; i += 32) {
		uint pixel_id = s_buffer[buf_offset + i] & 31;
		atomicAdd(s_vis_pixels[(LIX & ~31) + pixel_id], 1);
	}
}

void finishVisualizeSamples(ivec2 pixel_pos) {
	uint pixel_id = (pixel_pos.x & 7) + ((pixel_pos.y & 3) << 3);
	vec3 color = vec3(s_vis_pixels[(LIX & ~31) + pixel_id]) / 32.0;
	uint enc_col = encodeRGBA8(vec4(SATURATE(color), 1.0));
	outputPixel(pixel_pos, enc_col);
}

void visualizeFragmentCounts(uint hbid, ivec2 pixel_pos) {
	uint count = s_hblock_counts[hbid & (NUM_WARPS * 2 - 1)] >> 16;
	vec4 color = vec4(SATURATE(vec3(count) / 2048.0), 1.0);
	outputPixel(pixel_pos, encodeRGBA8(color));
}

void visualizeTriangleCounts(uint hbid, ivec2 pixel_pos) {
	uint count = s_block_tri_count[(hbid >> 1) & (NUM_WARPS - 1)];
	//count = s_block_row_tri_count[hbid >> (BLOCK_ROW_SHIFT + 1)];
	//count = s_bin_quad_count;

	vec3 color;
	if(count == 0)
		color = vec3(0);
	else if(count < 16)
		color = vec3(count + 16, 0, 0) / 32.0;
	else if(count < 64)
		color = vec3(count + 32, count + 32, 0) / 96.0;
	else
		color = vec3(0, count, 0.0) / 256.0;
	outputPixel(pixel_pos, encodeRGBA8(vec4(SATURATE(color), 1.0)));
}

void visualizeErrors(uint bid) {
	uint lbid = LIX >> 5;
	bid += lbid;

	uint color = 0xff000031;
	if((s_raster_error & (1 << lbid)) != 0)
		color += 0x64;

	uint bx = bid & 7, by = bid >> 3;
	ivec2 pixel_pos = ivec2((LIX & 7) + (bx << 3), ((LIX >> 3) & 3) + (by << 3));
	outputPixel(pixel_pos, color);
	outputPixel(pixel_pos + ivec2(0, 4), color);
}

void rasterBin(int bin_id) {
	INIT_CLOCK();

	const int num_blocks = (BIN_SIZE / BLOCK_SIZE) * (BIN_SIZE / BLOCK_SIZE);

	if(LIX < BLOCK_ROWS) {
		if(LIX == 0) {
			// TODO: optimize
			ivec2 bin_pos = ivec2(bin_id % BIN_COUNT_X, bin_id / BIN_COUNT_X) * BIN_SIZE;
			s_bin_pos = bin_pos;
			s_bin_quad_count = BIN_QUAD_COUNTS(bin_id);
			s_bin_quad_offset = BIN_QUAD_OFFSETS(bin_id);
			s_raster_error = 0;
		}

		s_block_row_tri_count[LIX] = 0;
	}
	barrier();
	processQuads();
	groupMemoryBarrier();
	barrier();
	UPDATE_CLOCK(0);

#ifdef SHADER_DEBUG
	if(s_raster_error != 0) {
		for(uint bid = LIX >> 6; bid < num_blocks; bid += NUM_WARPS / 2)
			visualizeErrors(bid);
		return;
	}
#endif

	//  bid: block (8x8) id; We have 8 x 8 = 64 blocks
	// hbid: half block (8x4) id; We have 8 x 16 = 128 half blocks
	//       half blocks which make a single block are stored one after another
	//       (upper block first)
	// lbid: local block id (range: 0 up to NUM_WARPS - 1)
	// Each block has 64 pixels, so we need 2 warps to process all pixels within a single block
	for(uint hbid = LIX >> 5; hbid < num_blocks * 2; hbid += NUM_WARPS) {
		barrier();
		if((hbid & NUM_WARPS) == 0) {
			uint bid = (hbid & ~(NUM_WARPS - 1)) >> 1;
			generateBlocks(bid);
			barrier();
			finalizeSegments();
			barrier();
			groupMemoryBarrier();

			if(s_raster_error != 0) {
				if(LIX == 0) {
					int id = atomicAdd(g_info.bin_level_counts[BIN_LEVEL_MEDIUM], 1);
					MEDIUM_LEVEL_BINS(id) = int(bin_id);
					s_medium_bin_count = max(s_medium_bin_count, id + 1);
					return;
				}
				/*visualizeErrors(bid);
				hbid += NUM_WARPS;
				barrier();
				if(LIX == 0)
					s_raster_error = 0;
				continue;*/
			}
		}
		UPDATE_CLOCK(1);

		ReductionContext context;
		initReduceSamples(context);
		//initVisualizeSamples();

		for(int segment_id = 0;; segment_id++) {
			uint counts = s_hblock_counts[hbid & (NUM_WARPS * 2 - 1)];
			int frag_count = min(SEGMENT_SIZE, int(counts >> 16) - (segment_id << SEGMENT_SHIFT));
			if(frag_count <= 0)
				break;

			loadSamples(hbid, segment_id);
			UPDATE_CLOCK(2);

			shadeAndReduceSamples(hbid, frag_count, context);
			UPDATE_CLOCK(5);

#ifdef ALPHA_THRESHOLD
			if(allInvocationsARB(context.out_trans < 1.0 / 255.0))
				break;
#endif
		}

		uint bx = (hbid >> 1) & BLOCK_ROWS_MASK;
		uint hby = (hbid & 1) + ((hbid >> (BLOCK_ROWS_SHIFT + 1)) << 1);
		ivec2 pixel_pos = ivec2((LIX & 7) + (bx << BLOCK_SHIFT), ((LIX >> 3) & 3) + (hby << 2));
		finishReduceSamples(pixel_pos, context);

		//finishVisualizeSamples(pixel_pos);
		//visualizeFragmentCounts(hbid, pixel_pos);
		//visualizeTriangleCounts(hbid, pixel_pos);
		UPDATE_CLOCK(6);
		barrier();
	}
}

int loadNextBin() {
	if(LIX == 0) {
		uint bin_idx = atomicAdd(g_info.a_small_bins, 1);
		s_bin_id = bin_idx < s_num_bins ? LOW_LEVEL_BINS(bin_idx) : -1;
		s_bin_raster_offset = s_bin_id << (BIN_SHIFT * 2);
	}
	barrier();
	return s_bin_id;
}

void main() {
	initTimers();
	if(LIX == 0) {
		s_num_bins = g_info.bin_level_counts[BIN_LEVEL_LOW];
		s_medium_bin_count = 0;
		s_ray_dir0 = frustum.ws_dir0 + (frustum.ws_dirx + frustum.ws_diry) * 0.5;
	}

	int bin_id = loadNextBin();
	while(bin_id != -1) {
		barrier();
		rasterBin(bin_id);
		bin_id = loadNextBin();
	}

	// If some of the bins are promoted to the next level, we have to adjust number of dispatches
	if(LIX == 0 && s_medium_bin_count > 0) {
		uint num_dispatches = min(s_medium_bin_count, MAX_DISPATCHES);
		atomicMax(g_info.bin_level_dispatches[BIN_LEVEL_MEDIUM][0], num_dispatches);
	}
	commitTimers();
}
