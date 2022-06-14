// $$include funcs lighting frustum viewport shading scanline timers

#if BIN_SIZE == 64
#define LSIZE 512
#define LSHIFT 9
#else
#define LSIZE 1024
#define LSHIFT 10
#endif

#define NUM_WARPS (LSIZE / 32)

#define WARP_STEP 32
#define WARP_MASK 31

#define BUFFER_SIZE (LSIZE * 8)

// Basic maximum value of tris per hblock
#define MAX_HBLOCK_TRIS0 256
#define MAX_HBLOCK_TRIS0_SHIFT 8

// Actual max value of tris per hblock,
// assuming using multiple warps per hblock in generateHBlocks
#define MAX_HBLOCK_TRIS 2048
#define MAX_HBLOCK_TRIS_SHIFT 11

#define MAX_GROUP_SIZE 8

#define MAX_HBLOCK_ROW_TRIS 8192
#define MAX_HBLOCK_ROW_TRIS_SHIFT 13

#define SEGMENT_SIZE 256
#define SEGMENT_SHIFT 8
#define INVALID_SEGMENT 0xffff

#define MAX_SEGMENTS_SHIFT 6
#define MAX_SEGMENTS 64

#define HBLOCK_WIDTH 8
#define HBLOCK_HEIGHT 4

#define HBLOCK_WIDTH_SHIFT 3
#define HBLOCK_HEIGHT_SHIFT 2

#define HBLOCK_ROWS (BIN_SIZE / HBLOCK_HEIGHT)
#define HBLOCK_ROWS_SHIFT (BIN_SHIFT - HBLOCK_HEIGHT_SHIFT)
#define HBLOCK_ROWS_MASK (HBLOCK_ROWS - 1)

#define HBLOCK_COLS (BIN_SIZE / HBLOCK_WIDTH)
#define HBLOCK_COLS_SHIFT (BIN_SHIFT - HBLOCK_WIDTH_SHIFT)
#define HBLOCK_COLS_MASK (HBLOCK_COLS - 1)

// Number of rows which can be processed with given amount of warps
#define HBLOCK_ROWS_STEP (NUM_WARPS / HBLOCK_COLS)
#define HBLOCK_ROWS_STEP_MASK (HBLOCK_ROWS_STEP - 1)

#define BIN_MASK (BIN_SIZE - 1)

layout(local_size_x = LSIZE) in;

#define WORKGROUP_32_SCRATCH_SIZE (64 * 1024)
#define WORKGROUP_32_SCRATCH_SHIFT 16

#define WORKGROUP_64_SCRATCH_SIZE (128 * 1024)
#define WORKGROUP_64_SCRATCH_SHIFT 17

uint currentHBlockRow(uint hby) {
#if BIN_SIZE == 64
	return hby & HBLOCK_ROWS_STEP_MASK;
#else
	return hby;
#endif
}

uint scratch32HBlockRowTrisOffset(uint hby) {
	return (gl_WorkGroupID.x << WORKGROUP_32_SCRATCH_SHIFT) +
		   currentHBlockRow(hby) * MAX_HBLOCK_ROW_TRIS;
}

uint scratch64HBlockRowTrisOffset(uint hby) {
	return (gl_WorkGroupID.x << WORKGROUP_64_SCRATCH_SHIFT) +
		   currentHBlockRow(hby) * MAX_HBLOCK_ROW_TRIS;
}

uint scratch64HBlockTrisOffset(uint lhbid) {
	return (gl_WorkGroupID.x << WORKGROUP_64_SCRATCH_SHIFT) + 64 * 1024 + lhbid * MAX_HBLOCK_TRIS;
}

// Once we've generated HBlock-tris we no longer need current HBlock-row-tris, so sorted HBlocks
// can overlap hblock-row-tris memory. TODO: make it work for 64x64
uint scratch64SortedHBlockTrisOffset(uint lhbid) {
	return (gl_WorkGroupID.x << WORKGROUP_64_SCRATCH_SHIFT) + lhbid * MAX_HBLOCK_TRIS;
}

shared int s_num_scratch_tris;
shared int s_num_bins, s_bin_id, s_bin_raster_offset;
shared uint s_bin_quad_count, s_bin_quad_offset;
shared uint s_bin_tri_count, s_bin_tri_offset;
shared ivec2 s_bin_pos;

shared uint s_hblock_row_tri_counts[HBLOCK_ROWS];
shared int s_hblock_tri_counts[NUM_WARPS];
shared uint s_hblock_frag_counts[NUM_WARPS];

shared uint s_hblock_max_tri_counts;

// How many warps do we need to process single half-block in generateHBlocks ?
// Acceptable values: 1, 2, 4, 8; More: trouble :(
shared uint s_hblock_group_size;
shared uint s_hblock_group_shift;
shared uint s_max_sort_rcount;

shared uint s_buffer[BUFFER_SIZE + 1];
shared uint s_mini_buffer[LSIZE];
shared uint s_segments[LSIZE * 2];
shared int s_raster_error;

// Only used when debugging
shared uint s_vis_pixels[LSIZE];

void outputPixel(ivec2 pixel_pos, uint color) {
	//color = tintColor(color, vec3(0.2, 0.3, 0.4), 0.8);
	g_raster_image[s_bin_raster_offset + pixel_pos.x + (pixel_pos.y << BIN_SHIFT)] = color;
}

uint max32(uint value, int width) {
	if(width >= 2)
		value = max(value, shuffleXorNV(value, 1, 32));
	if(width >= 4)
		value = max(value, shuffleXorNV(value, 2, 32));
	if(width >= 8)
		value = max(value, shuffleXorNV(value, 4, 32));
	if(width >= 16)
		value = max(value, shuffleXorNV(value, 8, 32));
	if(width >= 32)
		value = max(value, shuffleXorNV(value, 16, 32));
	return value;
}

void generateRowTris(uint tri_idx, int start_hby) {
	uint scan_offset = STORAGE_TRI_SCAN_OFFSET + tri_idx * 2;
	uvec4 val0 = g_uvec4_storage[scan_offset + 0];
	uvec4 val1 = g_uvec4_storage[scan_offset + 1];
	int min_hby = clamp(int(val0.w & 0xffff) - s_bin_pos.y, 0, BIN_MASK) >> HBLOCK_HEIGHT_SHIFT;
	int max_hby = clamp(int(val0.w >> 16) - s_bin_pos.y, 0, BIN_MASK) >> HBLOCK_HEIGHT_SHIFT;

	vec2 start = vec2(s_bin_pos.x, s_bin_pos.y + min_hby * 4);
	ScanlineParams scan = loadScanlineParamsRow(val0, val1, start);

#if HBLOCK_ROWS_STEP < HBLOCK_ROWS
	min_hby = max(start_hby, min_hby);
	max_hby = min(start_hby + HBLOCK_ROWS_STEP_MASK, max_hby);
	if(max_hby < min_hby)
		return;
#endif

	uint dst_offset_32 = scratch32HBlockRowTrisOffset(0);
	uint dst_offset_64 = scratch64HBlockRowTrisOffset(0);

	// TODO: is it worth it to make this loop more work-efficient?
	for(int hby = min_hby; hby <= max_hby; hby++) {
#define SCAN_STEP(id)                                                                              \
	int min##id = int(max(max(scan.min[0], scan.min[1]), max(scan.min[2], 0.0)));                  \
	int max##id = int(min(min(scan.max[0], scan.max[1]), min(scan.max[2], BIN_SIZE))) - 1;         \
	if(min##id > max##id)                                                                          \
		min##id = BIN_MASK, max##id = 0;                                                           \
	scan.min += scan.step, scan.max += scan.step;

#define BX_MASK_ROW(id)                                                                            \
	((((1 << HBLOCK_COLS) - 1) << (min##id >> HBLOCK_WIDTH_SHIFT)) &                               \
	 (((1 << HBLOCK_COLS) - 1) >> (HBLOCK_COLS_MASK - (max##id >> HBLOCK_WIDTH_SHIFT))))
		SCAN_STEP(0);
		SCAN_STEP(1);
		SCAN_STEP(2);
		SCAN_STEP(3);

		const int shift1 = BIN_SHIFT * 1, shift2 = BIN_SHIFT * 2, shift3 = BIN_SHIFT * 3;
		uint bx_mask = BX_MASK_ROW(0) | BX_MASK_ROW(1) | BX_MASK_ROW(2) | BX_MASK_ROW(3);
		uint min_bits = (min0 << 0) | (min1 << shift1) | (min2 << shift2) | (min3 << shift3);
		uint max_bits = (max0 << 0) | (max1 << shift1) | (max2 << shift2) | (max3 << shift3);

#undef BX_MASK_ROW
#undef SCAN_STEP

		if(bx_mask == 0)
			continue;

		uint hbid_row = currentHBlockRow(hby) << HBLOCK_COLS_SHIFT;
		uint min_hbid = findLSB(bx_mask), max_hbid = findMSB(bx_mask) + 1;
		// Accumulation is done at the end of processBlocks
		atomicAdd(s_hblock_tri_counts[hbid_row + min_hbid], 1);
		if(max_hbid < HBLOCK_COLS)
			atomicAdd(s_hblock_tri_counts[hbid_row + max_hbid], -1);

		uint row_tri_idx = atomicAdd(s_hblock_row_tri_counts[hby], 1);
		if(row_tri_idx >= MAX_HBLOCK_ROW_TRIS) {
			atomicOr(s_raster_error, 0xffffffff);
			return;
		}
		uint roffset = row_tri_idx + currentHBlockRow(hby) * MAX_HBLOCK_ROW_TRIS;
		g_scratch_32[dst_offset_32 + roffset] = tri_idx | (bx_mask << 24);
		g_scratch_64[dst_offset_64 + roffset] = uvec2(min_bits, max_bits);
	}
}

void processQuads(int start_hby) {
	if(LIX == 0)
		s_num_scratch_tris = 0;
	if(LIX < NUM_WARPS)
		s_hblock_tri_counts[LIX] = 0;
	barrier();

	// TODO: this loop is slooooow
	// TODO: divide big tris across different threads
	for(uint i = LIX >> 1; i < s_bin_quad_count; i += LSIZE / 2) {
		uint second_tri = LIX & 1;
		uint bin_quad_idx = g_bin_quads[s_bin_quad_offset + i];
		uint quad_idx = bin_quad_idx & 0xfffffff;
		uint cull_flag = (bin_quad_idx >> (30 + second_tri)) & 1;
		if(cull_flag == 1)
			continue;

		// TODO: store only if samples were generated
		// TODO: do triangle storing later
		uint tri_idx = quad_idx * 2 + second_tri;
		generateRowTris(tri_idx, start_hby);
	}
	for(uint i = LIX; i < s_bin_tri_count; i += LSIZE)
		generateRowTris(g_bin_tris[s_bin_tri_offset + i], start_hby);
	barrier();

	// Accumulating per hblock-counts for each hblock-row
	// Note: these are only estimates; very good estimates, but in some cases a single
	// triangle can have wide holes between pixels (because middle pixels don't hit pixel centers)
	if(LIX < NUM_WARPS) {
		uint hbx = LIX & HBLOCK_COLS_MASK;
		int value = s_hblock_tri_counts[LIX], temp;
		temp = shuffleUpNV(value, 1, HBLOCK_COLS), value += hbx >= 1 ? temp : 0;
		if(HBLOCK_COLS >= 4)
			temp = shuffleUpNV(value, 2, HBLOCK_COLS), value += hbx >= 2 ? temp : 0;
		if(HBLOCK_COLS >= 8)
			temp = shuffleUpNV(value, 4, HBLOCK_COLS), value += hbx >= 4 ? temp : 0;
		s_hblock_tri_counts[LIX] = value;
		uint max_value = max32(uint(value), NUM_WARPS);
		if(LIX == 0) {
			s_hblock_max_tri_counts = max_value;
			// rcount: count rounded up to next power of 2, minimum: 32
			uint rcount =
				(max_value & (max_value - 1)) == 0 ? max_value : (2 << findMSB(max_value));
			s_max_sort_rcount = max(32, rcount);

			//uint group_shift = max(int(log2(max_value)) - MAX_HBLOCK_TRIS0_SHIFT, 0);
			uint group_shift = max_value <= MAX_HBLOCK_TRIS0	 ? 0 :
							   max_value <= MAX_HBLOCK_TRIS0 * 2 ? 1 :
							   max_value <= MAX_HBLOCK_TRIS0 * 4 ? 2 :
							   max_value <= MAX_HBLOCK_TRIS0 * 8 ? 3 :
																	 4;
			uint group_size = 1 << group_shift;
			s_hblock_group_shift = group_shift;
			s_hblock_group_size = 1 << group_shift;
			if(group_size > MAX_GROUP_SIZE)
				s_raster_error = 0xffffffff; // TODO: better reporting
		}
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

void sortTris(uint lhbid, uint count, uint buf_offset, uint group_size, uint lid) {
	uint rcount = s_max_sort_rcount;
	for(uint i = lid + count; i < rcount; i += group_size)
		s_buffer[buf_offset + i] = 0xffffffff;

#ifdef VENDOR_NVIDIA
	for(uint i = lid; i < rcount; i += group_size) {
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
	barrier();

	for(uint k = start_k; k <= rcount; k = 2 * k) {
		for(uint j = k >> 1; j >= end_j; j = j >> 1) {
			for(uint i = lid; i < rcount; i += group_size * 2) {
				uint idx = (i & j) != 0 ? i + group_size - j : i;
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
		for(uint i = lid; i < rcount; i += group_size) {
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

// TODO: maybe process smaller amount of blocks at the same time?
// smaller chance that it will leave cache
void generateHBlocks(uint start_hbid) {
	uint group_size = s_hblock_group_size * 32;
	uint group_shift = s_hblock_group_shift;
	uint group_mask = group_size - 1;
	uint group_thread = LIX & group_mask;

	// TODO: better names for indices
	uint group_hbid = LIX >> (5 + group_shift);
	uint hbid = start_hbid + group_hbid;
	uint hby = hbid >> HBLOCK_COLS_SHIFT, hbx = hbid & HBLOCK_COLS_MASK;
	uint lhbid = hbid & (NUM_WARPS - 1);
	uint tri_count = s_hblock_row_tri_counts[hby];
	uint buf_offset = group_hbid << (MAX_HBLOCK_TRIS0_SHIFT + group_shift);

	uint src_offset_32 = scratch32HBlockRowTrisOffset(hby);
	uint src_offset_64 = scratch64HBlockRowTrisOffset(hby);

	{
		uint bx_bits_shift = 24 + hbx;
		uint thread_bit_mask = ~(0xffffffffu << (LIX & 31));
		uint block_tri_count = 0;

		if(group_thread < WARP_SIZE) {
			for(uint i = group_thread; i < tri_count; i += WARP_STEP) {
				uint bx_bit = (g_scratch_32[src_offset_32 + i] >> bx_bits_shift) & 1;
				uint bit_mask = uint(ballotARB(bx_bit != 0));
				if(bit_mask == 0)
					continue;

				uint warp_offset = bitCount(bit_mask & thread_bit_mask);
				if(bx_bit != 0) {
					uint tri_offset = block_tri_count + warp_offset;
					s_buffer[buf_offset + tri_offset] = i;
				}
				block_tri_count += bitCount(bit_mask);
			}
			if(group_thread == 0) {
				if(s_hblock_tri_counts[lhbid] < int(block_tri_count))
					RECORD(hbid, s_hblock_tri_counts[lhbid], block_tri_count, 0);
				s_hblock_tri_counts[lhbid] = int(block_tri_count);
			}
		}
		barrier();
	}

	uint dst_offset_64 = scratch64HBlockTrisOffset(lhbid);
	tri_count = s_hblock_tri_counts[lhbid];
	int startx = int(hbx << 3);
	vec2 block_pos = vec2(hbx << HBLOCK_WIDTH_SHIFT, hby << HBLOCK_HEIGHT_SHIFT) + vec2(s_bin_pos);

	for(uint i = group_thread; i < tri_count; i += group_size) {
		uint row_tri_idx = s_buffer[buf_offset + i];
		uint tri_idx = g_scratch_32[src_offset_32 + row_tri_idx] & 0xffffff;
		uvec2 tri_info = g_scratch_64[src_offset_64 + row_tri_idx];

		const ivec4 bin_shifts = ivec4(0, BIN_SHIFT, BIN_SHIFT * 2, BIN_SHIFT * 3);
		ivec4 xmin = max(((ivec4(tri_info.x) >> bin_shifts) & BIN_MASK) - startx, 0);
		ivec4 xmax = min(((ivec4(tri_info.y) >> bin_shifts) & BIN_MASK) - startx, 7);
		ivec4 count = max(xmax - xmin + 1, 0);
		vec4 cpx = vec4(xmin * 2 + count) * count;
		vec4 cpy = vec4(1.0, 3.0, 5.0, 7.0) * count;
		vec2 cpos = vec2(cpx[0] + cpx[1] + cpx[2] + cpx[3], cpy[0] + cpy[1] + cpy[2] + cpy[3]);
		uint num_frags = count[0] + count[1] + count[2] + count[3];
		uint min_bits =
			(xmin[0] & 7) | ((xmin[1] & 7) << 3) | ((xmin[2] & 7) << 6) | ((xmin[3] & 7) << 9);
		uint count_bits = (count[0] << 16) | (count[1] << 20) | (count[2] << 24) | (count[3] << 28);
		uint bits = min_bits | count_bits;

		cpos *= 0.5 / float(num_frags);
		cpos += block_pos;

		uint depth_offset = STORAGE_TRI_DEPTH_OFFSET + tri_idx;
		vec3 depth_eq = uintBitsToFloat(g_uvec4_storage[depth_offset].xyz);
		float ray_pos = depth_eq.x * cpos.x + (depth_eq.y * cpos.y + depth_eq.z);
		float depth = 0xffffe * SATURATE(inversesqrt(ray_pos + 1)); // 20 bits

		if(num_frags == 0) // This means that bx_mask is invalid
			RECORD(0, 0, 0, 0);

		g_scratch_64[dst_offset_64 + i] = uvec2(bits, tri_idx | (num_frags << 24));
		s_buffer[buf_offset + i] = i | (uint(depth) << 12);
	}
	barrier();

	sortTris(lhbid, tri_count, buf_offset, group_size, group_thread);

	barrier();
	groupMemoryBarrier();

#ifdef SHADER_DEBUG
	for(uint i = LIX & WARP_MASK; i < tri_count; i += WARP_STEP) {
		uint value = s_buffer[buf_offset + i];
		uint prev_value = i == 0 ? 0 : s_buffer[buf_offset + i - 1];
		if(value <= prev_value)
			RECORD(i, tri_count, prev_value, value);
	}
#endif

	for(uint i = group_thread; i < tri_count; i += group_size) {
		uint hblock_tri_idx = s_buffer[buf_offset + i] & 0xfff;
		uint num_frags = g_scratch_64[dst_offset_64 + hblock_tri_idx].y >> 24;

		// Computing triangle-ordered sample offsets within each block
		uint temp;
		temp = shuffleUpNV(num_frags, 1, 32), num_frags += (LIX & 31) >= 1 ? temp : 0;
		temp = shuffleUpNV(num_frags, 2, 32), num_frags += (LIX & 31) >= 2 ? temp : 0;
		temp = shuffleUpNV(num_frags, 4, 32), num_frags += (LIX & 31) >= 4 ? temp : 0;
		temp = shuffleUpNV(num_frags, 8, 32), num_frags += (LIX & 31) >= 8 ? temp : 0;
		temp = shuffleUpNV(num_frags, 16, 32), num_frags += (LIX & 31) >= 16 ? temp : 0;
		s_buffer[buf_offset + i] = num_frags | (hblock_tri_idx << 16);
	}
	barrier();

	// Computing prefix sum across whole half-blocks
	// wgsize in this context is 2x smaller than in general, because here,
	// we're processing 2 elements at a time.
	// Can handle at most wgsize * 32 elements (from 256 to 2048)
	if(LIX < 4 * NUM_WARPS) {
		uint wgsize = 4 << group_shift, wgmask = wgsize - 1;
		uint group_hbid = LIX >> (2 + group_shift), group_sub_idx = LIX & wgmask;
		uint buf_offset = group_hbid << (MAX_HBLOCK_TRIS0_SHIFT + group_shift);
		uint lhbid = (start_hbid + group_hbid) & (NUM_WARPS - 1);
		uint tri_count = s_hblock_tri_counts[lhbid];
		uint warp_offset = group_sub_idx << 6;

		uint value0 = 0;
		if(warp_offset < tri_count) {
			uint hblock_tri_idx = min(warp_offset + 31, tri_count - 1);
			value0 = s_buffer[buf_offset + hblock_tri_idx] & 0xffff;
		}
		uint value1 = 0;
		warp_offset += 32;
		if(warp_offset < tri_count) {
			uint hblock_tri_idx = min(warp_offset + 31, tri_count - 1);
			value1 = s_buffer[buf_offset + hblock_tri_idx] & 0xffff;
		}

		uint value = value0 + value1;
		uint sum = value, temp;
		temp = shuffleUpNV(sum, 1, wgsize), sum += group_sub_idx >= 1 ? temp : 0;
		temp = shuffleUpNV(sum, 2, wgsize), sum += group_sub_idx >= 2 ? temp : 0;
		if(wgsize >= 8)
			temp = shuffleUpNV(sum, 4, wgsize), sum += group_sub_idx >= 4 ? temp : 0;
		if(wgsize >= 16)
			temp = shuffleUpNV(sum, 8, wgsize), sum += group_sub_idx >= 8 ? temp : 0;
		if(wgsize >= 32)
			temp = shuffleUpNV(sum, 16, wgsize), sum += group_sub_idx >= 16 ? temp : 0;

		// TODO: report error if frags can't fit into segments
		if(group_sub_idx == wgmask)
			s_hblock_frag_counts[lhbid] = sum;
		s_mini_buffer[LIX * 2 + 0] = sum - value;
		s_mini_buffer[LIX * 2 + 1] = sum - value1;
	}
	barrier();
	if(s_raster_error != 0)
		return;

	// Storing triangle fragment offsets to scratch mem
	// Also finding first triangle for each segment
	src_offset_64 = dst_offset_64;
	dst_offset_64 = scratch64SortedHBlockTrisOffset(lhbid);
	uint seg_block_offset = lhbid << MAX_SEGMENTS_SHIFT;
	for(uint i = group_thread; i < tri_count; i += group_size) {
		uint tri_offset = 0;
		uint mini_offset = group_hbid << (3 + group_shift);
		if(i > 0) {
			uint prev = i - 1;
			tri_offset = s_buffer[buf_offset + prev] & 0xffff;
			tri_offset += s_mini_buffer[mini_offset + (prev >> 5)];
		}

		uint tri_value = s_buffer[buf_offset + i] + s_mini_buffer[mini_offset + (i >> 5)];
		uint hblock_tri_idx = tri_value >> 16;
		tri_value = (tri_value & 0xffff) - tri_offset;

		uint seg_offset = tri_offset & (SEGMENT_SIZE - 1);
		uint seg_id = tri_offset >> SEGMENT_SHIFT;
		bool first_seg = seg_offset == 0;
		if(seg_offset + tri_value > SEGMENT_SIZE)
			seg_id++, first_seg = true;
		if(seg_id >= MAX_SEGMENTS) {
			atomicOr(s_raster_error, 1 << lhbid);
			break;
		}
		seg_offset <<= 24;
		if(first_seg && tri_value > 0)
			s_segments[seg_block_offset + seg_id] = i | seg_offset;

		uvec2 tri_data = g_scratch_64[src_offset_64 + hblock_tri_idx];
		uint tri_idx = tri_data.y & 0xffffff;
		g_scratch_64[dst_offset_64 + i] = uvec2(tri_idx | seg_offset, tri_data.x);
	}
	barrier();
}

void finalizeSegments() {
	uint lhbid = LIX >> 5, seg_group_offset = lhbid << MAX_SEGMENTS_SHIFT;
	uint tri_count = s_hblock_tri_counts[lhbid];
	uint num_segments = (s_hblock_frag_counts[lhbid] + SEGMENT_SIZE - 1) >> SEGMENT_SHIFT;
	// TODO: make sure that num_segments <= MAX_SEGMENTS

	for(uint seg_id = LIX & WARP_MASK; seg_id < num_segments; seg_id += WARP_SIZE) {
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
	uint lhbid = hbid & (NUM_WARPS - 1);
	uint seg_group_offset = lhbid << MAX_SEGMENTS_SHIFT;
	uint segment_data = s_segments[seg_group_offset + segment_id];
	uint tri_count = segment_data >> 16, first_tri = segment_data & 0xffff;
	uint src_offset_64 = scratch64SortedHBlockTrisOffset(lhbid) + first_tri;
	uint buf_offset = (LIX >> 5) << SEGMENT_SHIFT;

	if(tri_count >= 32) {
		for(uint i = LIX & WARP_MASK; i < tri_count; i += WARP_STEP) {
			uvec2 tri_data = g_scratch_64[src_offset_64 + i];
			int tri_offset = int(tri_data.x >> 24);
			if(i == 0 && tri_offset != 0)
				tri_offset -= SEGMENT_SIZE;
			uint tri_idx = tri_data.x & 0xffffff;
			uvec4 countx = (tri_data.y >> uvec4(16, 20, 24, 28)) & 15;
			uvec4 minx = (tri_data.y >> uvec4(0, 3, 6, 9)) & 7;
			uint bits0 = uint((1 << countx[0]) - 1) << (minx[0] + 0);
			uint bits1 = uint((1 << countx[1]) - 1) << (minx[1] + 8);
			uint bits2 = uint((1 << countx[2]) - 1) << (minx[2] + 16);
			uint bits3 = uint((1 << countx[3]) - 1) << (minx[3] + 24);

			uint bits = bits0 | bits1 | bits2 | bits3;
			tri_idx <<= 5;
			while(bits != 0 && tri_offset < SEGMENT_SIZE) {
				uint pixel_id = findLSB(bits);
				bits &= ~(1u << pixel_id);
				if(tri_offset >= 0)
					s_buffer[buf_offset + tri_offset] = pixel_id | tri_idx;
				tri_offset++;
			}
		}
	} else {
		int y = int(LIX & 3), count_shift = 16 + (y << 2), min_shift = (y << 1) + y;
		int mask1 = y >= 1 ? ~0 : 0, mask2 = y >= 2 ? ~0 : 0;

		for(uint i = (LIX & WARP_MASK) >> 2; i < tri_count; i += WARP_STEP / 4) {
			uvec2 tri_data = g_scratch_64[src_offset_64 + i];
			int tri_offset = int(tri_data.x >> 24);
			if(i == 0 && tri_offset != 0)
				tri_offset -= SEGMENT_SIZE;
			uint tri_idx = tri_data.x & 0xffffff;

			int minx = int((tri_data.y >> min_shift) & 7);
			int countx = int((tri_data.y >> count_shift) & 15);
			int prevx = countx + (shuffleUpNV(countx, 1, 4) & mask1);
			prevx += (shuffleUpNV(prevx, 2, 4) & mask2);
			tri_offset += prevx - countx;
			countx = min(countx, SEGMENT_SIZE - tri_offset);
			if(tri_offset < 0)
				countx += tri_offset, minx -= tri_offset, tri_offset = 0;

			uint pixel_id = (y << 3) | minx;
			uint value = pixel_id | (tri_idx << 5);
			for(int j = 0; j < countx; j++)
				s_buffer[buf_offset + tri_offset++] = value++;
		}
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
	uint hbx = hbid & HBLOCK_COLS_MASK;
	uint hby = hbid >> HBLOCK_COLS_SHIFT;
	uint mini_offset = LIX & ~31;
	uint reduce_pixel_bit = 1u << (LIX & 31);
	ivec2 half_block_pos = ivec2(hbx << HBLOCK_WIDTH_SHIFT, hby << HBLOCK_HEIGHT_SHIFT) + s_bin_pos;
	vec3 out_color = decodeRGB10(ctx.out_color);

	for(uint i = 0; i < sample_count; i += WARP_STEP) {
		// TODO: we don't need s_mini_buffer here, we can use s_buffer, thus decreasing mini_buffer size
		s_mini_buffer[LIX] = 0;
		uvec2 sample_s;
		uint sample_id = i + (LIX & 31);
		if(sample_id < sample_count) {
			uint value = s_buffer[buf_offset + sample_id];
			uint sample_pixel_id = value & 31;
			uint tri_idx = value >> 5;
			ivec2 pix_pos = half_block_pos + ivec2(sample_pixel_id & 7, sample_pixel_id >> 3);
			float sample_depth;
			uint sample_color = shadeSample(pix_pos, tri_idx, sample_depth);
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

ivec2 computePixelPos(uint hbid) {
	uint hbx = hbid & HBLOCK_COLS_MASK, hby = hbid >> HBLOCK_COLS_SHIFT;
	return ivec2((LIX & 7) + (hbx << HBLOCK_WIDTH_SHIFT),
				 ((LIX >> 3) & 3) + (hby << HBLOCK_HEIGHT_SHIFT));
}

void initVisualizeSamples() { s_vis_pixels[LIX] = 0; }

void visualizeSamples(uint sample_count) {
	uint buf_offset = (LIX >> 5) << SEGMENT_SHIFT;
	for(uint i = LIX & 31; i < sample_count; i += WARP_STEP) {
		uint pixel_id = s_buffer[buf_offset + i] & 31;
		atomicAdd(s_vis_pixels[(LIX & ~WARP_MASK) + pixel_id], 1);
	}
}

void finishVisualizeSamples(ivec2 pixel_pos) {
	uint pixel_id = (pixel_pos.x & 7) + ((pixel_pos.y & 3) << 3);
	vec3 color = vec3(s_vis_pixels[(LIX & ~31) + pixel_id]) / 64.0;
	uint enc_col = encodeRGBA8(vec4(SATURATE(color), 1.0));
	outputPixel(pixel_pos, enc_col);
}

void visualizeAllSamples(uint hbid) {
	uint lhbid = hbid & (NUM_WARPS - 1);
	uint tri_count = s_hblock_tri_counts[lhbid];
	uint src_offset_64 = scratch64SortedHBlockTrisOffset(lhbid);

	int y = int(LIX & 3);
	uint count_shift = 16 + (y << 2), min_shift = (y << 1) + y;
	uint buf_offset = (LIX >> 5) << SEGMENT_SHIFT;

	s_vis_pixels[LIX] = 0;

	for(uint i = (LIX & WARP_MASK) >> 2; i < tri_count; i += WARP_STEP / 4) {
		uvec2 tri_data = g_scratch_64[src_offset_64 + i];
		int minx = int((tri_data.y >> min_shift) & 7);
		int countx = int((tri_data.y >> count_shift) & 15);
		uint pixel_id = (y << 3) | minx;
		for(int j = 0; j < countx; j++)
			atomicAdd(s_vis_pixels[(LIX & ~31) | (pixel_id + j)], 1);
	}

	finishVisualizeSamples(computePixelPos(hbid));
}

void visualizeFragmentCounts(uint hbid, ivec2 pixel_pos) {
	uint count = s_hblock_frag_counts[hbid & (NUM_WARPS - 1)];
	vec4 color = vec4(SATURATE(vec3(count) / 4096.0), 1.0);
	if(count > SEGMENT_SIZE * MAX_SEGMENTS)
		color.gb *= 0.25;
	outputPixel(pixel_pos, encodeRGBA8(color));
}

void visualizeTriangleCounts(uint hbid, ivec2 pixel_pos) {
	uint count = s_hblock_tri_counts[hbid & (NUM_WARPS - 1)];
	//count = s_hblock_row_tri_counts[hbid >> HBLOCK_COLS_SHIFT] / 8;
	//count = s_bin_quad_count / 16;

	vec3 color = vec3(count) / 512.0;
	outputPixel(pixel_pos, encodeRGBA8(vec4(SATURATE(color), 1.0)));
}

void visualizeErrors(uint hbid) {
	uint lhbid = hbid & (NUM_WARPS - 1);
	uint color = 0xff000000;
	if(s_raster_error != 0)
		color += 0xff;
	else {
		color += 0x30;
		if(s_hblock_tri_counts[lhbid] > MAX_HBLOCK_TRIS)
			color += 0x40;
	}

	outputPixel(computePixelPos(hbid), color);
}

void rasterBin(int bin_id) {
	START_TIMER();

	const int num_blocks = (BIN_SIZE / BLOCK_SIZE) * (BIN_SIZE / BLOCK_SIZE);

	if(LIX < HBLOCK_ROWS) {
		if(LIX == 0) {
			// TODO: optimize
			ivec2 bin_pos = ivec2(bin_id % BIN_COUNT_X, bin_id / BIN_COUNT_X) * BIN_SIZE;
			s_bin_pos = bin_pos;
			s_bin_quad_count = BIN_QUAD_COUNTS(bin_id);
			s_bin_quad_offset = BIN_QUAD_OFFSETS(bin_id);
			s_bin_tri_count = BIN_TRI_COUNTS(bin_id);
			s_bin_tri_offset = BIN_TRI_OFFSETS(bin_id);
			s_raster_error = 0;
		}

		s_hblock_row_tri_counts[LIX] = 0;
	}
	barrier();

	for(int start_hby = 0; start_hby < HBLOCK_ROWS; start_hby += HBLOCK_ROWS_STEP) {
		processQuads(start_hby);
		groupMemoryBarrier();
		barrier();
		UPDATE_TIMER(0);
		s_segments[LIX] = INVALID_SEGMENT;
		s_segments[LSIZE + LIX] = INVALID_SEGMENT;

		if(s_raster_error == 0) {
			int step = NUM_WARPS >> s_hblock_group_shift;
			for(int i = 0; i < s_hblock_group_size; i++)
				generateHBlocks(start_hby * HBLOCK_COLS + step * i);
		}
		groupMemoryBarrier();
		barrier();
		finalizeSegments();
		barrier();

		int hbid = start_hby * HBLOCK_COLS + int(LIX >> 5);
		if(s_raster_error != 0) {
			visualizeErrors(hbid);
			barrier();
			if(LIX == 0)
				s_raster_error = 0;
			barrier();
			continue;
		}
		UPDATE_TIMER(1);

		//visualizeAllSamples(hbid);
		ReductionContext context;
		initReduceSamples(context);

		for(int segment_id = 0;; segment_id++) {
			int frag_count = int(s_hblock_frag_counts[hbid & (NUM_WARPS - 1)]);
			frag_count = min(SEGMENT_SIZE, frag_count - (segment_id << SEGMENT_SHIFT));
			if(frag_count <= 0)
				break;

			loadSamples(hbid, segment_id);
			UPDATE_TIMER(2);

			shadeAndReduceSamples(hbid, frag_count, context);
			UPDATE_TIMER(3);

#ifdef ALPHA_THRESHOLD
			if(allInvocationsARB(context.out_trans < 1.0 / 255.0))
				break;
#endif
		}

		ivec2 pixel_pos = computePixelPos(hbid);
		finishReduceSamples(pixel_pos, context);
		//visualizeFragmentCounts(hbid, pixel_pos);
		//visualizeTriangleCounts(hbid, pixel_pos);
		UPDATE_TIMER(4);

		barrier();
	}
}

int loadNextBin() {
	if(LIX == 0) {
		uint bin_idx = atomicAdd(g_info.a_medium_bins, 1);
		s_bin_id = bin_idx < s_num_bins ? MEDIUM_LEVEL_BINS(bin_idx) : -1;
		s_bin_raster_offset = s_bin_id << (BIN_SHIFT * 2);
	}
	barrier();
	return s_bin_id;
}

void main() {
	INIT_TIMERS();
	if(LIX == 0)
		s_num_bins = g_info.bin_level_counts[BIN_LEVEL_MEDIUM];

	int bin_id = loadNextBin();
	while(bin_id != -1) {
		barrier();
		rasterBin(bin_id);
		bin_id = loadNextBin();
	}
	COMMIT_TIMERS(g_info.raster_timers);
}
