#include "shared/compute_funcs.glsl"
#include "shared/raster.glsl"
#include "shared/scanline.glsl"
#include "shared/shading.glsl"
#include "shared/timers.glsl"

#include "%shader_debug"
DEBUG_SETUP(1, 11)

#extension GL_KHR_shader_subgroup_ballot : require

#define LSIZE 1024
#define LSHIFT 10

#define NUM_WARPS (LSIZE / WARP_SIZE)
#define BUFFER_SIZE (LSIZE * 8)

// Basic maximum value of tris per rblock
#define MAX_RBLOCK_TRIS0 (8 * WARP_SIZE)
#define MAX_RBLOCK_TRIS0_SHIFT (WARP_SHIFT + 3)

// Actual max value of tris per rblock,
// assuming using multiple warps per rblock in generateRBlocks
#define MAX_RBLOCK_TRIS 4096
#define MAX_RBLOCK_TRIS_SHIFT 12

#define MAX_GROUP_SIZE 16

#define MAX_RBLOCK_ROW_TRIS 16384
#define MAX_RBLOCK_ROW_TRIS_SHIFT 14

#define SEGMENT_SIZE 256
#define SEGMENT_SHIFT 8
#define INVALID_SEGMENT 0xffff

#define MAX_SEGMENTS_SHIFT 6
#define MAX_SEGMENTS 64

// Number of rows which can be processed with given amount of warps
#define RBLOCK_ROWS_STEP (NUM_WARPS / RBLOCK_COLS)
#define RBLOCK_ROWS_STEP_MASK (RBLOCK_ROWS_STEP - 1)

layout(local_size_x = LSIZE) in;

#define WORKGROUP_32_SCRATCH_SIZE (128 * 1024)
#define WORKGROUP_32_SCRATCH_SHIFT 17

#define WORKGROUP_64_SCRATCH_SIZE (256 * 1024)
#define WORKGROUP_64_SCRATCH_SHIFT 18

uint currentRBlockRow(uint rby) {
	return LSIZE < BIN_SIZE * BIN_SIZE ? rby & RBLOCK_ROWS_STEP_MASK : rby;
}

uint scratch32RBlockRowTrisOffset(uint rby) {
	return (gl_WorkGroupID.x << WORKGROUP_32_SCRATCH_SHIFT) +
		   currentRBlockRow(rby) * MAX_RBLOCK_ROW_TRIS;
}

uint scratch64RBlockRowTrisOffset(uint rby) {
	return (gl_WorkGroupID.x << WORKGROUP_64_SCRATCH_SHIFT) +
		   currentRBlockRow(rby) * (MAX_RBLOCK_ROW_TRIS * (RBLOCK_HEIGHT == 8 ? 2 : 1));
}

uint scratch64RBlockTrisOffset(uint lrbid) {
	uint offset = 128 * 1024;
	return (gl_WorkGroupID.x << WORKGROUP_64_SCRATCH_SHIFT) + offset + lrbid * MAX_RBLOCK_TRIS;
}

#if RBLOCK_HEIGHT == 8
uint scratch32RBlockTrisOffset(uint lrbid) {
	return (gl_WorkGroupID.x << WORKGROUP_32_SCRATCH_SHIFT) + lrbid * MAX_RBLOCK_TRIS;
}
#endif

shared int s_num_scratch_tris;
shared int s_num_bins, s_bin_id;
shared uint s_bin_quad_count, s_bin_quad_offset;
shared uint s_bin_tri_count, s_bin_tri_offset;

shared uint s_rblock_row_tri_counts[RBLOCK_ROWS];
shared int s_rblock_tri_counts[NUM_WARPS];
shared uint s_rblock_frag_counts[NUM_WARPS];

shared uint s_rblock_max_tri_counts;

// How many warps do we need to process single half-block in generateRBlocks ?
// Acceptable values: 1, 2, 4, 8; More: trouble :(
shared uint s_rblock_group_size;
shared uint s_rblock_group_shift;
shared uint s_max_sort_rcount;

shared uint s_buffer[BUFFER_SIZE + 1];
shared uint s_mini_buffer[LSIZE * (WARP_SIZE == 64 ? 2 : 1)];
shared uint s_segments[LSIZE * (WARP_SIZE == 64 ? 1 : 2)];

shared int s_raster_error;

// Only used when debugging
shared uint s_vis_pixels[LSIZE];

void generateRowTris(uint tri_idx, int start_rby) {
	uint scan_offset = STORAGE_TRI_SCAN_OFFSET + tri_idx * 2;
	uvec4 val0 = g_uvec4_storage[scan_offset + 0];
	uvec4 val1 = g_uvec4_storage[scan_offset + 1];
	int min_rby = clamp(int(val0.w & 0xffff) - s_bin_pos.y, 0, BIN_MASK) >> RBLOCK_HEIGHT_SHIFT;
	int max_rby = clamp(int(val0.w >> 16) - s_bin_pos.y, 0, BIN_MASK) >> RBLOCK_HEIGHT_SHIFT;

	if(RBLOCK_ROWS_STEP < RBLOCK_ROWS) {
		min_rby = max(start_rby, min_rby);
		max_rby = min(start_rby + RBLOCK_ROWS_STEP_MASK, max_rby);
		if(max_rby < min_rby)
			return;
	}

	vec2 start = vec2(s_bin_pos.x, s_bin_pos.y + min_rby * RBLOCK_HEIGHT);
	ScanlineParams scan = loadScanlineParamsRow(val0, val1, start);

	uint dst_offset_32 = scratch32RBlockRowTrisOffset(0);
	uint dst_offset_64 = scratch64RBlockRowTrisOffset(0);

	// TODO: is it worth it to make this loop more work-efficient?
	for(int rby = min_rby; rby <= max_rby; rby++) {
#if RBLOCK_HEIGHT == 8
		uvec3 bits0 = rasterBinStep(scan);
		uvec3 bits1 = rasterBinStep(scan);
		uint bx_mask = bits0.z | bits1.z;
#else
		uvec3 bits = rasterBinStep(scan);
		uint bx_mask = bits.z;
#endif
		if(bx_mask == 0)
			continue;

		uint rbid_row = currentRBlockRow(rby) << RBLOCK_COLS_SHIFT;
		uint min_rbid = findLSB(bx_mask), max_rbid = findMSB(bx_mask) + 1;
		// Accumulation is done at the end of processBlocks
		atomicAdd(s_rblock_tri_counts[rbid_row + min_rbid], 1);
		if(max_rbid < RBLOCK_COLS)
			atomicAdd(s_rblock_tri_counts[rbid_row + max_rbid], -1);

		uint row_tri_idx = atomicAdd(s_rblock_row_tri_counts[rby], 1);
		if(row_tri_idx >= MAX_RBLOCK_ROW_TRIS) {
			atomicOr(s_raster_error, 0xffffffff);
			return;
		}

#if RBLOCK_HEIGHT == 8
		uint roffset = row_tri_idx + currentRBlockRow(rby) * (MAX_RBLOCK_ROW_TRIS * 2);
		g_scratch_64[dst_offset_64 + roffset] =
			uvec2(bits0.x | (bx_mask << 24), bits1.x | ((tri_idx & 0xff0000) << 8));
		g_scratch_64[dst_offset_64 + roffset + MAX_RBLOCK_ROW_TRIS] =
			uvec2(bits0.y | ((tri_idx & 0xff) << 24), bits1.y | ((tri_idx & 0xff00) << 16));
#else
		uint roffset = row_tri_idx + currentRBlockRow(rby) * MAX_RBLOCK_ROW_TRIS;
		g_scratch_32[dst_offset_32 + roffset] = tri_idx | (bx_mask << 24);
		g_scratch_64[dst_offset_64 + roffset] = bits.xy;
#endif
	}
}

void processQuads(int start_rby) {
	if(LIX == 0)
		s_num_scratch_tris = 0;
	if(LIX < NUM_WARPS)
		s_rblock_tri_counts[LIX] = 0;
	barrier();

	for(uint i = LIX >> 1; i < s_bin_quad_count; i += LSIZE / 2) {
		uint second_tri = LIX & 1;
		uint bin_quad_idx = g_bin_quads[s_bin_quad_offset + i];
		uint quad_idx = bin_quad_idx & 0xfffffff;
		uint cull_flag = (bin_quad_idx >> (30 + second_tri)) & 1;
		if(cull_flag == 1)
			continue;

		uint tri_idx = quad_idx * 2 + second_tri;
		generateRowTris(tri_idx, start_rby);
	}
	for(uint i = LIX; i < s_bin_tri_count; i += LSIZE)
		generateRowTris(g_bin_tris[s_bin_tri_offset + i], start_rby);
	barrier();

	// Accumulating per hblock-counts for each hblock-row
	// Note: these are only estimates; very good estimates, but in some cases a single
	// triangle can have wide holes between pixels (because middle pixels don't hit pixel centers)
	if(LIX < NUM_WARPS) {
		uint rbx = LIX & RBLOCK_COLS_MASK;
		int value = s_rblock_tri_counts[LIX], temp;
		temp = subgroupShuffleUp(value, 1), value += rbx >= 1 ? temp : 0;
		if(RBLOCK_COLS >= 4)
			temp = subgroupShuffleUp(value, 2), value += rbx >= 2 ? temp : 0;
		if(RBLOCK_COLS >= 8)
			temp = subgroupShuffleUp(value, 4), value += rbx >= 4 ? temp : 0;
		s_rblock_tri_counts[LIX] = value;
		uint max_value = subgroupMax_(uint(value), NUM_WARPS);
		if(LIX == 0) {
			s_rblock_max_tri_counts = max_value;
			// rcount: count rounded up to next power of 2, minimum: 32
			uint rcount =
				(max_value & (max_value - 1)) == 0 ? max_value : (2 << findMSB(max_value));
			s_max_sort_rcount = max(32, rcount);

			//uint group_shift = max(int(log2(max_value)) - MAX_RBLOCK_TRIS0_SHIFT, 0);
			uint group_shift = max_value <= MAX_RBLOCK_TRIS0	  ? 0 :
							   max_value <= MAX_RBLOCK_TRIS0 * 2  ? 1 :
							   max_value <= MAX_RBLOCK_TRIS0 * 4  ? 2 :
							   max_value <= MAX_RBLOCK_TRIS0 * 8  ? 3 :
							   max_value <= MAX_RBLOCK_TRIS0 * 16 ? 4 :
																	5;
			uint group_size = 1 << group_shift;
			s_rblock_group_shift = group_shift;
			s_rblock_group_size = group_size;
			if(group_size > MAX_GROUP_SIZE)
				s_raster_error = 0xffffffff; // TODO: better reporting
		}
	}
}

uint swap(uint x, int mask, uint dir) {
	uint y = subgroupShuffleXor(x, mask);
	return uint(x < y) == dir ? y : x;
}
uint bitExtract(uint value, int boffset) { return (value >> boffset) & 1; }
uint xorBits(uint value, int bit0, int bit1) { return ((value >> bit0) ^ (value >> bit1)) & 1; }

void sortTris(uint lrbid, uint count, uint buf_offset, uint group_size, uint lid) {
	uint rcount = s_max_sort_rcount;
	for(uint i = lid + count; i < rcount; i += group_size)
		s_buffer[buf_offset + i] = 0xffffffff;

	for(uint i = lid; i < rcount; i += group_size) {
		uint value = s_buffer[buf_offset + i];
		// TODO: register sort could be faster?
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
		if(group_size >= 64) {
			value = swap(value, 0x10, xorBits(lid, 5, 4)); // K = 32
			value = swap(value, 0x08, xorBits(lid, 5, 3));
			value = swap(value, 0x04, xorBits(lid, 5, 2));
			value = swap(value, 0x02, xorBits(lid, 5, 1));
			value = swap(value, 0x01, xorBits(lid, 5, 0));
		}
		s_buffer[buf_offset + i] = value;
	}
	int start_k = group_size >= 64 ? 64 : 32, end_j = 32;
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
	}
}

// TODO: maybe process smaller amount of blocks at the same time?
// smaller chance that it will leave cache
void generateRBlocks(uint start_rbid) {
	uint group_size = s_rblock_group_size * WARP_SIZE;
	uint group_shift = s_rblock_group_shift;
	uint group_mask = group_size - 1;
	uint group_thread = LIX & group_mask;

	// TODO: better names for indices
	uint group_rbid = LIX >> (WARP_SHIFT + group_shift);
	uint rbid = start_rbid + group_rbid;
	uint rby = rbid >> RBLOCK_COLS_SHIFT, rbx = rbid & RBLOCK_COLS_MASK;
	uint lrbid = rbid & (NUM_WARPS - 1);
	uint tri_count = s_rblock_row_tri_counts[rby];
	uint buf_offset = group_rbid << (MAX_RBLOCK_TRIS0_SHIFT + group_shift);

	uint src_offset_32 = scratch32RBlockRowTrisOffset(rby);
	uint src_offset_64 = scratch64RBlockRowTrisOffset(rby);

	{
		uint bx_bits_shift = 24 + rbx;
		uint thread_bit_mask = ~(0xffffffffu << (LIX & WARP_MASK));
		uint block_tri_count = 0;
		uint max_block_tris = MAX_RBLOCK_TRIS0 << group_shift;

		if(group_thread < WARP_SIZE) {
			for(uint i = group_thread; i < tri_count; i += WARP_SIZE) {
#if RBLOCK_HEIGHT == 8
				uint bx_bit = (g_scratch_64[src_offset_64 + i].x >> bx_bits_shift) & 1;
#else
				uint bx_bit = (g_scratch_32[src_offset_32 + i] >> bx_bits_shift) & 1;
#endif

#if WARP_SIZE == 32
				uint bit_mask = subgroupBallot(bx_bit != 0).x;
				if(bit_mask == 0)
					continue;
				uint warp_offset = bitCount(bit_mask & gl_SubgroupLtMask.x);
				uint bit_count = bitCount(bit_mask);
#else
				uvec2 bit_mask = subgroupBallot(bx_bit != 0).xy;
				if(bit_mask.x == 0 && bit_mask.y == 0)
					continue;
				uint warp_offset = bitCount(bit_mask.x & gl_SubgroupLtMask.x) +
								   bitCount(bit_mask.y & gl_SubgroupLtMask.y);
				uint bit_count = bitCount(bit_mask.x) + bitCount(bit_mask.y);
#endif

				if(bx_bit != 0) {
					uint tri_offset = block_tri_count + warp_offset;
					if(tri_offset < max_block_tris)
						s_buffer[buf_offset + tri_offset] = i;
				}
				block_tri_count += bit_count;
			}

			if(group_thread == 0) {
				if(s_rblock_tri_counts[lrbid] < int(block_tri_count))
					DEBUG_RECORD(rbid, s_rblock_tri_counts[lrbid], block_tri_count, 0);
				s_rblock_tri_counts[lrbid] = int(block_tri_count);
			}
		}
		barrier();
	}

	uint dst_offset_64 = scratch64RBlockTrisOffset(lrbid);
#if RBLOCK_HEIGHT == 8
	uint dst_offset_32 = scratch32RBlockTrisOffset(lrbid);
#endif
	tri_count = s_rblock_tri_counts[lrbid];
	int startx = int(rbx << RBLOCK_WIDTH_SHIFT);
	vec2 block_pos = vec2(rbx << RBLOCK_WIDTH_SHIFT, rby << RBLOCK_HEIGHT_SHIFT) + vec2(s_bin_pos);

	for(uint i = group_thread; i < tri_count; i += group_size) {
		uint row_tri_idx = s_buffer[buf_offset + i];
		vec2 cpos = vec2(0, 0);
		uint num_frags;

#if RBLOCK_HEIGHT == 8
		uvec2 tri_mins = g_scratch_64[src_offset_64 + row_tri_idx];
		uvec2 tri_maxs = g_scratch_64[src_offset_64 + row_tri_idx + MAX_RBLOCK_ROW_TRIS];
		uint tri_idx =
			(tri_maxs.x >> 24) | ((tri_maxs.y >> 16) & 0xff00) | ((tri_mins.y >> 8) & 0xff0000);
		uvec2 hnum_frags;
		uvec2 bits = uvec2(rasterBlock(tri_mins.x, tri_maxs.x, startx, hnum_frags.x, cpos),
						   rasterBlock(tri_mins.y, tri_maxs.y, startx, hnum_frags.y, cpos));
		num_frags = hnum_frags.x + hnum_frags.y;
		g_scratch_64[dst_offset_64 + i] = bits;
		g_scratch_32[dst_offset_32 + i] = tri_idx | (num_frags << 24);
#else
		uint tri_idx = g_scratch_32[src_offset_32 + row_tri_idx] & 0xffffff;
		uvec2 tri_info = g_scratch_64[src_offset_64 + row_tri_idx];
		uint bits = rasterBlock(tri_info.x, tri_info.y, startx, num_frags, cpos);
		g_scratch_64[dst_offset_64 + i] = uvec2(bits, tri_idx | (num_frags << 24));
#endif

		if(num_frags == 0) // This means that bx_mask is invalid
			DEBUG_RECORD(0, 0, 0, 0);
		uint depth = rasterBlockDepth(cpos * (0.5 / float(num_frags)) + block_pos, tri_idx);
		s_buffer[buf_offset + i] = i | (depth << 12);
	}
	barrier();

	sortTris(lrbid, tri_count, buf_offset, group_size, group_thread);

	barrier();
	groupMemoryBarrier();

#ifdef SHADER_DEBUG
	for(uint i = LIX & WARP_MASK; i < tri_count; i += WARP_SIZE) {
		uint value = s_buffer[buf_offset + i];
		uint prev_value = i == 0 ? 0 : s_buffer[buf_offset + i - 1];
		if(value <= prev_value)
			DEBUG_RECORD(i, tri_count, prev_value, value);
	}
#endif

	for(uint i = group_thread; i < tri_count; i += group_size) {
		uint rblock_tri_idx = s_buffer[buf_offset + i] & 0xfff;
#if RBLOCK_HEIGHT == 8
		uint num_frags = g_scratch_32[dst_offset_32 + rblock_tri_idx] >> 24;
#else
		uint num_frags = g_scratch_64[dst_offset_64 + rblock_tri_idx].y >> 24;
#endif

		// Computing triangle-ordered sample offsets within each block
		uint sum = subgroupInclusiveAddFast(num_frags);
		s_buffer[buf_offset + i] = (sum - num_frags) | (num_frags << 12) | (rblock_tri_idx << 20);
	}
	barrier();

	// Computing prefix sum across whole render-blocks. We're processing 4 elements
	// at a time, so that we can fit with 4096 tris/hblock (128 warps).
	if(LIX < 2 * NUM_WARPS) {
		uint wgsize = 2 << group_shift, wgmask = wgsize - 1;
		uint group_rbid = LIX >> (1 + group_shift), group_sub_idx = LIX & wgmask;
		uint buf_offset = group_rbid << (MAX_RBLOCK_TRIS0_SHIFT + group_shift);
		uint lrbid = (start_rbid + group_rbid) & (NUM_WARPS - 1);
		uint tri_count = s_rblock_tri_counts[lrbid];
		uint warp_offset = group_sub_idx << (WARP_SHIFT + 2);

		uvec4 values = uvec4(0);
		for(int j = 0; j < 4; j++) {
			if(warp_offset < tri_count) {
				uint rblock_tri_idx = min(warp_offset + WARP_MASK, tri_count - 1);
				uint last = s_buffer[buf_offset + rblock_tri_idx];
				values[j] = (last & 0xfff) + ((last >> 12) & 0xff);
			}
			warp_offset += WARP_SIZE;
		}

		uint value = values[0] + values[1] + values[2] + values[3];
		uint sum = value, temp;
		if(true)
			temp = subgroupShuffleUp(sum, 1), sum += group_sub_idx >= 1 ? temp : 0;
		if(wgsize >= 4)
			temp = subgroupShuffleUp(sum, 2), sum += group_sub_idx >= 2 ? temp : 0;
		if(wgsize >= 8)
			temp = subgroupShuffleUp(sum, 4), sum += group_sub_idx >= 4 ? temp : 0;
		if(wgsize >= 16)
			temp = subgroupShuffleUp(sum, 8), sum += group_sub_idx >= 8 ? temp : 0;
		if(wgsize >= 32)
			temp = subgroupShuffleUp(sum, 16), sum += group_sub_idx >= 16 ? temp : 0;

		if(group_sub_idx == wgmask) {
			updateStats(int(sum), int(tri_count));
			s_rblock_frag_counts[lrbid] = sum;
		}
		s_mini_buffer[LIX * 4 + 0] = sum - value, value -= values[0];
		s_mini_buffer[LIX * 4 + 1] = sum - value, value -= values[1];
		s_mini_buffer[LIX * 4 + 2] = sum - value;
		s_mini_buffer[LIX * 4 + 3] = sum - values[3];
	}
	barrier();
	if(s_raster_error != 0)
		return;

	// Finding triangles which start segments
	// Also storing hblock-tri data to temporary stored_tris array
	uint seg_block_offset = lrbid << MAX_SEGMENTS_SHIFT;
#if RBLOCK_HEIGHT == 8
	uvec3 stored_tris[8];
#else
	uvec2 stored_tris[8];
#endif
	uint stored_idx = 0;
	for(uint i = group_thread; i < tri_count; i += group_size) {
		uint current = s_buffer[buf_offset + i];
		uint tri_value = (current >> 12) & 255;
		uint rblock_tri_idx = current >> 20;

		uint mini_offset = group_rbid << (3 + group_shift);
		uint tri_offset = (current & 0xfff) + s_mini_buffer[mini_offset + (i >> WARP_SHIFT)];

		uint seg_offset = tri_offset & (SEGMENT_SIZE - 1);
		uint seg_id = tri_offset >> SEGMENT_SHIFT;
		bool first_seg = seg_offset == 0;
		if(seg_offset + tri_value > SEGMENT_SIZE)
			seg_id++, first_seg = true;
		if(seg_id >= MAX_SEGMENTS) {
			atomicOr(s_raster_error, 1 << lrbid);
			break;
		}
		seg_offset <<= 24;
		if(first_seg && tri_value > 0)
			s_segments[seg_block_offset + seg_id] = i | seg_offset;

#if RBLOCK_HEIGHT == 8
		uvec2 tri_data = g_scratch_64[dst_offset_64 + rblock_tri_idx];
		uint tri_info = g_scratch_32[dst_offset_32 + rblock_tri_idx];
		uint tri_idx = tri_info & 0xffffff;
		stored_tris[stored_idx++] = uvec3(tri_data, tri_idx | seg_offset);
#else
		uvec2 tri_data = g_scratch_64[dst_offset_64 + rblock_tri_idx];
		uint tri_idx = tri_data.y & 0xffffff;
		stored_tris[stored_idx++] = uvec2(tri_idx | seg_offset, tri_data.x);
#endif
	}
	barrier();

	// Reordering hblock-tris in scratch
	stored_idx = 0;
	for(uint i = group_thread; i < tri_count; i += group_size) {
#if RBLOCK_HEIGHT == 8
		g_scratch_64[dst_offset_64 + i] = stored_tris[stored_idx].xy;
		g_scratch_32[dst_offset_32 + i] = stored_tris[stored_idx++].z;
#else
		g_scratch_64[dst_offset_64 + i] = stored_tris[stored_idx++];
#endif
	}
	barrier();
}

void finalizeSegments() {
	uint lrbid = LIX >> WARP_SHIFT, seg_group_offset = lrbid << MAX_SEGMENTS_SHIFT;
	uint tri_count = s_rblock_tri_counts[lrbid];
	uint num_segments = (s_rblock_frag_counts[lrbid] + SEGMENT_SIZE - 1) >> SEGMENT_SHIFT;
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

void loadSamples(uint rbid, int segment_id) {
	uint lrbid = rbid & (NUM_WARPS - 1);
	uint seg_group_offset = lrbid << MAX_SEGMENTS_SHIFT;
	uint segment_data = s_segments[seg_group_offset + segment_id];
	uint tri_count = segment_data >> 16, first_tri = segment_data & 0xffff;
	uint src_offset_64 = scratch64RBlockTrisOffset(lrbid) + first_tri;
#if RBLOCK_HEIGHT == 8
	uint src_offset_32 = scratch32RBlockTrisOffset(lrbid) + first_tri;
#endif
	uint buf_offset = (LIX >> WARP_SHIFT) << SEGMENT_SHIFT;

	if(tri_count >= 32 && RBLOCK_HEIGHT == 4) {
		for(uint i = LIX & WARP_MASK; i < tri_count; i += WARP_SIZE) {
			uvec2 tri_data = g_scratch_64[src_offset_64 + i];
			int tri_offset = int(tri_data.x >> 24);
			if(i == 0 && tri_offset != 0)
				tri_offset -= SEGMENT_SIZE;
			uint tri_idx = tri_data.x & 0xffffff;

			uvec4 countx = (uvec4(tri_data.y) >> uvec4(16, 20, 24, 28)) & 15;
			uvec4 minx = (uvec4(tri_data.y) >> uvec4(0, 3, 6, 9)) & 7;
			uint bits0 = uint((1 << countx[0]) - 1) << (minx[0] + 0);
			uint bits1 = uint((1 << countx[1]) - 1) << (minx[1] + 8);
			uint bits2 = uint((1 << countx[2]) - 1) << (minx[2] + 16);
			uint bits3 = uint((1 << countx[3]) - 1) << (minx[3] + 24);

			uint bits = bits0 | bits1 | bits2 | bits3;
			tri_idx <<= 8;
			while(bits != 0 && tri_offset < SEGMENT_SIZE) {
				uint pixel_id = findLSB(bits);
				bits &= ~(1u << pixel_id);
				if(tri_offset >= 0)
					s_buffer[buf_offset + tri_offset] = pixel_id | tri_idx;
				tri_offset++;
			}
		}
	} else {
		int y = int(LIX & (RBLOCK_HEIGHT - 1)), count_shift = 16 + (y & 3) * 4,
			min_shift = (y & 3) * 3; // TODO: min_shift the same as count_shift
		int mask1 = y >= 1 ? ~0 : 0, mask2 = y >= 2 ? ~0 : 0, mask3 = y >= 4 ? ~0 : 0;

		for(uint i = (LIX & WARP_MASK) >> RBLOCK_HEIGHT_SHIFT; i < tri_count;
			i += WARP_SIZE / RBLOCK_HEIGHT) {
#if RBLOCK_HEIGHT == 8
			uint tri_data = g_scratch_64[src_offset_64 + i][y >> 2];
			uint tri_info = g_scratch_32[src_offset_32 + i];
			int tri_offset = int(tri_info >> 24);
			if(i == 0 && tri_offset != 0)
				tri_offset -= SEGMENT_SIZE;
			uint tri_idx = tri_info & 0xffffff;
			int minx = int((tri_data >> min_shift) & 7);
			int countx = int((tri_data >> count_shift) & 15);
#else
			uvec2 tri_data = g_scratch_64[src_offset_64 + i];
			int tri_offset = int(tri_data.x >> 24);
			if(i == 0 && tri_offset != 0)
				tri_offset -= SEGMENT_SIZE;
			uint tri_idx = tri_data.x & 0xffffff;
			int minx = int((tri_data.y >> min_shift) & 7);
			int countx = int((tri_data.y >> count_shift) & 15);
#endif

			int prevx = countx + (subgroupShuffleUp(countx, 1) & mask1);
			prevx += (subgroupShuffleUp(prevx, 2) & mask2);
#if RBLOCK_HEIGHT == 8
			prevx += (subgroupShuffleUp(prevx, 4) & mask3);
#endif

			tri_offset += prevx - countx;
			countx = min(countx, SEGMENT_SIZE - tri_offset);
			if(tri_offset < 0)
				countx += tri_offset, minx -= tri_offset, tri_offset = 0;

			uint pixel_id = (y << 3) | minx;
			uint value = pixel_id | (tri_idx << 8);
			for(int j = 0; j < countx; j++)
				s_buffer[buf_offset + tri_offset++] = value++;
		}
	}
}

ivec2 computePixelPos(uint rbid) {
	uint rbx = rbid & RBLOCK_COLS_MASK, rby = rbid >> RBLOCK_COLS_SHIFT;
	return ivec2((LIX & (RBLOCK_WIDTH - 1)) + (rbx << RBLOCK_WIDTH_SHIFT),
				 ((LIX >> RBLOCK_WIDTH_SHIFT) & (RBLOCK_HEIGHT - 1)) +
					 (rby << RBLOCK_HEIGHT_SHIFT));
}

void shadeAndReduceSamples(uint rbid, uint sample_count, in out ReductionContext ctx) {
	uint buf_offset = (LIX >> WARP_SHIFT) << SEGMENT_SHIFT;
	uint mini_offset =
		WARP_SIZE == 64 ? (LIX & ~WARP_MASK) + ((LIX & 32) != 0 ? LSIZE : 0) : LIX & ~WARP_MASK;
	// TODO: make it work the same way as in raster_low? maybe it's better to change in raster_low ?
	uint rbx = rbid & RBLOCK_COLS_MASK, rby = rbid >> RBLOCK_COLS_SHIFT;
	ivec2 rblock_pos = ivec2(rbx << RBLOCK_WIDTH_SHIFT, rby << RBLOCK_HEIGHT_SHIFT) + s_bin_pos;
	vec3 out_color = ctx.out_color;

	for(uint i = 0; i < sample_count; i += WARP_SIZE) {
		// TODO: we don't need s_mini_buffer here, we can use s_buffer, thus decreasing mini_buffer size
		s_mini_buffer[LIX] = 0;
		if(WARP_SIZE == 64)
			s_mini_buffer[LSIZE + LIX] = 0;
		uvec2 sample_s;
		uint sample_id = i + (LIX & WARP_MASK);
		if(sample_id < sample_count) {
			uint value = s_buffer[buf_offset + sample_id];
			uint sample_pixel_id = value & WARP_MASK;
			uint tri_idx = value >> 8;
			ivec2 pix_pos = rblock_pos + ivec2(sample_pixel_id & 7, sample_pixel_id >> 3);
			float sample_depth;
			uint sample_color = shadeSample(pix_pos, tri_idx, sample_depth);
			sample_s = uvec2(sample_color, floatBitsToUint(sample_depth));
#if WARP_SIZE == 32
			atomicOr(s_mini_buffer[mini_offset + sample_pixel_id], gl_SubgroupEqMask.x);
#else
			const uint pixel_bit = gl_SubgroupEqMask.x | gl_SubgroupEqMask.y;
			atomicOr(s_mini_buffer[mini_offset + sample_pixel_id], pixel_bit);
#endif
		}
		subgroupMemoryBarrierShared();

#if WARP_SIZE == 32
		uint pixel_bitmask = s_mini_buffer[LIX];
#else
		uvec2 pixel_bitmask = uvec2(s_mini_buffer[LIX], s_mini_buffer[LIX + LSIZE]);
#endif
		if(reduceSample(ctx, out_color, sample_s, pixel_bitmask))
			break;
	}

	// TODO: check if encode+decode for out_color is really needed (to save 2 regs)
	ctx.out_color = SATURATE(out_color);
}

void initVisualizeSamples() { s_vis_pixels[LIX] = 0; }

void visualizeSamples(uint sample_count) {
	uint buf_offset = (LIX >> WARP_SHIFT) << SEGMENT_SHIFT;
	for(uint i = LIX & WARP_MASK; i < sample_count; i += WARP_SIZE) {
		uint pixel_id = s_buffer[buf_offset + i] & WARP_MASK;
		atomicAdd(s_vis_pixels[(LIX & ~WARP_MASK) + pixel_id], 1);
	}
}

void finishVisualizeSamples(ivec2 pixel_pos) {
	uint pixel_id = (pixel_pos.x & (RBLOCK_WIDTH - 1)) +
					((pixel_pos.y & (RBLOCK_HEIGHT - 1)) << RBLOCK_WIDTH_SHIFT);
	vec3 color = vec3(s_vis_pixels[(LIX & ~WARP_MASK) + pixel_id]) / 32.0;
	outputPixel(pixel_pos, vec4(SATURATE(color), 1.0));
}

void visualizeAllSamples(uint rbid, ivec2 pixel_pos) {
	uint lrbid = rbid & (NUM_WARPS - 1);
	uint tri_count = s_rblock_tri_counts[lrbid];
	uint src_offset_64 = scratch64RBlockTrisOffset(lrbid);

	int y = int(LIX & 3);
	uint count_shift = 16 + (y << 2), min_shift = (y << 1) + y;
	uint buf_offset = (LIX >> 5) << SEGMENT_SHIFT;

	s_vis_pixels[LIX] = 0;

	for(uint i = (LIX & WARP_MASK) >> 2; i < tri_count; i += WARP_SIZE / 4) {
		uvec2 tri_data = g_scratch_64[src_offset_64 + i];
		int minx = int((tri_data.y >> min_shift) & 7);
		int countx = int((tri_data.y >> count_shift) & 15);
		uint pixel_id = (y << 3) | minx;
		for(int j = 0; j < countx; j++)
			atomicAdd(s_vis_pixels[(LIX & ~31) | (pixel_id + j)], 1);
	}

	finishVisualizeSamples(pixel_pos);
}

void visualizeBlockCounts(uint rbid, ivec2 pixel_pos) {
	uint lrbid = rbid & (WARP_SIZE == 64 ? NUM_WARPS - 1 : NUM_WARPS * 2 - 1);
	uint frag_count = s_rblock_frag_counts[rbid & (NUM_WARPS - 1)];
	uint tri_count = s_rblock_tri_counts[rbid & (NUM_WARPS - 1)];
	//tri_count = s_rblock_row_tri_counts[rbid >> RBLOCK_COLS_SHIFT] / 8;
	//tri_count = s_bin_quad_count / 16;

	/*uint seg_offset = lrbid << MAX_SEGMENTS_SHIFT;
	tri_count = 0, frag_count = 0;
	uint num_segments = 0;
	for(uint i = 0; i < MAX_SEGMENTS; i++) {
		uint segment = s_segments[seg_offset + i];
		num_segments++;
		if(segment == INVALID_SEGMENT)
			break;
		uint seg_tri_count = segment >> 16;
		uint seg_first_tri = segment & 0xffff;

		tri_count += seg_tri_count;
		uint src_offset_64 = scratch64RBlockTrisOffset(lrbid) + seg_first_tri;
		for(uint j = 0; j < seg_tri_count; j++) {
			uvec2 tri_data = g_scratch_64[src_offset_64 + j];
			uint count0 = tri_data.x >> 16, count1 = RBLOCK_HEIGHT == 8 ? 0 : tri_data.y >> 16;
			frag_count +=
				(count0 & 15) + ((count0 >> 4) & 15) + ((count0 >> 8) & 15) + ((count0 >> 12) & 15);
			frag_count +=
				(count1 & 15) + ((count1 >> 4) & 15) + ((count1 >> 8) & 15) + ((count1 >> 12) & 15);
		}
	}*/

	vec3 color;
	color = gradientColor(frag_count, uvec4(64, 256, 1024, 4096));
	//color = gradientColor(tri_count, uvec4(16, 64, 256, 1024));
	outputPixel(pixel_pos, vec4(SATURATE(color), 1.0));
}

// TODO: fixme
void visualizeErrors(uint rbid) {
	uint lrbid = rbid & (NUM_WARPS - 1);
	uint color = 0xff000000;
	if(s_raster_error != 0)
		color += 0xff;
	else {
		color += 0x30;
		if(s_rblock_tri_counts[lrbid] > MAX_RBLOCK_TRIS)
			color += 0x40;
	}

	//outputPixel(computePixelPos(rbid), decodeRGBA8(color));
}

void rasterBin(int bin_id) {
	START_TIMER();
	if(LIX < RBLOCK_ROWS) {
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

		s_rblock_row_tri_counts[LIX] = 0;
	}
	barrier();

	for(int start_rby = 0; start_rby < RBLOCK_ROWS; start_rby += RBLOCK_ROWS_STEP) {
		processQuads(start_rby);
		groupMemoryBarrier();
		barrier();
		UPDATE_TIMER(0);
		s_segments[LIX] = INVALID_SEGMENT;
#if WARP_SIZE == 32
		s_segments[LSIZE + LIX] = INVALID_SEGMENT;
#endif

		if(s_raster_error == 0) {
			int step = NUM_WARPS >> s_rblock_group_shift;
			for(int i = 0; i < s_rblock_group_size; i++)
				generateRBlocks(start_rby * RBLOCK_COLS + step * i);
		}
		groupMemoryBarrier();
		barrier();
		finalizeSegments();
		barrier();

		int rbid = start_rby * RBLOCK_COLS + int(LIX >> WARP_SHIFT);
		if(s_raster_error != 0) {
			visualizeErrors(rbid);
			barrier();
			if(LIX == 0)
				s_raster_error = 0;
			barrier();
			continue;
		}
		UPDATE_TIMER(1);

		//visualizeAllSamples(rbid);
		ReductionContext context;
		initReduceSamples(context);
		//initVisualizeSamples();

		for(int segment_id = 0;; segment_id++) {
			int frag_count = int(s_rblock_frag_counts[rbid & (NUM_WARPS - 1)]);
			frag_count = min(SEGMENT_SIZE, frag_count - (segment_id << SEGMENT_SHIFT));
			if(frag_count <= 0)
				break;

			loadSamples(rbid, segment_id);
			UPDATE_TIMER(2);

			shadeAndReduceSamples(rbid, frag_count, context);
			//visualizeSamples(frag_count);
			UPDATE_TIMER(3);

#ifdef ALPHA_THRESHOLD
			if(subgroupAll(context.out_trans < alpha_threshold))
				break;
#endif
		}

		ivec2 pixel_pos = computePixelPos(rbid);
		outputPixel(pixel_pos, finishReduceSamples(context));
		//finishVisualizeSamples(pixel_pos);
		//visualizeBlockCounts(rbid, pixel_pos);
		UPDATE_TIMER(4);

		barrier();
	}
}

int loadNextBin() {
	if(LIX == 0) {
		uint bin_idx = atomicAdd(g_info.a_high_bins, 1);
		s_bin_id = bin_idx < s_num_bins ? HIGH_LEVEL_BINS(bin_idx) : -1;
	}
	barrier();
	return s_bin_id;
}

void main() {
	INIT_TIMERS();
	if(LIX == 0)
		s_num_bins = g_info.bin_level_counts[BIN_LEVEL_HIGH];
	initStats();

	int bin_id = loadNextBin();
	while(bin_id != -1) {
		barrier();
		rasterBin(bin_id);
		bin_id = loadNextBin();
	}

	COMMIT_TIMERS(g_info.raster_timers);
	commitStats();
}
