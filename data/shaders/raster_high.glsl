#define LSIZE 1024
#define LSHIFT 10

#define MAX_SEGMENTS_SHIFT 6
#define MAX_SEGMENTS 64

#include "shared/raster.glsl"

#include "%shader_debug"
DEBUG_SETUP(1, 11)

// Basic maximum value of tris per rblock
#define MAX_RBLOCK_TRIS0 (8 * WARP_SIZE)
#define MAX_RBLOCK_TRIS0_SHIFT (WARP_SHIFT + 3)

#define MAX_GROUP_SIZE 16
#define MAX_GROUP_SHIFT 4

// Actual max value of tris per rblock,
// assuming using multiple warps per rblock in generateRBlocks
#define MAX_RBLOCK_TRIS (MAX_RBLOCK_TRIS0 * MAX_GROUP_SIZE)
#define MAX_RBLOCK_TRIS_SHIFT (MAX_RBLOCK_TRIS0_SHIFT + MAX_GROUP_SHIFT)

#define MAX_RBLOCK_ROW_TRIS 16384
#define MAX_RBLOCK_ROW_TRIS_SHIFT 14

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

uint scratch32RBlockTrisOffset(uint lrbid) {
	return (gl_WorkGroupID.x << WORKGROUP_32_SCRATCH_SHIFT) + lrbid * MAX_RBLOCK_TRIS;
}

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

shared int s_raster_error;

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
		if(RBLOCK_COLS >= 2)
			temp = subgroupShuffleUp(value, 1), value += rbx >= 1 ? temp : 0;
		if(RBLOCK_COLS >= 4)
			temp = subgroupShuffleUp(value, 2), value += rbx >= 2 ? temp : 0;
		if(RBLOCK_COLS >= 8)
			temp = subgroupShuffleUp(value, 4), value += rbx >= 4 ? temp : 0;
		subgroupMemoryBarrierShared();
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
				atomicOr(s_raster_error, 0xffffffff);
		}
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

	uvec2 rblock_pos = rasterBlockPos(rbid);
	uint lrbid = rbid & (NUM_WARPS - 1);
	uint tri_count = s_rblock_row_tri_counts[rblock_pos.y];
	uint buf_offset = group_rbid << (MAX_RBLOCK_TRIS0_SHIFT + group_shift);

	uint src_offset_32 = scratch32RBlockRowTrisOffset(rblock_pos.y);
	uint src_offset_64 = scratch64RBlockRowTrisOffset(rblock_pos.y);

	{
		uint bx_bits_shift = 24 + rblock_pos.x;
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
					DEBUG_RECORD(s_rblock_tri_counts[lrbid], block_tri_count, 0, tri_count);
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
	int startx = int(rblock_pos.x << RBLOCK_WIDTH_SHIFT);
	vec2 block_pos = vec2(rblock_pos << rasterBlockShift()) + vec2(s_bin_pos);

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
		s_buffer[buf_offset + i] = i | (RBLOCK_HEIGHT == 8 ? (depth >> 1) << 13 : depth << 12);
	}
	barrier();
	groupMemoryBarrier();
	sortBuffer(lrbid, tri_count, s_max_sort_rcount, buf_offset, group_size, group_thread, true);
	barrier();

#ifdef DEBUG_ENABLED
	for(uint i = LIX & WARP_MASK; i < tri_count; i += WARP_SIZE) {
		uint value = s_buffer[buf_offset + i];
		uint prev_value = i == 0 ? 0 : s_buffer[buf_offset + i - 1];
		if(value <= prev_value)
			DEBUG_RECORD(i, tri_count, prev_value, value);
	}
	barrier();
#endif

	for(uint i = group_thread; i < tri_count; i += group_size) {
		uint rblock_tri_idx = s_buffer[buf_offset + i] & (RBLOCK_HEIGHT == 8 ? 0x1fff : 0xfff);
#if RBLOCK_HEIGHT == 8
		uint num_frags = g_scratch_32[dst_offset_32 + rblock_tri_idx] >> 24;
#else
		uint num_frags = g_scratch_64[dst_offset_64 + rblock_tri_idx].y >> 24;
#endif

		// Computing triangle-ordered sample offsets within each block
		uint sum = subgroupInclusiveAddFast(num_frags);
		s_buffer[buf_offset + i] = (sum - num_frags) | (num_frags << 12) | (rblock_tri_idx << 19);
	}
	barrier();

	// Computing prefix sum across whole render-blocks. We're processing 4 elements
	// at a time, so that we can fit with MAX_RBLOCK_TRIS tris/hblock (128 warps).
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
				values[j] = (last & 0xfff) + ((last >> 12) & 0x7f);
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
		uint tri_value = (current >> 12) & 0x7f;
		uint rblock_tri_idx = current >> 19;

		uint mini_offset = group_rbid << (3 + group_shift);
		uint tri_offset = (current & 0xfff) + s_mini_buffer[mini_offset + (i >> WARP_SHIFT)];

		uint seg_offset = tri_offset & (SEGMENT_SIZE - 1);
		uint seg_id = tri_offset >> SEGMENT_SHIFT;
		bool first_seg = seg_offset == 0;
		if(seg_offset + tri_value > SEGMENT_SIZE)
			seg_id++, first_seg = true;
		if(seg_id >= MAX_SEGMENTS) {
			atomicOr(s_raster_error, 0xffffffff);
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
	if(s_raster_error != 0)
		return;

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
	if(num_segments > MAX_SEGMENTS)
		DEBUG_RECORD(num_segments, MAX_SEGMENTS, 0, 0);

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
	uint src_offset_64 = scratch64RBlockTrisOffset(lrbid);
	uint src_offset_32 = scratch32RBlockTrisOffset(lrbid);
	loadSamples(lrbid, segment_id, src_offset_32, src_offset_64, true);
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
		clearSegments();

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
			outputPixel(rasterBlockPixelPos(rbid), vec4(1.0, 0.0, 0.0, 0.0));
			barrier();
			if(LIX == 0)
				s_raster_error = 0;
			barrier();
			continue;
		}
		UPDATE_TIMER(1);

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

		ivec2 pixel_pos = rasterBlockPixelPos(rbid);
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
