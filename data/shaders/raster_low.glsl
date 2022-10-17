#define LSIZE 256
#define LSHIFT 8

#include "shared/raster.glsl"

#include "%shader_debug"
DEBUG_SETUP(1, 11)

#define MAX_BLOCK_ROW_TRIS 1024
#define MAX_BLOCK_TRIS 256
#define MAX_BLOCK_TRIS_SHIFT 8

#define MAX_SEGMENTS_SHIFT WARP_SHIFT
#define MAX_SEGMENTS WARP_SIZE

#define ACTIVE_RBLOCKS_MASK (RBLOCK_COUNT - 1)

layout(local_size_x = LSIZE) in;

// TODO: too much mem used
#define WORKGROUP_64_SCRATCH_SIZE (64 * 1024)
#define WORKGROUP_64_SCRATCH_SHIFT 16

#define WORKGROUP_32_SCRATCH_SIZE (64 * 1024)
#define WORKGROUP_32_SCRATCH_SHIFT 16

uint scratch64BlockRowTrisOffset(uint by) {
	return (gl_WorkGroupID.x << WORKGROUP_64_SCRATCH_SHIFT) + by * (MAX_BLOCK_ROW_TRIS * 2);
}

uint scratch64BlockTrisOffset(uint bid) {
	return (gl_WorkGroupID.x << WORKGROUP_64_SCRATCH_SHIFT) + 32 * 1024 +
		   bid * (MAX_BLOCK_TRIS * 2);
}

uint scratch64RasterBlockTrisOffset(uint rbid) {
	return (gl_WorkGroupID.x << WORKGROUP_64_SCRATCH_SHIFT) + 48 * 1024 + rbid * MAX_BLOCK_TRIS;
}

uint scratch32RasterBlockTrisOffset(uint rbid) {
	return (gl_WorkGroupID.x << WORKGROUP_32_SCRATCH_SHIFT) + rbid * MAX_BLOCK_TRIS;
}

shared int s_num_bins, s_bin_id;
shared uint s_bin_quad_count, s_bin_quad_offset;
shared uint s_bin_tri_count, s_bin_tri_offset;

shared uint s_block_row_tri_count[BLOCK_ROWS];
shared uint s_block_tri_count[NUM_WARPS];
shared uint s_rblock_counts[NUM_WARPS * 2];

shared int s_raster_error;
shared int s_promoted_bin_count;

void generateRowTris(uint tri_idx) {
	uint dst_offset_64 = scratch64BlockRowTrisOffset(0);

	uint scan_offset = STORAGE_TRI_SCAN_OFFSET + tri_idx * 2;
	uvec4 val0 = g_uvec4_storage[scan_offset + 0];
	uvec4 val1 = g_uvec4_storage[scan_offset + 1];
	int min_by = clamp(int(val0.w & 0xffff) - s_bin_pos.y, 0, BIN_MASK) >> BLOCK_SHIFT;
	int max_by = clamp(int(val0.w >> 16) - s_bin_pos.y, 0, BIN_MASK) >> BLOCK_SHIFT;

	vec2 start = vec2(s_bin_pos.x, s_bin_pos.y + min_by * BLOCK_SIZE);
	ScanlineParams scan = loadScanlineParamsRow(val0, val1, start);

	// TODO: is it worth it to make this loop more work-efficient?
	for(int by = min_by; by <= max_by; by++) {
		uvec3 bits0 = rasterBinStep(scan);
		uvec3 bits1 = rasterBinStep(scan);
		uint bx_mask = bits0.z | bits1.z;
		if(bx_mask == 0)
			continue;

		uint row_idx = atomicAdd(s_block_row_tri_count[by], 1);
		if(row_idx >= MAX_BLOCK_ROW_TRIS) {
			atomicOr(s_raster_error, 0xffffffff);
			return;
		}

		uint roffset = row_idx + by * (MAX_BLOCK_ROW_TRIS * 2);
		g_scratch_64[dst_offset_64 + roffset] =
			uvec2(bits0.x | (bx_mask << 24), bits1.x | ((tri_idx & 0xff0000) << 8));
		g_scratch_64[dst_offset_64 + roffset + MAX_BLOCK_ROW_TRIS] =
			uvec2(bits0.y | ((tri_idx & 0xff) << 24), bits1.y | ((tri_idx & 0xff00) << 16));
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
		generateRowTris(quad_idx * 2 + second_tri);
	}

	for(uint i = (LSIZE - 1) - LIX; i < s_bin_tri_count; i += LSIZE)
		generateRowTris(g_bin_tris[s_bin_tri_offset + i]);
}

shared uint s_sort_rcount[NUM_WARPS];

void prepareSortTris() {
	if(LIX < NUM_WARPS) {
		uint count = s_block_tri_count[LIX];
		// rcount: count rounded up to next power of 2; minimum: WARP_SIZE
		uint rcount = max(WARP_SIZE, (count & (count - 1)) == 0 ? count : (2 << findMSB(count)));
		s_sort_rcount[LIX] = rcount;
	}
}

uint swap(uint x, int mask, uint dir) {
	uint y = subgroupShuffleXor(x, mask);
	return uint(x < y) == dir ? y : x;
}
uint bitExtract(uint value, int boffset) { return (value >> boffset) & 1; }
uint xorBits(uint value, int bit0, int bit1) { return ((value >> bit0) ^ (value >> bit1)) & 1; }

void sortTris(uint lbid, uint count, uint buf_offset) {
	uint lid = LIX & WARP_MASK;
	uint rcount = s_sort_rcount[lbid];
	for(uint i = lid + count; i < rcount; i += WARP_SIZE)
		s_buffer[buf_offset + i] = 0xffffffff;

	for(uint i = lid; i < rcount; i += WARP_SIZE) {
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
#if WARP_SIZE >= 64
		value = swap(value, 0x10, xorBits(lid, 5, 4)); // K = 32
		value = swap(value, 0x08, xorBits(lid, 5, 3));
		value = swap(value, 0x04, xorBits(lid, 5, 2));
		value = swap(value, 0x02, xorBits(lid, 5, 1));
		value = swap(value, 0x01, xorBits(lid, 5, 0));
#endif
		s_buffer[buf_offset + i] = value;
	}
	int start_k = WARP_SIZE, end_j = WARP_SIZE;

	for(uint k = start_k; k <= rcount; k = 2 * k) {
		for(uint j = k >> 1; j >= end_j; j = j >> 1) {
			for(uint i = lid; i < rcount; i += WARP_SIZE * 2) {
				uint idx = (i & j) != 0 ? i + WARP_SIZE - j : i;
				uint lvalue = s_buffer[buf_offset + idx];
				uint rvalue = s_buffer[buf_offset + idx + j];
				if(((idx & k) != 0) == (lvalue.x < rvalue.x)) {
					s_buffer[buf_offset + idx] = rvalue;
					s_buffer[buf_offset + idx + j] = lvalue;
				}
			}
		}
		for(uint i = lid; i < rcount; i += WARP_SIZE) {
			uint bit = (i & k) == 0 ? 0 : 1;
			uint value = s_buffer[buf_offset + i];
#if WARP_SIZE == 64
			value = swap(value, 0x20, bit ^ bitExtract(lid, 5));
#endif
			value = swap(value, 0x10, bit ^ bitExtract(lid, 4));
			value = swap(value, 0x08, bit ^ bitExtract(lid, 3));
			value = swap(value, 0x04, bit ^ bitExtract(lid, 2));
			value = swap(value, 0x02, bit ^ bitExtract(lid, 1));
			value = swap(value, 0x01, bit ^ bitExtract(lid, 0));
			s_buffer[buf_offset + i] = value;
		}
	}
}

// TODO: maybe process smaller amount of blocks at the same time?
// smaller chance that it will leave cache
void generateBlocks(uint bid) {
	int lbid = int(LIX >> WARP_SHIFT);
	bid += lbid;
	uint by = bid >> BLOCK_ROWS_SHIFT, bx = bid & BLOCK_ROWS_MASK;

	uint src_offset_64 = scratch64BlockRowTrisOffset(by);
	uint tri_count = s_block_row_tri_count[by];
	uint buf_offset = lbid << MAX_BLOCK_TRIS_SHIFT;

	clearSegments();

	{
		uint bx_bits_shift = 24 + bx, block_tri_count = 0;
		// In some cases on integrated AMD gpu atomic-based version is faster
		for(uint i = gl_SubgroupInvocationID; i < tri_count; i += gl_SubgroupSize) {
			uint bx_bit = (g_scratch_64[src_offset_64 + i].x >> bx_bits_shift) & 1;

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
				if(tri_offset < MAX_BLOCK_TRIS)
					s_buffer[buf_offset + tri_offset] = i;
			}
			block_tri_count += bit_count;
		}

		if(gl_SubgroupInvocationID == 0) {
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
	int startx = int(bx << BLOCK_SHIFT);
	vec2 block_pos = vec2(s_bin_pos + ivec2(bx << BLOCK_SHIFT, by << BLOCK_SHIFT));

	for(uint i = LIX & WARP_MASK; i < tri_count; i += WARP_SIZE) {
		uint row_idx = s_buffer[buf_offset + i];

		uvec2 tri_mins = g_scratch_64[src_offset_64 + row_idx];
		uvec2 tri_maxs = g_scratch_64[src_offset_64 + row_idx + MAX_BLOCK_ROW_TRIS];
		uint tri_idx =
			(tri_maxs.x >> 24) | ((tri_maxs.y >> 16) & 0xff00) | ((tri_mins.y >> 8) & 0xff0000);

		uvec2 bits;
		uvec2 num_frags;
		vec2 cpos = vec2(0);

		bits.x = rasterBlock(tri_mins.x, tri_maxs.x, startx, num_frags.x, cpos);
		bits.y = rasterBlock(tri_mins.y, tri_maxs.y, startx, num_frags.y, cpos);

		uint num_all_frags = num_frags.x + num_frags.y;
		uint depth = rasterBlockDepth(cpos * (0.5 / float(num_all_frags)) + block_pos, tri_idx);

		if(num_all_frags == 0) // This means that bx_mask is invalid
			DEBUG_RECORD(0, 0, 0, 0);
		uint counts =
			WARP_SIZE == 64 ? num_frags.x + num_frags.y : num_frags.x | (num_frags.y << 6);
		g_scratch_64[dst_offset_64 + i] = bits;
		g_scratch_64[dst_offset_64 + i + MAX_BLOCK_TRIS] = uvec2(tri_idx, counts);

		// 12 bits for tile-tri index, 20 bits for depth
		s_buffer[buf_offset + i] = i | (depth << 12);
	}
	barrier();

	if(tri_count > RC_COLOR_SIZE)
		sortTris(lbid, tri_count, buf_offset);

	barrier();
	groupMemoryBarrier();

#ifdef DEBUG_ENABLED
	// Making sure that tris are properly ordered
	if(tri_count > 3)
		for(uint i = LIX & WARP_MASK; i < tri_count; i += WARP_SIZE) {
			uint value = s_buffer[buf_offset + i];
			uint prev_value = i == 0 ? 0 : s_buffer[buf_offset + i - 1];
			if(value <= prev_value)
				DEBUG_RECORD(i, tri_count, prev_value, value);
		}
#endif

	for(uint i = LIX & WARP_MASK; i < tri_count; i += WARP_SIZE) {
		uint idx = s_buffer[buf_offset + i] & 0xff;
		uint counts = g_scratch_64[dst_offset_64 + idx + MAX_BLOCK_TRIS].y;
#if WARP_SIZE == 32
		uint num_frags1 = counts & 63, num_frags2 = (counts >> 6) & 63;
		uint num_frags = num_frags1 | (num_frags2 << 12);
#else
		uint num_frags = counts;
#endif
		// Computing triangle-ordered sample offsets within each block
		num_frags = subgroupInclusiveAddFast(num_frags);
		s_buffer[buf_offset + i] = num_frags | (idx << 24);
	}
	barrier();

	// Computing prefix sum across whole blocks
	if(LIX < NUM_WARPS * NUM_WARPS) {
		uint lbid = LIX >> NUM_WARPS_SHIFT, warp_idx = LIX & NUM_WARPS_MASK;
		uint warp_offset = warp_idx << WARP_SHIFT;
		uint buf_offset = lbid << MAX_BLOCK_TRIS_SHIFT;
		uint tri_count = s_block_tri_count[lbid];
		uint value = 0;

		if(warp_offset < tri_count) {
			uint block_tri_idx = min(warp_offset + WARP_MASK, tri_count - 1);
			value = s_buffer[buf_offset + block_tri_idx];
		}
		value = WARP_SIZE == 64 ? value & 0xffffff : (value & 0xfff) | ((value & 0xfff000) << 4);
		uint sum = value, temp;
		temp = subgroupShuffleUp(sum, 1), sum += warp_idx >= 1 ? temp : 0;
		temp = subgroupShuffleUp(sum, 2), sum += warp_idx >= 2 ? temp : 0;
#if NUM_WARPS > 4
		temp = subgroupShuffleUp(sum, 4), sum += warp_idx >= 4 ? temp : 0;
#endif

		if(warp_idx == NUM_WARPS_MASK) {
#if WARP_SIZE == 32
			uint v0 = sum & 0xffff, v1 = sum >> 16;
			uint max_segs = (max(v0, v1) + SEGMENT_SIZE - 1) >> SEGMENT_SHIFT;
			updateStats(int(v0 + v1), int(tri_count * 2));
			s_rblock_counts[lbid * 2 + 0] = (v0 << 16) | tri_count;
			s_rblock_counts[lbid * 2 + 1] = (v1 << 16) | tri_count;
#else
			uint max_segs = (sum + SEGMENT_SIZE - 1) >> SEGMENT_SHIFT;
			s_rblock_counts[lbid] = (sum << 16) | tri_count;
			updateStats(int(sum), int(tri_count * 2));
#endif
			if(max_segs > MAX_SEGMENTS)
				atomicOr(s_raster_error, 0xffffffff);
		}

		s_mini_buffer[LIX] = sum - value;
	}
	barrier();
	if(s_raster_error != 0)
		return;

		// Storing triangle fragment offsets to scratch mem
		// Also finding first triangle for each segment

#if WARP_SIZE == 32
	src_offset_64 = dst_offset_64;
	dst_offset_64 = scratch64RasterBlockTrisOffset(lbid << 1);

	uint seg_block1_offset = lbid << (MAX_SEGMENTS_SHIFT + 1);
	uint seg_block2_offset = seg_block1_offset + MAX_SEGMENTS;
	for(uint i = LIX & WARP_MASK; i < tri_count; i += WARP_SIZE) {
		uint tri_offset = 0;
		if(i > 0) {
			uint prev = i - 1;
			tri_offset = s_buffer[buf_offset + prev] & 0xffffff;
			tri_offset = (tri_offset & 0xfff) | ((tri_offset & 0xfff000) << 4);
			tri_offset += s_mini_buffer[(lbid << NUM_WARPS_SHIFT) + (prev >> WARP_SHIFT)];
		}

		uint tri_value = s_buffer[buf_offset + i];
		uint block_tri_idx = tri_value >> 24;
		tri_value = (tri_value & 0xfff) | ((tri_value & 0xfff000) << 4);
		tri_value =
			(tri_value + s_mini_buffer[(lbid << NUM_WARPS_SHIFT) + (i >> WARP_SHIFT)]) - tri_offset;

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

		uint tri_idx = g_scratch_64[src_offset_64 + block_tri_idx + MAX_BLOCK_TRIS].x;
		uvec2 tri_data = g_scratch_64[src_offset_64 + block_tri_idx];
		g_scratch_64[dst_offset_64 + i] = uvec2(tri_idx | seg_offset0, tri_data.x);
		g_scratch_64[dst_offset_64 + i + MAX_BLOCK_TRIS] = uvec2(tri_idx | seg_offset1, tri_data.y);
	}
#else
	src_offset_64 = dst_offset_64;
	dst_offset_64 = scratch64RasterBlockTrisOffset(lbid);
	uint dst_offset_32 = scratch32RasterBlockTrisOffset(lbid);

	uint seg_block_offset = lbid << MAX_SEGMENTS_SHIFT;
	for(uint i = LIX & WARP_MASK; i < tri_count; i += WARP_SIZE) {
		uint tri_offset = 0;
		if(i > 0) {
			uint prev = i - 1;
			tri_offset = s_buffer[buf_offset + prev] & 0xffffff;
			tri_offset += s_mini_buffer[(lbid << NUM_WARPS_SHIFT) + (prev >> WARP_SHIFT)];
		}

		uint tri_value = s_buffer[buf_offset + i];
		uint block_tri_idx = tri_value >> 24;
		tri_value &= 0xffffff;
		tri_value =
			(tri_value + s_mini_buffer[(lbid << NUM_WARPS_SHIFT) + (i >> WARP_SHIFT)]) - tri_offset;
		uint seg_offset = tri_offset & (SEGMENT_SIZE - 1);
		uint seg_id = tri_offset >> SEGMENT_SHIFT;
		bool first_seg = seg_offset == 0;
		if(seg_offset + tri_value > SEGMENT_SIZE)
			seg_id++, first_seg = true;
		seg_offset <<= 24;
		if(first_seg && tri_value > 0)
			s_segments[seg_block_offset + seg_id] = i | seg_offset;

		uint tri_idx = g_scratch_64[src_offset_64 + block_tri_idx + MAX_BLOCK_TRIS].x & 0xffffff;
		uvec2 tri_data = g_scratch_64[src_offset_64 + block_tri_idx];
		g_scratch_32[dst_offset_32 + i] = tri_idx | seg_offset;
		g_scratch_64[dst_offset_64 + i] = tri_data;
	}
#endif
}

void finalizeSegments() {
	uint rbid = LIX >> 4, seg_group_offset = rbid << MAX_SEGMENTS_SHIFT;
	if(rbid < RBLOCK_COUNT) {
		uint counts = s_rblock_counts[rbid];
		uint num_segments = ((counts >> 16) + SEGMENT_SIZE - 1) >> SEGMENT_SHIFT;
		uint tri_count = counts & 0xffff;
		if(num_segments > MAX_SEGMENTS)
			DEBUG_RECORD(num_segments, MAX_SEGMENTS, 0, 0);

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
}

void loadSamples(uint rbid, int segment_id) {
#if WARP_SIZE == 32
	uint lrbid = rbid & ACTIVE_RBLOCKS_MASK;
	uint seg_group_offset = lrbid << MAX_SEGMENTS_SHIFT;
	uint segment_data = s_segments[seg_group_offset + segment_id];
	uint tri_count = segment_data >> 16, first_tri = segment_data & 0xffff;
	uint src_offset_64 = scratch64RasterBlockTrisOffset(lrbid) + first_tri;
	uint buf_offset = (LIX >> 5) << SEGMENT_SHIFT;

	int y = int(LIX & 3), row_shift = (y & 3) * 4;
	int mask1 = y >= 1 ? ~0 : 0, mask2 = y >= 2 ? ~0 : 0;

	for(uint i = (LIX & WARP_MASK) >> 2; i < tri_count; i += WARP_SIZE / 4) {
		uvec2 tri_data = g_scratch_64[src_offset_64 + i];
		int tri_offset = int(tri_data.x >> 24);
		if(i == 0 && tri_offset != 0)
			tri_offset -= SEGMENT_SIZE;
		uint tri_idx = tri_data.x & 0xffffff;

		uint row_data = tri_data.y >> row_shift;
		int minx = int(row_data & 15), countx = int((row_data >> 16) & 15);
		int prevx = countx + (subgroupShuffleUp(countx, 1) & mask1);
		prevx += (subgroupShuffleUp(prevx, 2) & mask2);
		tri_offset += prevx - countx;
		countx = min(countx, SEGMENT_SIZE - tri_offset);
		if(tri_offset < 0)
			countx += tri_offset, minx -= tri_offset, tri_offset = 0;

		uint pixel_id = (y << 3) | minx;
		uint value = pixel_id | (tri_idx << 8);
		for(int j = 0; j < countx; j++)
			s_buffer[buf_offset + tri_offset++] = value++;
	}
#else
	uint lrbid = rbid & ACTIVE_RBLOCKS_MASK;
	uint seg_group_offset = lrbid << MAX_SEGMENTS_SHIFT;
	uint segment_data = s_segments[seg_group_offset + segment_id];
	uint tri_count = segment_data >> 16, first_tri = segment_data & 0xffff;
	uint src_offset_32 = scratch32RasterBlockTrisOffset(lrbid) + first_tri;
	uint src_offset_64 = scratch64RasterBlockTrisOffset(lrbid) + first_tri;
	uint buf_offset = (LIX >> 6) << SEGMENT_SHIFT;

	int y = int(LIX & 7), row_shift = (y & 3) * 4;
	int mask1 = y >= 1 ? ~0 : 0, mask2 = y >= 2 ? ~0 : 0, mask3 = y >= 4 ? ~0 : 0;

	for(uint i = (LIX & WARP_MASK) >> 3; i < tri_count; i += WARP_SIZE / 8) {
		uint tri_data = g_scratch_64[src_offset_64 + i][y >> 2];
		uint tri_info = g_scratch_32[src_offset_32 + i];
		int tri_offset = int(tri_info >> 24);
		if(i == 0 && tri_offset != 0)
			tri_offset -= SEGMENT_SIZE;
		uint tri_idx = tri_info & 0xffffff;

		uint row_data = tri_data >> row_shift;
		int minx = int(row_data & 15), countx = int((row_data >> 16) & 15);
		int prevx = countx + (subgroupShuffleUp(countx, 1) & mask1);
		prevx += (subgroupShuffleUp(prevx, 2) & mask2);
		prevx += (subgroupShuffleUp(prevx, 4) & mask3);

		tri_offset += prevx - countx;
		countx = min(countx, SEGMENT_SIZE - tri_offset);
		if(tri_offset < 0)
			countx += tri_offset, minx -= tri_offset, tri_offset = 0;

		uint pixel_id = (y << 3) | minx;
		uint value = pixel_id | (tri_idx << 8);
		for(int j = 0; j < countx; j++)
			s_buffer[buf_offset + tri_offset++] = value++;
	}
#endif
	subgroupMemoryBarrierShared();
}

void visualizeBlockCounts(uint rbid, ivec2 pixel_pos) {
	uint lrbid = rbid & (WARP_SIZE == 64 ? NUM_WARPS - 1 : NUM_WARPS * 2 - 1);
	uint lbid = (WARP_SIZE == 64 ? rbid : rbid >> 1) & (NUM_WARPS - 1);
	uint frag_count = (WARP_SIZE == 64 ? 1 : 2) * (s_rblock_counts[lrbid] >> 16);
	uint tri_count = s_block_tri_count[lbid];
	//tri_count = s_block_row_tri_count[pixel_pos.y >> BLOCK_SHIFT];
	//tri_count = s_bin_quad_count * 2 + s_bin_tri_count;

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
		uint src_offset_64 = scratch64RasterBlockTrisOffset(lrbid) + seg_first_tri;
		for(uint j = 0; j < seg_tri_count; j++) {
			uvec2 tri_data = g_scratch_64[src_offset_64 + j];
			uint count0 = tri_data.x >> 16, count1 = tri_data.y >> 16;
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

	if(LIX < BLOCK_ROWS) {
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

		s_block_row_tri_count[LIX] = 0;
	}
	barrier();
	processQuads();
	groupMemoryBarrier();
	barrier();
	UPDATE_TIMER(0);

	const int num_rblocks = (BIN_SIZE / RBLOCK_WIDTH) * (BIN_SIZE / RBLOCK_HEIGHT);

	//  bid: block (8x8) id; We have 8 x 8 = 64 blocks
	// rbid: half block (8x4) id; We have 8 x 16 = 128 half blocks
	//       half blocks which make a single block are stored one after another
	//       (upper block first)
	// lbid: local block id (range: 0 up to NUM_WARPS - 1)
	// Each block has 64 pixels, so we need 2 warps to process all pixels within a single block
	for(uint rbid = LIX >> WARP_SHIFT; rbid < num_rblocks; rbid += NUM_WARPS) {
		barrier();
		if(WARP_SIZE == 64 || (rbid & NUM_WARPS) == 0) {
			uint bid = (rbid & ~(NUM_WARPS - 1)) >> (WARP_SIZE == 64 ? 0 : 1);
			generateBlocks(bid);
			barrier();
			// raster_low errors are not visualized, but propagated to high
			if(s_raster_error != 0) {
				if(LIX == 0) {
					int id = atomicAdd(g_info.bin_level_counts[BIN_LEVEL_HIGH], 1);
					HIGH_LEVEL_BINS(id) = int(bin_id);
					s_promoted_bin_count = max(s_promoted_bin_count, id + 1);
				}
				return;
			}
			finalizeSegments();
			groupMemoryBarrier();
			barrier();
		}
		UPDATE_TIMER(1);

		ReductionContext context;
		initReduceSamples(context);
		//initVisualizeSamples();

		for(int segment_id = 0;; segment_id++) {
			uint counts = s_rblock_counts[rbid & ACTIVE_RBLOCKS_MASK];
			int frag_count = min(SEGMENT_SIZE, int(counts >> 16) - (segment_id << SEGMENT_SHIFT));
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
		uint bin_idx = atomicAdd(g_info.a_small_bins, 1);
		s_bin_id = bin_idx < s_num_bins ? LOW_LEVEL_BINS(bin_idx) : -1;
	}
	barrier();
	return s_bin_id;
}

void main() {
	INIT_TIMERS();
	if(LIX == 0) {
		s_num_bins = g_info.bin_level_counts[BIN_LEVEL_LOW];
		s_promoted_bin_count = 0;
	}
	initStats();

	int bin_id = loadNextBin();
	while(bin_id != -1) {
		barrier();
		rasterBin(bin_id);
		bin_id = loadNextBin();
	}

	// If some of the bins are promoted to the next level, we have to adjust number of dispatches
	if(LIX == 0 && s_promoted_bin_count > 0) {
		uint num_dispatches = min(s_promoted_bin_count, MAX_DISPATCHES / 2);
		atomicMax(g_info.bin_level_dispatches[BIN_LEVEL_HIGH][0], num_dispatches);
	}

	COMMIT_TIMERS(g_info.raster_timers);
	commitStats();
}
