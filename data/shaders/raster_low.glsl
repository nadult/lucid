#define LSIZE 256
#define LSHIFT 8

#include "shared/raster.glsl"

#include "%shader_debug"
DEBUG_SETUP(1, 11)

#define MAX_BLOCK_ROW_TRIS 1024
#define MAX_BLOCK_TRIS 256
#define MAX_BLOCK_TRIS_SHIFT 8

layout(local_size_x = LSIZE) in;

// TODO: too much mem used
#define WORKGROUP_SCRATCH_SIZE (64 * 1024)
#define WORKGROUP_SCRATCH_SHIFT 16

uint scratch64BlockRowTrisOffset(uint by) {
	return (gl_WorkGroupID.x << WORKGROUP_SCRATCH_SHIFT) + by * (MAX_BLOCK_ROW_TRIS * 2);
}

uint scratch64BlockTrisOffset(uint bid) {
	return (gl_WorkGroupID.x << WORKGROUP_SCRATCH_SHIFT) + 32 * 1024 + bid * (MAX_BLOCK_TRIS * 2);
}

uint scratchRasterBlockTrisOffset(uint rbid) {
	return (gl_WorkGroupID.x << WORKGROUP_SCRATCH_SHIFT) + 48 * 1024 + rbid * MAX_BLOCK_TRIS;
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

// TODO: maybe process smaller amount of blocks at the same time?
// smaller chance that it will leave cache
void generateBlocks(uint bid) {
	int lbid = int(LIX >> WARP_SHIFT);
	bid += lbid;
	uint by = bid >> BLOCK_ROWS_SHIFT, bx = bid & BLOCK_ROWS_MASK;

	uint src_offset_64 = scratch64BlockRowTrisOffset(by);
	uint tri_count = s_block_row_tri_count[by];
	uint buf_offset = lbid << MAX_BLOCK_TRIS_SHIFT;
	const uint mini_offset = BASE_BUFFER_SIZE;

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
		sortBuffer(lbid, tri_count, s_sort_rcount[lbid], buf_offset, WARP_SIZE, LIX & WARP_MASK,
				   false);

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
			updateStats(int(v0 + v1), int(tri_count * 2));
			uint lrbid = blockIdToRaster(lbid);
			s_rblock_counts[lrbid] = (v0 << 16) | tri_count;
			s_rblock_counts[lrbid + RBLOCK_COLS] = (v1 << 16) | tri_count;
#else
			s_rblock_counts[lbid] = (sum << 16) | tri_count;
			updateStats(int(sum), int(tri_count * 2));
#endif
		}

		s_buffer[mini_offset + LIX] = sum - value;
	}
	barrier();
	if(s_raster_error != 0)
		return;

		// Storing triangle fragment offsets to scratch mem
		// Also finding first triangle for each segment

#if WARP_SIZE == 32
	src_offset_64 = dst_offset_64;
	uint lrbid0 = blockIdToRaster(lbid);
	uint lrbid1 = lrbid0 + RBLOCK_COLS;
	uint dst_offset0 = scratchRasterBlockTrisOffset(lrbid0);
	uint dst_offset1 = scratchRasterBlockTrisOffset(lrbid1);
	for(uint i = LIX & WARP_MASK; i < tri_count; i += WARP_SIZE) {
		uint tri_offset = 0;
		if(i > 0) {
			uint prev = i - 1;
			tri_offset = s_buffer[buf_offset + prev] & 0xffffff;
			tri_offset = (tri_offset & 0xfff) | ((tri_offset & 0xfff000) << 4);
			tri_offset += s_buffer[mini_offset + (lbid << NUM_WARPS_SHIFT) + (prev >> WARP_SHIFT)];
		}

		uint tri_value = s_buffer[buf_offset + i];
		uint block_tri_idx = tri_value >> 24;
		tri_value = (tri_value & 0xfff) | ((tri_value & 0xfff000) << 4);
		tri_value =
			(tri_value + s_buffer[mini_offset + (lbid << NUM_WARPS_SHIFT) + (i >> WARP_SHIFT)]) -
			tri_offset;

		uint tri_offset0 = tri_offset & 0xffff, tri_offset1 = tri_offset >> 16;
		uint segment_bits0 = (tri_offset0 & 0xf00) << 20, segment_bits1 = (tri_offset1 & 0xf00)
																		  << 20;

		uint tri_idx = g_scratch_64[src_offset_64 + block_tri_idx + MAX_BLOCK_TRIS].x << 8;
		uvec2 tri_data = g_scratch_64[src_offset_64 + block_tri_idx];
		g_scratch_64[dst_offset0 + i] =
			uvec2(tri_idx | (tri_offset0 & 0xff), tri_data.x | segment_bits0);
		g_scratch_64[dst_offset1 + i] =
			uvec2(tri_idx | (tri_offset1 & 0xff), tri_data.y | segment_bits1);
	}
#else
	src_offset_64 = dst_offset_64;
	uint dst_offset = scratchRasterBlockTrisOffset(lbid);

	for(uint i = LIX & WARP_MASK; i < tri_count; i += WARP_SIZE) {
		uint tri_offset = 0;
		if(i > 0) {
			uint prev = i - 1;
			tri_offset = s_buffer[buf_offset + prev] & 0xffffff;
			tri_offset += s_buffer[mini_offset + (lbid << NUM_WARPS_SHIFT) + (prev >> WARP_SHIFT)];
		}
		uint tri_value = s_buffer[buf_offset + i];
		uint block_tri_idx = tri_value >> 24;

		uint tri_idx = g_scratch_64[src_offset_64 + block_tri_idx + MAX_BLOCK_TRIS].x & 0xffffff;
		uvec2 tri_data = g_scratch_64[src_offset_64 + block_tri_idx];
		g_scratch_32[dst_offset + i] = tri_idx | (tri_offset & 0xff);
		g_scratch_64[dst_offset + i] = tri_data;
	}
#endif
}

void visualizeBlockCounts(uint rbid, ivec2 pixel_pos) {
	uint lrbid = rbid & (RBLOCK_COUNT - 1), lbid = blockIdFromRaster(lrbid);
	uint frag_count = (WARP_SIZE == 64 ? 1 : 2) * (s_rblock_counts[lrbid] >> 16);
	uint tri_count = s_block_tri_count[lbid];
	//tri_count = s_block_row_tri_count[pixel_pos.y >> BLOCK_SHIFT];
	//tri_count = s_bin_quad_count * 2 + s_bin_tri_count;

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
	barrier(); // TODO: stall (7%, conference)
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
			groupMemoryBarrier();
			barrier(); // TODO: stall (2.5%, conference)
		}
		UPDATE_TIMER(1);

		ReductionContext context;
		initReduceSamples(context);
		//initVisualizeSamples();

		uint cur_tri_idx = 0;
		for(int segment_id = 0;; segment_id++) {
			uint counts = s_rblock_counts[rbid & (RBLOCK_COUNT - 1)];
			int frag_count = min(SEGMENT_SIZE, int(counts >> 16) - segment_id * SEGMENT_SIZE);
			if(frag_count <= 0)
				break;

			uint lrbid = rbid & (RBLOCK_COUNT - 1);
			uint src_offset = scratchRasterBlockTrisOffset(lrbid);
			loadSamples(lrbid, cur_tri_idx, segment_id, counts, src_offset);
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
		barrier(); // TODO: stall (10.5%, conference)
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
