#define LSIZE 256
#define LSHIFT 8

#define BIN_LEVEL BIN_LEVEL_LOW

#include "shared/raster.glsl"

#include "%shader_debug"
DEBUG_SETUP(1, 11)

#define MAX_BLOCK_ROW_TRIS 1024
#define MAX_BLOCK_TRIS 256
#define MAX_BLOCK_TRIS_SHIFT 8

layout(local_size_x = LSIZE) in;

// TODO: too much mem used
#define WORKGROUP_SCRATCH_SIZE (16 * 1024)
#define WORKGROUP_SCRATCH_SHIFT 14

// More space needed only for 64x64 bins
uint scratchBlockRowOffset(uint by) {
	return (gl_WorkGroupID.x << WORKGROUP_SCRATCH_SHIFT) + by * (MAX_BLOCK_ROW_TRIS * 2);
}

// TODO: It could overlap with block row data
uint scratchRasterBlockOffset(uint rbid) {
	return (gl_WorkGroupID.x << WORKGROUP_SCRATCH_SHIFT) + 8 * 1024 + rbid * MAX_BLOCK_TRIS;
}

shared uint s_block_row_tri_count[BLOCK_ROWS];
shared uint s_block_tri_count[NUM_BLOCKS];
shared uint s_rblock_counts[NUM_RBLOCKS];
shared int s_promoted_bin_count;

void generateRowTris(uint tri_idx) {
	uint dst_offset = scratchBlockRowOffset(0);

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
		uint roffset = row_idx + by * (MAX_BLOCK_ROW_TRIS * 2);
		g_scratch_64[dst_offset + roffset] = uvec2(bits0.x | (bx_mask << 24), bits1.x);
		g_scratch_64[dst_offset + roffset + MAX_BLOCK_ROW_TRIS] =
			uvec2(bits0.y | ((tri_idx & 0xfff) << 20), bits1.y | ((tri_idx & 0xfff000) << 8));
	}
}

void processQuads() {
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

// TODO: maybe process smaller amount of blocks at the same time?
// smaller chance that it will leave cache
void generateBlocks(uint bid) {
	int lbid = int(LIX >> RASTER_SUBGROUP_SHIFT);
	uint by = bid >> BLOCK_ROWS_SHIFT, bx = bid & BLOCK_ROWS_MASK;

	uint rows_offset = scratchBlockRowOffset(by);
	uint tri_count = s_block_row_tri_count[by];
	uint buf_offset = lbid << MAX_BLOCK_TRIS_SHIFT;
	const uint mini_offset = BASE_BUFFER_SIZE;

	{
		uint bx_bits_mask = 1u << (24 + bx), tri_offset = 0;
		for(uint i = LIX & RASTER_SUBGROUP_MASK; i < tri_count; i += RASTER_SUBGROUP_SIZE) {
			uint bx_bits = g_scratch_64[rows_offset + i].x;
			if((bx_bits & bx_bits_mask) != 0) {
				tri_offset = atomicAdd(s_block_tri_count[bid], 1);
				if(tri_offset < MAX_BLOCK_TRIS)
					s_buffer[buf_offset + tri_offset] = i;
			}
		}
		subgroupMemoryBarrierShared();
		if(subgroupAny(tri_offset >= MAX_BLOCK_TRIS)) {
			if(gl_SubgroupInvocationID == 0)
				s_raster_error = ~0;
			return;
		}
	}

	tri_count = s_block_tri_count[bid];
	int startx = int(bx << BLOCK_SHIFT);
	vec2 block_pos = vec2(s_bin_pos + ivec2(bx << BLOCK_SHIFT, by << BLOCK_SHIFT));

	uint frag_count = 0;
	for(uint i = LIX & RASTER_SUBGROUP_MASK; i < tri_count; i += RASTER_SUBGROUP_SIZE) {
		uint row_tri_idx = s_buffer[buf_offset + i];

		uvec2 tri_mins = g_scratch_64[rows_offset + row_tri_idx];
		uvec2 tri_maxs = g_scratch_64[rows_offset + row_tri_idx + MAX_BLOCK_ROW_TRIS];
		uint tri_idx = (tri_maxs.x >> 20) | ((tri_maxs.y & 0xfff00000) >> 8);

		uvec2 num_frags;
		vec2 cpos = rasterHalfBlockCentroid(tri_mins.x, tri_maxs.x, startx, num_frags.x) +
					rasterHalfBlockCentroid(tri_mins.y, tri_maxs.y, startx, num_frags.y);

		uint num_block_frags = num_frags.x + num_frags.y;
		uint depth = rasterBlockDepth(cpos * (0.5 / float(num_block_frags)) + block_pos, tri_idx);
		if(num_block_frags == 0) // This means that bx_mask is invalid
			DEBUG_RECORD(0, 0, 0, 0);
		frag_count += num_frags.x | (num_frags.y << 16);

		// 12 bits for tile-tri index, 20 bits for depth
		s_buffer[buf_offset + i] = row_tri_idx | (depth << 12);
	}
	subgroupMemoryBarrier();

	frag_count = subgroupInclusiveAddFast32(frag_count);
	if((LIX & RASTER_SUBGROUP_MASK) == RASTER_SUBGROUP_MASK) {
		// TODO: separate tri_count & frag_count; use same counters in high & low?
		uint rbid = blockIdToRender(lbid + (bid & ~(NUM_RASTER_SUBGROUPS - 1)));
		uint v0 = frag_count & 0xffff, v1 = frag_count >> 16;
		s_rblock_counts[rbid] = (v0 << 16) | tri_count;
		s_rblock_counts[rbid + RBLOCK_COLS] = (v1 << 16) | tri_count;
	}

	if(tri_count > RC_COLOR_SIZE) {
		// rcount: count rounded up to the next power of 2; minimum: RASTER_SUBGROUP_SIZE
		uint rcount =
			max(RASTER_SUBGROUP_SIZE,
				(tri_count & (tri_count - 1)) == 0 ? tri_count : (2 << findMSB(tri_count)));
		sortBuffer(tri_count, rcount, buf_offset, RASTER_SUBGROUP_SIZE, LIX & RASTER_SUBGROUP_MASK,
				   false);
	}
	subgroupMemoryBarrierShared();

	// TODO: move to sortBuffer()
#ifdef DEBUG_ENABLED
	// Making sure that tris are properly ordered
	if(tri_count > RC_COLOR_SIZE)
		for(uint i = LIX & RASTER_SUBGROUP_MASK; i < tri_count; i += RASTER_SUBGROUP_SIZE) {
			uint value = s_buffer[buf_offset + i];
			uint prev_value = i == 0 ? 0 : s_buffer[buf_offset + i - 1];
			if(value <= prev_value)
				DEBUG_RECORD(i, tri_count, prev_value, value);
		}
#endif

	uint rbid0 = blockIdToRender(bid), rbid1 = rbid0 + RBLOCK_COLS;
	uint dst_offset0 = scratchRasterBlockOffset(rbid0);
	uint dst_offset1 = scratchRasterBlockOffset(rbid1);

	uint base_offset = 0;
	for(uint i = LIX & RASTER_SUBGROUP_MASK; i < tri_count; i += RASTER_SUBGROUP_SIZE) {
		uint row_idx = s_buffer[buf_offset + i] & 0xfff;
		uvec2 tri_mins = g_scratch_64[rows_offset + row_idx];
		uvec2 tri_maxs = g_scratch_64[rows_offset + row_idx + MAX_BLOCK_ROW_TRIS];
		uint tri_idx_shifted = ((tri_maxs.x >> 12) & 0xfff00) | (tri_maxs.y & 0xfff00000);
		uvec2 num_frags_half;
		uvec2 bits = uvec2(rasterHalfBlockBits(tri_mins.x, tri_maxs.x, startx, num_frags_half.x),
						   rasterHalfBlockBits(tri_mins.y, tri_maxs.y, startx, num_frags_half.y));
		uint num_frags = num_frags_half.x | (num_frags_half.y << 16);

		uint num_frags_accum = subgroupInclusiveAddFast32(num_frags);
		uint cur_offset = base_offset + num_frags_accum - num_frags;

#if WARP_SIZE == RASTER_SUBGROUP_SIZE
		base_offset += subgroupBroadcast(num_frags_accum, RASTER_SUBGROUP_MASK);
#else
		base_offset += subgroupShuffle(num_frags_accum, (LIX & 32) + RASTER_SUBGROUP_MASK);
#endif

		uint seg_offset0 = cur_offset & 0xff, seg_offset1 = (cur_offset & 0xff0000) >> 16;
		uint seg_high0 = (cur_offset & 0xf00) << 20, seg_high1 = (cur_offset & 0x0f000000) << 4;
		// TODO: optimize this?
		g_scratch_64[dst_offset0 + i] = uvec2(tri_idx_shifted | seg_offset0, bits.x | seg_high0);
		g_scratch_64[dst_offset1 + i] = uvec2(tri_idx_shifted | seg_offset1, bits.y | seg_high1);
	}
}

void visualizeBlockCounts(uint rbid, ivec2 pixel_pos) {
	uint frag_count = s_rblock_counts[rbid] >> 16;
	uint tri_count = s_rblock_counts[rbid] & 0xffff;
	//tri_count = s_block_tri_count[blockIdFromRender(rbid)];
	//tri_count = s_block_row_tri_count[pixel_pos.y >> BLOCK_SHIFT];
	//tri_count = s_bin_quad_count * 2 + s_bin_tri_count;

	vec3 color;
	color = gradientColor(frag_count, uvec4(8, 32, 128, 1024) * RASTER_SUBGROUP_SIZE);
	//color = gradientColor(tri_count, uvec4(16, 64, 256, 1024));

	outputPixel(pixel_pos, vec4(SATURATE(color), 1.0));
}

void rasterBin() {
	START_TIMER();

	if(LIX < NUM_BLOCKS) {
		s_block_tri_count[LIX] = 0;
		if(LIX < BLOCK_ROWS)
			s_block_row_tri_count[LIX] = 0;
	}
	barrier();
	processQuads();
	groupMemoryBarrier();
	barrier(); // TODO: stall (7%, conference)
	UPDATE_TIMER(0);

	const int num_blocks = (BIN_SIZE / BLOCK_SIZE) * (BIN_SIZE / BLOCK_SIZE);
	for(uint bid = LIX >> RASTER_SUBGROUP_SHIFT; bid < num_blocks; bid += NUM_RASTER_SUBGROUPS)
		generateBlocks(bid);

	barrier();
	// raster_low errors are not visualized, but propagated to high
	if(s_raster_error != 0) {
		if(LIX == 0) {
			int id = atomicAdd(g_info.bin_level_counts[BIN_LEVEL_HIGH], 1);
			HIGH_LEVEL_BINS(id) = s_bin_id;
			s_promoted_bin_count = max(s_promoted_bin_count, id + 1);
		}
		return;
	}
	groupMemoryBarrier();
	UPDATE_TIMER(1);

	for(uint rbid = LIX >> RASTER_SUBGROUP_SHIFT; rbid < NUM_RBLOCKS;
		rbid += NUM_RASTER_SUBGROUPS) {
		ReductionContext context;
		initReduceSamples(context);
		//initVisualizeSamples();

		uint temp_counts = s_rblock_counts[rbid];
		int frag_count = int(temp_counts >> 16);
		uint control_var = initUnpackSamples(temp_counts & 0xffff, temp_counts >> 16);
		while(frag_count > 0) {
			uint src_offset = scratchRasterBlockOffset(rbid);
			unpackSamples(control_var, src_offset);
			UPDATE_TIMER(2);

			shadeAndReduceSamples(rbid, min(frag_count, SEGMENT_SIZE), context);
			//visualizeSamples(min(frag_count, SEGMENT_SIZE));
			UPDATE_TIMER(3);

#ifdef ALPHA_THRESHOLD
			if(subgroupAll(context.out_trans < alpha_threshold))
				break;
#endif
			frag_count -= SEGMENT_SIZE;
		}

		ivec2 pixel_pos = renderBlockPixelPos(rbid);
		outputPixel(pixel_pos, finishReduceSamples(context));
		//finishVisualizeSamples(pixel_pos);
		//visualizeBlockCounts(rbid, pixel_pos);
		UPDATE_TIMER(4);
	}

	if(LIX >= LSIZE - NUM_RBLOCKS) {
		uint counts = s_rblock_counts[(LSIZE - 1) - LIX];
		updateStats(counts >> 16, counts & 0xffff);
	}

	// TODO: we should be able to start processing next bin before all warps have finished
	// but we would have to divide work in processQuads differently;
	// We could load bins in double-buffered fashion and once one bin is completely finished, we could load next one
	barrier(); // TODO: stall (10.5%, conference)
}

// TODO: consider removing persistent threads and using qcquire/unacquire for storage
void main() {
	INIT_TIMERS();
	initBinLoader(BIN_LEVEL_LOW);
	if(LIX == 0)
		s_promoted_bin_count = 0;
	initStats();

	while(loadNextBin(BIN_LEVEL_LOW))
		rasterBin();

	// If some of the bins are promoted to the next level, we have to adjust number of dispatches
	if(LIX == 0 && s_promoted_bin_count > 0) {
		uint num_dispatches = min(s_promoted_bin_count, MAX_DISPATCHES / 2);
		atomicMax(g_info.bin_level_dispatches[BIN_LEVEL_HIGH][0], num_dispatches);
	}

	COMMIT_TIMERS(g_info.raster_timers);
	commitStats();
}
