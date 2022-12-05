// raster_high prefers smaller bins; 32x32 max
// TODO: we could try 16x16 bins for raster_high and 64x64 for raster_low?
// raster_high would require additional phase in which 64x64 would be split into 16x16

#define LSIZE 1024
#define LSHIFT 10

#define BIN_LEVEL BIN_LEVEL_HIGH

#include "shared/raster.glsl"

#include "%shader_debug"
DEBUG_SETUP(1, 11)

// Basic maximum value of tris per rblock
#define MAX_HBLOCK_TRIS0 256
#define MAX_HBLOCK_TRIS0_SHIFT 8

#define MAX_GROUP_SIZE 16
#define MAX_GROUP_SHIFT 4

#define MAX_HBLOCK_TRIS (MAX_HBLOCK_TRIS0 * MAX_GROUP_SIZE)
#define MAX_HBLOCK_TRIS_SHIFT (MAX_HBLOCK_TRIS0_SHIFT + MAX_GROUP_SHIFT)

#define MAX_HBLOCK_ROW_TRIS 16384
#define MAX_HBLOCK_ROW_TRIS_SHIFT 14

layout(local_size_x = LSIZE) in;

#define WORKGROUP_SCRATCH_SIZE (256 * 1024)
#define WORKGROUP_SCRATCH_SHIFT 18

uint scratchHalfBlockRowTrisOffset(uint rby) {
	return (gl_WorkGroupID.x << WORKGROUP_SCRATCH_SHIFT) + rby * MAX_HBLOCK_ROW_TRIS;
}

uint scratchHalfBlockTrisOffset(uint hbid) {
	return (gl_WorkGroupID.x << WORKGROUP_SCRATCH_SHIFT) + 128 * 1024 + hbid * MAX_HBLOCK_TRIS;
}

shared uint s_hblock_row_tri_counts[HBLOCK_ROWS];
shared int s_hblock_tri_counts[NUM_RBLOCKS];
shared uint s_hblock_frag_counts[NUM_RBLOCKS];

shared uint s_hblock_group_size;
shared uint s_hblock_group_shift;
shared uint s_max_sort_rcount;

void generateRowTris(uint tri_idx) {
	uint scan_offset = STORAGE_TRI_SCAN_OFFSET + tri_idx * 2;
	uvec4 val0 = g_uvec4_storage[scan_offset + 0];
	uvec4 val1 = g_uvec4_storage[scan_offset + 1];
	int min_rby = clamp(int(val0.w & 0xffff) - s_bin_pos.y, 0, BIN_MASK) >> HBLOCK_HEIGHT_SHIFT;
	int max_rby = clamp(int(val0.w >> 16) - s_bin_pos.y, 0, BIN_MASK) >> HBLOCK_HEIGHT_SHIFT;

	vec2 start = vec2(s_bin_pos.x, s_bin_pos.y + min_rby * HBLOCK_HEIGHT);
	ScanlineParams scan = loadScanlineParamsRow(val0, val1, start);

	uint dst_offset = scratchHalfBlockRowTrisOffset(0);

	// TODO: is it worth it to make this loop more work-efficient?
	for(int rby = min_rby; rby <= max_rby; rby++) {
		uvec3 bits = rasterBinStep(scan);
		uint bx_mask = bits.z;
		if(bx_mask == 0)
			continue;

		uint hbid_row = rby << HBLOCK_COLS_SHIFT;
		uint min_hbid = findLSB(bx_mask), max_hbid = findMSB(bx_mask) + 1;
		// Accumulation is done at the end of processBlocks
		atomicAdd(s_hblock_tri_counts[hbid_row + min_hbid], 1);
		if(max_hbid < HBLOCK_COLS)
			atomicAdd(s_hblock_tri_counts[hbid_row + max_hbid], -1);

		uint row_tri_idx = atomicAdd(s_hblock_row_tri_counts[rby], 1);
		if(row_tri_idx >= MAX_HBLOCK_ROW_TRIS) {
			atomicOr(s_raster_error, 0xffffffff);
			return;
		}

		uint roffset = row_tri_idx + rby * MAX_HBLOCK_ROW_TRIS;
		g_scratch_32[dst_offset + roffset] = bx_mask;
		g_scratch_64[dst_offset + roffset] =
			uvec2(bits.x | ((tri_idx & 0xfff) << 20), bits.y | ((tri_idx & 0xfff000) << 8));
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

		uint tri_idx = quad_idx * 2 + second_tri;
		generateRowTris(tri_idx);
	}
	for(uint i = LIX; i < s_bin_tri_count; i += LSIZE)
		generateRowTris(g_bin_tris[s_bin_tri_offset + i]);
}

void computeRBlockGroups() {
	// Accumulating per hblock-counts for each hblock-row
	// Note: these are only estimates; very good estimates, but in some cases a single
	// triangle can have wide holes between pixels (because middle pixels don't hit pixel centers)
	if(LIX < NUM_RBLOCKS) {
		uint rbx = LIX & HBLOCK_COLS_MASK;
		int value = s_hblock_tri_counts[LIX], temp;
		if(HBLOCK_COLS >= 2)
			temp = subgroupShuffleUp(value, 1), value += rbx >= 1 ? temp : 0;
		if(HBLOCK_COLS >= 4)
			temp = subgroupShuffleUp(value, 2), value += rbx >= 2 ? temp : 0;
		if(HBLOCK_COLS >= 8)
			temp = subgroupShuffleUp(value, 4), value += rbx >= 4 ? temp : 0;
		subgroupMemoryBarrierShared();
		s_hblock_tri_counts[LIX] = 0;
		uint max_value = subgroupMax_(uint(value), NUM_HALFGROUPS);
		if(LIX == 0) {
			// rcount: count rounded up to next power of 2, minimum: 32
			uint rcount =
				(max_value & (max_value - 1)) == 0 ? max_value : (2 << findMSB(max_value));
			s_max_sort_rcount = max(32, rcount);

			//uint group_shift = max(int(log2(max_value)) - MAX_HBLOCK_TRIS0_SHIFT, 0);
			uint group_shift = max_value <= MAX_HBLOCK_TRIS0	  ? 0 :
							   max_value <= MAX_HBLOCK_TRIS0 * 2  ? 1 :
							   max_value <= MAX_HBLOCK_TRIS0 * 4  ? 2 :
							   max_value <= MAX_HBLOCK_TRIS0 * 8  ? 3 :
							   max_value <= MAX_HBLOCK_TRIS0 * 16 ? 4 :
																	5;
			uint group_size = 1 << group_shift;
			s_hblock_group_shift = group_shift;
			s_hblock_group_size = group_size;
			if(group_size > MAX_GROUP_SIZE)
				s_raster_error = 0xffffffff;
		}
	}
}

// TODO: maybe process smaller amount of blocks at the same time?
// smaller chance that it will leave cache
void generateRBlocks(uint start_hbid) {
	uint group_size = s_hblock_group_size * HALFGROUP_SIZE;
	uint group_shift = s_hblock_group_shift;
	uint group_mask = group_size - 1;
	uint group_thread = LIX & group_mask;

	// TODO: better names for indices
	uint group_hbid = LIX >> (HALFGROUP_SHIFT + group_shift);
	uint hbid = start_hbid + group_hbid;

	uvec2 hblock_pos = halfBlockPos(hbid);
	uint tri_count = s_hblock_row_tri_counts[hblock_pos.y];
	uint buf_offset = group_hbid << (MAX_HBLOCK_TRIS0_SHIFT + group_shift);
	uint src_offset = scratchHalfBlockRowTrisOffset(hblock_pos.y);

	// Filling s_buffer with tri indices
	uint bx_bits_mask = 1u << hblock_pos.x;
	for(uint i = group_thread; i < tri_count; i += group_size) {
		uint bx_bits = g_scratch_32[src_offset + i];
		if((bx_bits & bx_bits_mask) != 0) {
			uint tri_offset = atomicAdd(s_hblock_tri_counts[hbid], 1);
			s_buffer[buf_offset + tri_offset] = i;
		}
	}
	barrier();

	uint dst_offset = scratchHalfBlockTrisOffset(hbid);
	tri_count = s_hblock_tri_counts[hbid];
	int startx = int(hblock_pos.x << HBLOCK_WIDTH_SHIFT);
	vec2 block_pos = vec2(hblock_pos << halfBlockShift()) + vec2(s_bin_pos);

	for(uint i = group_thread; i < tri_count; i += group_size) {
		uint row_tri_idx = s_buffer[buf_offset + i];
		uint num_frags;

		uvec2 tri_info = g_scratch_64[src_offset + row_tri_idx];
		uint tri_idx = (tri_info.x >> 20) | ((tri_info.y & 0xfff00000) >> 8);
		vec2 cpos = rasterHalfBlockCentroid(tri_info.x, tri_info.y, startx, num_frags);

		if(num_frags == 0) // This means that bx_mask is invalid
			DEBUG_RECORD(0, 0, 0, 0);
		// TODO: make depth 18 bit from the start
		uint depth = rasterBlockDepth(cpos * (0.5 / float(num_frags)) + block_pos, tri_idx);
		s_buffer[buf_offset + i] = row_tri_idx | ((depth >> 2) << 14);
	}
	barrier();
	sortBuffer(tri_count, s_max_sort_rcount, buf_offset, group_size, group_thread, true);
	barrier();

	// TODO: move to sortBuffer()
#ifdef DEBUG_ENABLED
	for(uint i = LIX & HALFGROUP_MASK; i < tri_count; i += HALFGROUP_SIZE) {
		uint value = s_buffer[buf_offset + i];
		uint prev_value = i == 0 ? 0 : s_buffer[buf_offset + i - 1];
		if(value <= prev_value)
			DEBUG_RECORD(i, tri_count, prev_value, value);
	}
	barrier();
#endif

	for(uint i = group_thread; i < tri_count; i += group_size) {
		uint row_tri_idx = s_buffer[buf_offset + i] & (MAX_HBLOCK_ROW_TRIS - 1);
		uvec2 tri_info = g_scratch_64[src_offset + row_tri_idx];
		uint num_frags = rasterHalfBlockNumFrags(tri_info.x, tri_info.y, startx);
		uint num_frags_accum = subgroupInclusiveAddFast32(num_frags);
		s_buffer[buf_offset + i] = row_tri_idx | (num_frags << 14) | (num_frags_accum << 20);
	}
	barrier();

	// Computing prefix sum across whole render-blocks. We're processing 4 elements
	// at a time, so that we can fit with MAX_HBLOCK_TRIS tris/hblock (128 half-groups).
	if(LIX < 2 * NUM_HALFGROUPS) {
		uint wgsize = 2 << group_shift, wgmask = wgsize - 1;
		uint group_hbid = LIX >> (1 + group_shift), group_sub_idx = LIX & wgmask;
		uint buf_offset = group_hbid << (MAX_HBLOCK_TRIS0_SHIFT + group_shift);
		uint hbid = start_hbid + group_hbid;
		uint tri_count = s_hblock_tri_counts[hbid];
		uint group_offset = group_sub_idx << (HALFGROUP_SHIFT + 2);

		uvec4 values = uvec4(0);
		for(int j = 0; j < 4; j++) {
			if(group_offset < tri_count) {
				uint hblock_tri_idx = min(group_offset + HALFGROUP_MASK, tri_count - 1);
				values[j] = s_buffer[buf_offset + hblock_tri_idx] >> 20;
			}
			group_offset += HALFGROUP_SIZE;
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

		if(group_sub_idx == wgmask)
			s_hblock_frag_counts[hbid] = sum;
		s_buffer[BASE_BUFFER_SIZE + LIX * 4 + 0] = sum - value, value -= values[0];
		s_buffer[BASE_BUFFER_SIZE + LIX * 4 + 1] = sum - value, value -= values[1];
		s_buffer[BASE_BUFFER_SIZE + LIX * 4 + 2] = sum - value;
		s_buffer[BASE_BUFFER_SIZE + LIX * 4 + 3] = sum - values[3];
	}
	barrier();

	for(uint i = group_thread; i < tri_count; i += group_size) {
		uint current = s_buffer[buf_offset + i];
		uint row_tri_idx = current & 0x3fff;

		uint cur_offset =
			(current >> 20) - ((current >> 14) & 0x3f) +
			s_buffer[BASE_BUFFER_SIZE + (group_hbid << (3 + group_shift)) + (i >> HALFGROUP_SHIFT)];
		uvec2 tri_info = g_scratch_64[src_offset + row_tri_idx];
		uint tri_idx_shifted = ((tri_info.x >> 12) & 0xfff00) | (tri_info.y & 0xfff00000);
		uint num_frags, bits = rasterHalfBlockBits(tri_info.x, tri_info.y, startx, num_frags);

		uint seg_offset = cur_offset & 0xff, seg_high = (cur_offset & 0xf00) << 20;
		g_scratch_64[dst_offset + i] =
			uvec2(tri_idx_shifted | (cur_offset & 0xff), bits | seg_high);
	}
	barrier();
}

void visualizeBlockCounts(uint hbid, ivec2 pixel_pos) {
	uint frag_count = s_hblock_frag_counts[hbid];
	uint tri_count = s_hblock_tri_counts[hbid];
	//tri_count = s_hblock_row_tri_counts[hbid >> HBLOCK_COLS_SHIFT] / 8;
	//tri_count = s_bin_quad_count / 16;

	vec3 color;
	color = gradientColor(frag_count, uvec4(64, 256, 1024, 4096));
	//color = gradientColor(tri_count, uvec4(16, 64, 256, 1024));
	outputPixel(pixel_pos, vec4(SATURATE(color), 1.0));
}

void rasterBin() {
	START_TIMER();
	if(LIX < NUM_RBLOCKS) {
		s_hblock_tri_counts[LIX] = 0;
		if(LIX < HBLOCK_ROWS)
			s_hblock_row_tri_counts[LIX] = 0;
	}
	barrier();
	processQuads();
	groupMemoryBarrier();
	barrier();
	computeRBlockGroups();
	barrier();
	UPDATE_TIMER(0);

	if(s_raster_error == 0) {
		int step = NUM_HALFGROUPS >> s_hblock_group_shift;
		for(int i = 0; i < s_hblock_group_size; i++)
			generateRBlocks(step * i);
	}
	groupMemoryBarrier();
	barrier();
	if(LIX < NUM_RBLOCKS)
		updateStats(s_hblock_frag_counts[LIX], s_hblock_tri_counts[LIX]);

	int hbid = int(LIX >> HALFGROUP_SHIFT);
	if(s_raster_error != 0) {
		outputPixel(halfBlockPixelPos(hbid), vec4(1.0, 0.0, 0.0, 0.0));
		barrier();
		return;
	}
	UPDATE_TIMER(1);

	ReductionContext context;
	initReduceSamples(context);
	//initVisualizeSamples();

	uint control_var = initUnpackSamples(s_hblock_tri_counts[hbid], s_hblock_frag_counts[hbid]);
	int frag_count = int(s_hblock_frag_counts[hbid]);
	while(frag_count > 0) {
		unpackSamples(control_var, scratchHalfBlockTrisOffset(hbid));
		UPDATE_TIMER(2);

		shadeAndReduceSamples(hbid, min(frag_count, SEGMENT_SIZE), context);
		//visualizeSamples(min(frag_count, SEGMENT_SIZE));
		UPDATE_TIMER(3);

#ifdef ALPHA_THRESHOLD
		if(subgroupAll(context.out_trans < alpha_threshold))
			break;
#endif
		frag_count -= SEGMENT_SIZE;
	}

	ivec2 pixel_pos = halfBlockPixelPos(hbid);
	outputPixel(pixel_pos, finishReduceSamples(context));
	//finishVisualizeSamples(pixel_pos);
	//visualizeBlockCounts(hbid, pixel_pos);
	UPDATE_TIMER(4);

	barrier(); // TODO: large stall here (9%)
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
	initBinLoader(BIN_LEVEL_HIGH);
	initStats();

	while(loadNextBin(BIN_LEVEL_HIGH))
		rasterBin();

	COMMIT_TIMERS(g_info.raster_timers);
	commitStats();
}
