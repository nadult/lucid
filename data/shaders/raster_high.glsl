// raster_high prefers smaller bins; 32x32 max
// TODO: we could try 16x16 bins for raster_high and 64x64 for raster_low?
// raster_high would require additional phase in which 64x64 would be split into 16x16

#define LSIZE 1024
#define LSHIFT 10

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

layout(local_size_x = LSIZE) in;

#define WORKGROUP_SCRATCH_SIZE (256 * 1024)
#define WORKGROUP_SCRATCH_SHIFT 18

uint scratch32RBlockRowTrisOffset(uint rby) {
	return (gl_WorkGroupID.x << WORKGROUP_SCRATCH_SHIFT) + rby * MAX_RBLOCK_ROW_TRIS;
}

uint scratch64RBlockRowTrisOffset(uint rby) {
	return (gl_WorkGroupID.x << WORKGROUP_SCRATCH_SHIFT) +
		   rby * (MAX_RBLOCK_ROW_TRIS * (RBLOCK_HEIGHT == 8 ? 2 : 1));
}

uint scratchRasterBlockTrisOffset(uint rbid) {
	uint offset = 128 * 1024;
	return (gl_WorkGroupID.x << WORKGROUP_SCRATCH_SHIFT) + offset + rbid * MAX_RBLOCK_TRIS;
}

shared int s_num_bins, s_bin_id;
shared uint s_bin_quad_count, s_bin_quad_offset;
shared uint s_bin_tri_count, s_bin_tri_offset;

shared uint s_rblock_row_tri_counts[RBLOCK_ROWS];
shared int s_rblock_tri_counts[NUM_RBLOCKS];
shared uint s_rblock_frag_counts[NUM_RBLOCKS];

shared uint s_rblock_max_tri_counts;

// How many warps do we need to process single render-block in generateRBlocks ?
// Acceptable values: 1, 2, 4, 8; More: trouble :(
shared uint s_rblock_group_size;
shared uint s_rblock_group_shift;
shared uint s_max_sort_rcount;

shared int s_raster_error;

void generateRowTris(uint tri_idx) {
	uint scan_offset = STORAGE_TRI_SCAN_OFFSET + tri_idx * 2;
	uvec4 val0 = g_uvec4_storage[scan_offset + 0];
	uvec4 val1 = g_uvec4_storage[scan_offset + 1];
	int min_rby = clamp(int(val0.w & 0xffff) - s_bin_pos.y, 0, BIN_MASK) >> RBLOCK_HEIGHT_SHIFT;
	int max_rby = clamp(int(val0.w >> 16) - s_bin_pos.y, 0, BIN_MASK) >> RBLOCK_HEIGHT_SHIFT;

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

		uint rbid_row = rby << RBLOCK_COLS_SHIFT;
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
		uint roffset = row_tri_idx + rby * (MAX_RBLOCK_ROW_TRIS * 2);
		g_scratch_64[dst_offset_64 + roffset] =
			uvec2(bits0.x | (bx_mask << 24), bits1.x | ((tri_idx & 0xff0000) << 8));
		g_scratch_64[dst_offset_64 + roffset + MAX_RBLOCK_ROW_TRIS] =
			uvec2(bits0.y | ((tri_idx & 0xff) << 24), bits1.y | ((tri_idx & 0xff00) << 16));
#else
		uint roffset = row_tri_idx + rby * MAX_RBLOCK_ROW_TRIS;
		g_scratch_32[dst_offset_32 + roffset] = tri_idx | (bx_mask << 24);
		g_scratch_64[dst_offset_64 + roffset] = bits.xy;
#endif
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
		uint rbx = LIX & RBLOCK_COLS_MASK;
		int value = s_rblock_tri_counts[LIX], temp;
		if(RBLOCK_COLS >= 2)
			temp = subgroupShuffleUp(value, 1), value += rbx >= 1 ? temp : 0;
		if(RBLOCK_COLS >= 4)
			temp = subgroupShuffleUp(value, 2), value += rbx >= 2 ? temp : 0;
		if(RBLOCK_COLS >= 8)
			temp = subgroupShuffleUp(value, 4), value += rbx >= 4 ? temp : 0;
		subgroupMemoryBarrierShared();
		s_rblock_tri_counts[LIX] = 0;
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
	const uint mini_offset = BASE_BUFFER_SIZE;

	// TODO: better names for indices
	uint group_rbid = LIX >> (WARP_SHIFT + group_shift);
	uint rbid = start_rbid + group_rbid;

	uvec2 rblock_pos = renderBlockPos(rbid);
	uint tri_count = s_rblock_row_tri_counts[rblock_pos.y];
	uint buf_offset = group_rbid << (MAX_RBLOCK_TRIS0_SHIFT + group_shift);

	uint src_offset_32 = scratch32RBlockRowTrisOffset(rblock_pos.y);
	uint src_offset_64 = scratch64RBlockRowTrisOffset(rblock_pos.y);

	// Filling s_buffer with tri indices
	uint bx_bits_shift = 24 + rblock_pos.x;
	for(uint i = group_thread; i < tri_count; i += group_size) {
#if RBLOCK_HEIGHT == 8
		uint bx_bit = (g_scratch_64[src_offset_64 + i].x >> bx_bits_shift) & 1;
#else
		uint bx_bit = (g_scratch_32[src_offset_32 + i] >> bx_bits_shift) & 1;
#endif
		if(bx_bit != 0) {
			uint tri_offset = atomicAdd(s_rblock_tri_counts[rbid], 1);
			s_buffer[buf_offset + tri_offset] = i;
		}
	}
	barrier();

	uint dst_offset = scratchRasterBlockTrisOffset(rbid);
	tri_count = s_rblock_tri_counts[rbid];
	int startx = int(rblock_pos.x << RBLOCK_WIDTH_SHIFT);
	vec2 block_pos = vec2(rblock_pos << renderBlockShift()) + vec2(s_bin_pos);
	const uint rblock_tri_mask = RBLOCK_HEIGHT == 8 ? 0x1fff : 0xfff;

	uint frag_count = 0;
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
		uvec2 bits = uvec2(rasterHalfBlock(tri_mins.x, tri_maxs.x, startx, hnum_frags.x, cpos),
						   rasterHalfBlock(tri_mins.y, tri_maxs.y, startx, hnum_frags.y, cpos));
		num_frags = hnum_frags.x + hnum_frags.y;
		g_scratch_64[dst_offset + i] = bits;
		g_scratch_32[dst_offset + i] = tri_idx << 8;
#else
		uint tri_idx = g_scratch_32[src_offset_32 + row_tri_idx] & 0xffffff;
		uvec2 tri_info = g_scratch_64[src_offset_64 + row_tri_idx];
		uint bits = rasterHalfBlock(tri_info.x, tri_info.y, startx, num_frags, cpos);
		g_scratch_64[dst_offset + i] = uvec2(tri_idx << 8, bits);
#endif

		if(num_frags == 0) // This means that bx_mask is invalid
			DEBUG_RECORD(0, 0, 0, 0);
		frag_count += num_frags;
		uint depth = rasterBlockDepth(cpos * (0.5 / float(num_frags)) + block_pos, tri_idx);
		s_buffer[buf_offset + i] = i | (RBLOCK_HEIGHT == 8 ? (depth >> 1) << 13 : depth << 12);
	}
	barrier();
	groupMemoryBarrier();
	frag_count = subgroupInclusiveAddFast(frag_count);
	if(gl_SubgroupInvocationID == WARP_SIZE - 1)
		atomicAdd(s_rblock_frag_counts[rbid], frag_count);
	sortBuffer(tri_count, s_max_sort_rcount, buf_offset, group_size, group_thread, true);
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

	// Reordering rblocks in scratch
	// TODO: optimize it further (unroll?)
#if RBLOCK_HEIGHT == 4
	uint stored_rblocks[8];
	uint stored_idx = 0;
	for(uint i = group_thread; i < tri_count; i += group_size) {
		uint rblock_tri_idx = s_buffer[buf_offset + i] & rblock_tri_mask;
		uvec2 tri_data = g_scratch_64[dst_offset + rblock_tri_idx];
		stored_rblocks[stored_idx++] = tri_data.x;
		s_buffer[buf_offset + i] = tri_data.y;
	}
	barrier();
	stored_idx = 0;
	for(uint i = group_thread; i < tri_count; i += group_size)
		g_scratch_64[dst_offset + i] =
			uvec2(stored_rblocks[stored_idx++], s_buffer[buf_offset + i]);
#else
	uvec2 stored_rblocks[8];
	uint stored_idx = 0;
	for(uint i = group_thread; i < tri_count; i += group_size) {
		uint rblock_tri_idx = s_buffer[buf_offset + i] & rblock_tri_mask;
		uvec2 tri_data = g_scratch_64[dst_offset + rblock_tri_idx];
		stored_rblocks[stored_idx++] = tri_data;
		s_buffer[buf_offset + i] = g_scratch_32[dst_offset + rblock_tri_idx];
	}
	barrier();
	stored_idx = 0;
	for(uint i = group_thread; i < tri_count; i += group_size) {
		g_scratch_64[dst_offset + i] = stored_rblocks[stored_idx++];
		g_scratch_32[dst_offset + i] = s_buffer[buf_offset + i];
	}
#endif
	barrier();
}

void visualizeBlockCounts(uint rbid, ivec2 pixel_pos) {
	uint frag_count = s_rblock_frag_counts[rbid];
	uint tri_count = s_rblock_tri_counts[rbid];
	//tri_count = s_rblock_row_tri_counts[rbid >> RBLOCK_COLS_SHIFT] / 8;
	//tri_count = s_bin_quad_count / 16;

	vec3 color;
	color = gradientColor(frag_count, uvec4(64, 256, 1024, 4096));
	//color = gradientColor(tri_count, uvec4(16, 64, 256, 1024));
	outputPixel(pixel_pos, vec4(SATURATE(color), 1.0));
}

void rasterBin(int bin_id) {
	START_TIMER();
	if(LIX < NUM_RBLOCKS) {
		s_rblock_tri_counts[LIX] = 0;
		s_rblock_frag_counts[LIX] = 0;
		if(LIX < RBLOCK_ROWS)
			s_rblock_row_tri_counts[LIX] = 0;
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
	}
	barrier();
	processQuads();
	groupMemoryBarrier();
	barrier();
	computeRBlockGroups();
	barrier();
	UPDATE_TIMER(0);

	if(s_raster_error == 0) {
		int step = NUM_WARPS >> s_rblock_group_shift;
		for(int i = 0; i < s_rblock_group_size; i++)
			generateRBlocks(step * i);
	}
	groupMemoryBarrier();
	barrier();

	int rbid = int(LIX >> WARP_SHIFT);
	if(s_raster_error != 0) {
		outputPixel(renderBlockPixelPos(rbid), vec4(1.0, 0.0, 0.0, 0.0));
		barrier();
		return;
	}
	UPDATE_TIMER(1);

	ReductionContext context;
	initReduceSamples(context);
	//initVisualizeSamples();

	uint cur_tri_idx = initUnpackSamples(s_rblock_tri_counts[rbid], s_rblock_frag_counts[rbid]);
	for(int segment_id = 0;; segment_id++) {
		int frag_count = int(s_rblock_frag_counts[rbid]);
		frag_count = min(SEGMENT_SIZE, frag_count - (segment_id * SEGMENT_SIZE));
		if(frag_count <= 0)
			break;

		uint src_offset = scratchRasterBlockTrisOffset(rbid);
		uint tri_count = s_rblock_tri_counts[rbid];
		unpackSamples(cur_tri_idx, tri_count, src_offset);
		UPDATE_TIMER(2);

		shadeAndReduceSamples(rbid, frag_count, context);
		//visualizeSamples(frag_count);
		UPDATE_TIMER(3);

#ifdef ALPHA_THRESHOLD
		if(subgroupAll(context.out_trans < alpha_threshold))
			break;
#endif
	}

	ivec2 pixel_pos = renderBlockPixelPos(rbid);
	outputPixel(pixel_pos, finishReduceSamples(context));
	//finishVisualizeSamples(pixel_pos);
	//visualizeBlockCounts(rbid, pixel_pos);
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
