// $$include funcs declarations

// TODO: don't run too many groups if we have small amount of data (indirect dispatch)

#define LID gl_LocalInvocationID
#define LIX gl_LocalInvocationIndex
#define WGID gl_WorkGroupID

#define LSIZE BINNING_LSIZE

layout(local_size_x = LSIZE) in;
layout(std430, binding = 0) buffer buf0_ { uint g_quad_aabbs[]; };
BIN_COUNTERS_BUFFER(1); // TODO: coherent ?

shared int s_bins[BIN_COUNT];
shared int s_rows[BIN_COUNT_Y];
shared int s_large_bins[LARGE_BIN_COUNT];

//#define COMPUTE_WARP_DIVERGENCE
shared int s_warp_divergence;

void countSmallQuadBins(uint quad_idx) {
	const int shift = BIN_SIZE == 64 ? BIN_SHIFT - TILE_SHIFT : 0;
	ivec4 aabb = ivec4(decodeAABB32(g_quad_aabbs[quad_idx]) >> shift);
	int bsx = aabb[0], bsy = aabb[1], bex = aabb[2], bey = aabb[3];

	// Handling only tris with bin area 1 to 3:
	atomicAdd(s_bins[bsy * BIN_COUNT_X + bsx], 1);
	if(bex != bsx || bey != bsy)
		atomicAdd(s_bins[bey * BIN_COUNT_X + bex], 1);
	int bmx = (bsx + bex) >> 1, bmy = (bsy + bey) >> 1;
	if(bmx > bsx || bmy > bsy)
		atomicAdd(s_bins[bmy * BIN_COUNT_X + bmx], 1);
}

void countLargeQuadBins(uint quad_idx) {
	const int shift = BIN_SIZE == 64 ? BIN_SHIFT - TILE_SHIFT : 0;
	ivec4 aabb = ivec4(decodeAABB32(g_quad_aabbs[quad_idx]) >> shift);
	int bsx = aabb[0], bsy = aabb[1], bex = aabb[2], bey = aabb[3];

#ifdef COMPUTE_WARP_DIVERGENCE
	// Estimating divergence within warp; TODO: for 16 threads only?
	int area = (bex - bsx + 1) * (bey - bsy + 1);
	int avg_area = int(area);
	uint mask = uint(ballotARB(true)), cur_bit = LIX & 31;
	avg_area += ((mask & (cur_bit ^ 1)) != 0 ? 0xffffffff : 0) & shuffleXorNV(avg_area, 1, 32);
	avg_area += ((mask & (cur_bit ^ 2)) != 0 ? 0xffffffff : 0) & shuffleXorNV(avg_area, 2, 32);
	avg_area += ((mask & (cur_bit ^ 4)) != 0 ? 0xffffffff : 0) & shuffleXorNV(avg_area, 4, 32);
	avg_area += ((mask & (cur_bit ^ 8)) != 0 ? 0xffffffff : 0) & shuffleXorNV(avg_area, 8, 32);
	avg_area += ((mask & (cur_bit ^ 16)) != 0 ? 0xffffffff : 0) & shuffleXorNV(avg_area, 16, 32);
	atomicAdd(s_warp_divergence, int(32.0 * abs(float(area) - float(avg_area) / bitCount(mask))));
#endif

	const int large_shift = LARGE_BIN_SHIFT - BIN_SHIFT;
	int lbsx = bsx >> large_shift, lbex = (bex >> large_shift) + 1;
	int lbsy = bsy >> large_shift, lbey = (bey >> large_shift) + 1;
	atomicAdd(s_large_bins[lbsx + lbsy * BIN_COUNT_X], 1);
	if(lbex < BIN_COUNT_X)
		atomicAdd(s_large_bins[lbex + lbsy * BIN_COUNT_X], -1);
	if(lbey < BIN_COUNT_Y)
		atomicAdd(s_large_bins[lbsx + lbey * BIN_COUNT_X], -1);
	if(lbex < BIN_COUNT_X && lbey < BIN_COUNT_Y)
		atomicAdd(s_large_bins[lbex + lbey * BIN_COUNT_X], 1);

	bex++, bey++;
	atomicAdd(s_bins[bsx + bsy * BIN_COUNT_X], 1);
	if(bex < BIN_COUNT_X)
		atomicAdd(s_bins[bex + bsy * BIN_COUNT_X], -1);
	if(bey < BIN_COUNT_Y)
		atomicAdd(s_bins[bsx + bey * BIN_COUNT_X], -1);
	if(bex < BIN_COUNT_X && bey < BIN_COUNT_Y)
		atomicAdd(s_bins[bex + bey * BIN_COUNT_X], 1);
}

void computeOffsets() {
	// Computing quad offsets for each bin
	if(LIX < BIN_COUNT_Y) {
		int yoffset = int(LIX * BIN_COUNT_X), value = 0;
		for(int x = 0; x < BIN_COUNT_X; x++) {
			value += BIN_QUAD_COUNTS(x + yoffset);
			BIN_QUAD_OFFSETS(x + yoffset) = value;
		}
		s_rows[LIX] = value;
	}
	barrier();
	groupMemoryBarrier();
	if(LIX < BIN_COUNT_X) {
		int x = int(LIX), sum_rows = 0;
		for(int y = 0, yoffset = 0; y < BIN_COUNT_Y; y++, yoffset += BIN_COUNT_X) {
			int value = BIN_QUAD_OFFSETS(x + yoffset) + sum_rows - BIN_QUAD_COUNTS(x + yoffset);
			BIN_QUAD_OFFSETS(x + yoffset) = value;
			BIN_QUAD_OFFSETS_TEMP(x + yoffset) = value;
			sum_rows += s_rows[y];
		}
	}
	barrier();

	/*
	// Computing quad offsets for large bins
	for(uint i = LIX; i < LARGE_BIN_COUNT; i += LSIZE)
		s_large_bins[i] = LARGE_BIN_QUAD_COUNTS(i);
	barrier();
	if(LIX < LARGE_BIN_COUNT_Y) {
		uint yoffset = LIX * LARGE_BIN_COUNT_X;
		for(int x = 1; x < LARGE_BIN_COUNT_X; x++)
			s_large_bins[x + yoffset] += s_large_bins[x - 1 + yoffset];
		s_rows[LIX] = s_large_bins[LARGE_BIN_COUNT_X - 1 + yoffset];
	}
	barrier();
	if(LIX < LARGE_BIN_COUNT_X) {
		int prev_sum = 0;
		for(int y = 1; y < LARGE_BIN_COUNT_Y; y++) {
			prev_sum += s_rows[y - 1];
			s_large_bins[LIX + y * LARGE_BIN_COUNT_X] += prev_sum;
		}
	}
	barrier();
	for(uint i = LIX; i < LARGE_BIN_COUNT; i += LSIZE) {
		int cur_offset = s_large_bins[i] - LARGE_BIN_QUAD_COUNTS(i);
		LARGE_BIN_QUAD_OFFSETS(i) = cur_offset;
		LARGE_BIN_QUAD_OFFSETS_TEMP(i) = cur_offset;
	}*/
}

shared int s_num_quads[2];
shared int s_quads_offset, s_processed_small_quads;
shared uint s_num_finished;
shared int s_item_id;

void main() {
	if(LIX < 2) {
		s_num_quads[LIX] = g_bins.num_visible_quads[LIX];
		s_item_id = 0;
		s_processed_small_quads = 0;
#ifdef COMPUTE_WARP_DIVERGENCE
		s_warp_divergence = 0;
#endif
	}

	// Initializing bin counters
	for(uint i = LIX; i < BIN_COUNT; i += LSIZE)
		s_bins[i] = 0;
	barrier();

	// Counting large quads
	int num_quads = s_num_quads[1];
	while(s_item_id < MAX_BIN_WORKGROUP_ITEMS - 1) {
		if(LIX == 0) {
			int quads_offset = atomicAdd(g_bins.num_estimated_visible_quads[1], LSIZE);
			if(quads_offset < num_quads)
				BIN_WORKGROUP_ITEMS(WGID.x, s_item_id++) = -quads_offset - 1;
			s_quads_offset = quads_offset;
		}
		barrier();

		int quad_offset = s_quads_offset;
		if(quad_offset >= num_quads)
			break;

		int quad_idx = quad_offset + int(LIX);
		if(quad_idx < num_quads)
			countLargeQuadBins((MAX_QUADS - 1) - quad_idx);
		barrier();
	}

	barrier();
	{
		// Accumulating large quad counts across columns
		int bx = int(LIX), lbx = int(LIX - BIN_COUNT_X);
		if(bx < BIN_COUNT_X) {
			int accum = 0;
			for(uint by = 0; by < BIN_COUNT_Y; by++) {
				accum += s_bins[bx + by * BIN_COUNT_X];
				s_bins[bx + by * BIN_COUNT_X] = accum;
			}
		}
		if(lbx >= 0 && lbx < LARGE_BIN_COUNT_X) {
			int accum = 0;
			for(uint lby = 0; lby < LARGE_BIN_COUNT_Y; lby++) {
				accum += s_large_bins[lbx + lby * LARGE_BIN_COUNT_X];
				s_large_bins[lbx + lby * LARGE_BIN_COUNT_X] = accum;
			}
		}
		barrier();

		// Accumulating large quad counts across rows
		int by = int(LIX), lby = int(LIX - BIN_COUNT_X);
		if(by < BIN_COUNT_Y) {
			int accum = 0;
			for(uint bx = 0; bx < BIN_COUNT_X; bx++) {
				accum += s_bins[bx + by * BIN_COUNT_X];
				s_bins[bx + by * BIN_COUNT_X] = accum;
			}
		}
		if(lby >= 0 && lby < LARGE_BIN_COUNT_Y) {
			int accum = 0;
			for(uint lbx = 0; lbx < LARGE_BIN_COUNT_X; lbx++) {
				accum += s_large_bins[lbx + lby * LARGE_BIN_COUNT_X];
				s_large_bins[lbx + lby * LARGE_BIN_COUNT_X] = accum;
			}
		}
	}
	barrier();

	for(uint i = LIX; i < LARGE_BIN_COUNT; i += LSIZE)
		atomicAdd(LARGE_BIN_QUAD_COUNTS(i), s_large_bins[i]);

	// Counting small quads
	num_quads = s_num_quads[0];
	while(s_item_id < MAX_BIN_WORKGROUP_ITEMS - 1) {
		if(LIX == 0) {
			int quads_offset = atomicAdd(g_bins.num_estimated_visible_quads[0], LSIZE);
			if(quads_offset < num_quads) {
				BIN_WORKGROUP_ITEMS(WGID.x, s_item_id++) = quads_offset;
				s_processed_small_quads += min(num_quads - quads_offset, LSIZE);
			}
			s_quads_offset = quads_offset;
		}
		barrier();

		int quad_offset = s_quads_offset;
		if(quad_offset >= num_quads)
			break;

		int quad_idx = quad_offset + int(LIX);
		if(quad_idx < num_quads)
			countSmallQuadBins(quad_idx);
		barrier();
	}

	barrier();

	// Copying bin counters to global memory buffer
	for(uint i = LIX; i < BIN_COUNT; i += LSIZE) {
		if(s_bins[i] > 0)
			atomicAdd(BIN_QUAD_COUNTS(i), s_bins[i]);
		BIN_WORKGROUP_COUNTS(WGID.x, i) = s_bins[i];
	}
	barrier();

	// Groups which didn't do any estimation can quit early
	if(s_item_id == 0)
		return;

	if(LIX == 0 && s_item_id > 0) {
		s_num_finished = atomicAdd(g_bins.a_dispatcher_active_thread_groups, 1);
		s_processed_small_quads +=
			atomicAdd(g_bins.a_dispatcher_processed_quads[0], s_processed_small_quads);
		BIN_WORKGROUP_ITEMS(WGID.x, MAX_BIN_WORKGROUP_ITEMS - 1) = s_item_id;

#ifdef COMPUTE_WARP_DIVERGENCE
		atomicAdd(g_bins.temp[0], s_warp_divergence >> 5);
#endif
	}
	barrier();

	// Last group is responsible for computing offsets
	if(s_processed_small_quads == num_quads) {
		groupMemoryBarrier();
		computeOffsets();
	}
}
