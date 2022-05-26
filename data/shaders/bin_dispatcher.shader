// $$include funcs declarations

// TODO: don't run too many groups if we have small amount of data (indirect dispatch)

#define LID gl_LocalInvocationID
#define LIX gl_LocalInvocationIndex
#define WGID gl_WorkGroupID

#define LSIZE BINNING_LSIZE
#define LSHIFT BINNING_LSHIFT

#define ITEM_STEPS 1
#define ITEM_SIZE (LSIZE * ITEM_STEPS)

layout(local_size_x = LSIZE) in;
layout(std430, binding = 0) buffer buf0_ { uint g_quad_aabbs[]; };
coherent BIN_COUNTERS_BUFFER(1);
layout(std430, binding = 3) buffer buf3_ { uint g_bin_quads[]; };
layout(std430, binding = 4) buffer buf4_ { uint g_bin_row_quads[]; };

shared int s_bins[BIN_COUNT];
shared int s_rows[BIN_COUNT_Y + 1];
shared int s_row_warps[32];

// Constants useful for efficient processing of bin counts
const int xbin_warps = LSIZE / BIN_COUNT_X, ybin_warps = LSIZE / BIN_COUNT_Y;
const int xbin_step = int(log2(xbin_warps)), ybin_step = int(log2(ybin_warps));
const int xbin_count = 1 << xbin_step, ybin_count = 1 << ybin_step;

void countSmallQuadBins(uint quad_idx) {
	const int shift = BIN_SIZE == 64 ? BIN_SHIFT - TILE_SHIFT : 0;
	ivec4 aabb = ivec4(decodeAABB32(g_quad_aabbs[quad_idx]) >> shift);
	int bsx = aabb[0], bsy = aabb[1], bex = aabb[2], bey = aabb[3];
	int area = (bex - bsx + 1) * (bey - bsy + 1);

	for(int by = bsy; by <= bey; by++)
		for(int bx = bsx; bx <= bex; bx++)
			atomicAdd(s_bins[by * BIN_COUNT_X + bx], 1);

	// Handling only tris with bin area 1 to 3:
	/*atomicAdd(s_bins[bsy * BIN_COUNT_X + bsx], 1);
	if(bex != bsx || bey != bsy)
		atomicAdd(s_bins[bey * BIN_COUNT_X + bex], 1);
	int bmx = (bsx + bex) >> 1, bmy = (bsy + bey) >> 1;
	if(bmx > bsx || bmy > bsy)
		atomicAdd(s_bins[bmy * BIN_COUNT_X + bmx], 1);*/
}

void countLargeQuadBins(uint quad_idx) {
	const int shift = BIN_SIZE == 64 ? BIN_SHIFT - TILE_SHIFT : 0;
	ivec4 aabb = ivec4(decodeAABB32(g_quad_aabbs[quad_idx]) >> shift);
	int bsx = aabb[0], bsy = aabb[1], bex = aabb[2], bey = aabb[3];

	// TODO: maybe it makes no sense to compute bins and rows separately?
	// We could compute rows from bins
	bex++, bey++;
	atomicAdd(s_bins[bsx + bsy * BIN_COUNT_X], 1);
	atomicAdd(s_rows[bsy], 1);
	if(bex < BIN_COUNT_X)
		atomicAdd(s_bins[bex + bsy * BIN_COUNT_X], -1);
	if(bey < BIN_COUNT_Y) {
		atomicAdd(s_bins[bsx + bey * BIN_COUNT_X], -1);
		atomicAdd(s_rows[bey], -1);
	}
	if(bex < BIN_COUNT_X && bey < BIN_COUNT_Y)
		atomicAdd(s_bins[bex + bey * BIN_COUNT_X], 1);
}

void dispatchQuad(int quad_idx) {
	uvec4 aabb = decodeAABB32(g_quad_aabbs[quad_idx]);
#if BIN_SIZE == 64
	// Encodes tile ranges within a single bin
	uint tile_ranges =
		(aabb[0] & 0x3) | ((aabb[1] & 0x3) << 2) | ((aabb[2] & 0x3) << 4) | ((aabb[3] & 0x3) << 6);
	aabb >>= BIN_SHIFT - TILE_SHIFT;
#endif
	uint bsx = aabb[0], bsy = aabb[1], bex = aabb[2], bey = aabb[3];

	for(uint by = bsy; by <= bey; by++) {
#if BIN_SIZE == 64
		uint tile_ranges_cury = (tile_ranges | (by < bey ? 0xc0 : 0)) & (by > bsy ? 0xf3 : 0xff);
#endif
		for(uint bx = bsx; bx <= bex; bx++) {
			uint bin_id = bx + by * BIN_COUNT_X;
			uint quad_offset = atomicAdd(s_bins[bin_id], 1);
#if BIN_SIZE == 64
			uint tile_ranges_cur =
				(tile_ranges_cury | (bx < bex ? 0x30 : 0)) & (bx > bsx ? 0xfc : 0xff);
			g_bin_quads[quad_offset] = (quad_idx & 0xffffff) | (tile_ranges_cur << 24);
#else
			g_bin_quads[quad_offset] = quad_idx;
#endif
		}
	}
}

int prefixSum32(int accum) {
	int temp;
	uint sub_id = LIX & 31;
	temp = shuffleUpNV(accum, 1, 32), accum += sub_id >= 1 ? temp : 0;
	temp = shuffleUpNV(accum, 2, 32), accum += sub_id >= 2 ? temp : 0;
	temp = shuffleUpNV(accum, 4, 32), accum += sub_id >= 4 ? temp : 0;
	temp = shuffleUpNV(accum, 8, 32), accum += sub_id >= 8 ? temp : 0;
	temp = shuffleUpNV(accum, 16, 32), accum += sub_id >= 16 ? temp : 0;
	return accum;
}

void accumulateLargeQuadCounts() {
	// Accumulating large quad counts across columns
	if(LIX < BIN_COUNT_X * xbin_count) {
		uint bx = LIX >> xbin_step;
		int accum = 0;
		for(uint by = 0, suby = LIX & (xbin_count - 1); by < BIN_COUNT_Y; by += xbin_count) {
			uint idx = bx + (by + suby) * BIN_COUNT_X;
			int cur = by + suby < BIN_COUNT_Y ? s_bins[idx] : 0, temp;
			for(int i = 1; i < xbin_count; i <<= 1) {
				temp = shuffleUpNV(cur, i, xbin_count);
				cur += suby >= i ? temp : 0;
			}
			cur += accum;
			if(by + suby < BIN_COUNT_Y)
				s_bins[idx] = cur;
			accum = shuffleNV(cur, xbin_count - 1, xbin_count);
		}
	}

	// Accumulating bin rows (within warps)
	if(LIX < BIN_COUNT_Y) {
		int accum = prefixSum32(s_rows[LIX]);
		s_rows[LIX] = accum;
		if((LIX & 31) == 31)
			s_row_warps[LIX >> 5] = accum;
	}
	barrier();

	// Accumulating large quad counts across rows
	if(LIX < BIN_COUNT_Y * ybin_count) {
		uint by = LIX >> ybin_step;
		int accum = 0;
		for(uint bx = 0, subx = LIX & (ybin_count - 1); bx < BIN_COUNT_X; bx += ybin_count) {
			uint idx = bx + subx + by * BIN_COUNT_X;
			int cur = bx + subx < BIN_COUNT_X ? s_bins[idx] : 0, temp;
			for(int i = 1; i < ybin_count; i <<= 1) {
				temp = shuffleUpNV(cur, i, ybin_count);
				cur += subx >= i ? temp : 0;
			}
			cur += accum;
			if(bx + subx < BIN_COUNT_X)
				s_bins[idx] = cur;
			accum = shuffleNV(cur, ybin_count - 1, ybin_count);
		}
	}
	// Accumulating bin rows (adding previous warps)
	if(LIX < BIN_COUNT_Y) {
		int prev_warps = 0, warp_id = int(LIX >> 5);
		for(int i = 0; i < warp_id; i++)
			prev_warps += s_row_warps[i];
		atomicAdd(BIN_ROWS_LARGE_QUAD_COUNTS(LIX), s_rows[LIX] + prev_warps);
	}
	barrier();
	for(uint i = LIX; i < BIN_COUNT; i += LSIZE)
		if(s_bins[i] > 0) {
			atomicAdd(BIN_QUAD_COUNTS(i), s_bins[i]);
			s_bins[i] = 0;
		}
	barrier();
}

void computeOffsets() {
	// TODO: no need for groupMemoryBarrier ?

	// Computing offsets of large quad counts within bin rows
	if(LIX < BIN_COUNT_Y) {
		int accum = prefixSum32(BIN_ROWS_LARGE_QUAD_COUNTS(LIX));
		s_rows[LIX] = accum;
		if((LIX & 31) == 31)
			s_row_warps[LIX >> 5] = accum;
	}

	// Computing bin quad counts offsets
	if(LIX < BIN_COUNT_Y * ybin_count) {
		uint by = LIX >> ybin_step;
		int accum = 0;
		for(uint bx = 0, subx = LIX & (ybin_count - 1); bx < BIN_COUNT_X; bx += ybin_count) {
			uint idx = bx + subx + by * BIN_COUNT_X;
			int cur = bx + subx < BIN_COUNT_X ? BIN_QUAD_COUNTS(idx) : 0, temp;
			for(int i = 1; i < ybin_count; i <<= 1) {
				temp = shuffleUpNV(cur, i, ybin_count);
				cur += subx >= i ? temp : 0;
			}
			cur += accum;
			if(bx + subx < BIN_COUNT_X)
				BIN_QUAD_OFFSETS(idx) = cur;
			accum = shuffleNV(cur, ybin_count - 1, ybin_count);
		}
	}
	barrier();
	if(LIX < BIN_COUNT_Y) {
		// Finishing computation of bin rows offsets
		int prev_warps = 0, warp_id = int(LIX >> 5);
		for(int i = 0; i < warp_id; i++)
			prev_warps += s_row_warps[i];
		int count = BIN_ROWS_LARGE_QUAD_COUNTS(LIX);
		int value = s_rows[LIX] + prev_warps;
		BIN_ROWS_LARGE_QUAD_OFFSETS(LIX) = value - count;
		BIN_ROWS_LARGE_QUAD_OFFSETS_TEMP(LIX) = value - count;
		if(LIX == BIN_COUNT_Y - 1)
			g_bins.num_bin_rows_quads = value;

		// Storing accumulated bin quad counts for each row
		s_rows[LIX] = BIN_QUAD_OFFSETS((BIN_COUNT_X - 1) + LIX * BIN_COUNT_X);
	}
	barrier();
	if(LIX < BIN_COUNT_X * xbin_count) {
		uint bx = LIX >> xbin_step;
		int accum = 0;
		for(uint by = 0, suby = LIX & (xbin_count - 1); by < BIN_COUNT_Y; by += xbin_count) {
			uint idx = by + suby;
			int cur = idx < BIN_COUNT_Y && idx > 0 ? s_rows[idx - 1] : 0, temp;
			for(int i = 1; i < xbin_count; i <<= 1) {
				temp = shuffleUpNV(cur, i, xbin_count);
				cur += suby >= i ? temp : 0;
			}
			cur += accum;
			if(idx < BIN_COUNT_Y) {
				uint bidx = bx + idx * BIN_COUNT_X;
				int value = BIN_QUAD_OFFSETS(bidx) + cur - BIN_QUAD_COUNTS(bidx);
				BIN_QUAD_OFFSETS(bidx) = value;
				BIN_QUAD_OFFSETS_TEMP(bidx) = value;
			}
			accum = shuffleNV(cur, xbin_count - 1, xbin_count);
		}
	}
	barrier();
}

shared int s_num_quads[2], s_num_bin_row_quads;
shared int s_quads_offset;
shared uint s_num_finished;
shared int s_item_id, s_num_items;
shared int s_num_processed_items, s_num_all_items;

// TODO: Zamiast tego jedna tablica w gmem?
// TODO: 16 bits is enough for workgroup items ?
shared int s_workgroup_items[MAX_BIN_WORKGROUP_ITEMS];

void main() {
	if(LIX < 2) {
		s_num_quads[LIX] = g_bins.num_visible_quads[LIX];
		// TODO: we can probably improve perf by dividing work more evenly
		// Two types of quads make it a bit tricky
		if(LIX == 0) {
			s_num_all_items = (s_num_quads[0] + ITEM_SIZE - 1) / ITEM_SIZE;
			s_item_id = 0;
		}
	}
	for(uint i = LIX; i < BIN_COUNT; i += LSIZE)
		s_bins[i] = 0;
	if(LIX < BIN_COUNT_Y)
		s_rows[LIX] = 0;
	barrier();

	// Computing large quads bin coverage
	int num_quads = s_num_quads[1];
	while(num_quads > 0) {
		if(LIX == 0)
			s_quads_offset = atomicAdd(g_bins.num_estimated_visible_quads[1], ITEM_SIZE);
		barrier();

		int quad_offset = s_quads_offset;
		if(quad_offset >= num_quads)
			break;

		for(int s = 0; s < ITEM_STEPS; s++) {
			int quad_idx = quad_offset + (LSIZE * s) + int(LIX);
			if(quad_idx >= num_quads)
				break;
			countLargeQuadBins((MAX_QUADS - 1) - quad_idx);
		}
		barrier();
	}

	barrier();
	accumulateLargeQuadCounts();

	// Computing small quads bin coverage
	num_quads = s_num_quads[0];
	while(s_item_id < MAX_BIN_WORKGROUP_ITEMS - 1) {
		if(LIX == 0) {
			int quads_offset = atomicAdd(g_bins.num_estimated_visible_quads[0], ITEM_SIZE);
			if(quads_offset < num_quads)
				s_workgroup_items[s_item_id++] = quads_offset;
			s_quads_offset = quads_offset;
		}
		barrier();

		int quad_offset = s_quads_offset;
		if(quad_offset >= num_quads)
			break;

		for(int s = 0; s < ITEM_STEPS; s++) {
			int quad_idx = quad_offset + (LSIZE * s) + int(LIX);
			if(quad_idx >= num_quads)
				break;
			countSmallQuadBins(quad_idx);
		}
		barrier();
	}

	barrier();

	// Thread groups which didn't do any estimation can quit early:
	// they won't participate in dispatching either
	// TODO: they could participate in large quad processing?
	if(s_item_id == 0)
		return;

	// Copying bin counters to global memory buffer
	for(uint i = LIX; i < BIN_COUNT; i += LSIZE)
		if(s_bins[i] > 0)
			atomicAdd(BIN_QUAD_COUNTS(i), s_bins[i]);
	barrier();

	// Finishing estimation phase
	if(LIX == 0) {
		s_num_finished = atomicAdd(g_bins.a_dispatcher_active_thread_groups, 1);
		s_num_processed_items =
			atomicAdd(g_bins.a_dispatcher_processed_items, s_item_id) + s_item_id;
		s_num_items = s_item_id;
		g_bins.dispatcher_item_counts[s_num_finished] = s_num_items;
		s_item_id = 0;
	}
	barrier();

	// Last group is responsible for computing bin offsets
	if(s_num_processed_items == s_num_all_items) {
		groupMemoryBarrier();
		computeOffsets();
		if(LIX == 0)
			atomicExchange(g_bins.a_dispatcher_phase, 1);
	}

	// Waiting until all bin offsets are computed
	// TODO: can we do something useful while waiting?
	if(LIX == 0)
		while(g_bins.a_dispatcher_phase == 0)
			;
	barrier();

	// Reserving space for quad indices in bins
	for(uint i = LIX; i < BIN_COUNT; i += LSIZE)
		if(s_bins[i] > 0)
			s_bins[i] = atomicAdd(BIN_QUAD_OFFSETS_TEMP(i), s_bins[i]);
	barrier();

	// Dispatching small quads
	num_quads = s_num_quads[0];
	while(s_item_id < s_num_items) {
		barrier();
		if(LIX == 0)
			s_quads_offset = s_workgroup_items[s_item_id++];
		barrier();
		int quads_offset = s_quads_offset;

		for(int s = 0; s < ITEM_STEPS; s++) {
			int quad_idx = quads_offset + LSIZE * s + int(LIX);
			if(quad_idx >= num_quads)
				break;
			dispatchQuad(quad_idx);
		}
	}
	barrier();

	num_quads = s_num_quads[1];
	if(num_quads == 0)
		return;
	// Dispatching large quads to bin rows
	while(true) {
		if(LIX == 0)
			s_quads_offset = atomicAdd(g_bins.num_dispatched_visible_quads[0], LSIZE);
		barrier();

		int quad_offset = s_quads_offset;
		if(quad_offset >= num_quads)
			break;

		if(LIX < BIN_COUNT_Y)
			s_rows[LIX] = 0;
		barrier();

		int large_quad_idx = quad_offset + int(LIX);
		int quad_idx = (MAX_QUADS - 1) - large_quad_idx;
		uint bsy, bey;
		if(large_quad_idx < num_quads) {
			const uint shift = BIN_SIZE == 64 ? BIN_SHIFT - TILE_SHIFT : 0;
			uvec4 aabb = decodeAABB32(g_quad_aabbs[quad_idx]) >> shift;
			bsy = aabb[1], bey = aabb[3];
			atomicAdd(s_rows[bsy], 1);
			atomicAdd(s_rows[bey + 1], -1);

			atomicAdd(g_bins.num_bin_large_quads, int((aabb[2] - aabb[0] + 1) * (bey - bsy + 1)));
		}
		barrier();
		// Accumulating current quad counts for each row (within warps)
		if(LIX < BIN_COUNT_Y) {
			int accum = prefixSum32(s_rows[LIX]);
			s_rows[LIX] = accum;
			if((LIX & 31) == 31)
				s_row_warps[LIX >> 5] = accum;
		}
		barrier();
		// Reserving ranges for quad indices in rows
		if(LIX < BIN_COUNT_Y) {
			int prev_warps = 0, warp_id = int(LIX >> 5);
			for(int i = 0; i < warp_id; i++)
				prev_warps += s_row_warps[i];
			int value = s_rows[LIX] + prev_warps;
			if(value > 0)
				s_rows[LIX] = atomicAdd(BIN_ROWS_LARGE_QUAD_OFFSETS_TEMP(LIX), value);
		}
		barrier();
		// Dispatching quad indices to bin rows
		// TODO: this work can be more fairly divided among threads
		if(large_quad_idx < num_quads)
			for(uint by = bsy; by <= bey; by++) {
				uint row_quad_offset = atomicAdd(s_rows[by], 1);
				g_bin_row_quads[row_quad_offset] = (quad_idx & 0xffffff) | (by << 24);
			}

		barrier();
	}
}
