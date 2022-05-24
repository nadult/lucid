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
layout(std430, binding = 3) buffer buf2_ { uint g_bin_quads[]; };

shared int s_bins[BIN_COUNT];
shared int s_rows[BIN_COUNT_Y];
shared int s_large_bins[LARGE_BIN_COUNT];

// Constants useful for efficient processing of bin counts
const int xbin_warps = LSIZE / BIN_COUNT_X, ybin_warps = LSIZE / BIN_COUNT_Y;
const int xbin_step = int(log2(xbin_warps)), ybin_step = int(log2(ybin_warps));
const int xbin_count = 1 << xbin_step, ybin_count = 1 << ybin_step;

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

	/*if(lbx >= 0 && lbx < LARGE_BIN_COUNT_X) {
		int accum = 0;
		for(uint lby = 0; lby < LARGE_BIN_COUNT_Y; lby++) {
			accum += s_large_bins[lbx + lby * LARGE_BIN_COUNT_X];
			s_large_bins[lbx + lby * LARGE_BIN_COUNT_X] = accum;
		}
	}*/
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

	/*if(lby >= 0 && lby < LARGE_BIN_COUNT_Y) {
		int accum = 0;
		for(uint lbx = 0; lbx < LARGE_BIN_COUNT_X; lbx++) {
			accum += s_large_bins[lbx + lby * LARGE_BIN_COUNT_X];
			s_large_bins[lbx + lby * LARGE_BIN_COUNT_X] = accum;
		}
	}*/

	barrier();
	/*for(uint i = LIX; i < LARGE_BIN_COUNT; i += LSIZE)
		atomicAdd(LARGE_BIN_QUAD_COUNTS(i), s_large_bins[i]);*/
}

void computeOffsets() {
	// TODO: no need for groupMemoryBarrier ?

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
	if(LIX < BIN_COUNT_Y)
		s_rows[LIX] = BIN_QUAD_OFFSETS((BIN_COUNT_X - 1) + LIX * BIN_COUNT_X);
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

shared int s_num_quads[2], s_num_cur_quads, s_size_type;
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
			int num_small_items = (s_num_quads[0] + ITEM_SIZE - 1) / ITEM_SIZE;
			int num_large_items = (s_num_quads[1] + ITEM_SIZE - 1) / ITEM_SIZE;
			s_num_all_items = num_small_items + num_large_items;
			s_item_id = 0;
		}
	}
	for(uint i = LIX; i < BIN_COUNT; i += LSIZE)
		s_bins[i] = 0;
	barrier();

	// Computing large quads bin coverage
	int num_quads = s_num_quads[1];
	while(s_item_id < MAX_BIN_WORKGROUP_ITEMS - 1) {
		if(LIX == 0) {
			int quads_offset = atomicAdd(g_bins.num_estimated_visible_quads[1], ITEM_SIZE);
			if(quads_offset < num_quads)
				s_workgroup_items[s_item_id++] = -quads_offset - 1;
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
	if(LIX == 0)
		while(g_bins.a_dispatcher_phase == 0)
			;
	barrier();

	// Reserving space for quad indices in bins
	for(uint i = LIX; i < BIN_COUNT; i += LSIZE)
		if(s_bins[i] > 0)
			s_bins[i] = atomicAdd(BIN_QUAD_OFFSETS_TEMP(i), s_bins[i]);
	barrier();

	// Dispatching quads
	while(s_item_id < s_num_items) {
		barrier();
		if(LIX == 0) {
			int quads_offset = s_workgroup_items[s_item_id++];
			int size_type = quads_offset < 0 ? 1 : 0;
			s_num_cur_quads = s_num_quads[size_type];
			s_quads_offset = quads_offset < 0 ? -quads_offset - 1 : quads_offset;
			s_size_type = size_type;
		}
		barrier();
		int quads_offset = s_quads_offset;
		int num_quads = s_num_cur_quads;

		for(int s = 0; s < ITEM_STEPS; s++) {
			int quad_idx = quads_offset + LSIZE * s + int(LIX);
			if(quad_idx >= num_quads)
				break;

			if(s_size_type == 1)
				quad_idx = (MAX_QUADS - 1) - quad_idx;
			dispatchQuad(quad_idx);
		}
	}
}
