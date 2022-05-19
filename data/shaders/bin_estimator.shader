// $$include data

// TODO: don't run too many groups if we have small amount of data (indirect dispatch)

#define LID gl_LocalInvocationID
#define LIX gl_LocalInvocationIndex
#define WGID gl_WorkGroupID

#define LSIZE 512

layout(local_size_x = LSIZE) in;
layout(std430, binding = 0) buffer buf0_ { uint g_quad_aabbs[]; };
BIN_COUNTERS_BUFFER(1); // TODO: coherent ?

shared int s_counts[BIN_COUNT];
shared int s_rows[BIN_COUNT_Y];

//#define COMPUTE_WARP_DIVERGENCE
shared int s_warp_divergence;

void countSmallQuadBins(uint quad_idx) {
	uint aabb = g_quad_aabbs[quad_idx];
	int tsx = int(aabb & 0xff), tsy = int((aabb >> 8) & 0xff);
	int tex = int((aabb >> 16) & 0xff), tey = int((aabb >> 24));
	int bsx = tsx >> 2, bsy = tsy >> 2;
	int bex = tex >> 2, bey = tey >> 2;

	// Handling only tris with bin area 1 to 3:
	atomicAdd(s_counts[bsy * BIN_COUNT_X + bsx], 1);
	if(bex != bsx || bey != bsy)
		atomicAdd(s_counts[bey * BIN_COUNT_X + bex], 1);
	int bmx = (bsx + bex) >> 1, bmy = (bsy + bey) >> 1;
	if(bmx > bsx || bmy > bsy)
		atomicAdd(s_counts[bmy * BIN_COUNT_X + bmx], 1);
}

void countLargeQuadBins(uint quad_idx) {
	uint aabb = g_quad_aabbs[quad_idx];
	int tsx = int(aabb & 0xff), tsy = int((aabb >> 8) & 0xff);
	int tex = int((aabb >> 16) & 0xff), tey = int((aabb >> 24));

	int bsx = tsx >> 2, bsy = tsy >> 2;
	int bex = tex >> 2, bey = tey >> 2;
	// ASSERT(bsx >= 0 && bsy >= 0);
	// ASSERT(bex <= BIN_COUNT_X && bey <= BIN_COUNT_Y);

#ifdef COMPUTE_WARP_DIVERGENCE
	// Estimating divergence within warp
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

	bex++, bey++;
	atomicAdd(s_counts[bsx + bsy * BIN_COUNT_X], 1);
	if(bex < BIN_COUNT_X)
		atomicAdd(s_counts[bex + bsy * BIN_COUNT_X], -1);
	if(bey < BIN_COUNT_Y)
		atomicAdd(s_counts[bsx + bey * BIN_COUNT_X], -1);
	if(bex < BIN_COUNT_X && bey < BIN_COUNT_Y)
		atomicAdd(s_counts[bex + bey * BIN_COUNT_X], 1);
}

void computeOffsets() {
	// Accumulating large quad counts
	for(uint i = LIX; i < BIN_COUNT; i += LSIZE)
		s_counts[i] = BIN_QUAD_OFFSETS(i);
	barrier();
	if(LIX < BIN_COUNT_X) {
		int accum = 0;
		for(uint y = 0; y < BIN_COUNT_Y; y++) {
			accum += s_counts[LIX + y * BIN_COUNT_X];
			s_counts[LIX + y * BIN_COUNT_X] = accum;
		}
	}
	barrier();
	if(LIX < BIN_COUNT_Y) {
		int accum = 0;
		for(uint x = 0; x < BIN_COUNT_X; x++) {
			accum += s_counts[x + LIX * BIN_COUNT_X];
			s_counts[x + LIX * BIN_COUNT_X] = accum;
		}
	}
	barrier();
	// Adding small quad counts; updating final quad counts
	for(uint i = LIX; i < BIN_COUNT; i += LSIZE) {
		s_counts[i] += BIN_QUAD_COUNTS(i);
		BIN_QUAD_COUNTS(i) = s_counts[i];
	}
	barrier();

	// Computing per-bin tri offsets
	if(LIX < BIN_COUNT_Y) {
		for(int x = 1; x < BIN_COUNT_X; x++)
			s_counts[x + LIX * BIN_COUNT_X] += s_counts[x - 1 + LIX * BIN_COUNT_X];
		s_rows[LIX] = s_counts[BIN_COUNT_X - 1 + LIX * BIN_COUNT_X];
	}
	barrier();
	if(LIX < BIN_COUNT_X) {
		int prev_sum = 0;
		for(int y = 1; y < BIN_COUNT_Y; y++) {
			prev_sum += s_rows[y - 1];
			s_counts[LIX + y * BIN_COUNT_X] += prev_sum;
		}
	}
	barrier();
	for(uint i = LIX; i < BIN_COUNT; i += LSIZE) {
		int cur_offset = s_counts[i] - BIN_QUAD_COUNTS(i);
		BIN_QUAD_OFFSETS(i) = cur_offset;
		BIN_QUAD_OFFSETS_TEMP(i) = cur_offset;
	}
	barrier();

	if(LIX == 0) {
		g_bins.num_estimated_quads = s_counts[BIN_COUNT - 1];
		g_bins.num_binned_quads = 0;
	}
}

shared int s_num_quads[2];
shared int s_quads_offset;
shared uint s_num_finished;

void main() {
	if(LIX < 2) {
		s_num_quads[LIX] = g_bins.num_visible_quads[LIX];
#ifdef COMPUTE_WARP_DIVERGENCE
		s_warp_divergence = 0;
#endif
	}

	// Initializing bin counters
	for(uint i = LIX; i < BIN_COUNT; i += LSIZE)
		s_counts[i] = 0;
	barrier();

	// Processing small tris
	int num_quads = s_num_quads[0];
	while(true) {
		if(LIX == 0)
			s_quads_offset = atomicAdd(g_bins.num_estimated_visible_quads[0], LSIZE);
		barrier();

		int quad_offset = s_quads_offset;
		if(quad_offset >= num_quads)
			break;

		int quad_idx = quad_offset + int(LIX);
		if(quad_idx < num_quads)
			countSmallQuadBins(quad_idx);
		barrier();
	}

	// Adding small bin counters to global memory buffer
	for(uint i = LIX; i < BIN_COUNT; i += LSIZE)
		if(s_counts[i] > 0) {
			atomicAdd(BIN_QUAD_COUNTS(i), s_counts[i]);
			s_counts[i] = 0;
		}
	barrier();

	// Processing big tris
	num_quads = s_num_quads[1];
	while(true) {
		if(LIX == 0)
			s_quads_offset = atomicAdd(g_bins.num_estimated_visible_quads[1], LSIZE);
		barrier();

		int quad_offset = s_quads_offset;
		if(quad_offset >= num_quads)
			break;

		int quad_idx = quad_offset + int(LIX);
		if(quad_idx < num_quads)
			countLargeQuadBins((MAX_QUADS - 1) - quad_idx);
		barrier();
	}

	// Adding large bin counters to global memory buffer;
	// We're storing it temporarily in offsets buffer
	for(uint i = LIX; i < BIN_COUNT; i += LSIZE)
		if(s_counts[i] != 0)
			atomicAdd(BIN_QUAD_OFFSETS(i), s_counts[i]);

#ifdef COMPUTE_WARP_DIVERGENCE
	if(LIX == 0)
		atomicAdd(g_bins.temp[0], s_warp_divergence >> 5);
#endif

	// TODO: additional group memory barrier to make sure that changes in mem are propagated to all other groups?
	barrier();

	// TODO: kill redundant work groups earlier
	if(LIX == 0)
		s_num_finished = atomicAdd(g_bins.num_finished_bin_groups, 1);
	barrier();

	// Last group is responsible for computing offsets
	if(s_num_finished == gl_NumWorkGroups.x - 1) {
		groupMemoryBarrier();
		computeOffsets();
	}
}
