// $$include data

#define LID gl_LocalInvocationID
#define LIX gl_LocalInvocationIndex
#define WGID gl_WorkGroupID

// TODO: this is too small for large number of bins!
#define LSIZE 512

layout(local_size_x = LSIZE) in;
layout(std430, binding = 0) buffer buf0_ { uint g_quad_aabbs[]; };
BIN_COUNTERS_BUFFER(1);

shared int s_num_quads, s_max_quads;
shared int s_quads_offset;
shared uint s_num_finished;

shared int s_counts[BIN_COUNT];
shared int s_rows[BIN_COUNT_Y];

void countQuadBins(uint quad_idx) {
	uint aabb = g_quad_aabbs[quad_idx];
	int tsx = int(aabb & 0xff), tsy = int((aabb >> 8) & 0xff);
	int tex = int((aabb >> 16) & 0xff), tey = int((aabb >> 24));

	int bsx = tsx >> 2, bsy = tsy >> 2;
	int bex = tex >> 2, bey = tey >> 2;
	// ASSERT(bsx >= 0 && bsy >= 0);
	// ASSERT(bex <= BIN_COUNT_X && bey <= BIN_COUNT_Y);

	for(int by = bsy; by <= bey; by++)
		for(int bx = bsx; bx <= bex; bx++)
			atomicAdd(s_counts[bx + by * BIN_COUNT_X], 1);
}

void estimateBins() {
	int max_quads = s_max_quads;
	if(LIX == 0)
		s_quads_offset = atomicAdd(g_bins.num_binned_quads, max_quads);
	barrier();

	int quads_offset = s_quads_offset;
	int num_quads = clamp(s_num_quads - quads_offset, 0, max_quads);
	if(num_quads == 0)
		return;

	for(uint i = LIX; i < BIN_COUNT; i += LSIZE)
		s_counts[i] = 0;
	barrier();

	for(int i = int(LIX); i < num_quads; i += LSIZE)
		countQuadBins(quads_offset + i);
	barrier();

	for(uint i = LIX; i < BIN_COUNT; i += LSIZE)
		if(s_counts[i] > 0)
			atomicAdd(BIN_QUAD_COUNTS(i), s_counts[i]);
}

void computeOffsets() {
	// Loading tri counts
	for(uint i = LIX; i < BIN_COUNT; i += LSIZE)
		s_counts[i] = BIN_QUAD_COUNTS(i);
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

void main() {
	if(LIX == 0) {
		int num_quads = g_bins.num_visible_quads;
		s_num_quads = num_quads;
		// TODO: does it make sense to make this value smaller than LSIZE ?
		int quads_per_wg = int((num_quads + gl_NumWorkGroups.x - 1) / gl_NumWorkGroups.x);
		s_max_quads = max(quads_per_wg, LSIZE);
	}
	barrier();

	estimateBins();
	barrier();
	if(LIX == 0)
		s_num_finished = atomicAdd(g_bins.num_finished_bin_groups, 1);
	barrier();

	// Last group is responsible for computing offsets
	if(s_num_finished == gl_NumWorkGroups.x - 1) {
		groupMemoryBarrier();
		computeOffsets();
	}
}
