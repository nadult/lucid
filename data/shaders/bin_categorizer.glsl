#include "shared/compute_funcs.glsl"
#include "shared/funcs.glsl"
#include "shared/structures.glsl"

#define LSIZE BIN_CATEGORIZER_LSIZE
layout(local_size_x = 512, local_size_x_id = BIN_CATEGORIZER_LSIZE_ID) in;

coherent layout(std430, set = 0, binding = 0) buffer lucid_info_ {
	LucidInfo g_info;
	int g_counts[];
};
layout(binding = 1) uniform lucid_config_ { LucidConfig u_config; };

shared int s_bin_level_counts[BIN_LEVELS_COUNT];
shared int s_bins[BIN_COUNT];
shared int s_temp[BIN_COUNT / WARP_SIZE + 1], s_temp2[WARP_SIZE];

// TODO: we could run it in a separate kernel during categorization (independent tasks)
void computeOffsets(const bool quads_mode) {
	for(uint idx = LIX; idx < BIN_COUNT; idx += LSIZE) {
		int value = quads_mode ? BIN_QUAD_COUNTS(idx) : BIN_TRI_COUNTS(idx);
		int accum = subgroupInclusiveAddFast(value);
		s_bins[idx] = accum - value;
		if((idx & WARP_MASK) == WARP_MASK)
			s_temp[idx >> WARP_SHIFT] = accum;
	}
	barrier();
	if(LIX < BIN_COUNT / WARP_SIZE)
		s_temp[LIX] = subgroupInclusiveAddFast(s_temp[LIX]);
	barrier();
	if(LIX < (BIN_COUNT / WARP_SIZE) / WARP_SIZE)
		s_temp2[LIX] = subgroupInclusiveAddFast(s_temp[(LIX << WARP_SHIFT) + WARP_MASK]);
	barrier();
	if(LIX < BIN_COUNT / WARP_SIZE && LIX >= WARP_SIZE)
		s_temp[LIX] += s_temp2[int(LIX >> WARP_SHIFT) - 1];
	barrier();
	for(uint idx = LIX; idx < BIN_COUNT; idx += LSIZE) {
		int widx = int(idx >> WARP_SHIFT) - 1;
		int accum = s_bins[idx];
		if(widx >= 0)
			accum += s_temp[widx];
		if(quads_mode) {
			BIN_QUAD_OFFSETS(idx) = accum;
			BIN_QUAD_OFFSETS_TEMP(idx) = accum;
		} else {
			BIN_TRI_OFFSETS(idx) = accum;
			BIN_TRI_OFFSETS_TEMP(idx) = accum;
		}
	}
}

void main() {
	if(LIX < BIN_LEVELS_COUNT)
		s_bin_level_counts[LIX] = 0;
	computeOffsets(true);
	barrier();
	computeOffsets(false);

	for(uint i = LIX; i < BIN_COUNT; i += LSIZE) {
		int num_quads = BIN_QUAD_COUNTS(i);
		int num_tris = BIN_TRI_COUNTS(i) + num_quads * 2;

		// TODO: add micro phase (< 64/128?)
		if(num_tris == 0) {
			atomicAdd(s_bin_level_counts[0], 1);
		} else if(num_tris < 1024) {
			// TODO: On gallery, dragon, san-miguel setting limit to 512 for low increases perf, why?
			int id = atomicAdd(s_bin_level_counts[BIN_LEVEL_LOW], 1);
			LOW_LEVEL_BINS(id) = int(i);
		} else if(true) {
			int id = atomicAdd(s_bin_level_counts[BIN_LEVEL_HIGH], 1);
			HIGH_LEVEL_BINS(id) = int(i);
		}
	}

	barrier();
	if(LIX < BIN_LEVELS_COUNT) {
		g_info.bin_level_counts[LIX] = s_bin_level_counts[LIX];
		int max_dispatches = MAX_DISPATCHES >> (LIX == BIN_LEVEL_HIGH ? 1 : 0);
		g_info.bin_level_dispatches[LIX][0] = min(s_bin_level_counts[LIX], max_dispatches);
		g_info.bin_level_dispatches[LIX][1] = 1;
		g_info.bin_level_dispatches[LIX][2] = 1;
	}
}
