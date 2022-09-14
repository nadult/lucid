#include "shared/funcs.glsl"
#include "shared/structures.glsl"

#define LSIZE BIN_CATEGORIZER_LSIZE
layout(local_size_x = 512, local_size_x_id = BIN_CATEGORIZER_LSIZE_ID) in;

#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_shuffle_relative : require
#extension GL_ARB_shader_group_vote : require

coherent layout(std430, set = 0, binding = 0) buffer lucid_info_ {
	LucidInfo g_info;
	int g_counts[];
};
layout(binding = 1) uniform lucid_config_ { LucidConfig u_config; };
layout(std430, set = 1, binding = 0) buffer buf0_ { uint g_compose_quads[]; };

shared int s_bin_level_counts[BIN_LEVELS_COUNT];
shared int s_bins[BIN_COUNT];
shared int s_temp[BIN_COUNT / WARP_SIZE + 1], s_temp2[WARP_SIZE];

int prefixSum(int accum) {
	int temp;
	uint thread_id = LIX & WARP_MASK; // TODO: use gl_
	temp = subgroupShuffleUp(accum, 1), accum += thread_id >= 1 ? temp : 0;
	temp = subgroupShuffleUp(accum, 2), accum += thread_id >= 2 ? temp : 0;
	temp = subgroupShuffleUp(accum, 4), accum += thread_id >= 4 ? temp : 0;
	temp = subgroupShuffleUp(accum, 8), accum += thread_id >= 8 ? temp : 0;
#if WARP_SIZE >= 32
	temp = subgroupShuffleUp(accum, 16), accum += thread_id >= 16 ? temp : 0;
#endif
#if WARP_SIZE >= 64
	temp = subgroupShuffleUp(accum, 32), accum += thread_id >= 32 ? temp : 0;
#endif
	return accum;
}

void computeOffsets() {
	for(uint idx = LIX; idx < BIN_COUNT; idx += LSIZE) {
		int value = s_bins[idx];
		int accum = prefixSum(value);
		s_bins[idx] = accum - value;
		if((idx & WARP_MASK) == WARP_MASK)
			s_temp[idx >> WARP_SHIFT] = accum;
	}
	barrier();
	if(LIX < BIN_COUNT / WARP_SIZE)
		s_temp[LIX] = prefixSum(s_temp[LIX]);
	barrier();
	if(LIX < (BIN_COUNT / WARP_SIZE) / WARP_SIZE)
		s_temp2[LIX] = prefixSum(s_temp[(LIX << WARP_SHIFT) + WARP_MASK]);
	barrier();
	if(LIX < BIN_COUNT / WARP_SIZE && LIX >= WARP_SIZE)
		s_temp[LIX] += s_temp2[int(LIX >> WARP_SHIFT) - 1];
	barrier();
	for(uint idx = LIX; idx < BIN_COUNT; idx += LSIZE) {
		int widx = int(idx >> WARP_SHIFT) - 1;
		int accum = s_bins[idx];
		if(widx >= 0)
			accum += s_temp[widx];
		s_bins[idx] = accum;
	}
}

shared uint s_num_bin_workgroups;

void accumulateSmallQuads() {
	for(uint i = LIX; i < BIN_COUNT; i += LSIZE)
		s_bins[i] = BIN_QUAD_COUNTS(i);
	barrier();
	computeOffsets();
	barrier();
	for(uint i = LIX; i < BIN_COUNT; i += LSIZE) {
		BIN_QUAD_OFFSETS(i) = s_bins[i];
		BIN_QUAD_OFFSETS_TEMP(i) = s_bins[i];
	}
}

void accumulateLargeTris() {
	for(uint i = LIX; i < BIN_COUNT; i += LSIZE)
		s_bins[i] = BIN_TRI_COUNTS(i);
	barrier();
	computeOffsets();
	barrier();
	for(uint i = LIX; i < BIN_COUNT; i += LSIZE) {
		BIN_TRI_OFFSETS(i) = s_bins[i];
		BIN_TRI_OFFSETS_TEMP(i) = s_bins[i];
	}
}

void main() {
	if(LIX < BIN_LEVELS_COUNT)
		s_bin_level_counts[LIX] = 0;
	if(LIX == 0)
		s_num_bin_workgroups = g_info.num_binning_dispatches[0];
	barrier();

	accumulateSmallQuads();
	barrier();

	accumulateLargeTris();
	barrier();

	groupMemoryBarrier();
	barrier();

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

		uint bin_id = i << 16;
		uint mask = num_tris == 0 ? 0 : 0xffffffff;
		g_compose_quads[i * 4 + 0] = mask & (bin_id);
		g_compose_quads[i * 4 + 1] = mask & (bin_id + (BIN_SIZE << 8));
		g_compose_quads[i * 4 + 2] = mask & (bin_id + BIN_SIZE + (BIN_SIZE << 8));
		g_compose_quads[i * 4 + 3] = mask & (bin_id + BIN_SIZE);
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
