#include "shared/funcs.glsl"
#include "shared/structures.glsl"

#define LSIZE BIN_CATEGORIZER_LSIZE
layout(local_size_x = 512, local_size_x_id = 17) in;

coherent layout(std430, binding = 0) buffer lucid_info_ {
	LucidInfo g_info;
	int g_counts[];
};
layout(binding = 1) uniform lucid_config_ { LucidConfig u_config; };
layout(std430, binding = 2) buffer buf1_ { uint g_compose_quads[]; };

shared int s_bin_level_counts[BIN_LEVELS_COUNT];

void main() {
	if(LIX < BIN_LEVELS_COUNT)
		s_bin_level_counts[LIX] = 0;
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
