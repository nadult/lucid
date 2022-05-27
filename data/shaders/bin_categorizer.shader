// $$include funcs structures

#define LIX gl_LocalInvocationIndex
#define LID gl_LocalInvocationID

#if BIN_SIZE == 64
#define LSIZE 128
#else
#define LSIZE 512
#endif

layout(local_size_x = LSIZE) in;
layout(std430, binding = 1) buffer buf1_ { uint g_compose_quads[]; };

shared int s_bin_level_counts[BIN_LEVELS_COUNT];

void main() {
	if(LIX < BIN_LEVELS_COUNT)
		s_bin_level_counts[LIX] = 0;
	barrier();

	// TODO: problem w tym, że aby dobrze porozdzielac taski musimy znac liczbe sampli...
	//       da się to jakoś oszacować? co najmniej w tile dispatcherze...
	for(uint i = LIX; i < BIN_COUNT; i += LSIZE) {
		int num_quads = BIN_QUAD_COUNTS(i);

		// TODO: add micro phase (< 64/128?)
		// TODO: we need accurate count :(
		int num_tris = num_quads * 2;
		if(num_tris == 0) {
			atomicAdd(s_bin_level_counts[0], 1);
		} else if(num_tris < 2048) {
			int id = atomicAdd(s_bin_level_counts[BIN_LEVEL_LOW], 1);
			LOW_LEVEL_BINS(id) = int(i);
		} else if(num_tris < 32 * 1024) {
			int id = atomicAdd(s_bin_level_counts[BIN_LEVEL_MEDIUM], 1);
			MEDIUM_LEVEL_BINS(id) = int(i);
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
		g_info.bin_level_dispatches[LIX][0] = min(s_bin_level_counts[LIX], MAX_DISPATCHES);
		g_info.bin_level_dispatches[LIX][1] = 1;
		g_info.bin_level_dispatches[LIX][2] = 1;
	}
}
