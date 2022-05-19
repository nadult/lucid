// $$include data

#define LID gl_LocalInvocationID
#define LIX gl_LocalInvocationIndex
#define WGID gl_WorkGroupID

#define LSIZE 128

BIN_COUNTERS_BUFFER(0);
TILE_COUNTERS_BUFFER(1);
layout(std430, binding = 0) buffer buf0_alias_ { uint g_bins_plain[]; };
layout(std430, binding = 1) buffer buf1_alias_ { uint g_tiles_plain[]; };
layout(local_size_x = LSIZE) in;

void main() {
	if(LIX < 64) {
		g_bins_plain[LIX] = 0;
		g_tiles_plain[LIX] = 0;
	}
	for(uint i = LIX; i < BIN_COUNT; i += LSIZE) {
		BIN_QUAD_COUNTS(i) = 0;
		BIN_QUAD_OFFSETS(i) = 0;
	}
}
