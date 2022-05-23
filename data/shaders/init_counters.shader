// $$include declarations

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
	if(LIX < BIN_COUNTERS_SIZE)
		g_bins_plain[LIX] = 0;
	if(LIX < TILE_COUNTERS_SIZE)
		g_tiles_plain[LIX] = 0;
	for(uint i = LIX; i < BIN_COUNT; i += LSIZE)
		BIN_QUAD_COUNTS(i) = 0;
	for(uint i = LIX; i < LARGE_BIN_COUNT; i += LSIZE)
		LARGE_BIN_QUAD_COUNTS(i) = 0;
	for(uint i = LIX; i < MAX_DISPATCHES; i += LSIZE)
		BIN_WORKGROUP_ITEMS(WGID.x, MAX_BIN_WORKGROUP_ITEMS - 1) = 0;
}
