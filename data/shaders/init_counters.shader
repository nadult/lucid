// $$include declarations

#define LID gl_LocalInvocationID
#define LIX gl_LocalInvocationIndex
#define WGID gl_WorkGroupID

#define LSIZE 256

BIN_COUNTERS_BUFFER(0);
layout(std430, binding = 0) buffer buf0_alias_ { uint g_bins_plain[]; };
layout(local_size_x = LSIZE) in;

void main() {
	if(LIX < BIN_COUNTERS_SIZE)
		g_bins_plain[LIX] = 0;
	for(uint i = LIX; i < BIN_COUNT; i += LSIZE)
		BIN_QUAD_COUNTS(i) = 0;
}
