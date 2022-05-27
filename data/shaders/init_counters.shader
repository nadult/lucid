// $$include structures

#define LID gl_LocalInvocationID
#define LIX gl_LocalInvocationIndex
#define WGID gl_WorkGroupID

#define LSIZE 256

layout(std430, binding = 0) buffer buf0_alias_ { uint g_info_plain[]; };
layout(local_size_x = LSIZE) in;

void main() {
	for(uint i = LIX; i < LUCID_INFO_SIZE; i += LSIZE)
		g_info_plain[i] = 0;
	for(uint i = LIX; i < BIN_COUNT; i += LSIZE)
		BIN_QUAD_COUNTS(i) = 0;
}
