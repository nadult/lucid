// $$include data

#define LID gl_LocalInvocationID
#define LIX gl_LocalInvocationIndex
#define WGID gl_WorkGroupID

#define LSIZE 32

layout(std430, binding = 0) buffer buf0_ { BinCounters  g_bins; };
layout(std430, binding = 1) buffer buf1_ { TileCounters g_tiles; };
layout(std430, binding = 0) buffer buf0_alias_ { uint g_bins_plain[]; };
layout(std430, binding = 1) buffer buf1_alias_ { uint g_tiles_plain[]; };
layout(local_size_x = LSIZE) in;

uniform int num_verts;

void main() {
	if(LIX < 32) {
		g_bins_plain[LIX] = 0;
		g_tiles_plain[LIX] = 0;
	}
	for(uint i = LIX; i < BIN_COUNT; i += LSIZE)
		g_bins.bin_quad_counts[i] = 0;
}
