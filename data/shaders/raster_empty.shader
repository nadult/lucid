// $$include funcs data

#define LIX gl_LocalInvocationIndex
#define LID gl_LocalInvocationID

#define LSIZE 128

layout(local_size_x = LSIZE) in;
layout(binding = 0, r32ui) uniform uimage2D final_raster;

layout(std430, binding = 6) buffer buf6_ { BinCounters g_bins; };

void rasterBin(int bin_id) {
	ivec2 bin_pos = ivec2(bin_id % BIN_COUNT_X, bin_id / BIN_COUNT_X) * BIN_SIZE;
	for(uint i = LIX; i < BIN_SIZE * BIN_SIZE; i += LSIZE) {
		ivec2 pixel_pos = ivec2(i & (BIN_SIZE - 1), i >> BIN_SHIFT);
		imageStore(final_raster, bin_pos + pixel_pos, uvec4(0, 0, 0, 0));
	}
}

shared int s_num_bins, s_bin_id;

int loadNextBin() {
	if(LIX == 0) {
		uint bin_idx = atomicAdd(g_bins.empty_bin_counter, 1);
		s_bin_id = bin_idx < s_num_bins? g_bins.empty_bins[bin_idx] : -1;
	}
	barrier();
	return s_bin_id;
}

void main() {
	if(LIX == 0)
		s_num_bins = g_bins.num_empty_bins;
	int bin_id = loadNextBin();
	while(bin_id != -1) {
		barrier();
		rasterBin(bin_id);
		bin_id = loadNextBin();
	}
}
