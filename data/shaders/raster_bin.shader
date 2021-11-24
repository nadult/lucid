// $$include funcs lighting frustum viewport data

#define LIX gl_LocalInvocationIndex
#define LID gl_LocalInvocationID

#define LSIZE 256
#define LSHIFT 8

layout(local_size_x = LSIZE) in;
layout(binding = 0, r32ui) uniform uimage2D final_raster;

layout(std430, binding = 6) buffer buf6_ { BinCounters g_bins; };
layout(std430, binding = 7) buffer buf7_ { TileCounters g_tiles; };

shared ivec2 s_bin_pos;

void rasterInvalidBin(vec3 color)
{
	for(uint i = LIX; i < BIN_SIZE * BIN_SIZE; i += LSIZE) {
		ivec2 pixel_pos = ivec2(i & (BIN_SIZE - 1), i >> BIN_SHIFT);
		vec4 color = vec4(color, 1.0);
		uint enc_col = encodeRGBA8(color);
		imageStore(final_raster, s_bin_pos + pixel_pos, uvec4(enc_col, 0, 0, 0));
	}
}

void rasterBins(int bin_id) {
	if(LIX < TILES_PER_BIN) {
		ivec2 bin_pos = ivec2(bin_id % BIN_COUNT_X, bin_id / BIN_COUNT_X) * BIN_SIZE;
		if(LIX == 0)
			s_bin_pos = bin_pos;
	}
	barrier();
	rasterInvalidBin(vec3(0.7, 0.7, 0.7));
}

shared int s_num_bins, s_bin_id;

int loadNextBin() {
	if(LIX == 0) {
		uint bin_idx = atomicAdd(g_tiles.small_bin_counter, 1);
		s_bin_id = bin_idx < s_num_bins? g_bins.small_bins[bin_idx] : -1;
	}
	barrier();
	return s_bin_id;
}

void main() {
	if(LIX == 0)
		s_num_bins = g_bins.num_small_bins;
	int bin_id = loadNextBin();
	while(bin_id != -1) {
		barrier();
		rasterBins(bin_id);
		bin_id = loadNextBin();
	}
}
