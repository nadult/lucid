// $$include funcs data

#define LIX gl_LocalInvocationIndex
#define LID gl_LocalInvocationID
#define LSIZE 128

layout(local_size_x = LSIZE) in;
layout(std430, binding = 1) buffer buf1_ { BinCounters g_bins; };
layout(std430, binding = 2) buffer buf2_ { vec2 g_bin_quads[]; };

shared int s_num_empty_bins, s_num_small_bins, s_num_medium_bins;
shared int s_num_big_bins, s_num_tiled_bins;

uniform bool tile_all_bins;
uniform ivec2 bin_counts;
uniform vec2 bin_scale;

void main() {
	if(LIX == 0) {
		s_num_empty_bins = 0;
		s_num_small_bins = 0;
		s_num_medium_bins = 0;
		s_num_big_bins = 0;
		s_num_tiled_bins = 0;
	}
	barrier();

	// TODO: problem w tym, że aby dobrze porozdzielac taski musimy znac liczbe sampli...
	//       da się to jakoś oszacować? co najmniej w tile dispatcherze...
	for(uint i = LIX; i < BIN_COUNT; i += LSIZE) {
		int num_quads = g_bins.bin_quad_counts[i];

		int bin_y = int(i / bin_counts.x);
		int bin_x = int(i - bin_y * bin_counts.x);
		
		// TODO: we need accurate count :(
		int num_tris = num_quads * 2;
		if(num_tris == 0) {
			int id = atomicAdd(s_num_empty_bins, 1);
			if(tile_all_bins)
				g_bins.tiled_bins[atomicAdd(s_num_tiled_bins, 1)] = int(i);
		}
		else if(num_tris < 2048) {
			g_bins.small_bins[atomicAdd(s_num_small_bins, 1)] = int(i);
			if(tile_all_bins)
				g_bins.tiled_bins[atomicAdd(s_num_tiled_bins, 1)] = int(i);
		}
		else if(num_tris < 6 * 1024) {
			g_bins.medium_bins[atomicAdd(s_num_medium_bins, 1)] = int(i);
			g_bins.tiled_bins[atomicAdd(s_num_tiled_bins, 1)] = int(i);
		}
		else if(true) {
			g_bins.big_bins[atomicAdd(s_num_big_bins, 1)] = int(i);
			g_bins.tiled_bins[atomicAdd(s_num_tiled_bins, 1)] = int(i);
		}

		vec2 bin_pos = (num_tris == 0? vec2(-100, -100) : vec2(bin_x, bin_y)) * bin_scale;
		g_bin_quads[i * 4 + 0] = bin_pos;
		g_bin_quads[i * 4 + 1] = bin_pos + vec2(0.0, bin_scale.y);
		g_bin_quads[i * 4 + 2] = bin_pos + bin_scale;
		g_bin_quads[i * 4 + 3] = bin_pos + vec2(bin_scale.x, 0.0);
	}

	barrier();
	if(LIX == 0) {
		g_bins.num_empty_bins = s_num_empty_bins;
		g_bins.num_small_bins = s_num_small_bins;
		g_bins.num_medium_bins = s_num_medium_bins;
		g_bins.num_big_bins = s_num_big_bins;
		g_bins.num_tiled_bins = s_num_tiled_bins;
	}
}
