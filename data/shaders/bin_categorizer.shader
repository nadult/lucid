// $$include funcs data

#define LIX gl_LocalInvocationIndex
#define LID gl_LocalInvocationID
#define LSIZE 128

layout(local_size_x = LSIZE) in;
layout(std430, binding = 1) buffer buf1_ { BinCounters g_bins; };

shared int s_num_small_bins, s_num_medium_bins, s_num_big_bins;

void main() {
	if(LIX == 0) {
		s_num_small_bins = 0;
		s_num_medium_bins = 0;
		s_num_big_bins = 0;
	}
	barrier();

	// TODO: special shader for empty bins?
	// TODO: problem w tym, że aby dobrze porozdzielac taski musimy znac liczbe sampli...
	//       da się to jakoś oszacować? co najmniej w tile dispatcherze...
	for(uint i = LIX; i < BIN_COUNT; i += LSIZE) {
		int num_quads = g_bins.bin_quad_counts[i];
		
		// TODO: we need accurate count :(
		int num_tris = num_quads * 2;
		if(num_tris < 256) {
			int idx = atomicAdd(s_num_small_bins, 1);
			g_bins.small_bins[idx] = int(i);
		}
		else if(num_tris < 6 * 1024) {
			int idx = atomicAdd(s_num_medium_bins, 1);
			g_bins.medium_bins[idx] = int(i);
		}
		else if(true) {
			int idx = atomicAdd(s_num_big_bins, 1);
			g_bins.big_bins[idx] = int(i);
		}
	}

	barrier();
	if(LIX == 0) {
		g_bins.num_small_bins = s_num_small_bins;
		g_bins.num_medium_bins = s_num_medium_bins;
		g_bins.num_big_bins = s_num_big_bins;
	}
}
