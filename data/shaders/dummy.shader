// $$include funcs data

// This program isn't doing anything useful
// It can be used for measuring performance of simple constructs

// Iteration over all bins (Geforce 1050):
//   workgroups:   32      64      128     256      512
//   64 threads:   10 us   7 us    5 us    6 us     7 us
//  128 threads:   10 us   7 us    7 us    7 us     9 us
//  256 threads:   10 us   9 us   10 us   11 us    14 us
//  512 threads:   15 us  15 us   17 us   20 us    26 us
// 1024 threads:   27 us  30 us   32 us   39 us    51 us
// 1536 threads:   53 us  56 us   62 us   75 us   100 us

#define LIX gl_LocalInvocationIndex
#define LID gl_LocalInvocationID
#define LSIZE 256

layout(local_size_x = LSIZE) in;

BIN_COUNTERS_BUFFER(0);
TILE_COUNTERS_BUFFER(1);

// Using this SMEM variable directly increases running time by 7%
shared int s_bin_id;
shared uint s_bin_tri_count;

void countTris(int bin_id) {
	if(LIX < 16)
		atomicAdd(s_bin_tri_count, TILE_TRI_COUNTS(bin_id, LIX));
}

int loadNextBin() {
	if(LIX == 0)
		s_bin_id = int(atomicAdd(g_tiles.final_raster_bin_counter, 1));
	barrier();
	return s_bin_id;
}

void main() {
	if(LIX == 0)
		s_bin_tri_count = 0;
	int bin_id = loadNextBin();
	while(bin_id < BIN_COUNT) {
		barrier();
		countTris(bin_id);
		bin_id = loadNextBin();
	}
	if(LIX == 0)
		g_bins.temp[0] = int(s_bin_tri_count);
}
