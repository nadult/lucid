// $$include funcs data

#define LIX gl_LocalInvocationIndex
#define LID gl_LocalInvocationID
#define LSIZE 128

layout(local_size_x = LSIZE) in;

BIN_COUNTERS_BUFFER(1);
layout(std430, binding = 2) buffer buf2_ { uint g_bin_quads[]; };

shared int s_num_empty_bins, s_num_small_bins, s_num_medium_bins;
shared int s_num_big_bins, s_num_tiled_bins;

uniform bool tile_all_bins;

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
		int num_quads = BIN_QUAD_COUNTS(i);

		// TODO: add micro phase (< 32?)
		// TODO: we need accurate count :(
		int num_tris = num_quads * 2;
		if(num_tris == 0) {
			// TODO: we're not using BIN_EMPTY_BINS ?
			int id = atomicAdd(s_num_empty_bins, 1);
			if(tile_all_bins)
				BIN_TILED_BINS(atomicAdd(s_num_tiled_bins, 1)) = int(i);
		} else if(num_tris < 2048) {
			BIN_SMALL_BINS(atomicAdd(s_num_small_bins, 1)) = int(i);
			if(tile_all_bins)
				BIN_TILED_BINS(atomicAdd(s_num_tiled_bins, 1)) = int(i);
		} else if(num_tris < 16 * 1024) {
			BIN_MEDIUM_BINS(atomicAdd(s_num_medium_bins, 1)) = int(i);
			BIN_TILED_BINS(atomicAdd(s_num_tiled_bins, 1)) = int(i);
		} else if(true) {
			BIN_BIG_BINS(atomicAdd(s_num_big_bins, 1)) = int(i);
			BIN_TILED_BINS(atomicAdd(s_num_tiled_bins, 1)) = int(i);
		}

		uint bin_id = i << 16;
		uint mask = num_tris == 0 ? 0 : 0xffffffff;
		g_bin_quads[i * 4 + 0] = mask & (bin_id);
		g_bin_quads[i * 4 + 1] = mask & (bin_id + (BIN_SIZE << 8));
		g_bin_quads[i * 4 + 2] = mask & (bin_id + BIN_SIZE + (BIN_SIZE << 8));
		g_bin_quads[i * 4 + 3] = mask & (bin_id + BIN_SIZE);
	}

	barrier();
	if(LIX == 0) {
		g_bins.num_empty_bins = s_num_empty_bins;
		g_bins.num_small_bins = s_num_small_bins;
		g_bins.num_medium_bins = s_num_medium_bins;
		g_bins.num_big_bins = s_num_big_bins;
		g_bins.num_tiled_bins = s_num_tiled_bins;

#define SET_NUM_DISPATCHES(name, value)                                                            \
	{                                                                                              \
		g_bins.name[0] = min((value), MAX_DISPATCHES);                                             \
		g_bins.name[1] = 1;                                                                        \
		g_bins.name[2] = 1;                                                                        \
	}

		SET_NUM_DISPATCHES(num_tiling_dispatches, s_num_tiled_bins);
		SET_NUM_DISPATCHES(num_bin_raster_dispatches, s_num_small_bins);
		SET_NUM_DISPATCHES(num_tile_raster_dispatches, s_num_medium_bins);
		SET_NUM_DISPATCHES(num_block_raster_dispatches, s_num_big_bins);

#undef SET_NUM_DISPATCHES
	}
}
