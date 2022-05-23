// $$include structures

#define BIN_COUNTERS_BUFFER(idx)                                                                   \
	layout(std430, binding = idx) buffer buf##idx##_ {                                             \
		BinCounters g_bins;                                                                        \
		int g_bins_counts[];                                                                       \
	}

#define BIN_QUAD_COUNTS(idx) g_bins_counts[BIN_COUNT * 0 + (idx)]
#define BIN_QUAD_OFFSETS(idx) g_bins_counts[BIN_COUNT * 1 + (idx)]
#define BIN_QUAD_OFFSETS_TEMP(idx) g_bins_counts[BIN_COUNT * 2 + (idx)]

#define LARGE_BIN_QUAD_COUNTS(idx) g_bins_counts[BIN_COUNT * 3 + (idx)]
#define LARGE_BIN_QUAD_OFFSETS(idx) g_bins_counts[BIN_COUNT * 4 + (idx)]
#define LARGE_BIN_QUAD_OFFSETS_TEMP(idx) g_bins_counts[BIN_COUNT * 5 + (idx)]

// Bins are categorized based on quad count
#define BIN_SMALL_BINS(idx) g_bins_counts[BIN_COUNT * 6 + (idx)]
#define BIN_MEDIUM_BINS(idx) g_bins_counts[BIN_COUNT * 7 + (idx)]
#define BIN_BIG_BINS(idx) g_bins_counts[BIN_COUNT * 8 + (idx)]
#define BIN_TILED_BINS(idx) g_bins_counts[BIN_COUNT * 9 + (idx)]

#define BIN_WORKGROUP_COUNTS(wg_id, idx) g_bins_counts[BIN_COUNT * (10 + (wg_id)) + (idx)]
#define BIN_WORKGROUP_ITEMS(wg_id, idx)                                                            \
	g_bins_counts[BIN_COUNT * (10 + MAX_DISPATCHES) + (wg_id)*MAX_BIN_WORKGROUP_ITEMS + (idx)]

#define TILE_COUNTERS_BUFFER(idx)                                                                  \
	layout(std430, binding = idx) buffer buf##idx##_ {                                             \
		TileCounters g_tiles;                                                                      \
		uint g_tiles_counts[];                                                                     \
	}

// TODO: 16-bit counters?
#define TILE_TRI_COUNTS(bin, tile)                                                                 \
	g_tiles_counts[BIN_COUNT * TILES_PER_BIN * 0 + (((bin)*TILES_PER_BIN) + (tile))]
#define TILE_TRI_OFFSETS(bin, tile)                                                                \
	g_tiles_counts[BIN_COUNT * TILES_PER_BIN * 1 + (((bin)*TILES_PER_BIN) + (tile))]
#define TILE_BLOCK_TRI_COUNTS(bin, tile)                                                           \
	g_tiles_counts[BIN_COUNT * TILES_PER_BIN * 2 + (((bin)*TILES_PER_BIN) + (tile))]
#define TILE_BLOCK_TRI_OFFSETS(bin, tile)                                                          \
	g_tiles_counts[BIN_COUNT * TILES_PER_BIN * 3 + (((bin)*TILES_PER_BIN) + (tile))]
