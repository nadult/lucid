// $$include structures

#define BIN_COUNTERS_BUFFER(idx)                                                                   \
	layout(std430, binding = idx) buffer buf##idx##_ {                                             \
		BinCounters g_bins;                                                                        \
		int g_bins_counts[];                                                                       \
	}

#define BIN_QUAD_COUNTS(idx) g_bins_counts[BIN_COUNT * 0 + (idx)]
#define BIN_QUAD_OFFSETS(idx) g_bins_counts[BIN_COUNT * 1 + (idx)]
#define BIN_QUAD_OFFSETS_TEMP(idx) g_bins_counts[BIN_COUNT * 2 + (idx)]

// Bins are categorized based on quad count
#define BIN_SMALL_BINS(idx) g_bins_counts[BIN_COUNT * 3 + (idx)]
#define BIN_MEDIUM_BINS(idx) g_bins_counts[BIN_COUNT * 4 + (idx)]
#define BIN_BIG_BINS(idx) g_bins_counts[BIN_COUNT * 5 + (idx)]
#define BIN_TILED_BINS(idx) g_bins_counts[BIN_COUNT * 6 + (idx)]