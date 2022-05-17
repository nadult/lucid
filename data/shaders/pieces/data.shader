// $$include macros

struct TriangleData {
	float pos[9];
	uint color;
	uint normal;
};

struct InstanceData {
	int index_offset;
	int vertex_offset;
	int num_quads;
	uint flags, temp;
	uint color;
	// TODO: materials?
};

#define REJECTED_OTHER				0
#define REJECTED_BACKFACE			1
#define REJECTED_FRUSTUM			2
#define REJECTED_BETWEEN_SAMPLES	3

#define REJECTED_TYPE_COUNT			4

struct BinCounters {
	int num_binned_quads;
	int num_input_quads;
	int num_estimated_quads;
	int num_verts;

	uint num_rejected_quads[4];

	int num_empty_bins;
	int num_small_bins;
	int num_medium_bins;
	int num_big_bins;
	int num_tiled_bins;

	uint empty_bin_counter;
	uint small_bin_counter;

	uint timings[8];

	int num_visible_quads; // TODO: move

	int temp[8];
};

#define BIN_COUNTERS_BUFFER(idx)                                                                   \
	layout(std430, binding = idx) buffer buf##idx##_ {                                             \
		BinCounters g_bins;                                                                        \
		int g_bins_counts[];                                                                       \
	}

#define BIN_QUAD_COUNTS(idx) g_bins_counts[BIN_COUNT * 0 + (idx)]
#define BIN_QUAD_OFFSETS(idx) g_bins_counts[BIN_COUNT * 1 + (idx)]
#define BIN_QUAD_OFFSETS_TEMP(idx) g_bins_counts[BIN_COUNT * 2 + (idx)]

// Bins are categorized based on quad count
#define BIN_EMPTY_BINS(idx) g_bins_counts[BIN_COUNT * 3 + (idx)]
#define BIN_SMALL_BINS(idx) g_bins_counts[BIN_COUNT * 4 + (idx)]
#define BIN_MEDIUM_BINS(idx) g_bins_counts[BIN_COUNT * 5 + (idx)]
#define BIN_BIG_BINS(idx) g_bins_counts[BIN_COUNT * 6 + (idx)]
#define BIN_TILED_BINS(idx) g_bins_counts[BIN_COUNT * 7 + (idx)]

// TODO: better explanation of stats
struct TileCounters {
	uint tile_dispatch_bin_counter;
	uint mask_raster_bin_counter;
	uint final_raster_bin_counter;
	uint sorted_bin_counter;

	uint medium_bin_counter;
	uint big_bin_counter;
	uint tiled_bin_counter;
	uint temp1[1];

	// This is only rough estimation (some tile-tris don't pass edge test)
	uint num_tile_tris;

	// Number of triangles per block summed across all blocks
	uint num_block_tris;
	uint num_invalid_pixels;
	uint num_invalid_blocks;
	uint num_invalid_tiles;

	uint num_tile_tris_with_no_blocks;
	uint num_processed_block_rows;

	uint max_tris_per_block;
	uint max_tris_per_tile;
	uint max_row_tris_per_tile;
	uint max_block_tris_per_tile;

	uint num_fragments;
	uint max_fragments_per_tile;
	uint max_fragments_per_pixel;

	uint temp2[10];
};

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
