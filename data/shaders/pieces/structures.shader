// $$include macros

struct InstanceData {
	int index_offset;
	int vertex_offset;
	int num_quads;
	uint flags, temp;
	uint color;
};

struct BinCounters {
	int num_input_quads;
	int num_verts;

	uint num_rejected_quads[4];

	int num_empty_bins;
	int num_small_bins;
	int num_medium_bins;
	int num_big_bins;
	int num_tiled_bins;

	uint timings[8];

	int num_visible_quads[2]; // TODO: move
	int num_estimated_visible_quads[2]; // TODO: naming
	int num_dispatched_visible_quads[3];
	int num_bin_rows_quads;
	int num_bin_large_quads;

	int num_finished_setup_groups;
	int num_finished_bin_groups;

	// Atomics
	uint a_small_bins;
	int a_dispatcher_active_thread_groups;
	int a_dispatcher_processed_items;
	uint a_dispatcher_phase;

	uint num_binning_dispatches[3];
	uint num_tiling_dispatches[3];
	uint num_bin_raster_dispatches[3];
	uint num_tile_raster_dispatches[3];
	uint num_block_raster_dispatches[3];

	int dispatcher_item_counts[128];
	int dispatcher_timings[128];
	int temp[15 + 64];
};

#define BIN_COUNTERS_SIZE 384

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

#define TILE_COUNTERS_SIZE 32