// $$include macros

struct InstanceData {
	int index_offset;
	int vertex_offset;
	int num_quads;
	uint flags, temp;
	uint color;
};

// TODO: better name for that
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

	int num_finished_setup_groups;
	int num_finished_bin_groups;

	// Atomics
	uint a_small_bins;
	int a_dispatcher_active_thread_groups;
	int a_dispatcher_processed_items;
	uint a_dispatcher_phase;
	uint a_dummy_counter;

	uint num_binning_dispatches[3];
	uint num_tiling_dispatches[3];
	uint num_bin_raster_dispatches[3];
	uint num_tile_raster_dispatches[3];
	uint num_block_raster_dispatches[3];

	int dispatcher_item_counts[128];
	int dispatcher_timings[128];
	int temp[16 + 64];
};

#define BIN_COUNTERS_SIZE 384
