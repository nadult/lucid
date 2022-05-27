// $$include macros

struct InstanceData {
	int index_offset;
	int vertex_offset;
	int num_quads;
	uint flags, temp;
	uint color;
};

#define BIN_LEVEL_EMPTY 0
#define BIN_LEVEL_MICRO 1
#define BIN_LEVEL_LOW 2
#define BIN_LEVEL_MEDIUM 3
#define BIN_LEVEL_HIGH 4

#define BIN_LEVELS_COUNT 5
#define REJECTION_TYPE_COUNT 4

// This structure contains all the necessary counters, atomics, etc.
// In shader code it's available as g_info; In the same buffer after
// this structure some basic per-bin counters are also kept (g_counts)
struct LucidInfo {
	int num_input_quads;
	int num_verts;

	uint num_rejected_quads[REJECTION_TYPE_COUNT];
	uint timings[8];

	int bin_level_counts[BIN_LEVELS_COUNT];

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
	uint bin_level_dispatches[BIN_LEVELS_COUNT][3];

	int dispatcher_item_counts[128];
	int dispatcher_timings[128];
	int temp[13 + 64];
};

#define LUCID_INFO_SIZE 384

#ifndef __cplusplus

// LucidInfo is always bound at index 0 for consistency
// TODO: we don't need coherent everywhere
coherent layout(std430, binding = 0) buffer buf0_ {
	LucidInfo g_info;
	int g_counts[];
};

#endif