// $$include macros

struct InstanceData {
	int index_offset;
	int vertex_offset;
	int num_quads;
	uint flags;
};

#define BIN_LEVEL_EMPTY 0
#define BIN_LEVEL_MICRO 1
#define BIN_LEVEL_LOW 2
#define BIN_LEVEL_MEDIUM 3
#define BIN_LEVEL_HIGH 4

#define BIN_LEVELS_COUNT 5
#define REJECTION_TYPE_COUNT 4

#define TIMERS_COUNT 8

// This structure contains all the necessary counters, atomics, etc.
// In shader code it's available as g_info; In the same SSBO just after
// this structure some basic per-bin counters are also kept (g_counts)
// TODO: consistent naming (count or num)
struct LucidInfo {
	int num_input_quads;
	int num_visible_quads[2];
	int num_estimated_quads[2];

	int bin_level_counts[BIN_LEVELS_COUNT];

	// Atomic counters
	uint a_small_bins, a_medium_bins;
	uint a_setup_work_groups;
	uint a_dummy_counter;
	int a_bin_dispatcher_work_groups[2];
	int a_bin_dispatcher_items[2];
	int a_bin_dispatcher_phase[2];

	// Counters for indirect dispatch
	uint num_binning_dispatches[3];
	uint bin_level_dispatches[BIN_LEVELS_COUNT][3];

	// Statistics, timings, etc. (secondary data)
	uint num_rejected_quads[REJECTION_TYPE_COUNT];
	uint timers[TIMERS_COUNT];

	// TODO:
	uint num_fragments;
	uint avg_fragments_per_hblock_tri;

	int dispatcher_task_counts[128];
	int dispatcher_timers[128];
	int temp[12 + 64];
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