#ifndef _STRUCTURES_GLSL_
#define _STRUCTURES_GLSL_

#include "definitions.glsl"

struct InstanceData {
	int index_offset;
	int vertex_offset;
	int num_quads;
	uint flags;
};

struct Lighting {
	vec4 ambient_color;
	vec4 sun_color;
	vec4 sun_dir;
	float sun_power, ambient_power;
};

// All vectors in world space
struct Frustum {
	vec4 ws_origins[4], ws_dirs[4];
	vec4 ws_origin0, ws_dir0;
	vec4 ws_dirx, ws_diry;
};

struct SimpleDrawCall {
	mat4 proj_view_matrix;
	vec4 material_color;
	vec2 uv_rect_pos;
	vec2 uv_rect_size;
	uint draw_call_opts;
	vec4 world_camera_pos;
};

struct Rect {
	vec2 pos, size;
	vec2 min_uv, max_uv;
};

struct Viewport {
	mat4 proj_matrix;
	vec2 size, inv_size;
	float near_plane, far_plane;
	float inv_far_plane;
};

#define LUCID_INFO_MAX_DISPATCHES 256
#define LUCID_INFO_SIZE (128 + 4 * LUCID_INFO_MAX_DISPATCHES)

// This structure contains all the necessary counters, atomics, etc.
// In shader code it's available as g_info; In the same SSBO just after
// this structure some basic per-bin counters are also kept (g_counts)
// TODO: consistent naming (count or num)
struct LucidInfo {
	int num_input_quads;
	int num_visible_quads[2];
	int num_counted_quads[2];

	int bin_level_counts[BIN_LEVELS_COUNT];

	// Atomic counters
	uint a_small_bins, a_high_bins;
	uint a_setup_work_groups;
	uint a_dummy_counter;

	// Counters for indirect dispatch
	uint num_binning_dispatches[3];
	uint bin_level_dispatches[BIN_LEVELS_COUNT][3];

	// Statistics, timings, etc. (secondary data)
	uint num_rejected_quads[REJECTION_TYPE_COUNT];

	uint setup_timers[TIMERS_COUNT];
	uint raster_timers[TIMERS_COUNT];
	uint bin_dispatcher_timers[TIMERS_COUNT];

	uint stats[STATS_COUNT];

	int dispatcher_first_batch[2][LUCID_INFO_MAX_DISPATCHES];
	int dispatcher_num_batches[2][LUCID_INFO_MAX_DISPATCHES];

	int temp[64];
};

// This structure keeps uniform data passed to Lucid shaders
struct LucidConfig {
	Frustum frustum;
	mat4 view_proj_matrix;
	Lighting lighting;
	vec4 background_color;

	uint enable_backface_culling;
	int num_instances;
	int instance_packet_size;
};

struct PathTracerInfo {
	int temp[256];
};

// This structure keeps uniform data passed to Lucid shaders
struct PathTracerConfig {
	Frustum frustum;
	mat4 view_proj_matrix;
	Lighting lighting;
	vec4 background_color;

	uint num_nodes;
	uint num_triangles;
	uint max_depth;
};

#endif
