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

	int temp[24];

	int bin_quad_counts[BIN_COUNT];
	int bin_quad_offsets[BIN_COUNT];
	int bin_quad_offsets_temp[BIN_COUNT];
};

// TODO: better explanation of stats
struct TileCounters {
	uint tile_dispatch_bin_counter;
	uint mask_raster_bin_counter;
	uint final_raster_bin_counter;
	uint sorted_bin_counter;
	uint temp1[4];

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

	// TODO: 16-bit counters
	uint  tile_tri_counts  [BIN_COUNT][TILES_PER_BIN];
	uint  tile_tri_offsets [BIN_COUNT][TILES_PER_BIN];
	uint  tile_block_tri_counts [BIN_COUNT][TILES_PER_BIN];
	uint  tile_block_tri_offsets[BIN_COUNT][TILES_PER_BIN];
};
