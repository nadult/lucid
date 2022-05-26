#pragma once

#include "lucid_base.h"
#include "program.h"
#include <fwk/gfx/color.h>

// TODO: better handling of phases
// TODO: ability to change options without recreating renderer
DEFINE_ENUM(LucidRenderOpt, check_tiles, debug_masks, debug_bin_dispatcher, debug_raster,
			new_raster, raster_timings, additive_blending, visualize_errors, alpha_threshold,
			bin_size_32);
using LucidRenderOpts = EnumFlags<LucidRenderOpt>;

struct RasterBlockInfo {
	string description() const;

	vector<bool> selected_tile_tris;
	int2 bin_pos, tile_pos, block_pos;
	int num_block_tris = 0;
	int num_merged_block_tris = 0;
	int num_sub_block_tris = 0;
	int num_tile_tris = 0;
};

struct RasterTileInfo {
	vector<Triangle3F> tris;
	vector<array<u32, 3>> tri_verts;
	vector<u32> tri_indices;
	vector<u32> tri_instances;

	vector<vector<int>> triNeighbourMap(int max_dist) const;

	int2 bin_pos, tile_pos;
	int bin_id, tile_id;
};

class LucidRenderer {
  public:
	using Opt = LucidRenderOpt;
	using Opts = LucidRenderOpts;
	using Context = RenderContext;

	static constexpr int max_width = 2560, max_height = 2048;
	static constexpr int max_quads = 10 * 1024 * 1024, max_verts = 12 * 1024 * 1024;
	static constexpr int max_instance_quads = 1024;

	LucidRenderer();
	FWK_MOVABLE_CLASS(LucidRenderer)
	Ex<void> exConstruct(Opts, int2 view_size);

	void render(const Context &);

	auto opts() const { return m_opts; }

	void printTriangleSizeHistogram() const;
	void printHistograms() const;
	Image masksSnapshot();

	void analyzeMaskRasterizer() const;
	RasterTileInfo introspectTile(CSpan<float3> verts, int2 full_tile_pos) const;
	RasterBlockInfo introspectBlock4x4(const RasterTileInfo &, int2 full_block_pos,
									   bool merge_masks) const;
	RasterBlockInfo introspectBlock8x8(const RasterTileInfo &, int2 full_block_pos,
									   bool merge_masks) const;
	vector<StatsGroup> getStats() const;

	int binSize() const { return m_bin_size; }
	int tileSize() const { return m_tile_size; }

  private:
	void initCounters(const Context &);
	void uploadInstances(const Context &);
	void transformVerts(const Context &);

	void setupQuads(const Context &);
	void computeBins(const Context &);
	void computeTiles(const Context &);
	void rasterizeMasks(const Context &);
	void sortMasks(const Context &);
	void rasterizeFinal(const Context &);
	void bindRaster(const Context &);
	void rasterBin(const Context &);
	void rasterTile(const Context &);
	void rasterBlock(const Context &);
	void compose(const Context &);

	// Does nothing useful; can be used for measuring
	// performance of simple constructs
	void dummyIterateBins(const Context &);

	// .x: x0123, .y: y01, .z: y23
	static array<uint, 256> computeCentroids4x4();

	void checkTiles();
	void debugMasks();
	void copyCounters();

	struct BinBlockStats;
	BinBlockStats computeBlockStats(int bin_id, CSpan<u32>, CSpan<u32>, CSpan<u32>) const;

	Opts m_opts;

	// TODO: m_ prefix
	Program init_counters_program, setup_program;
	Program bin_dispatcher_program, tile_dispatcher_program;
	Program bin_categorizer_program;
	Program final_raster_program, mask_raster_program;
	Program raster_bin_program, raster_tile_program, raster_block_program;
	Program compose_program, dummy_program;

	PBuffer m_errors, m_scratch_32, m_scratch_64, m_instance_data, m_uv_rects;
	PBuffer m_quad_indices, m_quad_aabbs, m_tri_aabbs;
	PBuffer m_bin_counters, m_tile_counters, m_block_counts, m_block_offsets;
	PBuffer m_bin_quads, m_tile_tris, m_block_tris, m_block_tri_keys, m_raster_image;
	array<Pair<PBuffer>, 3> m_old_counters;

	PFramebuffer m_initial_fbo;

	int m_bin_size, m_tile_size, m_block_size;
	int m_blocks_per_bin, m_blocks_per_tile, m_tiles_per_bin;

	int2 m_bin_counts;
	int m_bin_count, m_tile_count;

	int2 m_size;
	int m_num_instances = 0, m_num_quads = 0;

	FrustumRays m_frustum_rays;
	Matrix4 m_view_proj_matrix;
	PBuffer m_compose_quads;
	PVertexArray m_compose_quads_vao;
};
