#pragma once

#include "lucid_base.h"
#include "program.h"
#include <fwk/gfx/color.h>

// TODO: better handling of phases
// TODO: ability to change options without recreating renderer
DEFINE_ENUM(LucidRenderOpt, check_bins, check_tiles, debug_masks, debug_raster, new_raster);
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

	static constexpr int block_size = 4, tile_size = 16, bin_size = 64;
	static constexpr int blocks_per_bin = square(bin_size / block_size),
						 tiles_per_bin = square(bin_size / tile_size),
						 blocks_per_tile = square(tile_size / block_size);

	static constexpr int max_width = 2560, max_height = 2048;
	static constexpr int max_quads = 10 * 1024 * 1024, max_verts = 12 * 1024 * 1024;

	static_assert(isPowerOfTwo(bin_size));

	LucidRenderer();
	FWK_MOVABLE_CLASS(LucidRenderer)
	Ex<void> exConstruct(Opts, int2 view_size);

	void render(const Context &);

	struct InstanceData {
		i32 index_offset;
		i32 vertex_offset;
		i32 num_quads;
		u32 flags, temp;
		IColor color;
	};

	static_assert(sizeof(InstanceData) == sizeof(i32) * 6);

	auto opts() const { return m_opts; }

	void printHistograms() const;
	Image masksSnapshot();

	void analyzeMaskRasterizer() const;
	RasterTileInfo introspectTile(CSpan<float3> verts, int2 full_tile_pos) const;
	RasterBlockInfo introspectBlock4x4(const RasterTileInfo &, int2 full_block_pos,
									   bool merge_masks) const;
	RasterBlockInfo introspectBlock8x8(const RasterTileInfo &, int2 full_block_pos,
									   bool merge_masks) const;
	vector<StatsGroup> getStats() const;

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
	void newRaster(const Context &);
	void compose(const Context &);

	// Does nothing useful; can be used for measuring
	// performance of simple constructs
	void dummyIterateBins(const Context &);

	// .x: x0123, .y: y01, .z: y23
	static array<uint, 256> computeCentroids4x4();

	void checkBins();
	void checkTiles();
	void debugMasks(bool sort_phase);
	void copyCounters();

	struct BinBlockStats;
	BinBlockStats computeBlockStats(int bin_id, CSpan<u32>, CSpan<u32>, CSpan<u32>) const;

	Opts m_opts;

	// TODO: m_ prefix
	Program init_counters_program, setup_program;
	Program bin_estimator_program, bin_dispatcher_program, tile_dispatcher_program;
	Program final_raster_program, mask_raster_program, sort_program, new_raster_program;
	Program compose_program, dummy_program;

	PTexture m_raster_image;

	PBuffer m_errors, m_scratch, m_instance_data, m_uv_rects;
	PBuffer m_quad_indices, m_quad_aabbs, m_tri_aabbs;
	PBuffer m_bin_counters, m_tile_counters, m_block_counts, m_block_offsets;
	PBuffer m_bin_quads, m_tile_tris, m_block_tris, m_block_tri_keys;
	array<Pair<PBuffer>, 3> m_old_counters;

	PFramebuffer m_initial_fbo;
	int2 m_size;
	int m_num_instances = 0, m_num_quads = 0;
	int2 m_bin_counts;

	FrustumRays m_frustum_rays;
	Matrix4 m_view_proj_matrix;
	PVertexArray m_rect_vao;
};
