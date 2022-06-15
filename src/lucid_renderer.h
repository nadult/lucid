#pragma once

#include "lucid_base.h"
#include "program.h"
#include <fwk/gfx/color.h>

// TODO: better handling of phases
// TODO: ability to change options without recreating renderer
DEFINE_ENUM(LucidRenderOpt, debug_bin_dispatcher, debug_raster, timers, additive_blending,
			visualize_errors, alpha_threshold, bin_size_64);
using LucidRenderOpts = EnumFlags<LucidRenderOpt>;

class LucidRenderer {
  public:
	using Opt = LucidRenderOpt;
	using Opts = LucidRenderOpts;
	using Context = RenderContext;

	static constexpr int max_width = 2560, max_height = 2048;
	static constexpr int max_quads = 10 * 1024 * 1024, max_verts = 12 * 1024 * 1024;
	static constexpr int max_instance_quads = 1024;
	static constexpr int max_visible_quads = 2 * 1024 * 1024;

	LucidRenderer();
	FWK_MOVABLE_CLASS(LucidRenderer)
	Ex<void> exConstruct(Opts, int2 view_size);

	void render(const Context &);

	auto opts() const { return m_opts; }

	void printTriangleSizeHistogram() const;
	void printHistograms() const;
	vector<StatsGroup> getStats() const;

	int binSize() const { return m_bin_size; }
	int tileSize() const { return m_tile_size; }

  private:
	void uploadInstances(const Context &);
	void quadSetup(const Context &);
	void computeBins(const Context &);
	void bindRasterCommon(const Context &);
	void bindRaster(Program &, const Context &);
	void rasterLow(const Context &);
	void rasterMedium(const Context &);
	void compose(const Context &);

	void copyInfo(int num_skip_frames);

	// Does nothing useful; can be used for measuring
	// performance of simple constructs
	void dummyIterateBins(const Context &);

	void debugProgram(Program &, ZStr title);

	Opts m_opts;

	Program p_quad_setup;
	Program p_bin_dispatcher, p_bin_categorizer;
	Program p_raster_low, p_raster_medium;
	Program p_compose, p_dummy;

	PBuffer m_instances, m_instance_colors, m_instance_uv_rects;
	PBuffer m_errors, m_scratch_32, m_scratch_64;
	PBuffer m_info, m_bin_quads, m_bin_tris, m_raster_image;
	PBuffer m_uint_storage, m_uvec4_storage;
	array<PBuffer, 3> m_old_info;

	PFramebuffer m_initial_fbo;

	int m_bin_size, m_tile_size, m_block_size;
	int m_blocks_per_bin, m_blocks_per_tile, m_tiles_per_bin;
	int m_max_dispatches;

	int2 m_bin_counts;
	int m_bin_count, m_tile_count;

	int2 m_size;
	int m_num_instances = 0, m_num_quads = 0;

	FrustumRays m_frustum_rays;
	Matrix4 m_view_proj_matrix;
	PBuffer m_compose_quads;
	PVertexArray m_compose_quads_vao;
};
