#include "lucid_renderer.h"

#include "scene.h"
#include "shading.h"
#include <fwk/gfx/camera.h>
#include <fwk/gfx/draw_call.h>
#include <fwk/gfx/gl_buffer.h>
#include <fwk/gfx/gl_format.h>
#include <fwk/gfx/gl_framebuffer.h>
#include <fwk/gfx/gl_program.h>
#include <fwk/gfx/gl_shader.h>
#include <fwk/gfx/gl_texture.h>
#include <fwk/gfx/gl_vertex_array.h>
#include <fwk/gfx/image.h>
#include <fwk/gfx/opengl.h>
#include <fwk/gfx/render_list.h>
#include <fwk/gfx/shader_debug.h>
#include <fwk/gfx/shader_defs.h>
#include <fwk/hash_set.h>
#include <fwk/io/file_system.h>
#include <fwk/math/ray.h>

// TODO: opisać różnego rodzaju definicje/nazwy używane w kodzie

// TODO: dużo specyficznych przypadków do obsłużenia:
// - clipping: może zaimplementować dodatkowe passy do obsługi clipowanych trójkątów ?
//   chyba nie ma sensu; chodzi tylko o to, żeby trójkąty które powinny być przycięte nas nie kosztowały
//   jeśli możemy je usunąć inaczej to nie ma problemu
// - overlappowanie się ścianek
//   ich jest generalnie mało więc na wydajność to raczej nie wpłynie, a
// - filtrowanie bardzo cienkich trójkątów które mieszczą się pomiędzy środkami pikseli
// - lepszy load balancing w rasteryzerze; małe trójkąty mogą być szczególnie kosztowne:
//   wystarczy zahaczenie o 1 piksel, żeby wszystkie wątki w warpie go rozpaatrywały

// TODO: system do automatycznego assignowania buforów
// TODO: program mógłby zarządzać różnymi wersjami programu (z włączonymi/wyłączonymi różnymi flagami)
//       różne wersje by się kompilowały on demand?

// TODO: ssao, msaa support?
// TODO: lots of options for optimization...
// TODO: still slow on integrated GPU; complex logic, atomic operations, etc. should be avoided
// TODO: sometimes when resizing only trans renderer's result is visible
//
// - pick such parameters for buffer sizes so that different counters will fit in shared memory
// - pick such local_size's that they will map properly on typical hardware (32/64 or 16 threads?)
// - there are definitely load balancing issues in many places, fix them
// - different versions of sort & raster shaders for different number of triangles in tile?

// Main idea: order of triangles in most of neighbouring pixels is very similar if not the same.
// Because of that it makes no sense to perform the sorting only at the pixel level. It's better
// to pre-sort at tile-level.

/* Random notes:

  Średnie ilości trójkątów zależnie od wielkości kafla (stare dane, bez dobrego cullingu):
  Im większy kafel, tym więcej trójkątów musimy przetważać per pixel; To zwiększa czas
  sortowania i dodatkowo wydajne rozdzielanie trójkątów między piksele jest trudne
   Wielkość kafla:   instancje w sumie:   per-pixel w kaflu:
   32x32 (40x33)            420K               318
   16x16 (80x66)            566K               107
    8x8 (160x132)           886K               41
    4x4:(320x258)           1670K              20
    2x2:(639x515)           3738K              11
   1x1:(1278x1030)           10M               8

  Ilość operacji sortowania (z grubsza); Jak mamy duże listy, to sortujemy bitonic-sortem
  (N * log^2N); małe listy powinno się dać szybciej: N * logN (TODO: do zweryfikowania)
                           N * logN^2          N * logN
      1x1:   1278 * 1030 * 72    = 94M |   * 24   = 31.5M
      2x2:   639 * 515   * 131   = 43M |   * 38   = 12.5M
      4x4:   320 * 258   * 373   = 30M |   * 86   = 7M
      16x16: 80 * 66     * 5000  = 25M |   * 721  = 4M
	  32x32  40 * 33     * 22000 = 29M |   * 2643 = 3.5M

	  Czyli jest jakiś sweet spot wielkości kafla (zależny od widoku) jeśli minimalizujemy
	  ilość operacji sortowania.

  Tile counts per resolution:
                      16x16     32x32    64x64      128x128
  1280 * 720  * 1x:    3600      920      240          60
  1920 * 1080 * 1x:    8160     2040      510         135
  1280 * 720  * 4x:   14400     3680      920         240
  1920 * 1080 * 4x:   32640     8160     2040         405

  Triangle distributions on different scenes (pairs of values: 1x1 (single), >1x1 (multi))
                   64x64 max   |  64x64 avg  |   8x8 max   |   8x8 avg
  hairball (3M): 221000, 81000 | 20500, 7700 | 5570, 35700 |   83, 1163
  gallery  (1M):  48900,  3150 |  3220,  400 | 1540,  1150 |   35,   56
  dragon (800K):  21600,  1756 |  5260,  670 | 1080,   540 |   64,  107
*/

void setupView(const IRect &viewport, PFramebuffer fbo) {
	if(fbo)
		DASSERT(fbo->size().x >= viewport.width() && fbo->size().y >= viewport.height());
	glViewport(viewport.x(), viewport.y(), viewport.width(), viewport.height());
	if(fbo)
		fbo->bind();
	else
		GlFramebuffer::unbind();
}

LucidRenderer::LucidRenderer() = default;
FWK_MOVABLE_CLASS_IMPL(LucidRenderer)

Ex<void> LucidRenderer::exConstruct(Opts opts, int2 view_size) {
	m_opts = opts;
	m_size = view_size;
	m_bin_counts = (view_size + int2(bin_size - 1)) / bin_size;
	int bin_count = m_bin_counts.x * m_bin_counts.y;
	int tile_count = bin_count * tiles_per_bin;

	// TODO: this takes a lot of memory
	// TODO: what should we do when quads won't fit?
	// TODO: better estimate needed
	// TODO: properly handle situations when limits were reached
	uint max_bin_quads = max_quads * 3 / 2;
	uint max_tile_tris = max_quads * 3;
	uint max_block_tris = max_quads * 2;

	// TODO: LucidRenderer constructed 2x at the beginning

	m_quad_indices.emplace(BufferType::shader_storage, max_quads * 4 * sizeof(u32));
	m_quad_aabbs.emplace(BufferType::shader_storage, max_quads * sizeof(u32));
	m_tri_aabbs.emplace(BufferType::shader_storage, max_quads * sizeof(int4));

	uint bin_counters_size = (bin_count * 8 + 256) * sizeof(u32);
	uint tile_counters_size = (bin_count * 16 * 4 + 256) * sizeof(u32);
	m_bin_counters.emplace(BufferType::shader_storage, bin_counters_size);
	m_tile_counters.emplace(BufferType::shader_storage, tile_counters_size);

	m_block_counts.emplace(BufferType::shader_storage, tile_count * 16 * sizeof(u32));
	m_block_offsets.emplace(BufferType::shader_storage, tile_count * 16 * sizeof(u32));

	m_bin_quads.emplace(BufferType::shader_storage, max_bin_quads * sizeof(u32));
	m_tile_tris.emplace(BufferType::shader_storage, max_tile_tris * sizeof(u32),
						BufferUsage::dynamic_read);
	m_block_tris.emplace(BufferType::shader_storage, max_block_tris * sizeof(u32),
						 BufferUsage::dynamic_read);
	m_block_tri_keys.emplace(BufferType::shader_storage, max_block_tris * sizeof(u32));
	m_scratch.emplace(BufferType::shader_storage,
					  (256 * 1024) * 128 * 2 * sizeof(u32)); // TODO: control size
	m_raster_image.emplace(BufferType::shader_storage, bin_count * square(bin_size) * sizeof(u32));

	if(m_opts & (Opt::check_bins | Opt::check_tiles | Opt::debug_masks | Opt::debug_raster))
		m_errors.emplace(BufferType::shader_storage, 1024 * 1024 * 4, BufferUsage::dynamic_read);

	ShaderDefs defs;
	defs["VIEWPORT_SIZE_X"] = view_size.x;
	defs["VIEWPORT_SIZE_Y"] = view_size.y;
	defs["BIN_COUNT"] = bin_count;
	defs["BIN_COUNT_X"] = m_bin_counts.x;
	defs["BIN_COUNT_Y"] = m_bin_counts.y;
	defs["BIN_SIZE"] = bin_size;
	defs["TILE_SIZE"] = tile_size;
	defs["BLOCK_SIZE"] = block_size;
	defs["BIN_SHIFT"] = log2(bin_size);
	defs["TILE_SHIFT"] = log2(tile_size);
	defs["BLOCK_SHIFT"] = log2(block_size);
	defs["MAX_LSIZE"] = gl_info->max_compute_work_group_size.x;
	defs["XTILES_PER_BIN"] = bin_size / tile_size;
	defs["TILES_PER_BIN"] = tiles_per_bin;
	defs["BLOCKS_PER_TILE"] = blocks_per_tile;
	defs["BLOCKS_PER_BIN"] = blocks_per_bin;
	defs["MAX_LSIZE"] = gl_info->limits[GlLimit::max_compute_work_group_invocations];

	init_counters_program = EX_PASS(Program::makeCompute("init_counters", defs));
	setup_program = EX_PASS(Program::makeCompute("setup", defs));
	bin_estimator_program = EX_PASS(Program::makeCompute(
		"bin_estimator", defs, mask(m_opts & Opt::check_bins, ProgramOpt::debug)));
	bin_dispatcher_program = EX_PASS(Program::makeCompute(
		"bin_dispatcher", defs, mask(m_opts & Opt::check_bins, ProgramOpt::debug)));
	bin_categorizer_program = EX_PASS(Program::makeCompute("bin_categorizer", defs));

	tile_dispatcher_program = EX_PASS(Program::makeCompute(
		"tile_dispatcher", defs, mask(m_opts & Opt::check_tiles, ProgramOpt::debug)));
	final_raster_program = EX_PASS(Program::makeCompute("final_raster", defs));
	mask_raster_program = EX_PASS(Program::makeCompute(
		"mask_raster", defs, mask(m_opts & Opt::debug_masks, ProgramOpt::debug)));
	raster_bin_program =
		EX_PASS(Program::makeCompute("raster_bin_handling_collisions", defs,
									 mask(m_opts & Opt::debug_raster, ProgramOpt::debug)));
	raster_tile_program = EX_PASS(Program::makeCompute(
		"raster_tile", defs, mask(m_opts & Opt::debug_raster, ProgramOpt::debug)));
	raster_block_program = EX_PASS(Program::makeCompute(
		"raster_block", defs, mask(m_opts & Opt::debug_raster, ProgramOpt::debug)));
	//	sort_program = EX_PASS(Program::makeCompute(
	//		"mask_sort", defs, mask(m_opts & Opt::debug_masks, ProgramOpt::debug)));
	dummy_program = EX_PASS(Program::makeCompute("dummy", defs));

	if(auto disas = raster_bin_program.getDisassembly()) {
		mkdirRecursive("temp").ignore();
		saveFile("temp/raster_bin.asm", *disas).ignore();
	}

	compose_program = EX_PASS(Program::make("compose", "", {"in_pos"}));

	vector<u16> indices(bin_count * 6);
	DASSERT(bin_count * 4 * 4 <= 64 * 1024);
	for(int i = 0; i < bin_count; i++) {
		int offsets[6] = {0, 1, 2, 0, 2, 3};
		for(int j = 0; j < 6; j++) {
			int value = offsets[j] + i * 4;
			indices[i * 6 + j] = offsets[j] + i * 4;
		}
	}
	auto ibuffer = GlBuffer::make(BufferType::element_array, indices);
	m_compose_quads =
		GlBuffer::make(BufferType::array, bin_count * 4 * sizeof(uint), ImmBufferFlags());

	m_compose_quads_vao = GlVertexArray::make();
	m_compose_quads_vao->set({m_compose_quads}, defaultVertexAttribs<uint>(), ibuffer,
							 IndexType::uint16);

	return {};
}

void LucidRenderer::render(const Context &ctx) {
	PERF_GPU_SCOPE();
	testGlError("LucidRenderer::render init");

	m_bin_counters->invalidate();
	m_tile_counters->invalidate();

	m_view_proj_matrix = ctx.camera.matrix();
	m_frustum_rays = ctx.camera;

	initCounters(ctx);
	uploadInstances(ctx);
	setupQuads(ctx);

	computeBins(ctx);
	if(m_opts & Opt::check_bins)
		checkBins();

	computeTiles(ctx);
	if(m_opts & Opt::check_tiles)
		checkTiles();

	if(false)
		dummyIterateBins(ctx);

	if(m_opts & Opt::new_raster) {
		bindRaster(ctx);
		rasterBin(ctx);
		rasterTile(ctx);
		rasterBlock(ctx);
	} else {
		rasterizeMasks(ctx);
		if(m_opts & Opt::debug_masks)
			debugMasks(false);
		rasterizeFinal(ctx);
	}

	copyCounters();

	// TODO: is this needed?
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

	compose(ctx);
	testGlError("LucidRenderer::render finish");
}

void LucidRenderer::initCounters(const Context &ctx) {
	PERF_GPU_SCOPE();

	// TODO: num_verts should be computed on gpu (only those vertices which are actually used
	// should be transformed)
	auto vbuffers = ctx.vao->buffers();
	DASSERT(vbuffers.size() == 4);
	int num_verts = vbuffers[0]->size() / sizeof(float3);

	m_bin_counters->bindIndex(0);
	m_tile_counters->bindIndex(1);
	init_counters_program.use();
	init_counters_program["num_verts"] = num_verts;
	if(m_opts & Opt::check_bins)
		shaderDebugUseBuffer(m_errors);
	glDispatchCompute(1, 1, 1);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void LucidRenderer::uploadInstances(const Context &ctx) {
	PERF_GPU_SCOPE();

	vector<InstanceData> instances;
	instances.reserve(32 * 1024);
	vector<float4> uv_rects;
	uv_rects.reserve(32 * 1024);

	vector<IColor> mat_colors = transform(
		ctx.materials, [](auto &mat) { return IColor(FColor(mat.diffuse, mat.opacity)); });
	int max_instance_quads = gl_info->limits[GlLimit::max_compute_work_group_invocations];

	m_num_quads = 0;
	for(auto &dc : ctx.dcs) {
		int num_quads = dc.num_quads;
		if(m_num_quads + num_quads > max_quads)
			num_quads = max_quads - m_num_quads;
		if(num_quads == 0)
			break;

		InstanceData out;
		out.index_offset = dc.quad_offset * 4;
		out.vertex_offset = 0;
		out.num_quads = num_quads;
		out.flags = (uint(dc.opts.bits) & 0xffff);
		out.color = mat_colors[dc.material_id];
		out.temp = 0;
		float4 uv_rect(dc.uv_rect.x(), dc.uv_rect.y(), dc.uv_rect.width(), dc.uv_rect.height());
		m_num_quads += dc.num_quads;

		if(num_quads <= max_instance_quads) {
			instances.emplace_back(out);
			uv_rects.emplace_back(uv_rect);
		} else {
			for(int i = 0; i < num_quads; i += max_instance_quads) {
				out.index_offset = dc.quad_offset * 4 + i * 4;
				out.num_quads = min(max_instance_quads, num_quads - i);
				instances.emplace_back(out);
				uv_rects.emplace_back(uv_rect);
			}
		}
	}

	m_instance_data.emplace(BufferType::shader_storage, instances);
	m_uv_rects.emplace(BufferType::shader_storage, uv_rects);
	m_num_instances = instances.size();
}

void LucidRenderer::setupQuads(const Context &ctx) {
	// TODO: co robić z trójkątami, które są na tyle małe, że wogóle ich nie widać nawet w pełnej rozdziałce?
	// TODO: backface-culling ?

	PERF_GPU_SCOPE();
	ctx.quads_ib->bindIndexAs(0, BufferType::shader_storage);
	m_instance_data->bindIndex(1);
	auto vbuffers = ctx.vao->buffers();
	DASSERT(vbuffers.size() == 4);
	vbuffers[0]->bindIndexAs(2, BufferType::shader_storage);

	m_bin_counters->bindIndex(3);
	m_quad_indices->bindIndex(4);
	m_quad_aabbs->bindIndex(5);
	m_tri_aabbs->bindIndex(6);

	auto &program = setup_program;
	program["enable_backface_culling"] = ctx.config.backface_culling ? 1 : 0;
	program["num_instances"] = m_num_instances;
	program["view_proj_matrix"] = m_view_proj_matrix;
	program.setFrustum(ctx.camera);
	program.use();

	glDispatchCompute((m_num_instances + 31) / 32, 1, 1);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void LucidRenderer::computeBins(const Context &ctx) {
	PERF_GPU_SCOPE();

	// TODO: Why adding more on intel causes problems?
	// TODO: properly get number of compute units (use opencl?)
	// https://tinyurl.com/o7s9ph3
	int max_dispatches = gl_info->vendor == GlVendor::intel ? 32 : 128;

	m_quad_aabbs->bindIndex(0);
	m_bin_counters->bindIndex(1);
	m_tile_counters->bindIndex(2);
	m_bin_quads->bindIndex(3);

	bin_estimator_program.use();
	if(m_opts & Opt::check_bins)
		shaderDebugUseBuffer(m_errors);

	PERF_CHILD_SCOPE("estimator phase 1");
	bin_estimator_program["phase"] = 1u;
	glDispatchCompute(max_dispatches, 1, 1);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

	PERF_SIBLING_SCOPE("estimator phase 2");
	bin_estimator_program["phase"] = 2u;
	glDispatchCompute(1, 1, 1);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

	bin_dispatcher_program.use();
	if(m_opts & Opt::check_bins)
		shaderDebugUseBuffer(m_errors);

	PERF_SIBLING_SCOPE("bin dispatching phase");
	// Why adding more slows everything down?
	// TODO: how to optimize this ?
	// 1ms for now for dragon...
	glDispatchCompute(max_dispatches, 1, 1);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

	PERF_SIBLING_SCOPE("bin categorizing phase");
	bin_categorizer_program.use();
	bin_categorizer_program["tile_all_bins"] = !(m_opts & Opt::new_raster);
	m_compose_quads->bindIndexAs(2, BufferType::shader_storage);
	glDispatchCompute(1, 1, 1);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void LucidRenderer::computeTiles(const Context &ctx) {
	PERF_GPU_SCOPE();

	// TODO: Why adding more on intel causes problems?
	// TODO: properly get number of compute units (use opencl?)
	// https://tinyurl.com/o7s9ph3
	int max_dispatches = gl_info->vendor == GlVendor::intel ? 32 : 128;

	m_bin_counters->bindIndex(1);
	m_tile_counters->bindIndex(2);
	m_bin_quads->bindIndex(3);
	m_tile_tris->bindIndex(4);
	auto vbuffers = ctx.vao->buffers();
	DASSERT(vbuffers.size() == 4);
	vbuffers[0]->bindIndexAs(5, BufferType::shader_storage);
	m_quad_indices->bindIndex(6);
	m_tri_aabbs->bindIndex(7);

	tile_dispatcher_program.use();
	tile_dispatcher_program.setFrustum(ctx.camera);
	if(m_opts & Opt::check_tiles)
		shaderDebugUseBuffer(m_errors);
	glDispatchCompute(max_dispatches, 1, 1);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void LucidRenderer::checkBins() {
	vector<int> bin_quad_counts, bin_quad_offsets, bin_quad_offsets2;
	int num_binned_quads, num_input_quads;
	int bin_count = m_bin_counts.x * m_bin_counts.y;

	{
		auto vals = m_bin_counters->map<int>(AccessMode::read_only);
		num_input_quads = vals[1];
		num_binned_quads = vals[2];
		vals = vals.subSpan(32);
		bin_quad_counts = vals.subSpan(bin_count * 0, bin_count * 1);
		bin_quad_offsets = vals.subSpan(bin_count * 1, bin_count * 2);
		bin_quad_offsets2 = vals.subSpan(bin_count * 2, bin_count * 3);
		m_bin_counters->unmap();
	}

	int max_width = 40, max_height = 60;
	if(auto dim = consoleDimensions()) {
		max_width = (dim->x - 1) / 6;
		max_height = dim->y * 3 / 4;
	}

	print("Checking bins:\n");
	print("Input quads:%\nBinned quads:%\n", num_input_quads, num_binned_quads);
	print("\nBin quad counts:\n");
	for(int y = 0; y < min(max_height, m_bin_counts.y); y++) {
		for(int x = 0; x < min(max_width, m_bin_counts.x); x++) {
			int count = bin_quad_counts[x + y * m_bin_counts.x];
			printf("%6d ", count);
		}
		printf("\n");
	}

	int cur_offset = 0, num_errors = 0;
	for(int y = 0; y < m_bin_counts.y; y++) {
		for(int x = 0; x < m_bin_counts.x; x++) {
			int idx = x + y * m_bin_counts.x;
			if(cur_offset != bin_quad_offsets[idx] && num_errors++ < 1)
				print("Invalid offset at (%, %): % (should be: %)\n", x, y, bin_quad_offsets[idx],
					  cur_offset);
			if(bin_quad_counts[idx] != bin_quad_offsets2[idx] - bin_quad_offsets[idx] &&
			   num_errors++ < 1)
				print("Quad count & estimate does not match at (%, %)\n", x, y);
			cur_offset += bin_quad_counts[idx];
		}
	}
	print("\n");
}

void LucidRenderer::checkTiles() {
	int bin_count = m_bin_counts.x * m_bin_counts.y;
	vector<int> tile_tri_counts;
	{
		auto vals = m_tile_counters->map<int>(AccessMode::read_only);
		vals = vals.subSpan(32);
		tile_tri_counts = vals.subSpan(0, bin_count * tiles_per_bin);
		m_tile_counters->unmap();
	}

	print("Per-tile triangle count histogram:\n");
	vector<pair<int, int>> histogram(32);
	pair<int, int> sum = {0, 0};

	for(int value : tile_tri_counts) {
		int i = value == 0 ? 0 : value <= 16 ? 1 : int(log2(value - 1)) - 2;
		if(histogram.inRange(i)) {
			histogram[i].first++;
			histogram[i].second += value;
		}
	}

	for(auto &elem : histogram) {
		sum.first += elem.first;
		sum.second += elem.second;
	}

	for(int i : intRange(histogram)) {
		if(!histogram[i].first)
			continue;
		int level = i == 0 ? 0 : i == 1 ? 16 : 32 << (i - 2);
		printf("%s %8d: ", i == 0 ? "  " : "<=", level);
		printf("%5d tiles (%5.2f %%); ", histogram[i].first,
			   double(histogram[i].first) / sum.first * 100.0);
		printf("%7d tris total (%5.2f %%)\n", histogram[i].second,
			   double(histogram[i].second) / sum.second * 100.0);
	}
	printf("\n");
}

void LucidRenderer::dummyIterateBins(const Context &ctx) {
	PERF_GPU_SCOPE();

	m_bin_counters->bindIndex(0);
	m_tile_counters->bindIndex(1);
	dummy_program.use();
	glDispatchCompute(512, 1, 1);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void LucidRenderer::rasterizeMasks(const Context &ctx) {
	PERF_GPU_SCOPE();

	m_tri_aabbs->bindIndex(2);
	auto vbuffers = ctx.vao->buffers();
	DASSERT(vbuffers.size() == 4);
	vbuffers[0]->bindIndexAs(3, BufferType::shader_storage);
	m_quad_indices->bindIndex(4);
	m_bin_counters->bindIndex(5);
	m_tile_counters->bindIndex(6);
	m_block_counts->bindIndex(7);
	m_block_offsets->bindIndex(8);
	m_tile_tris->bindIndex(9);
	m_block_tris->bindIndex(10);
	m_block_tri_keys->bindIndex(11);
	m_scratch->bindIndex(12);

	mask_raster_program.use();
	mask_raster_program.setFrustum(ctx.camera);
	mask_raster_program.setViewport(ctx.camera, m_size);
	mask_raster_program["mask_centroids[0]"] = computeCentroids4x4();

	if(m_opts & Opt::debug_masks)
		shaderDebugUseBuffer(m_errors);

	// TODO: lepiej by było, jakby było to bardziej zintegrowane z GlProgramem
	// - dispatch też mógłby być funkcją debuggera);
	// - Inny debugger dla compute i inny dla pozostałych shaderów
	// - możliwość przekazywania konkretnych wartości (np. 4 różne wartości?)
	// - jakaś klasa do prostej introspekcji linii kodu programu
	glDispatchCompute(128, 1, 1);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void LucidRenderer::sortMasks(const Context &ctx) {
	PERF_GPU_SCOPE();

	m_tile_counters->bindIndex(0);
	m_block_counts->bindIndex(1);
	m_block_tris->bindIndex(2);
	m_block_tri_keys->bindIndex(3);
	sort_program.use();

	if(m_opts & Opt::debug_masks)
		shaderDebugUseBuffer(m_errors);

	// TODO: spawn workgroups only to saturate SMs
	glDispatchCompute(128, 1, 1);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void LucidRenderer::debugMasks(bool sort_phase) {
	auto source_ranges = (sort_phase ? sort_program : mask_raster_program).sourceRanges();
	auto records = shaderDebugRecords(m_errors, {256, 1, 1}, {128, 1, 1}, 256, source_ranges);
	if(records) {
		makeSorted(records);
		print("mask_% shader debug messages reported:\n", sort_phase ? "sort" : "raster");
		for(auto &record : records)
			print("%\n", record);
	}
}

struct LucidRenderer::BinBlockStats {
	double avg_max_pixel_depth = 0, avg_min_pixel_depth = 0, avg_tris_per_block = 0;
	int max_pixel_depth = 0, max_tris_per_block = 0, max_blocktris_per_tile = 0;
	int num_nonempty_blocks = 0, num_fragments = 0, unique_tris_sum = 0;
	int num_blocktris = 0;

	bool empty() const { return num_blocktris == 0; }
	void computeAverages() {
		if(num_nonempty_blocks) {
			avg_min_pixel_depth /= num_nonempty_blocks;
			avg_max_pixel_depth /= num_nonempty_blocks;
			avg_tris_per_block /= num_nonempty_blocks;
		}
	}

	static BinBlockStats sum(CSpan<BinBlockStats> stats) {
		BinBlockStats out;
		for(auto elem : stats) {
			if(elem.empty())
				continue;
#define ACCUM(var) out.var += elem.var;
#define ACCUM_MAX(var) out.var = max(out.var, elem.var);
			ACCUM(avg_max_pixel_depth);
			ACCUM(avg_min_pixel_depth);
			ACCUM(avg_tris_per_block);
			ACCUM_MAX(max_pixel_depth);
			ACCUM_MAX(max_tris_per_block);
			ACCUM_MAX(max_blocktris_per_tile);
			ACCUM(num_nonempty_blocks);
			ACCUM(num_fragments);
			ACCUM(unique_tris_sum);
			ACCUM(num_blocktris);
#undef ACCUM_MAX
#undef ACCUM
		}
		return out;
	}
};

auto LucidRenderer::computeBlockStats(int bin_id, CSpan<u32> block_instances,
									  CSpan<u32> block_counts, CSpan<u32> block_offsets) const
	-> BinBlockStats {
	BinBlockStats stats;

	for(int block_id = 0; block_id < blocks_per_bin; block_id++)
		stats.num_blocktris += block_counts[bin_id * blocks_per_bin + block_id];
	if(stats.num_blocktris == 0)
		return stats;

	vector<int> unique_tris_map(65536, -1);

	for(int tile_id = 0; tile_id < tiles_per_bin; tile_id++) {
		int tindex = bin_id * 16 + tile_id;
		int num_blocktris = 0;

		for(int block_id = 0; block_id < blocks_per_tile; block_id++) {
			int bindex = bin_id * blocks_per_bin + tile_id * blocks_per_tile + block_id;
			int offset = block_offsets[bindex];
			int count = block_counts[bindex];
			if(count == 0)
				continue;

			int pixel_stacks[16];
			fill(pixel_stacks, 0);

			for(int i = 0; i < count; i++) {
				u32 value = block_instances[offset + i];
				for(int j = 0; j < 16; j++)
					if(value & (1 << j))
						pixel_stacks[j]++;

				auto &unique = unique_tris_map[value >> 16];
				if(unique != tindex) {
					unique = tindex;
					stats.unique_tris_sum++;
				}
			}

			int max_depth = 0, min_depth = pixel_stacks[0];
			for(auto val : pixel_stacks) {
				max_depth = max(max_depth, val);
				min_depth = min(min_depth, val);
				stats.num_fragments += val;
			}
			stats.avg_max_pixel_depth += max_depth;
			stats.avg_min_pixel_depth += min_depth;
			stats.max_pixel_depth = max(stats.max_pixel_depth, max_depth);
			stats.avg_tris_per_block += count;
			stats.max_tris_per_block = max(stats.max_tris_per_block, count);
			num_blocktris += count;
			stats.num_nonempty_blocks++;
		}
		stats.max_blocktris_per_tile = max(stats.max_blocktris_per_tile, num_blocktris);
	}

	return stats;
}

void LucidRenderer::analyzeMaskRasterizer() const {
	vector<u32> block_counts, block_offsets, block_instances, tile_counters;
	int bin_count = m_bin_counts.x * m_bin_counts.y;
	int num_block_tris = 0;

	{
		PERF_GPU_SCOPE("download gpu data");
		tile_counters = m_tile_counters->download<u32>();
		num_block_tris = tile_counters[9];
		block_counts = m_block_counts->download<u32>();
		block_offsets = m_block_offsets->download<u32>();
		// TODO: why does it take 100ms to download m_block_tris?
		block_instances = m_block_tris->download<u32>(num_block_tris);
	}

	vector<int> offset_coverage(num_block_tris, 0);
	for(int bin_id = 0; bin_id < bin_count; bin_id++)
		for(int block_id = 0; block_id < blocks_per_bin; block_id++) {
			int offset = block_offsets[bin_id * blocks_per_bin + block_id];
			int count = block_counts[bin_id * blocks_per_bin + block_id];
			if(offset + count > num_block_tris) {
				print("Offset out of bounds: % > %\n", offset + count, num_block_tris);
				return;
			}

			for(int i = 0; i < count; i++)
				offset_coverage[offset + i]++;
		}
	if(anyOf(offset_coverage, [](int v) { return v > 1; })) {
		print("Overlapping offsets detected\n");
		return;
	}

	vector<BinBlockStats> bin_stats(bin_count);
#pragma omp parallel for
	for(int bin_id = 0; bin_id < bin_count; bin_id++)
		bin_stats[bin_id] = computeBlockStats(bin_id, block_instances, block_counts, block_offsets);
	auto stats = BinBlockStats::sum(bin_stats);
	stats.computeAverages();

	print("Mask rasterization statistics: ------------------------------------\n");
	print("\nTotal tri-blocks: %\n", num_block_tris);
	if(stats.num_blocktris != num_block_tris)
		print("Invalid number of summed block tris: %\n", stats.num_blocktris);
	print("Block-tris / unique tris per tile ratio: %\n",
		  double(num_block_tris) / stats.unique_tris_sum);
	print("Total fragments rasterized: %\n", stats.num_fragments);
	print("\nStats for non-empty blocks:\n");
	printf("    Max pixel depth: %d\n", stats.max_pixel_depth);
	printf("Avg min pixel depth: %.2f\n", stats.avg_min_pixel_depth);
	printf("Avg max pixel depth: %.2f\n\n", stats.avg_max_pixel_depth);
	printf("         Max tris per block: %8d\n", stats.max_tris_per_block);
	printf("    Max block-tris per tile: %8d\n", stats.max_blocktris_per_tile);
	printf("         Avg tris per block: %8.2f\n", stats.avg_tris_per_block);
	printf("Avg fragments per tri-block: %8.2f\n", double(stats.num_fragments) / num_block_tris);
	print("\nBlock-tris per bin:\n");

	int max_width = 40, max_height = 60;
	if(auto dim = consoleDimensions()) {
		max_width = (dim->x - 1) / 6;
		max_height = dim->y * 3 / 4;
	}

	for(int y = 0; y < min(max_height, m_bin_counts.y); y++) {
		for(int x = 0; x < min(max_width, m_bin_counts.x); x++) {
			int bin_id = x + y * m_bin_counts.x;
			int count = 0;
			for(int block_id = 0; block_id < blocks_per_bin; block_id++)
				count += block_counts[bin_id * blocks_per_bin + block_id];
			printf("%6d ", count);
		}
		printf("\n");
	}
	print("\n");
}

string RasterBlockInfo::description() const {
	TextFormatter fmt;
	fmt("bin:% tile:% block:%\ntris/block:% tris/tile:%\n", bin_pos, tile_pos, block_pos,
		num_block_tris, num_tile_tris);
	if(num_sub_block_tris)
		fmt.stdFormat("total tris/sub block:%d (%.2f %%)\n", num_sub_block_tris,
					  double(num_sub_block_tris) / num_block_tris * 100);
	if(num_merged_block_tris)
		fmt.stdFormat("merged tris/block:%d (%.2f %%)", num_merged_block_tris,
					  double(num_merged_block_tris) / num_block_tris * 100);
	return fmt.text();
}

vector<vector<int>> RasterTileInfo::triNeighbourMap(int max_dist) const {
	DASSERT(max_dist >= 1);
	vector<vector<int>> neighbours(tri_verts.size());
	HashMap<int, vector<int>> vertex_tri_map(tri_verts.size() * 2);
	for(auto i : intRange(tri_verts))
		for(auto v : tri_verts[i])
			vertex_tri_map[v].emplace_back(i);

	for(int i : intRange(tri_verts)) {
		auto &tri = tri_verts[i];
		auto is_neighbour = [&](int nid) {
			auto &ntri = tri_verts[nid];
			int num_shared = 0;
			for(auto v : tri)
				if(isOneOf(v, ntri))
					num_shared++;
			return num_shared >= 2;
		};

		for(auto v : tri)
			for(auto nid : vertex_tri_map[v])
				if(nid > i && is_neighbour(nid)) {
					neighbours[i].emplace_back(nid);
					neighbours[nid].emplace_back(i);
				}
	}

	max_dist--;
	for(int r = 0; r < max_dist; r++) {
		vector<vector<int>> temp = neighbours;
		for(int i : intRange(tri_verts))
			for(auto n1 : neighbours[i])
				for(auto n2 : neighbours[i])
					if(n1 != n2 && !isOneOf(n2, temp[n1])) {
						temp[n1].emplace_back(n2);
						temp[n2].emplace_back(n1);
					}
		neighbours = move(temp);
	}

	return neighbours;
}

RasterTileInfo LucidRenderer::introspectTile(CSpan<float3> verts, int2 full_tile_pos) const {
	RasterTileInfo out;

	PERF_GPU_SCOPE();
	glMemoryBarrier(GL_ALL_BARRIER_BITS);

	int2 bin_pos = full_tile_pos / 4;
	int2 tile_pos = full_tile_pos - bin_pos * 4;
	out.bin_pos = bin_pos;
	out.tile_pos = tile_pos;

	out.bin_id = bin_pos.x + bin_pos.y * m_bin_counts.x;
	out.tile_id = out.bin_id * tiles_per_bin + tile_pos.x + tile_pos.y * (bin_size / tile_size);
	int bin_count = m_bin_counts.x * m_bin_counts.y;

	vector<u32> tile_tri_indices, block_tri_masks[4];
	vector<Triangle3F> tile_tris;
	vector<int> tile_tri_instances;
	vector<array<uint, 3>> tile_tri_verts;

	auto tile_counters = m_tile_counters->map<u32>(AccessMode::read_only);
	int num_tile_tris = tile_counters[32 + out.tile_id];
	int tile_tri_offset = tile_counters[32 + bin_count * tiles_per_bin + out.tile_id];
	m_tile_counters->unmap();

	if(num_tile_tris)
		tile_tri_indices = m_tile_tris->download<u32>(num_tile_tris, tile_tri_offset);

	auto quad_indices = m_quad_indices->map<u32>(AccessMode::read_only);
	out.tris.reserve(tile_tri_indices.size());
	out.tri_verts.reserve(tile_tri_indices.size());
	out.tri_instances.reserve(tile_tri_indices.size());

	for(auto idx : tile_tri_indices) {
		bool second_tri = idx & 0x80000000;
		idx &= 0xffffff;
		u32 v0 = quad_indices[idx * 4 + 0], v1 = quad_indices[idx * 4 + 1];
		u32 v2 = quad_indices[idx * 4 + 2], v3 = quad_indices[idx * 4 + 3];
		uint instance_id = (v0 >> 26) | ((v1 >> 20) & 0xfc0) | ((v2 >> 14) & 0x3f000);
		v0 &= 0x3ffffff, v1 &= 0x3ffffff, v2 &= 0x3ffffff, v3 &= 0x3ffffff;
		out.tri_verts.emplace_back(v0, second_tri ? v2 : v1, second_tri ? v3 : v2);
		out.tris.emplace_back(verts[v0], verts[second_tri ? v2 : v1], verts[second_tri ? v3 : v2]);
		out.tri_instances.emplace_back(instance_id);
	}

	out.tri_indices = move(tile_tri_indices);
	m_quad_indices->unmap();

	return out;
}

RasterBlockInfo LucidRenderer::introspectBlock4x4(const RasterTileInfo &tile, int2 full_block_pos,
												  bool merge_masks) const {
	RasterBlockInfo out;
	PERF_GPU_SCOPE();
	int2 tile_pos = full_block_pos / 4;
	int2 bin_pos = tile_pos / 4;
	int2 block_pos = full_block_pos - tile_pos * 4;
	tile_pos -= bin_pos * 4;
	out.bin_pos = bin_pos;
	out.tile_pos = tile_pos;
	out.block_pos = block_pos;

	int bin_id = bin_pos.x + bin_pos.y * m_bin_counts.x;
	int tile_id = bin_id * tiles_per_bin + tile_pos.x + tile_pos.y * (bin_size / tile_size);
	int block_id = tile_id * blocks_per_tile + block_pos.x + block_pos.y * (tile_size / block_size);
	int bin_count = m_bin_counts.x * m_bin_counts.y;

	vector<u32> masks;

	out.num_tile_tris = tile.tris.size();
	out.num_block_tris = m_block_counts->map<u32>(AccessMode::read_only)[block_id];
	m_block_counts->unmap();
	int block_tri_offset = m_block_offsets->map<u32>(AccessMode::read_only)[block_id];
	m_block_offsets->unmap();
	if(out.num_block_tris)
		masks = m_block_tris->download<u32>(out.num_block_tris, block_tri_offset);
	vector<Pair<float>> mask_depths;

	// Note: it makes no sense without merging non-overlapping masks into layers

	for(int i : intRange(masks)) {
		u32 mask = masks[i] & 0xffff;
		u32 local_idx = masks[i] >> 16;
		auto &tri = tile.tris[local_idx];
		float depth_min = inf, depth_max = -inf;
		if(tri.degenerate())
			continue;
		Plane3F plane(tri);

		for(int j = 0; j < 16; j++) {
			if((mask & (1 << j)) == 0)
				continue;
			int2 pos = full_block_pos * 4 + int2(j % 4, j / 4);
			float3 ray_origin = m_frustum_rays.origin0;
			float3 ray_dir = normalize(m_frustum_rays.dir0 + m_frustum_rays.dirx * float(pos.x) +
									   m_frustum_rays.diry * float(pos.y));

			auto isect = Ray3F(ray_origin, ray_dir).isectParam(plane);
			if(isect.isPoint()) {
				depth_min = min(depth_min, isect.asPoint());
				depth_max = max(depth_max, isect.asPoint());
			}
		}
		mask_depths.emplace_back(depth_min, depth_max);
	}

	// 1: overlaps, 2: is overlapped
	vector<bool> mask_overlaps(mask_depths.size(), false);
	for(int i : intRange(mask_depths)) {
		for(int j = i + 1; j < mask_depths.size(); j++)
			if(mask_depths[j].first > mask_depths[i].second)
				mask_overlaps[i] = mask_overlaps[j] = true;
		for(int j = i - 1; j >= 0; j--)
			if(mask_depths[j].second < mask_depths[i].first)
				mask_overlaps[i] = mask_overlaps[j] = true;
	}

	out.selected_tile_tris.resize(tile.tris.size());
	for(int i : intRange(masks)) {
		u32 local_idx = masks[i] >> 16;
		out.selected_tile_tris[local_idx] = true;
	}

	if(!masks)
		return out;

	print("Triangle masks: %", masks.size());
	/*for(int i : intRange(masks)) {
		printf("%4d: %c %f - %f", i, mask_overlaps[i] ? 'X' : ' ', mask_depths[i].first,
			   mask_depths[i].second);
		uint local_idx = masks[i] >> 16;
		print("%\n", tile.tris[local_idx]);
	}*/
	int max_row_size = 16;
	for(int i = 0; i < masks.size(); i += max_row_size) {
		printf("\n");
		int row_size = min(masks.size() - i, max_row_size);
		for(int j = 0; j < row_size; j++) {
			uint local_idx = masks[i + j] >> 16;
			uint tri_idx = tile.tri_indices[local_idx];
			printf(" %4d  ", i + j);
			//printf("%c%6d ", tri_idx & 0x80000000 ? '*' : ' ', tri_idx & 0xffffff);
		}
		printf("\n");
		for(int iy = 0; iy < 4; iy++) {
			for(int j = 0; j < row_size; j++) {
				u32 mask = masks[i + j];
				int y = 3 - iy;
				printf(" ");
				for(int x = 0; x < 4; x++)
					printf("%c", mask & (1 << (x + y * 4)) ? 'X' : '.');
				printf(j + 1 == row_size ? "\n" : "  ");
			}
		}
	}
	print("\n");
	vector<pair<int, u16>> mask_layers;
	mask_layers.emplace_back(0, 0);
	for(int i : intRange(masks)) {
		auto &layer = mask_layers.back();
		auto mask = masks[i] & 0xffff;
		if((layer.second & mask) == 0) {
			layer.first++;
			layer.second |= mask;
		} else {
			mask_layers.emplace_back(1, mask);
		}
	}

	int num_fragments = 0;
	for(auto &mask : masks)
		num_fragments += countBits(mask & 0xffff);
	print("masks:% fragments:% layers:%\n", masks.size(), num_fragments, mask_layers.size());
	printf("fragments/mask:%.2f fragments/layer:%.2f masks/layer:%.2f\n",
		   double(num_fragments) / masks.size(), double(num_fragments) / mask_layers.size(),
		   double(masks.size()) / mask_layers.size());
	for(int i = 0; i < mask_layers.size(); i += 4) {
		int row_size = min(mask_layers.size() - i, 4);
		for(int j = 0; j < row_size; j++) {
			auto &layer = mask_layers[i + j];
			printf("[%03d]: %2d tri %2d pix %s", i + j, layer.first, countBits((u32)layer.second),
				   j + 1 == row_size ? "\n" : "| ");
		}
	}
	print("\n");

	if(!merge_masks)
		return out;

	struct MergedMask4x4 {
		vector<u16> tri_ids;
		array<u8, 16> indices;
		u16 bits = 0;
	};

	auto neighbour_map = tile.triNeighbourMap(4);
	auto are_compatible_tris = [&](int id0, int id1) -> bool {
		return true || isOneOf(id1, neighbour_map[id0]);
	};

	// We have to make sure that depth ranges don't get too mixed up...
	vector<MergedMask4x4> mmasks;
	constexpr int max_merged_tris = 15;
	for(auto &mask : masks) {
		int mmask_idx = -1;
		u16 bits = u16(mask & 0xffff);
		u16 tri_id = u16(mask >> 16);

		for(int i = max(0, mmasks.size() - 1); i < mmasks.size(); i++)
			if((mmasks[i].bits & bits) == 0 && mmasks[i].tri_ids.size() < max_merged_tris) {
				bool compatible = true;
				for(auto tid : mmasks[i].tri_ids)
					if(!are_compatible_tris(tri_id, tid)) {
						compatible = false;
						break;
					}
				if(compatible) {
					mmask_idx = i;
					break;
				}
			}
		if(mmask_idx == -1) {
			MergedMask4x4 new_mask;
			new_mask.bits = 0;
			fill(new_mask.indices, 255);
			mmask_idx = mmasks.size();
			mmasks.emplace_back(new_mask);
		}

		auto &mmask = mmasks[mmask_idx];
		mmask.bits |= bits;
		uint index = mmask.tri_ids.size();
		mmask.tri_ids.emplace_back(tri_id);
		for(int i : intRange(64))
			if(bits & (1ull << i))
				mmask.indices[i] = index;
	}

	out.num_merged_block_tris = mmasks.size();

	printf("\nMerged masks: %d (%.2f %%)\n", mmasks.size(),
		   double(mmasks.size()) / masks.size() * 100.0);
	max_row_size = 6;
	for(int i = 0; i < mmasks.size(); i += max_row_size) {
		printf("\n");
		int row_size = min(mmasks.size() - i, max_row_size);
		for(int j = 0; j < row_size; j++) {
			int num_tris = (int)mmasks[i + j].tri_ids.size();
			printf("  %4d (%d)    %s", i + j, num_tris, num_tris < 10 ? " " : "");
		}
		printf("\n");
		for(int iy = 0; iy < 4; iy++) {
			for(int j = 0; j < row_size; j++) {
				auto &mmask = mmasks[i + j];
				int y = 3 - iy;
				printf(" ");
				for(int x = 0; x < 4; x++) {
					int index = x + y * 4;
					if(mmask.indices[index] != 255)
						printf("%2d ", mmask.indices[index]);
					else
						printf(" . ");
				}
				printf(j + 1 == row_size ? "\n" : "  ");
			}
		}
	}
	print("\n");

	return out;
}

RasterBlockInfo LucidRenderer::introspectBlock8x8(const RasterTileInfo &tile,
												  int2 full_block8x8_pos, bool merge_masks) const {
	RasterBlockInfo out;
	PERF_GPU_SCOPE();
	int2 tile_pos = full_block8x8_pos / 2;
	int2 bin_pos = tile_pos / 4;
	int2 block8x8_pos = full_block8x8_pos - tile_pos * 2;
	tile_pos -= bin_pos * 4;
	out.bin_pos = bin_pos;
	out.tile_pos = tile_pos;
	out.block_pos = block8x8_pos;

	int bin_id = bin_pos.x + bin_pos.y * m_bin_counts.x;
	int tile_id = bin_id * tiles_per_bin + tile_pos.x + tile_pos.y * (bin_size / tile_size);
	int block_ids[4];
	for(int i = 0; i < 4; i++) {
		int2 bpos = int2(block8x8_pos.x * 2 + i % 2, block8x8_pos.y * 2 + i / 2);
		block_ids[i] = tile_id * blocks_per_tile + bpos.x + bpos.y * (tile_size / block_size);
	}
	int bin_count = m_bin_counts.x * m_bin_counts.y;
	int block_tri_counts[4], block_tri_offsets[4];

	vector<u32> block_tri_masks[4];

	{
		auto block_counts = m_block_counts->map<u32>(AccessMode::read_only);
		auto block_offsets = m_block_offsets->map<u32>(AccessMode::read_only);
		auto block_tris = m_block_tris->map<u32>(AccessMode::read_only);
		out.num_tile_tris = tile.tris.size();
		for(int i = 0; i < 4; i++) {
			block_tri_counts[i] = block_counts[block_ids[i]];
			block_tri_offsets[i] = block_offsets[block_ids[i]];
			if(block_tri_counts[i]) {
				block_tri_masks[i] = block_tris.subSpan(block_tri_offsets[i],
														block_tri_offsets[i] + block_tri_counts[i]);
				out.num_sub_block_tris += block_tri_counts[i];
			}
		}
		m_block_counts->unmap();
		m_block_offsets->unmap();
		m_block_tris->unmap();
	}

	struct Mask8x8 {
		u64 bits;
		int tri_id;
		Pair<float> depth = {0, 0};
	};

	vector<Mask8x8> masks8x8;
	array<int, 64> layer_depth;
	fill(layer_depth, 0);

	{
		vector<int> tri_ids;
		for(auto &masks : block_tri_masks)
			for(auto mask : masks) {
				int tri_id = mask >> 16;
				if(!isOneOf(tri_id, tri_ids))
					tri_ids.emplace_back(tri_id);
			}

		for(auto tri_id : tri_ids) {
			u64 mask8x8 = 0;
			for(int i = 0; i < 4; i++) {
				int bx = i % 2, by = i / 2;
				u32 cur_mask = 0;
				for(auto mask : block_tri_masks[i])
					if((mask >> 16) == tri_id) {
						cur_mask = mask;
						break;
					}

				if(cur_mask)
					for(int y = 0; y < 4; y++) {
						uint row_bits = (cur_mask >> (y * 4)) & 0xf;
						mask8x8 |= u64(row_bits) << (bx * 4 + y * 8 + by * 32);
					}
			}
			for(int i : intRange(64))
				if(mask8x8 & (1ull << i))
					layer_depth[i]++;
			masks8x8.emplace_back(mask8x8, tri_id);
		}
	}

	out.num_block_tris = masks8x8.size();

	// TODO: We're assuming that backface culling is enabled?
	// Note: it makes no sense without merging non-overlapping masks into layers
	for(auto &mask : masks8x8) {
		auto &tri = tile.tris[mask.tri_id];
		float depth_min = inf, depth_max = -inf;
		Plane3F plane(tri);

		for(int j = 0; j < 64; j++) {
			if((mask.bits & (1ull << j)) == 0)
				continue;
			int2 pos = full_block8x8_pos * 8 + int2(j % 8, j / 8);
			float3 ray_origin = m_frustum_rays.origin0;
			float3 ray_dir = normalize(m_frustum_rays.dir0 + m_frustum_rays.dirx * float(pos.x) +
									   m_frustum_rays.diry * float(pos.y));

			auto isect = Ray3F(ray_origin, ray_dir).isectParam(plane);
			if(isect.isPoint()) {
				depth_min = min(depth_min, isect.asPoint());
				depth_max = max(depth_max, isect.asPoint());
			}
		}
		mask.depth = {depth_min, depth_max};
	}

	std::sort(begin(masks8x8), end(masks8x8),
			  [](const Mask8x8 &a, const Mask8x8 &b) { return a.depth.first < b.depth.first; });

	struct MergedMask8x8 {
		vector<int> tri_ids;
		array<u8, 64> indices;
		Pair<float> depth = {inf, -inf};
		u64 bits;
	};

	auto neighbour_map = tile.triNeighbourMap(4);
	auto are_compatible_tris = [&](int id0, int id1) -> bool {
		return true || isOneOf(id1, neighbour_map[id0]);
	};

	// We have to make sure that depth ranges don't get too mixed up...
	vector<MergedMask8x8> mmasks;
	constexpr int max_merged_tris = 15;
	for(auto &mask : masks8x8) {
		int mmask_idx = -1;

		for(int i = max(0, mmasks.size() - 1); i < mmasks.size(); i++)
			if((mmasks[i].bits & mask.bits) == 0 && mmasks[i].tri_ids.size() < max_merged_tris) {
				bool compatible = true;
				for(auto tid : mmasks[i].tri_ids)
					if(!are_compatible_tris(mask.tri_id, tid)) {
						compatible = false;
						break;
					}
				if(compatible) {
					mmask_idx = i;
					break;
				}
			}
		if(mmask_idx == -1) {
			MergedMask8x8 new_mask;
			new_mask.bits = 0;
			fill(new_mask.indices, 255);
			mmask_idx = mmasks.size();
			mmasks.emplace_back(new_mask);
		}

		auto &mmask = mmasks[mmask_idx];
		mmask.bits |= mask.bits;
		uint index = mmask.tri_ids.size();
		mmask.tri_ids.emplace_back(mask.tri_id);
		mmask.depth.first = min(mmask.depth.first, mask.depth.first);
		mmask.depth.second = max(mmask.depth.second, mask.depth.second);
		for(int i : intRange(64))
			if(mask.bits & (1ull << i))
				mmask.indices[i] = index;
	}
	out.num_merged_block_tris = mmasks.size();

	print("Triangle masks: %\n", masks8x8.size());
	int max_row_size = 8;
	for(int i = 0; i < masks8x8.size(); i += max_row_size) {
		int row_size = min(masks8x8.size() - i, max_row_size);
		for(int j = 0; j < row_size; j++)
			printf("      %3d      ", i + j);
		printf("\n");
		for(int j = 0; j < row_size; j++)
			printf(" %6.2f:%6.2f ", masks8x8[i + j].depth.first, masks8x8[i + j].depth.second);
		printf("\n");
		for(int iy = 0; iy < 8; iy++) {
			for(int j = 0; j < row_size; j++) {
				u64 mask = masks8x8[i + j].bits;
				int y = 7 - iy;
				printf("    ");
				for(int x = 0; x < 8; x++)
					printf("%c", mask & (1ull << (x + y * 8)) ? 'X' : '.');
				printf(j + 1 == row_size ? "\n" : "   ");
			}
		}
	}

	if(merge_masks) {
		printf("\nMerged masks: %d (%.2f %%)\n", mmasks.size(),
			   double(mmasks.size()) / masks8x8.size() * 100.0);
		max_row_size = 4;
		for(int i = 0; i < mmasks.size(); i += max_row_size) {
			printf("\n");
			int row_size = min(mmasks.size() - i, max_row_size);
			for(int j = 0; j < row_size; j++)
				printf("        %4d:%4d          ", i + j, (int)mmasks[i + j].tri_ids.size());
			printf("\n");
			for(int j = 0; j < row_size; j++)
				printf("       %6.2f:%6.2f       ", mmasks[i + j].depth.first,
					   mmasks[i + j].depth.second);
			printf("\n");
			for(int iy = 0; iy < 8; iy++) {
				for(int j = 0; j < row_size; j++) {
					auto &mmask = mmasks[i + j];
					int y = 7 - iy;
					printf(" ");
					for(int x = 0; x < 8; x++) {
						int index = x + y * 8;
						if(mmask.indices[index] != 255)
							printf("%2d ", mmask.indices[index]);
						else
							printf(" . ");
					}
					printf(j + 1 == row_size ? "\n" : "  ");
				}
			}
		}
	}

	print("\nLayer depths: (min/max: %)\n", minMax(layer_depth));
	for(int y = 0; y < 8; y++) {
		for(int x = 0; x < 8; x++) {
			printf("%3d ", layer_depth[x + y * 8]);
		}
		printf("\n");
	}

	out.selected_tile_tris.resize(tile.tris.size());
	for(auto &mask : masks8x8)
		out.selected_tile_tris[mask.tri_id] = true;

	return out;
}

Image LucidRenderer::masksSnapshot() {
	auto block_counts = m_block_counts->download<u32>();
	auto block_offsets = m_block_offsets->download<u32>();
	auto tile_counters = m_tile_counters->download<u32>();

	int num_block_tris = tile_counters[9];
	int bin_count = m_bin_counts.x * m_bin_counts.y;

	Image image(m_bin_counts * bin_size, ColorId::black);
	auto block_instances = m_block_tris->download<u32>(num_block_tris);
	for(int bin_id = 0; bin_id < bin_count; bin_id++) {
		int2 bin_pos = int2(bin_id % m_bin_counts.x, bin_id / m_bin_counts.x) * bin_size;

		for(int tile_id = 0; tile_id < tiles_per_bin; tile_id++) {
			int2 tile_pos = bin_pos + int2(tile_id % 4, tile_id / 4) * tile_size;

			for(int block_id = 0; block_id < blocks_per_tile; block_id++) {
				int2 block_pos = tile_pos + int2(block_id % 4, block_id / 4) * block_size;
				int bindex = bin_id * blocks_per_bin + tile_id * blocks_per_tile + block_id;
				auto offset = block_offsets[bindex];
				int count = block_counts[bindex];

				for(int i = 0; i < count; i++) {
					u32 value = block_instances[offset + i];
					for(int y = 0; y < 4; y++)
						for(int x = 0; x < 4; x++)
							if(value & (1 << (x + y * 4))) {
								int2 pixel_pos = block_pos + int2(x, y);
								auto &pixel =
									image({pixel_pos.x, image.height() - 1 - pixel_pos.y});
								pixel.r = min(255, pixel.r + 64);
								pixel.g = min(255, pixel.g + 8);
								pixel.b = min(255, pixel.b + 1);
							}
				}
			}
		}
	}

	return image;
}

void LucidRenderer::bindRaster(const Context &ctx) {
	m_tri_aabbs->bindIndex(0);
	m_quad_indices->bindIndex(1);
	auto vbuffers = ctx.vao->buffers();
	DASSERT(vbuffers.size() == 4);
	vbuffers[0]->bindIndexAs(2, BufferType::shader_storage);
	if(auto tex_vb = vbuffers[2])
		tex_vb->bindIndexAs(3, BufferType::shader_storage);
	if(auto col_vb = vbuffers[1])
		col_vb->bindIndexAs(4, BufferType::shader_storage);
	if(auto nrm_vb = vbuffers[3])
		nrm_vb->bindIndexAs(5, BufferType::shader_storage);

	m_bin_counters->bindIndex(6);
	m_tile_counters->bindIndex(7);
	m_tile_tris->bindIndex(8);
	m_scratch->bindIndex(9);
	m_instance_data->bindIndex(10);
	m_uv_rects->bindIndex(11);
	m_raster_image->bindIndex(12); // TODO: too many bindings
}

void LucidRenderer::rasterBlock(const Context &ctx) {
	PERF_GPU_SCOPE();
	raster_block_program.use();

	GlTexture::bind({ctx.opaque_tex, ctx.trans_tex});
	//GlTexture::bind({ctx.depth_buffer, ctx.shadows.map});

	ctx.lighting.setUniforms(raster_block_program.glProgram());
	raster_block_program.setFrustum(ctx.camera);
	raster_block_program.setViewport(ctx.camera, m_size);
	raster_block_program.setShadows(ctx.shadows.matrix, ctx.shadows.enable);
	ctx.lighting.setUniforms(raster_block_program.glProgram());

	glDispatchCompute(64, 1, 1);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void LucidRenderer::rasterTile(const Context &ctx) {
	PERF_GPU_SCOPE();

	raster_tile_program.use();

	GlTexture::bind({ctx.opaque_tex, ctx.trans_tex});
	//GlTexture::bind({ctx.depth_buffer, ctx.shadows.map});

	ctx.lighting.setUniforms(raster_tile_program.glProgram());
	raster_tile_program.setFrustum(ctx.camera);
	raster_tile_program.setViewport(ctx.camera, m_size);
	raster_tile_program.setShadows(ctx.shadows.matrix, ctx.shadows.enable);
	ctx.lighting.setUniforms(raster_tile_program.glProgram());

	if(m_opts & Opt::debug_raster)
		shaderDebugUseBuffer(m_errors);

	// TODO: lepiej by było, jakby było to bardziej zintegrowane z GlProgramem
	// - dispatch też mógłby być funkcją debuggera);
	// - Inny debugger dla compute i inny dla pozostałych shaderów
	// - możliwość przekazywania konkretnych wartości (np. 4 różne wartości?)
	// - jakaś klasa do prostej introspekcji linii kodu programu
	glDispatchCompute(64, 1, 1);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

	if(m_opts & Opt::debug_raster) {
		auto source_ranges = raster_tile_program.sourceRanges();
		auto records = shaderDebugRecords(m_errors, {256, 1, 1}, {128, 1, 1}, 256, source_ranges);
		if(records) {
			makeSorted(records);
			print("raster_tile shader debug messages reported:\n");
			for(auto &record : records)
				print("%\n", record);
		}
	}
}

void LucidRenderer::rasterBin(const Context &ctx) {
	PERF_GPU_SCOPE();

	raster_bin_program.use();
	m_bin_quads->bindIndex(8);

	GlTexture::bind({ctx.opaque_tex, ctx.trans_tex});
	//GlTexture::bind({ctx.depth_buffer, ctx.shadows.map});

	ctx.lighting.setUniforms(raster_bin_program.glProgram());
	raster_bin_program.setFrustum(ctx.camera);
	raster_bin_program.setViewport(ctx.camera, m_size);
	raster_bin_program.setShadows(ctx.shadows.matrix, ctx.shadows.enable);
	raster_bin_program["additive_blending"] = ctx.config.additive_blending;
	ctx.lighting.setUniforms(raster_bin_program.glProgram());

	if(m_opts & Opt::debug_raster)
		shaderDebugUseBuffer(m_errors);

	glDispatchCompute(128, 1, 1);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	m_tile_tris->bindIndex(8);

	if(m_opts & Opt::debug_raster) {
		auto source_ranges = raster_tile_program.sourceRanges();
		auto records = shaderDebugRecords(m_errors, {256, 1, 1}, {128, 1, 1}, 256, source_ranges);
		if(records) {
			makeSorted(records);
			print("raster_bin shader debug messages reported:\n");
			for(auto &record : records)
				print("%\n", record);
		}
	}
}

void LucidRenderer::rasterizeFinal(const Context &ctx) {
	PERF_GPU_SCOPE();

	m_instance_data->bindIndex(0);
	m_quad_indices->bindIndex(1);
	auto vbuffers = ctx.vao->buffers();
	DASSERT(vbuffers.size() == 4);
	vbuffers[0]->bindIndexAs(2, BufferType::shader_storage);
	if(auto tex_vb = vbuffers[2])
		tex_vb->bindIndexAs(3, BufferType::shader_storage);
	if(auto col_vb = vbuffers[1])
		col_vb->bindIndexAs(4, BufferType::shader_storage);
	if(auto nrm_vb = vbuffers[3])
		nrm_vb->bindIndexAs(5, BufferType::shader_storage);

	m_tile_counters->bindIndex(6);
	m_block_counts->bindIndex(7);
	m_block_offsets->bindIndex(8);

	m_tile_tris->bindIndex(9);
	m_block_tris->bindIndex(10);
	m_uv_rects->bindIndex(11);
	m_raster_image->bindIndex(12);
	final_raster_program.use();

	GlTexture::bind({ctx.opaque_tex, ctx.trans_tex});
	//GlTexture::bind({ctx.depth_buffer, ctx.shadows.map});

	final_raster_program["fog_multiplier"] = 0.3f / ctx.camera.params().ortho_scale;

	ctx.lighting.setUniforms(final_raster_program.glProgram());
	final_raster_program.setFrustum(ctx.camera);
	final_raster_program.setViewport(ctx.camera, m_size);
	final_raster_program.setShadows(ctx.shadows.matrix, ctx.shadows.enable);
	ctx.lighting.setUniforms(final_raster_program.glProgram());

	// TODO: lepiej by było, jakby było to bardziej zintegrowane z GlProgramem
	// - dispatch też mógłby być funkcją debuggera);
	// - Inny debugger dla compute i inny dla pozostałych shaderów
	// - możliwość przekazywania konkretnych wartości (np. 4 różne wartości?)
	// - jakaś klasa do prostej introspekcji linii kodu programu
	glDispatchCompute(128, 1, 1);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

// Final stage: blits transparent pixels onto main framebuffer
void LucidRenderer::compose(const Context &ctx) {
	PERF_GPU_SCOPE();

	DASSERT(!ctx.out_fbo || ctx.out_fbo->size() == m_size);
	glDrawBuffer(GL_BACK);
	setupView(IRect(m_size), ctx.out_fbo);
	glEnable(GL_BLEND);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	glDepthMask(0);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	compose_program.setFullscreenRect();
	compose_program.use();
	compose_program["bin_counts"] = m_bin_counts;
	compose_program["screen_scale"] = float2(1.0) / float2(m_size);
	m_raster_image->bindIndex(0);
	m_compose_quads_vao->draw(PrimitiveType::triangles, m_bin_counts.x * m_bin_counts.y * 6);
	glDisable(GL_BLEND);
}

void LucidRenderer::copyCounters() {
	auto last = m_old_counters.back();
	for(int i = m_old_counters.size() - 1; i > 0; i--)
		m_old_counters[i] = m_old_counters[i - 1];
	m_old_counters[0] = last;

	if(!m_old_counters[0].first) {
		m_old_counters[0].first.emplace(BufferType::copy_read, m_bin_counters->size());
		m_old_counters[0].second.emplace(BufferType::copy_read, m_tile_counters->size());
	}

	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	m_bin_counters->copyTo(m_old_counters[0].first, 0, 0, m_old_counters[0].first->size());
	m_tile_counters->copyTo(m_old_counters[0].second, 0, 0, m_old_counters[0].second->size());
}

vector<StatsGroup> LucidRenderer::getStats() const {
	vector<StatsGroup> out;

	if(!m_old_counters.back().first)
		return out;

	// TODO: double/triple buffering to avoid stall
	int bin_count = m_bin_counts.x * m_bin_counts.y;
	auto bin_counters = m_old_counters.back().first->download<u32>();
	auto tile_counters =
		m_old_counters.back().second->download<u32>(32 + bin_count * tiles_per_bin);

	int num_pixels = m_size.x * m_size.y;
	int num_tiles =
		((m_size.x + tile_size - 1) / tile_size) * ((m_size.y + tile_size - 1) / tile_size);
	int num_blocks =
		((m_size.x + block_size - 1) / block_size) * ((m_size.y + block_size - 1) / block_size);

	int num_input_quads = bin_counters[1];
	int num_invalid_pixels = tile_counters[10];
	int num_invalid_blocks = tile_counters[11];
	int num_invalid_tiles = tile_counters[12];

	int tile_tris_estimate = tile_counters[8];
	int num_block_tris = tile_counters[9];
	int num_block_rows = tile_counters[14];

	int num_tile_tris = 0, num_nonempty_tiles = 0, num_nonempty_bins = 0;
	int max_quads_per_bin = 0, num_bin_quads = 0;
	for(int b = 0; b < bin_count; b++) {
		bool empty = true;
		for(int t = 0; t < 16; t++) {
			int count = tile_counters[32 + b * tiles_per_bin + t];
			num_tile_tris += count;
			if(count) {
				num_nonempty_tiles++;
				empty = false;
			}
		}
		if(!empty)
			num_nonempty_bins++;
		int bin_quads = bin_counters[32 + b];
		max_quads_per_bin = max(max_quads_per_bin, bin_quads);
		num_bin_quads += bin_quads;
	}

	int num_rejected[4];
	for(int i = 0; i < 4; i++)
		num_rejected[i] = bin_counters[4 + i];
	num_rejected[0] += num_rejected[1] + num_rejected[2] + num_rejected[3];

	auto rejected_info =
		stdFormat("%d (%.2f %%)", num_rejected[0], double(num_rejected[0]) / num_input_quads * 100);
	auto rejection_details = format("backface: %\nfrustum: %\nbetween-samples: %", num_rejected[1],
									num_rejected[2], num_rejected[3]);

	vector<StatsRow> timings;
	Str timer_names[] = {"generate rows", "generate blocks", "load samples", "shade samples",
						 "reduce samples"};
	u64 total = 0;
	for(int i : intRange(timer_names))
		total += bin_counters[15 + i];
	if(total)
		for(int i : intRange(timer_names)) {
			auto value = bin_counters[15 + i];
			timings.emplace_back(timer_names[i], stdFormat("%.2f %%", double(value) / total * 100));
		}

	vector<StatsRow> basic_rows = {
		{"input instances", toString(m_num_instances)},
		{"input quads", toString(num_input_quads)},
		{"rejected quads", rejected_info, rejection_details},
		{"bin-quads", toString(num_bin_quads), "Per-bin quads"},
		{"tile-tris", toString(num_tile_tris), "Per-tile triangles"},
		{"estimated tile-tris", toString(tile_tris_estimate),
		 "Estimating space needed for tile-tris is inaccurate"},
		{"empty tile-tris",
		 stdFormat("%d (%.2f %%)", tile_counters[13],
				   double(tile_counters[13]) / num_tile_tris * 100.0),
		 "Per-tile triangles which generate no samples"},
		{"row-tris", toString(num_block_rows), "Block rows generated for each per-tile triangle"},
		{"block-tris", toString(num_block_tris),
		 "Per-block triangle instances with at least 1 sample"},
		{"fragments", toString(tile_counters[19])},
	};

	vector<StatsRow> avg_rows = {
		{"quads / non-empty bin", stdFormat("%.2f", double(num_bin_quads) / num_nonempty_bins)},
		{"tile-tris / non-empty tile",
		 stdFormat("%.2f", double(num_tile_tris) / num_nonempty_tiles)},
		{"row-tris / non-empty tile",
		 stdFormat("%.2f", double(num_block_rows) / num_nonempty_tiles)},
		{"block-tris / non-empty tile",
		 stdFormat("%.2f", double(num_block_tris) / num_nonempty_tiles)},
		{"block-tris / block*",
		 stdFormat("%.2f", double(num_block_tris) / (num_nonempty_tiles * blocks_per_tile)),
		 "Counting all blocks in non-empty tiles"},
		{"block-tris / pixel*",
		 stdFormat("%.2f", double(num_block_tris) / (num_nonempty_tiles * square(tile_size))),
		 "Counting all pixels in non-empty tiles"},
	};

	vector<StatsRow> max_rows = {
		{"max quads / bin", toString(max_quads_per_bin)},
		{"max tile-tris / tile", toString(tile_counters[16])},
		{"max row-tris / tile", toString(tile_counters[17])},
		{"max block-tris / tile", toString(tile_counters[18])},
		{"max block-tris / block", toString(tile_counters[15])},
		{"max fragments / tile", toString(tile_counters[20])},
		{"max fragments / pixel", toString(tile_counters[21])},
	};

	vector<StatsRow> invalid_rows = {
		{"invalid pixels", stdFormat("%d (%.3f %%)", num_invalid_pixels,
									 float(num_invalid_pixels) / num_pixels * 100.0)},
		{"invalid blocks", stdFormat("%d (%.3f %%)", num_invalid_blocks,
									 float(num_invalid_blocks) / num_blocks * 100.0)},
		{"invalid tiles", stdFormat("%d (%.3f %%)", num_invalid_tiles,
									float(num_invalid_tiles) / num_tiles * 100.0)},
	};

	if(timings)
		out.emplace_back(move(timings), "", 130);
	out.emplace_back(move(basic_rows), "", 130);
	out.emplace_back(move(avg_rows), "Averages per non-empty bin/tile", 130);
	out.emplace_back(move(max_rows), "", 130);
	out.emplace_back(move(invalid_rows), "", 130);
	return out;
}

vector<int> generateRangeHistogram(CSpan<float2> ranges, int res) {
	vector<int> counts(res, 0);
	for(auto &range : ranges) {
		int begin = int(range[0] * res), end = int(range[1] * res) + 1;
		if(counts.inRange(begin))
			counts[begin]++;
		if(counts.inRange(end))
			counts[end]--;
	}
	int counter = 0;
	for(auto &val : counts) {
		int next = counter + val;
		val = counter + val;
		counter = next;
	}
	return counts;
}

vector<int> generateMinHistogram(CSpan<float2> ranges, int res) {
	vector<int> counts(res, 0);
	for(auto &range : ranges) {
		int begin = int(range[0] * res);
		if(counts.inRange(begin))
			counts[begin]++;
	}
	return counts;
}

void LucidRenderer::printHistograms() const {
	vector<IRect> quads;
	{
		auto quad_aabbs = m_quad_aabbs->download<u32>(m_num_quads);
		auto tri_aabbs = m_tri_aabbs->download<int4>(m_num_quads);
		quads.resize(m_num_quads);
		for(int i : intRange(quads)) {
			if(quad_aabbs[i] == ~0u)
				continue;
			auto enc = tri_aabbs[i];
			int2 min0(u32(enc[0]) & 0xffff, u32(enc[0]) >> 16);
			int2 max0(u32(enc[1]) & 0xffff, u32(enc[1]) >> 16);
			int2 min1(u32(enc[2]) & 0xffff, u32(enc[2]) >> 16);
			int2 max1(u32(enc[3]) & 0xffff, u32(enc[3]) >> 16);

			max0 = vmax(max0, min0);
			max1 = vmax(max1, min1);
			quads[i] = {vmin(min0, min1), vmax(max0, max1)};
		}
	}

	constexpr int max_dim = 16;
	auto get_quad = [&](IRect &out, int idx, int shift) {
		auto &quad = quads[idx];
		if(quad.max() == quad.min())
			return false;
		int gsx = quad.x() >> shift, gsy = quad.y() >> shift;
		int gex = quad.ex() >> shift, gey = quad.ey() >> shift;
		out = {gsx, gsy, gex, gey};
		return true;
	};

	int num_active_quads = 0;
	for(int i = 0; i < m_num_quads; i++) {
		IRect rect;
		if(get_quad(rect, i, 0))
			num_active_quads++;
	}

	print("Active quads: %\nAll quads: % (% rejected)\n", num_active_quads, m_num_quads,
		  m_num_quads - num_active_quads);
	print("Quad size distribution:\n");
	for(int gsize = 128; gsize >= 4; gsize /= 2) {
		int counts[max_dim * max_dim] = {
			0,
		};
		int shift = log2(gsize);

		for(int i = 0; i < m_num_quads; i++) {
			IRect rect;
			if(!get_quad(rect, i, shift))
				continue;
			int gw = min(rect.width(), max_dim - 1);
			int gh = min(rect.height(), max_dim - 1);
			counts[gw + gh * max_dim]++;
		}

		int max_x = 0, max_y = 0;
		for(int y = 0; y < max_dim; y++)
			for(int x = 0; x < max_dim; x++)
				if(counts[x + y * max_dim]) {
					max_x = max(max_x, x);
					max_y = max(max_y, y);
				}

		printf("[%4d]: ", gsize);
		for(int x = 0; x <= max_x; x++)
			printf("   X=%02d ", x + 1);
		printf("\n");
		for(int y = 0; y <= max_y; y++) {
			printf("  Y=%02d  ", y + 1);
			for(int x = 0; x <= max_x; x++)
				printf("%7d ", counts[x + y * max_dim]);
			printf("\n");
		}
		printf("\n");
	}

	print("Avg / Max quads per tile (empty tiles are ignored):\n");
	print("Single: 1x1 (fitting into group)\nMulti: > 1x1 (not fitting into group)\n");

	for(int gsize = 128; gsize >= 4; gsize /= 2) {
		int2 num_groups = (m_size + int2(gsize - 1)) / gsize;
		int shift = log2(gsize);

		vector<int> multi_counts(num_groups.x * num_groups.y, 0);
		vector<int> single_counts(multi_counts.size(), 0);

		for(int i = 0; i < m_num_quads; i++) {
			IRect rect;
			if(!get_quad(rect, i, shift))
				continue;
			auto &dst = rect.width() == 0 && rect.height() == 0 ? single_counts : multi_counts;
			for(int y = rect.y(); y <= rect.ey(); y++)
				for(int x = rect.x(); x <= rect.ex(); x++)
					dst[x + y * num_groups.x]++;
		}

		int num_not_empty = 0;
		for(int i : intRange(multi_counts))
			if(multi_counts[i] || single_counts[i])
				num_not_empty++;

		int smax = 0, ssum = 0, mmax = 0, msum = 0;
		for(auto &gval : multi_counts) {
			mmax = max(mmax, gval);
			msum += gval;
		}
		for(auto &gval : single_counts) {
			smax = max(smax, gval);
			ssum += gval;
		}
		printf("[%4d]:\n  max: single:%d multi:%d\n", gsize, smax, mmax);
		printf("  avg: single:%d multi:%d \n", int(double(ssum) / num_not_empty),
			   int(double(msum) / num_not_empty));
	}
}

array<uint, 256> LucidRenderer::computeCentroids4x4() {
	array<uint, 256> out;
	for(uint mask = 0; mask < 256; mask++) {
		uint x0123 = 1 * countBits(mask & 0x11) + 3 * countBits(mask & 0x22) +
					 5 * countBits(mask & 0x44) + 7 * countBits(mask & 0x88);
		uint y01 = 1 * countBits(mask & 0xf) + 3 * countBits(mask & 0xf0);
		uint y23 = 5 * countBits(mask & 0xf) + 7 * countBits(mask & 0xf0);
		out[mask] = x0123 | (y01 << 8) | (y23 << 16);
	}
	return out;
}
