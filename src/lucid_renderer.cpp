#include "lucid_renderer.h"

#include "scene.h"
#include "shader_structs.h"
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
	m_bin_size = opts & Opt::bin_size_32 ? 32 : 64;
	m_tile_size = opts & Opt::bin_size_32 ? 8 : 16;
	m_block_size = 4;

	m_blocks_per_bin = square(m_bin_size / m_block_size);
	m_blocks_per_tile = square(m_tile_size / m_block_size);
	m_tiles_per_bin = square(m_bin_size / m_tile_size);

	m_bin_counts = (view_size + int2(m_bin_size - 1)) / m_bin_size;
	m_bin_count = m_bin_counts.x * m_bin_counts.y;
	m_tile_count = m_bin_count * m_tiles_per_bin;

	// TODO: Why adding more on intel causes problems?
	// TODO: properly get number of compute units (use opencl?)
	// TODO: max dispatches should also depend on lsize
	// https://tinyurl.com/o7s9ph3
	m_max_dispatches = gl_info->vendor == GlVendor::intel ? 32 : 128;
	DASSERT(m_max_dispatches <= sizeof(shader::LucidInfo::dispatcher_task_counts) / sizeof(u32));

	m_opts = opts;
	m_size = view_size;

	// TODO: this takes a lot of memory
	// TODO: what should we do when quads won't fit?
	// TODO: better estimate needed; it should depend on bin size
	// TODO: properly handle situations when limits are reached
	uint max_bin_quads = max_quads * 3 / 2;
	uint max_bin_tris = max_quads;

	uint bin_counters_size = LUCID_INFO_SIZE + m_bin_count * 10;
	m_info.emplace(BufferType::shader_storage, bin_counters_size * sizeof(u32));
	m_bin_quads.emplace(BufferType::shader_storage, max_bin_quads * sizeof(u32));
	m_bin_tris.emplace(BufferType::shader_storage, max_bin_tris * sizeof(u32));

	int max_visible_tris = max_visible_quads * 2;
	int uvec4_storage_size = max_visible_tris * 5 + max_visible_quads * 4;
	m_uvec4_storage.emplace(BufferType::shader_storage, uvec4_storage_size * sizeof(int4));
	m_uint_storage.emplace(BufferType::shader_storage, max_visible_tris * sizeof(u32));

	uint scratch_32_size = 64 * 1024 * m_max_dispatches * sizeof(u32);
	uint scratch_64_size = (128 * 1024) * m_max_dispatches * sizeof(u64);
	scratch_64_size = max<uint>(scratch_64_size, max_visible_quads * sizeof(u32));

	// TODO: control size of scratch mem
	m_scratch_32.emplace(BufferType::shader_storage, scratch_32_size);
	m_scratch_64.emplace(BufferType::shader_storage, scratch_64_size);
	m_raster_image.emplace(BufferType::shader_storage,
						   m_bin_count * square(m_bin_size) * sizeof(u32));

	if(m_opts & (Opt::debug_bin_dispatcher | Opt::debug_raster))
		m_errors.emplace(BufferType::shader_storage, 1024 * 1024 * 4, BufferUsage::dynamic_read);

	ShaderDefs defs;
	defs["VIEWPORT_SIZE_X"] = view_size.x;
	defs["VIEWPORT_SIZE_Y"] = view_size.y;
	defs["BIN_COUNT"] = m_bin_count;
	defs["BIN_COUNT_X"] = m_bin_counts.x;
	defs["BIN_COUNT_Y"] = m_bin_counts.y;
	defs["BIN_SIZE"] = m_bin_size;
	defs["TILE_SIZE"] = m_tile_size;
	defs["BLOCK_SIZE"] = m_block_size;
	defs["BIN_SHIFT"] = log2(m_bin_size);
	defs["TILE_SHIFT"] = log2(m_tile_size);
	defs["BLOCK_SHIFT"] = log2(m_block_size);
	defs["XTILES_PER_BIN"] = m_bin_size / m_tile_size;
	defs["TILES_PER_BIN"] = m_tiles_per_bin;
	defs["BLOCKS_PER_TILE"] = m_blocks_per_tile;
	defs["BLOCKS_PER_BIN"] = m_blocks_per_bin;
	defs["MAX_INSTANCE_QUADS"] = max_instance_quads;
	defs["MAX_QUADS"] = max_quads;
	defs["MAX_VISIBLE_QUADS"] = max_visible_quads;
	defs["MAX_VISIBLE_QUADS_SHIFT"] = (int)log2(max_visible_quads);
	defs["MAX_VISIBLE_TRIS"] = max_visible_quads * 2;
	defs["MAX_DISPATCHES"] = m_max_dispatches;

	int bin_dispatcher_lsize = m_bin_size == 64 ? 512 : 1024;
	defs["BIN_DISPATCHER_LSIZE"] = bin_dispatcher_lsize;
	defs["BIN_DISPATCHER_LSHIFT"] = (int)log2(bin_dispatcher_lsize);
	p_quad_setup = EX_PASS(Program::makeCompute("quad_setup", defs));
	p_bin_categorizer = EX_PASS(Program::makeCompute("bin_categorizer", defs));

	if(m_opts & Opt::timers)
		defs["ENABLE_TIMERS"] = 1;
	p_bin_dispatcher = EX_PASS(Program::makeCompute(
		"bin_dispatcher", defs, mask(m_opts & Opt::debug_bin_dispatcher, ProgramOpt::debug)));

	if(m_opts & Opt::additive_blending)
		defs["ADDITIVE_BLENDING"] = 1;
	if(m_opts & Opt::visualize_errors)
		defs["VISUALIZE_ERRORS"] = 1;
	if(m_opts & Opt::alpha_threshold)
		defs["ALPHA_THRESHOLD"] = 1;

	auto raster_opts = mask(m_opts & Opt::debug_raster, ProgramOpt::debug);
	p_raster_low = EX_PASS(Program::makeCompute("raster_low", defs, raster_opts));
	p_raster_medium = EX_PASS(Program::makeCompute("raster_medium", defs, raster_opts));

	mkdirRecursive("temp").ignore();
	if(auto disas = p_quad_setup.getDisassembly())
		saveFile("temp/quad_setup.asm", *disas).ignore();
	if(auto disas = p_bin_dispatcher.getDisassembly())
		saveFile("temp/bin_dispatcher.asm", *disas).ignore();
	if(auto disas = p_raster_low.getDisassembly())
		saveFile("temp/raster_low.asm", *disas).ignore();
	if(auto disas = p_raster_medium.getDisassembly())
		saveFile("temp/raster_medium.asm", *disas).ignore();

	ShaderDefs compose_defs;
	compose_defs["BIN_SIZE"] = m_bin_size;
	compose_defs["BIN_SHIFT"] = log2(m_bin_size);
	p_compose = EX_PASS(Program::make("compose", compose_defs, {"in_pos"}));
	//p_dummy = EX_PASS(Program::makeCompute("dummy", defs));

	vector<u16> indices(m_bin_count * 6);
	DASSERT(m_bin_count * 4 * 4 <= 64 * 1024);
	for(int i = 0; i < m_bin_count; i++) {
		int offsets[6] = {0, 1, 2, 0, 2, 3};
		for(int j = 0; j < 6; j++) {
			int value = offsets[j] + i * 4;
			indices[i * 6 + j] = offsets[j] + i * 4;
		}
	}
	auto ibuffer = GlBuffer::make(BufferType::element_array, indices);
	m_compose_quads =
		GlBuffer::make(BufferType::array, m_bin_count * 4 * sizeof(uint), ImmBufferFlags());

	m_compose_quads_vao = GlVertexArray::make();
	m_compose_quads_vao->set({m_compose_quads}, defaultVertexAttribs<uint>(), ibuffer,
							 IndexType::uint16);

	return {};
}

void LucidRenderer::debugProgram(Program &program, ZStr name) {
	auto source_ranges = program.sourceRanges();
	ShaderDebugInfo records(m_errors, 1024, source_ranges);
	if(records)
		print("% debug records: %\n", name, records);
}

static void dispatchIndirect(int bin_counters_offset) {
	glDispatchComputeIndirect((GLintptr)(bin_counters_offset * sizeof(int)));
}

void LucidRenderer::render(const Context &ctx) {
	PERF_GPU_SCOPE();
	testGlError("LucidRenderer::render init");

	m_info->invalidate();
	m_info->clear(GlFormat::r32i, 0, 0, (LUCID_INFO_SIZE + m_bin_count * 6) * sizeof(u32));

	glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, m_info->id());
	m_view_proj_matrix = ctx.camera.matrix();
	m_frustum_rays = ctx.camera;

	uploadInstances(ctx);
	quadSetup(ctx);
	computeBins(ctx);
	if(false)
		dummyIterateBins(ctx);

	bindRasterCommon(ctx);
	rasterLow(ctx);
	rasterMedium(ctx);
	copyInfo();

	// TODO: is this needed?
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
	glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, 0);

	compose(ctx);
	testGlError("LucidRenderer::render finish");
}

void LucidRenderer::uploadInstances(const Context &ctx) {
	PERF_GPU_SCOPE();

	using InstanceData = shader::InstanceData;
	vector<InstanceData> instances;
	vector<uint> colors;
	vector<float4> uv_rects;
	colors.reserve(16 * 1024);
	instances.reserve(16 * 1024);
	uv_rects.reserve(16 * 1024);

	vector<IColor> mat_colors = transform(
		ctx.materials, [](auto &mat) { return IColor(FColor(mat.diffuse, mat.opacity)); });
	//int max_instance_quads = gl_info->limits[GlLimit::max_compute_work_group_invocations];

	m_num_quads = 0;
	for(auto &dc : ctx.dcs) {
		int num_quads = dc.num_quads;
		if(m_num_quads + num_quads > max_quads)
			num_quads = max_quads - m_num_quads;
		if(num_quads == 0)
			break;

		auto opts = dc.opts;
		u32 color = u32(mat_colors[dc.material_id]);
		if(color != 0xffffffff)
			opts |= DrawCallOpt::has_inst_color;

		InstanceData out;
		out.index_offset = dc.quad_offset * 4;
		out.vertex_offset = 0;
		out.num_quads = num_quads;
		out.flags = (uint(opts.bits) & 0xffff);
		float4 uv_rect(dc.uv_rect.x(), dc.uv_rect.y(), dc.uv_rect.width(), dc.uv_rect.height());
		m_num_quads += dc.num_quads;

		if(num_quads <= max_instance_quads) {
			instances.emplace_back(out);
			uv_rects.emplace_back(uv_rect);
			colors.emplace_back(color);
		} else {
			for(int i = 0; i < num_quads; i += max_instance_quads) {
				out.index_offset = dc.quad_offset * 4 + i * 4;
				out.num_quads = min(max_instance_quads, num_quads - i);
				instances.emplace_back(out);
				uv_rects.emplace_back(uv_rect);
				colors.emplace_back(color);
			}
		}
	}

	m_instances.emplace(BufferType::shader_storage, instances);
	m_instance_colors.emplace(BufferType::shader_storage, colors);
	m_instance_uv_rects.emplace(BufferType::shader_storage, uv_rects);
	m_num_instances = instances.size();
}

void LucidRenderer::quadSetup(const Context &ctx) {
	// TODO: co robić z trójkątami, które są na tyle małe, że wogóle ich nie widać nawet w pełnej rozdziałce?
	// TODO: backface-culling ?

	PERF_GPU_SCOPE();

	m_info->bindIndex(0);
	m_instances->bindIndex(1);

	ctx.quads_ib->bindIndexAs(2, BufferType::shader_storage);
	auto vbuffers = ctx.vao->buffers();
	DASSERT(vbuffers.size() == 4);
	vbuffers[0]->bindIndexAs(3, BufferType::shader_storage);
	if(auto tex_vb = vbuffers[2])
		tex_vb->bindIndexAs(4, BufferType::shader_storage);
	if(auto col_vb = vbuffers[1])
		col_vb->bindIndexAs(5, BufferType::shader_storage);
	if(auto nrm_vb = vbuffers[3])
		nrm_vb->bindIndexAs(6, BufferType::shader_storage);

	m_scratch_64->bindIndex(7);
	m_uvec4_storage->bindIndex(8);
	m_uint_storage->bindIndex(9);

	p_quad_setup["enable_backface_culling"] = ctx.config.backface_culling;
	p_quad_setup["view_proj_matrix"] = m_view_proj_matrix;
	p_quad_setup.setFrustum(ctx.camera);
	p_quad_setup.use();

	int max_dispatches = m_max_dispatches / 2; // TODO: tweak this properly...
	int packet_size = clamp(m_num_instances / max_dispatches, 1, 2);

	p_quad_setup["u_num_instances"] = m_num_instances;
	p_quad_setup["u_packet_size"] = packet_size;
	glDispatchCompute((m_num_instances + packet_size - 1) / packet_size, 1, 1);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_COMMAND_BARRIER_BIT);
}

void LucidRenderer::computeBins(const Context &ctx) {
	PERF_GPU_SCOPE();

	m_info->bindIndex(0);

	m_scratch_64->bindIndex(1);
	m_bin_quads->bindIndex(2);
	m_bin_tris->bindIndex(3);
	m_scratch_32->bindIndex(4);
	m_uvec4_storage->bindIndex(5);

	PERF_CHILD_SCOPE("dispatcher phase");
	p_bin_dispatcher.use();
	p_bin_dispatcher.setFrustum(ctx.camera);

	if(m_opts & Opt::debug_bin_dispatcher)
		shaderDebugUseBuffer(m_errors);
	dispatchIndirect(LUCID_INFO_MEMBER_OFFSET(num_binning_dispatches));
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	if(m_opts & Opt::debug_bin_dispatcher)
		debugProgram(p_bin_dispatcher, "bin_dispatcher");

	PERF_SIBLING_SCOPE("categorizer phase");
	p_bin_categorizer.use();
	m_compose_quads->bindIndexAs(1, BufferType::shader_storage);
	glDispatchCompute(1, 1, 1);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_COMMAND_BARRIER_BIT);
}

void LucidRenderer::dummyIterateBins(const Context &ctx) {
	PERF_GPU_SCOPE();

	m_info->bindIndex(0);
	p_dummy.use();
	glDispatchCompute(512, 1, 1);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void LucidRenderer::bindRasterCommon(const Context &ctx) {
	m_info->bindIndex(0);
	m_bin_quads->bindIndex(1);
	m_bin_tris->bindIndex(2);

	m_scratch_32->bindIndex(3);
	m_scratch_64->bindIndex(4);
	m_instance_colors->bindIndex(5);
	m_instance_uv_rects->bindIndex(6);
	m_uvec4_storage->bindIndex(7);
	m_uint_storage->bindIndex(8);

	m_raster_image->bindIndex(9);
}

void LucidRenderer::bindRaster(Program &program, const Context &ctx) {
	program.use();

	GlTexture::bind({ctx.opaque_tex, ctx.trans_tex});
	//GlTexture::bind({ctx.depth_buffer, ctx.shadows.map});

	ctx.lighting.setUniforms(program.glProgram());
	program.setFrustum(ctx.camera);
	program.setViewport(ctx.camera, m_size);
	program.setShadows(ctx.shadows.matrix, ctx.shadows.enable);
	program["background_color"] = u32(ctx.config.background_color);
	ctx.lighting.setUniforms(program.glProgram());
}

void LucidRenderer::rasterLow(const Context &ctx) {
	PERF_GPU_SCOPE();
	bindRaster(p_raster_low, ctx);
	if(m_opts & Opt::debug_raster)
		shaderDebugUseBuffer(m_errors);
	dispatchIndirect(LUCID_INFO_MEMBER_OFFSET(bin_level_dispatches[BIN_LEVEL_LOW]));
	if(m_opts & Opt::debug_raster)
		debugProgram(p_raster_low, "raster_low");
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_COMMAND_BARRIER_BIT);
}

void LucidRenderer::rasterMedium(const Context &ctx) {
	PERF_GPU_SCOPE();
	bindRaster(p_raster_medium, ctx);
	if(m_opts & Opt::debug_raster)
		shaderDebugUseBuffer(m_errors);
	dispatchIndirect(LUCID_INFO_MEMBER_OFFSET(bin_level_dispatches[BIN_LEVEL_MEDIUM]));
	if(m_opts & Opt::debug_raster)
		debugProgram(p_raster_medium, "raster_medium");
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void LucidRenderer::compose(const Context &ctx) {
	PERF_GPU_SCOPE();

	DASSERT(!ctx.out_fbo || ctx.out_fbo->size() == m_size);
	glDrawBuffer(GL_BACK);
	setupView(IRect(m_size), ctx.out_fbo);
	glDisable(GL_BLEND);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	glDepthMask(0);
	p_compose.setFullscreenRect();
	p_compose.use();
	p_compose["bin_counts"] = m_bin_counts;
	p_compose["screen_scale"] = float2(1.0) / float2(m_size);
	m_raster_image->bindIndex(0);
	m_compose_quads_vao->draw(PrimitiveType::triangles, m_bin_counts.x * m_bin_counts.y * 6);
}

void LucidRenderer::copyInfo() {
	//static int iter = 0;
	//if(iter++ % 30 != 0)
	//	return;

	auto last = m_old_info.back();
	for(int i = m_old_info.size() - 1; i > 0; i--)
		m_old_info[i] = m_old_info[i - 1];
	m_old_info[0] = last;

	if(!m_old_info[0]) {
		m_old_info[0].emplace(BufferType::copy_read, m_info->size());
	}

	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	m_info->copyTo(m_old_info[0], 0, 0, m_old_info[0]->size());
}

vector<StatsGroup> LucidRenderer::getStats() const {
	vector<StatsGroup> out;

	if(!m_old_info.back())
		return out;

	auto bin_counters = m_old_info.back()->download<u32>();
	shader::LucidInfo bins;
	memcpy(&bins, bin_counters.data(), sizeof(bins));
	bin_counters.erase(bin_counters.begin(), bin_counters.begin() + LUCID_INFO_SIZE);

	CSpan<uint> bin_quad_counts = cspan(bin_counters.data() + m_bin_count * 0, m_bin_count);
	CSpan<uint> bin_quad_offsets = cspan(bin_counters.data() + m_bin_count * 1, m_bin_count);
	CSpan<uint> bin_quad_offsets_temp = cspan(bin_counters.data() + m_bin_count * 2, m_bin_count);

	CSpan<uint> bin_tri_counts = cspan(bin_counters.data() + m_bin_count * 3, m_bin_count);
	CSpan<uint> bin_tri_offsets = cspan(bin_counters.data() + m_bin_count * 4, m_bin_count);
	CSpan<uint> bin_tri_offsets_temp = cspan(bin_counters.data() + m_bin_count * 5, m_bin_count);

	// Checking bins quad offsets
	for(uint i = 0; i < m_bin_count; i++) {
		int cur_value = bin_quad_counts[i];
		int cur_offset = bin_quad_offsets[i];
		int cur_offset_temp = bin_quad_offsets_temp[i];

		if(cur_offset_temp != cur_offset + cur_value)
			print("Invalid bin quad offset [%]: % != % (offset:% + count:%)\n", i, cur_offset_temp,
				  cur_offset + cur_value, cur_offset, cur_value);
	}

	// Checking bins tris offsets
	for(uint i = 0; i < m_bin_count; i++) {
		int cur_value = bin_tri_counts[i];
		int cur_offset = bin_tri_offsets[i];
		int cur_offset_temp = bin_tri_offsets_temp[i];

		if(cur_offset_temp != cur_offset + cur_value)
			print("Invalid bin tri offset [%]: % != % (offset:% + count:%)\n", i, cur_offset_temp,
				  cur_offset + cur_value, cur_offset, cur_value);
	}

	int num_pixels = m_size.x * m_size.y;
	int num_tiles =
		((m_size.x + m_tile_size - 1) / m_tile_size) * ((m_size.y + m_tile_size - 1) / m_tile_size);
	int num_blocks = ((m_size.x + m_block_size - 1) / m_block_size) *
					 ((m_size.y + m_block_size - 1) / m_block_size);

	int max_quads_per_bin = max(bin_quad_counts);
	int num_bin_quads = accumulate(bin_quad_counts);
	int num_bin_tris = accumulate(bin_tri_counts);

	// When raster algorithm cannot process some bin because there is too many tris
	// per block or too many samples, then such bin will be promoted to next level.
	// This will result in some bins being processed twice
	int num_promoted_bins = accumulate(bins.bin_level_counts) - m_bin_count;

	int num_visible_total = bins.num_visible_quads[0] + bins.num_visible_quads[1];
	auto visible_info = stdFormat("%d (%.2f %%)", num_visible_total,
								  double(num_visible_total) / bins.num_input_quads * 100);
	auto visible_details =
		stdFormat("%d small; %d big", bins.num_visible_quads[0], bins.num_visible_quads[1]);

	bins.num_rejected_quads[0] +=
		bins.num_rejected_quads[1] + bins.num_rejected_quads[2] + bins.num_rejected_quads[3];
	auto rejected_info = stdFormat("%d (%.2f %%)", bins.num_rejected_quads[0],
								   double(bins.num_rejected_quads[0]) / bins.num_input_quads * 100);
	auto rejection_details =
		format("backface: %\nfrustum: %\nbetween-samples: %", bins.num_rejected_quads[1],
			   bins.num_rejected_quads[2], bins.num_rejected_quads[3]);

	vector<StatsRow> timers;
	Str timer_names[] = {"generate rows",  "generate blocks",  "load samples", "shade samples",
						 "reduce samples", "shade and reduce", "finish reduce"};
	static_assert(arraySize(timer_names) < arraySize(bins.timers));

	u64 total_time = 0;
	for(int i : intRange(timer_names))
		total_time += bins.timers[i];
	if(total_time > 0)
		for(int i : intRange(timer_names)) {
			auto value = bins.timers[i];
			if(value == 0)
				continue;
			timers.emplace_back(timer_names[i],
								stdFormat("%.2f %%", double(value) / total_time * 100));
		}

	auto format_percentage = [](int value, int total) {
		return stdFormat("%d (%.0f %%)", value, value * 100.0 / total);
	};

	vector<StatsRow> bin_level_rows = {
		{"empty bins", format_percentage(bins.bin_level_counts[0], m_bin_count)},
		{"micro level bins",
		 format_percentage(bins.bin_level_counts[BIN_LEVEL_MICRO], m_bin_count)},
		{"low level bins", format_percentage(bins.bin_level_counts[BIN_LEVEL_LOW], m_bin_count)},
		{"medium level bins",
		 format_percentage(bins.bin_level_counts[BIN_LEVEL_MEDIUM], m_bin_count)},
		{"high level bins", format_percentage(bins.bin_level_counts[BIN_LEVEL_HIGH], m_bin_count)},
		{"promoted bins", format_percentage(num_promoted_bins, m_bin_count)},
	};

	// TODO: fix it
	string bin_dispatcher_info =
		toString(span(bins.dispatcher_task_counts, bins.a_bin_dispatcher_work_groups[1]));
	if(m_opts & Opt::timers) {
		auto timers = span(bins.dispatcher_timers, bins.a_bin_dispatcher_work_groups[1]);
		float sum = accumulate(timers);
		float average = sum / timers.size();

		float var = 0.0;
		for(auto value : timers)
			var += square(value - average);
		var /= timers.size() * square(average);
		bin_dispatcher_info += stdFormat("\nComputation variance: %.4f", var);
	}

	vector<StatsRow> basic_rows = {
		{"input instances", toString(m_num_instances)},
		{"bin dispatcher work-groups", toString(bins.a_bin_dispatcher_work_groups[1]),
		 bin_dispatcher_info},
		{"input quads", toString(bins.num_input_quads)},
		{"visible quads", visible_info, visible_details},
		{"rejected quads", rejected_info, rejection_details},
		{"bin quads", toString(num_bin_quads), "Total per-bin quads"},
		{"bin tris", toString(num_bin_tris), "Total per-bin tris"},
		{"max quads / bin", toString(max_quads_per_bin)},
	};

	// TODO: add better stats once rasterizez is working on all levels

	if(timers)
		out.emplace_back(move(timers), "", 130);
	out.emplace_back(move(bin_level_rows), "Bins categorized by quad density levels:", 130);
	out.emplace_back(move(basic_rows), "", 130);
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

void LucidRenderer::printTriangleSizeHistogram() const {
	HashMap<int, int> tri_height_histogram;

	/*auto quad_aabbs = m_quad_aabbs->download<u32>(m_num_quads);
	auto tri_aabbs = m_tri_aabbs->download<int4>(m_num_quads);
	for(int i : intRange(quad_aabbs)) {
		if(quad_aabbs[i] == ~0u)
			continue;
		auto enc = tri_aabbs[i];
		int2 min0(u32(enc[0]) & 0xffff, u32(enc[0]) >> 16);
		int2 max0(u32(enc[1]) & 0xffff, u32(enc[1]) >> 16);
		int2 min1(u32(enc[2]) & 0xffff, u32(enc[2]) >> 16);
		int2 max1(u32(enc[3]) & 0xffff, u32(enc[3]) >> 16);
		int2 sizes[2] = {max0 - min0 + int2(1, 1), max1 - min1 + int2(1, 1)};
		for(auto &size : sizes)
			if(size.y > 0)
				tri_height_histogram[size.y]++;
	}

	vector<Pair<int>> pairs;
	int total = 0;
	for(auto &pair : tri_height_histogram) {
		pairs.emplace_back(pair.value, pair.key);
		total += pair.value;
	}
	std::sort(begin(pairs), end(pairs), std::greater<Pair<int>>());

	printf("Triangle pixel-height histogram:\n");
	for(auto &pair : pairs) {
		printf("%4d: %.2f%% (%d)\n", pair.second, float(pair.first) * 100.0 / total, pair.first);
	}*/
}

void LucidRenderer::printHistograms() const {
	vector<IRect> quads;
	/*{
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
	}*/

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
