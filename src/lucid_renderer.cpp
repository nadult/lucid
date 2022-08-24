#include "lucid_renderer.h"

#include "scene.h"
#include "shader_structs.h"
#include "shading.h"
#include <fwk/gfx/camera.h>
#include <fwk/gfx/image.h>
#include <fwk/gfx/shader_compiler.h>
#include <fwk/gfx/shader_debug.h>
#include <fwk/hash_set.h>
#include <fwk/io/file_system.h>
#include <fwk/vulkan/vulkan_buffer.h>
#include <fwk/vulkan/vulkan_device.h>
#include <fwk/vulkan/vulkan_pipeline.h>
#include <fwk/vulkan/vulkan_swap_chain.h>

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

LucidRenderer::LucidRenderer() = default;
FWK_MOVABLE_CLASS_IMPL(LucidRenderer)

void LucidRenderer::addShaderDefs(ShaderCompiler &compiler) {
	vector<Pair<string>> vsh_macros = {{"VERTEX_SHADER", "1"}};
	vector<Pair<string>> fsh_macros = {{"FRAGMENT_SHADER", "1"}};

	compiler.add({"compose_vert", VShaderStage::vertex, "compose.glsl", vsh_macros});
	compiler.add({"compose_frag", VShaderStage::fragment, "compose.glsl", fsh_macros});

	compiler.add({"quad_setup", VShaderStage::compute, "quad_setup.glsl"});
	compiler.add({"bin_dispatcher", VShaderStage::compute, "bin_dispatcher.glsl"});
	compiler.add({"bin_categorizer", VShaderStage::compute, "bin_categorizer.glsl"});
	compiler.add({"raster_low", VShaderStage::compute, "raster_low.glsl"});
	compiler.add({"raster_high", VShaderStage::compute, "raster_high.glsl"});
}

static Ex<PVPipeline> makeComputePipeline(VulkanDevice &device, ShaderCompiler &compiler,
										  const shader::SpecializationConstants &consts,
										  ZStr def_name) {
	VComputePipelineSetup setup;
	setup.compute_module = EX_PASS(compiler.createShaderModule(device.ref(), def_name));
	setup.spec_constants.emplace_back(consts, 0u);
	return VulkanPipeline::create(device.ref(), setup);
}

Ex<void> LucidRenderer::exConstruct(VulkanDevice &device, ShaderCompiler &compiler,
									VColorAttachment color_att, Opts opts, int2 view_size) {
	m_bin_size = opts & Opt::bin_size_64 ? 64 : 32;
	m_block_size = 8;

	m_bin_counts = (view_size + int2(m_bin_size - 1)) / m_bin_size;
	m_bin_count = m_bin_counts.x * m_bin_counts.y;

	// TODO: Why adding more on intel causes problems?
	// TODO: properly get number of compute units (use opencl?)
	// TODO: max dispatches should also depend on lsize
	// https://tinyurl.com/o7s9ph3
	m_max_dispatches = 128; //device. gl_info->vendor == GlVendor::intel ? 32 : 128;
	DASSERT(m_max_dispatches <= sizeof(shader::LucidInfo::dispatcher_task_counts) / sizeof(u32));

	m_opts = opts;
	m_size = view_size;

	shader::SpecializationConstants consts;
	consts.VIEWPORT_SIZE_X = view_size.x;
	consts.VIEWPORT_SIZE_Y = view_size.y;
	consts.BIN_COUNT = m_bin_count;
	consts.BIN_COUNT_X = m_bin_counts.x;
	consts.BIN_COUNT_Y = m_bin_counts.y;
	consts.BIN_SIZE = m_bin_size;
	consts.BIN_SHIFT = log2(m_bin_size);
	consts.MAX_QUADS = max_quads;
	consts.MAX_VISIBLE_QUADS = max_visible_quads;
	consts.MAX_VISIBLE_QUADS_SHIFT = log2(max_visible_quads);
	consts.MAX_VISIBLE_TRIS = max_visible_quads * 2;
	consts.MAX_DISPATCHES = m_max_dispatches;
	consts.RENDER_OPTIONS = m_opts.bits;

	int bin_dispatcher_lsize = m_bin_size == 64 ? 512 : 1024;
	consts.BIN_DISPATCHER_LSIZE = bin_dispatcher_lsize;
	consts.BIN_DISPATCHER_LSHIFT = log2(bin_dispatcher_lsize);
	consts.BIN_DISPATCHER_XBIN_STEP = log2(bin_dispatcher_lsize / m_bin_counts.x);
	consts.BIN_DISPATCHER_YBIN_STEP = log2(bin_dispatcher_lsize / m_bin_counts.y);
	consts.BIN_CATEGORIZER_LSIZE = m_bin_size == 64 ? 128 : 512;

	// TODO: this takes a lot of memory
	// TODO: what should we do when quads won't fit?
	// TODO: better estimate needed; it should depend on bin size
	// TODO: properly handle situations when limits are reached
	uint max_bin_quads = max_quads * 3 / 2;
	uint max_bin_tris = max_quads;

	auto usage = VBufferUsage::storage_buffer;
	auto usage_copyable = VBufferUsage::storage_buffer | VBufferUsage::transfer_src;
	m_bin_quads = EX_PASS(VulkanBuffer::create<u32>(device, max_bin_quads, usage));
	m_bin_tris = EX_PASS(VulkanBuffer::create<u32>(device, max_bin_tris, usage));

	int max_visible_tris = max_visible_quads * 2;
	int uvec4_storage_size = max_visible_tris * 5 + max_visible_quads * 4; // 480MB ...
	m_uvec4_storage = EX_PASS(VulkanBuffer::create<int4>(device, uvec4_storage_size, usage));
	m_uint_storage = EX_PASS(VulkanBuffer::create<u32>(device, max_visible_tris, usage));

	uint scratch_32_size = 64 * 1024 * m_max_dispatches * sizeof(u32);
	uint scratch_64_size = (128 * 1024) * m_max_dispatches * sizeof(u64);
	scratch_64_size = max<uint>(scratch_64_size, max_visible_quads * sizeof(u32));

	// TODO: control size of scratch mem
	m_scratch_32 = EX_PASS(VulkanBuffer::create(device, scratch_32_size, usage));
	m_scratch_64 = EX_PASS(VulkanBuffer::create(device, scratch_64_size, usage));
	m_raster_image =
		EX_PASS(VulkanBuffer::create<u32>(device, m_bin_count * square(m_bin_size), usage));
	m_compose_quads = EX_PASS(VulkanBuffer::create<u32>(
		device, m_bin_count * 4,
		VBufferUsage::vertex_buffer | VBufferUsage::storage_buffer | VBufferUsage::transfer_dst));
	vector<u16> indices(m_bin_count * 6);
	DASSERT(m_bin_count * 4 * 4 <= 64 * 1024);
	for(int i = 0; i < m_bin_count; i++) {
		int offsets[6] = {0, 1, 2, 0, 2, 3};
		for(int j = 0; j < 6; j++) {
			int value = offsets[j] + i * 4;
			indices[i * 6 + j] = offsets[j] + i * 4;
		}
	}
	m_compose_ibuffer = EX_PASS(VulkanBuffer::createAndUpload(
		device, indices, VBufferUsage::index_buffer | VBufferUsage::transfer_dst));

	if(m_opts & (Opt::debug_bin_dispatcher | Opt::debug_raster))
		m_errors = EX_PASS(VulkanBuffer::create<u32>(device, 1024 * 1024, usage_copyable));

	p_bin_categorizer = EX_PASS(makeComputePipeline(device, compiler, consts, "bin_categorizer"));

	p_quad_setup = EX_PASS(makeComputePipeline(device, compiler, consts, "quad_setup"));
	p_bin_dispatcher = EX_PASS(makeComputePipeline(device, compiler, consts, "bin_dispatcher"));

	p_raster_low = EX_PASS(makeComputePipeline(device, compiler, consts, "raster_low"));
	//p_raster_high = EX_PASS(makeComputePipeline(device, compiler, consts, "raster_high"));

	// TODO: disassemble spirv
	/*mkdirRecursive("temp").ignore();
	if(auto disas = p_quad_setup.getDisassembly())
		saveFile("temp/quad_setup.asm", *disas).ignore();
	if(auto disas = p_bin_dispatcher.getDisassembly())
		saveFile("temp/bin_dispatcher.asm", *disas).ignore();
	if(auto disas = p_raster_low.getDisassembly())
		saveFile("temp/raster_low.asm", *disas).ignore();
	if(auto disas = p_raster_high.getDisassembly())
		saveFile("temp/raster_high.asm", *disas).ignore();*/

	color_att.sync = VColorSyncStd::clear;
	m_render_pass = device.getRenderPass({color_att});

	VPipelineSetup compose_setup;
	auto device_ref = device.ref();
	compose_setup.shader_modules = {
		{EX_PASS(compiler.createShaderModule(device_ref, "compose_vert")),
		 EX_PASS(compiler.createShaderModule(device_ref, "compose_frag"))}};
	compose_setup.render_pass = m_render_pass;
	compose_setup.vertex_attribs = {vertexAttrib<uint>(0, 0)};
	compose_setup.vertex_bindings = {vertexBinding<uint>(0)};
	compose_setup.spec_constants.emplace_back(consts, 0u);
	p_compose = EX_PASS(VulkanPipeline::create(device_ref, compose_setup));

	return {};
}

// TODO
/*void LucidRenderer::debugProgram(Program &program, ZStr name) {
	auto source_ranges = program.sourceRanges();
	ShaderDebugInfo records(m_errors, 1024, source_ranges);
	if(records)
		print("% debug records: %\n", name, records);
}*/

void LucidRenderer::render(const Context &ctx) {
	auto &cmds = ctx.device.cmdQueue();
	PERF_GPU_SCOPE(cmds);

	setupInputData(ctx).check();

	uploadInstances(ctx).check();
	quadSetup(ctx);
	computeBins(ctx);

	//bindRasterCommon(ctx);
	rasterLow(ctx);
	//rasterHigh(ctx);
	downloadInfo(ctx, 8).check();

	compose(ctx);
}

Ex<> LucidRenderer::uploadInstances(const Context &ctx) {
	auto &cmds = ctx.device.cmdQueue();
	PERF_SCOPE();

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

	auto usage = VBufferUsage::storage_buffer | VBufferUsage::transfer_dst;
	auto mem_usage = VMemoryUsage::frame;
	m_instances = EX_PASS(VulkanBuffer::createAndUpload(ctx.device, instances, usage, mem_usage));
	m_instance_colors =
		EX_PASS(VulkanBuffer::createAndUpload(ctx.device, colors, usage, mem_usage));
	m_instance_uv_rects =
		EX_PASS(VulkanBuffer::createAndUpload(ctx.device, uv_rects, usage, mem_usage));
	m_num_instances = instances.size();
	int max_dispatches = m_max_dispatches / 2; // TODO: tweak this properly...
	m_instance_packet_size = clamp(m_num_instances / max_dispatches, 1, 2);

	return {};
}

Ex<> LucidRenderer::setupInputData(const Context &ctx) {
	auto &cmds = ctx.device.cmdQueue();
	uint bin_counters_size = LUCID_INFO_SIZE + m_bin_count * 10;
	auto info_usage = VBufferUsage::storage_buffer | VBufferUsage::transfer_src |
					  VBufferUsage::transfer_dst | VBufferUsage::indirect_buffer;
	auto config_usage = VBufferUsage::uniform_buffer | VBufferUsage::transfer_dst;
	auto mem_usage = VMemoryUsage::frame;
	m_info =
		EX_PASS(VulkanBuffer::create<u32>(ctx.device, bin_counters_size, info_usage, mem_usage));
	cmds.fill(m_info.subSpan(0, LUCID_INFO_SIZE + m_bin_count * 6), 0);

	struct shader::LucidConfig config;
	config.frustum = FrustumInfo(ctx.camera);
	config.view_proj_matrix = ctx.camera.matrix();
	config.lighting = ctx.lighting;
	config.background_color = (float4)FColor(ctx.config.background_color);
	config.num_instances = m_num_instances;
	config.enable_backface_culling = ctx.config.backface_culling;
	config.instance_packet_size = m_instance_packet_size;
	m_config = EX_PASS(
		VulkanBuffer::createAndUpload(ctx.device, cspan(&config, 1), config_usage, mem_usage));

	return {};
}

void LucidRenderer::quadSetup(const Context &ctx) {
	// TODO: co robić z trójkątami, które są na tyle małe, że wogóle ich nie widać nawet w pełnej rozdziałce?
	// TODO: backface-culling ?

	auto &cmds = ctx.device.cmdQueue();

	cmds.barrier(VPipeStage::bottom, VPipeStage::top, VAccess::memory_write, VAccess::memory_read);
	PERF_GPU_SCOPE(cmds);
	cmds.bind(p_quad_setup);

	// TODO: descriptor set may be optimized out, what should we do in such a case?
	auto ds = cmds.bindDS(0);
	ds.set(0, m_info);
	ds.set(1, VDescriptorType::uniform_buffer, m_config);
	ds = cmds.bindDS(1);
	ds.set(0, m_instances, ctx.quads_ib, ctx.verts.pos, ctx.verts.tex, ctx.verts.col, ctx.verts.nrm,
		   m_scratch_64, m_uvec4_storage, m_uint_storage);

	int num_workgroups = (m_num_instances + m_instance_packet_size - 1) / m_instance_packet_size;
	cmds.dispatchCompute({num_workgroups, 1, 1});
}

void LucidRenderer::computeBins(const Context &ctx) {
	auto &cmds = ctx.device.cmdQueue();
	PERF_GPU_SCOPE(cmds);

	cmds.barrier(VPipeStage::all_graphics | VPipeStage::compute_shader,
				 VPipeStage::draw_indirect | VPipeStage::compute_shader,
				 VAccess::memory_write | VAccess::shader_write,
				 VAccess::memory_read | VAccess::shader_read | VAccess::indirect_command_read);
	cmds.bind(p_bin_dispatcher);
	auto ds = cmds.bindDS(0);
	ds.set(0, m_info);
	ds.set(1, VDescriptorType::uniform_buffer, m_config);
	ds = cmds.bindDS(1);
	ds.set(0, m_scratch_64, m_bin_quads, m_bin_tris, m_scratch_32, m_uvec4_storage);

	PERF_CHILD_SCOPE("dispatcher phase");

	cmds.dispatchComputeIndirect(m_info, LUCID_INFO_MEMBER_OFFSET(num_binning_dispatches));
	//if(m_opts & Opt::debug_bin_dispatcher)
	//	debugProgram(p_bin_dispatcher, "bin_dispatcher");

	PERF_SIBLING_SCOPE("categorizer phase");
	cmds.bind(p_bin_categorizer);
	ds = cmds.bindDS(1);
	ds.set(0, m_compose_quads);
	cmds.dispatchCompute({1, 1, 1});
}

/*void LucidRenderer::bindRasterCommon(const Context &ctx) {
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
	program["background_color"] = u32(ctx.config.background_color);
	ctx.lighting.setUniforms(program.glProgram());
}*/

void LucidRenderer::rasterLow(const Context &ctx) {
	auto &cmds = ctx.device.cmdQueue();
	PERF_GPU_SCOPE(cmds);

	/*	bindRaster(p_raster_low, ctx);
	if(m_opts & Opt::debug_raster)
		shaderDebugUseBuffer(m_errors);
	dispatchIndirect(LUCID_INFO_MEMBER_OFFSET(bin_level_dispatches[BIN_LEVEL_LOW]));
	if(m_opts & Opt::debug_raster)
		debugProgram(p_raster_low, "raster_low");
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_COMMAND_BARRIER_BIT);*/
}

/*void LucidRenderer::rasterHigh(const Context &ctx) {
	auto &cmds = ctx.device.cmdQueue();
	PERF_GPU_SCOPE(cmds);

	bindRaster(p_raster_high, ctx);
	if(m_opts & Opt::debug_raster)
		shaderDebugUseBuffer(m_errors);
	dispatchIndirect(LUCID_INFO_MEMBER_OFFSET(bin_level_dispatches[BIN_LEVEL_HIGH]));
	if(m_opts & Opt::debug_raster)
		debugProgram(p_raster_high, "raster_high");
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}*/

void LucidRenderer::compose(const Context &ctx) {
	auto &cmds = ctx.device.cmdQueue();
	PERF_GPU_SCOPE(cmds);

	auto swap_chain = ctx.device.swapChain();
	auto framebuffer = ctx.device.getFramebuffer({swap_chain->acquiredImage()});
	cmds.beginRenderPass(framebuffer, m_render_pass, none, {FColor(0.0, 0.0, 0.2)});

	/*DASSERT(!ctx.out_fbo || ctx.out_fbo->size() == m_size);
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
	m_compose_quads_vao->draw(PrimitiveType::triangles, m_bin_counts.x * m_bin_counts.y * 6);*/

	cmds.endRenderPass();
}

Ex<> LucidRenderer::downloadInfo(const Context &ctx, int num_skip_frames) {
	auto &cmds = ctx.device.cmdQueue();
	while(m_info_downloads) {
		auto id = m_info_downloads.front();
		auto data = cmds.retrieve<u32>(id);
		if(!data)
			break;
		m_last_info = move(data);
		m_info_downloads.erase(m_info_downloads.begin());
	}

	static int frame_counter = 0;
	if(num_skip_frames && frame_counter++ % (num_skip_frames + 1) != 0)
		return {};
	m_info_downloads.emplace_back(EX_PASS(cmds.download(m_info)));
	return {};
}

static vector<StatsRow> processTimers(CSpan<uint> timers, CSpan<Str> names) {
	vector<StatsRow> out;
	DASSERT(names.size() <= timers.size());
	u64 total = 0;
	for(int i : intRange(names))
		total += timers[i];
	if(total > 0)
		for(int i : intRange(names)) {
			auto value = timers[i];
			if(value == 0)
				continue;
			out.emplace_back(names[i], stdFormat("%.2f %%", double(value) / total * 100));
		}
	return out;
}

static string formatLarge(i64 value) {
	int v0 = value / 1e9;
	value -= i64(v0) * 1e9;
	int v1 = value / 1e6;
	value -= v1 * 1e6;
	int v2 = value / 1e3;
	value -= v2 * 1e3;
	TextFormatter fmt;
	if(v0)
		fmt.stdFormat("%d,%03d,%03d,%03d", v0, v1, v2, int(value));
	else if(v1)
		fmt.stdFormat("%d,%03d,%03d", v1, v2, int(value));
	else if(v2)
		fmt.stdFormat("%d,%03d", v2, int(value));
	else
		fmt << value;
	return fmt.text();
};

vector<StatsGroup> LucidRenderer::getStats() const {
	PERF_SCOPE();

	vector<StatsGroup> out;

	if(!m_last_info)
		return out;

	auto bin_counters = m_last_info;
	shader::LucidInfo info;
	memcpy(&info, bin_counters.data(), sizeof(info));
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
	int num_blocks = ((m_size.x + m_block_size - 1) / m_block_size) *
					 ((m_size.y + m_block_size - 1) / m_block_size);

	int max_quads_per_bin = max(bin_quad_counts);
	int max_tris_per_bin = max(bin_tri_counts);
	int num_bin_quads = accumulate(bin_quad_counts);
	int num_bin_tris = accumulate(bin_tri_counts);

	// When raster algorithm cannot process some bin because there is too many tris
	// per block or too many samples, then such bin will be promoted to next level.
	// This will result in some bins being processed twice
	int num_promoted_bins = accumulate(info.bin_level_counts) - m_bin_count;

	int num_visible_total = info.num_visible_quads[0] + info.num_visible_quads[1];
	auto visible_info =
		formatLarge(num_visible_total) +
		stdFormat(" (%.2f %%)", double(num_visible_total) / info.num_input_quads * 100);

	auto visible_details = format("% small; % large", formatLarge(info.num_visible_quads[0]),
								  formatLarge(info.num_visible_quads[1]));

	info.num_rejected_quads[0] +=
		info.num_rejected_quads[1] + info.num_rejected_quads[2] + info.num_rejected_quads[3];
	auto rejected_info =
		formatLarge(info.num_rejected_quads[0]) +
		stdFormat(" (%.2f %%)", double(info.num_rejected_quads[0]) / info.num_input_quads * 100);
	auto rejection_details = format(
		"backface: %\nfrustum: %\nbetween-samples: %", formatLarge(info.num_rejected_quads[1]),
		formatLarge(info.num_rejected_quads[2]), formatLarge(info.num_rejected_quads[3]));

	auto setup_timers = processTimers(info.setup_timers, {"init & finish", "process input quads",
														  "store tri data", "store quad data"});
	auto bin_dispatcher_timers =
		processTimers(info.bin_dispatcher_timers,
					  {"estimate small quads", "compute small quad offsets", "dispatch small quads",
					   "estimate large tris", "compute large tris offsets", "dispatch large tris"});
	auto raster_timers =
		processTimers(info.raster_timers, {"generate rows", "generate blocks", "load samples",
										   "shade and reduce", "finish reduce"});

	auto format_percentage = [](int value, int total) {
		return stdFormat("%d (%.0f %%)", value, value * 100.0 / total);
	};

	vector<StatsRow> bin_level_rows = {
		{"empty bins", format_percentage(info.bin_level_counts[0], m_bin_count)},
		{"micro level bins",
		 format_percentage(info.bin_level_counts[BIN_LEVEL_MICRO], m_bin_count)},
		{"low level bins", format_percentage(info.bin_level_counts[BIN_LEVEL_LOW], m_bin_count)},
		{"high level bins", format_percentage(info.bin_level_counts[BIN_LEVEL_HIGH], m_bin_count)},
		{"promoted bins", format_percentage(num_promoted_bins, m_bin_count)},
	};

	// TODO: fix it
	int num_bin_dispatcher_work_groups = max(info.a_bin_dispatcher_work_groups);
	string bin_dispatcher_info =
		toString(span(info.dispatcher_task_counts, num_bin_dispatcher_work_groups));

	auto fragment_info = stdFormat("%.3f avg fragments / pixel\n%.3f avg fragments / hblock",
								   double(info.num_fragments) / (m_size.x * m_size.y),
								   double(info.num_fragments) / info.num_half_blocks);

	vector<StatsRow> basic_rows = {
		{"input instances", formatLarge(m_num_instances)},
		{"bin dispatcher work-groups", toString(num_bin_dispatcher_work_groups),
		 bin_dispatcher_info},
		{"input quads", formatLarge(info.num_input_quads)},
		{"visible quads", visible_info, visible_details},
		{"rejected quads", rejected_info, rejection_details},
		{"bin quads", formatLarge(num_bin_quads), "Total per-bin quads"},
		{"bin tris", formatLarge(num_bin_tris), "Total per-bin tris"},
		{"max small quads / bin", formatLarge(max_quads_per_bin)},
		{"max large tris / bin", formatLarge(max_tris_per_bin)},
		{"half-blocks", formatLarge(info.num_half_blocks)},
		{"fragments", formatLarge(info.num_fragments), fragment_info},
	};

	// TODO: add better stats once rasterizez is working on all levels

	if(setup_timers)
		out.emplace_back(move(setup_timers), "quad_setup timers", 130);
	if(bin_dispatcher_timers)
		out.emplace_back(move(bin_dispatcher_timers), "bin_dispatcher timers", 130);
	if(raster_timers)
		out.emplace_back(move(raster_timers), "raster_low & raster_high timers", 130);

	out.emplace_back(move(bin_level_rows), "Bins categorized by quad density levels:", 130);
	out.emplace_back(move(basic_rows), "", 130);

	return out;
}
