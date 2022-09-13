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
#include <fwk/vulkan/vulkan_instance.h>

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

vector<Pair<string>> sharedShaderMacros(VulkanDevice &device) {
	auto warp_size = device.physInfo().subgroup_props.subgroupSize;
	return {{"WARP_SIZE", toString(warp_size)}, {"WARP_SHIFT", toString(log2(warp_size))}};
}

void LucidRenderer::addShaderDefs(VulkanDevice &device, ShaderCompiler &compiler) {
	vector<Pair<string>> vsh_macros = {{"VERTEX_SHADER", "1"}};
	vector<Pair<string>> fsh_macros = {{"FRAGMENT_SHADER", "1"}};
	vector<Pair<string>> debug_macros = {{"DEBUG_ENABLED", ""}};
	vector<Pair<string>> base_macros = sharedShaderMacros(device);
	insertBack(vsh_macros, base_macros);
	insertBack(fsh_macros, base_macros);
	insertBack(debug_macros, base_macros);

	compiler.add({"compose_vert", VShaderStage::vertex, "compose.glsl", vsh_macros});
	compiler.add({"compose_frag", VShaderStage::fragment, "compose.glsl", fsh_macros});

	auto add_debugable = [&](ZStr name, ZStr file_name) {
		compiler.add({name, VShaderStage::compute, file_name, base_macros});
		auto dbg_name = format("%_debug", name);
		compiler.add({dbg_name, VShaderStage::compute, file_name, debug_macros});
	};

	add_debugable("quad_setup", "quad_setup.glsl");
	add_debugable("bin_dispatcher", "bin_dispatcher.glsl");
	compiler.add({"bin_categorizer", VShaderStage::compute, "bin_categorizer.glsl", base_macros});
	add_debugable("raster_low", "raster_low.glsl");
	add_debugable("raster_high", "raster_high.glsl");
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
	consts.MAX_VISIBLE_QUADS = max_visible_quads;
	consts.MAX_VISIBLE_QUADS_SHIFT = log2(max_visible_quads);
	consts.MAX_VISIBLE_TRIS = max_visible_quads * 2;
	consts.MAX_DISPATCHES = m_max_dispatches;
	consts.RENDER_OPTIONS = m_opts.bits;

	int bin_dispatcher_lsize = m_bin_size == 64 ? 512 : 1024;
	consts.BIN_DISPATCHER_LSIZE = bin_dispatcher_lsize;
	consts.BIN_DISPATCHER_LSHIFT = log2(bin_dispatcher_lsize);
	consts.BIN_CATEGORIZER_LSIZE = m_bin_size == 64 ? 128 : 512;

	// TODO: this takes a lot of memory
	// TODO: what should we do when quads won't fit?
	// TODO: better estimate needed; it should depend on bin size
	// TODO: properly handle situations when limits are reached
	uint max_bin_quads = max_visible_quads * 2;
	uint max_bin_tris = max_visible_quads * 2;

	auto usage = VBufferUsage::storage_buffer;
	auto usage_copyable = VBufferUsage::storage_buffer | VBufferUsage::transfer_src;
	m_bin_quads = EX_PASS(
		VulkanBuffer::create<u32>(device, max_bin_quads, usage | VBufferUsage::transfer_src));
	m_bin_tris = EX_PASS(VulkanBuffer::create<u32>(device, max_bin_tris, usage));

	int max_visible_tris = max_visible_quads * 2;
	int uvec4_storage_size = max_visible_tris * 5 + max_visible_quads * 4; // 480MB ...
	m_uvec4_storage = EX_PASS(
		VulkanBuffer::create<int4>(device, uvec4_storage_size, usage | VBufferUsage::transfer_src));
	m_uint_storage = EX_PASS(VulkanBuffer::create<u32>(device, max_visible_tris, usage));

	uint scratch_32_size = 64 * 1024 * m_max_dispatches * sizeof(u32);
	uint scratch_64_size = (128 * 1024) * m_max_dispatches * sizeof(u64);
	scratch_64_size = max<uint>(scratch_64_size, max_visible_quads * sizeof(u32));

	// TODO: control size of scratch mem
	m_scratch_32 = EX_PASS(VulkanBuffer::create(device, scratch_32_size, usage));
	m_scratch_64 =
		EX_PASS(VulkanBuffer::create(device, scratch_64_size, usage | VBufferUsage::transfer_src));
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

	p_quad_setup = EX_PASS(
		makeComputePipeline(device, compiler, consts,
							opts & Opt::debug_quad_setup ? "quad_setup_debug" : "quad_setup"));
	p_bin_dispatcher = EX_PASS(makeComputePipeline(
		device, compiler, consts,
		opts & Opt::debug_bin_dispatcher ? "bin_dispatcher_debug" : "bin_dispatcher"));

	p_raster_low = EX_PASS(makeComputePipeline(
		device, compiler, consts, opts & Opt::debug_raster ? "raster_low_debug" : "raster_low"));
	//p_raster_high = EX_PASS(makeComputePipeline(device, compiler, consts, "raster_high"));

	if(opts & (Opt::debug_bin_dispatcher | Opt::debug_quad_setup | Opt::debug_raster)) {
		auto usage =
			VBufferUsage::storage_buffer | VBufferUsage::transfer_dst | VBufferUsage::transfer_src;
		auto mem_usage = VMemoryUsage::temporary;
		m_debug_buffer = EX_PASS(VulkanBuffer::create<u32>(device, 1024 * 1024, usage, mem_usage));
	}

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

	uint bin_counters_size = LUCID_INFO_SIZE + m_bin_count * 10;
	for(int i : intRange(num_frames)) {
		auto instance_usage = VBufferUsage::storage_buffer | VBufferUsage::transfer_dst;
		auto info_usage = VBufferUsage::storage_buffer | VBufferUsage::transfer_src |
						  VBufferUsage::transfer_dst | VBufferUsage::indirect_buffer;
		auto config_usage = VBufferUsage::uniform_buffer | VBufferUsage::transfer_dst;
		auto mem_usage = VMemoryUsage::temporary;
		m_frame_info[i] =
			EX_PASS(VulkanBuffer::create<u32>(device, bin_counters_size, info_usage, mem_usage));
		m_frame_config[i] =
			EX_PASS(VulkanBuffer::create<shader::LucidConfig>(device, 1, config_usage, mem_usage));
		u32 instance_data_size =
			max_instances * (sizeof(shader::InstanceData) + sizeof(u32) + sizeof(float4));
		m_frame_instance_data[i] =
			EX_PASS(VulkanBuffer::create(device, instance_data_size, instance_usage, mem_usage));
	}

	return {};
}

void LucidRenderer::render(const Context &ctx) {
	auto &cmds = ctx.device.cmdQueue();
	PERF_GPU_SCOPE(cmds);

	cmds.fullBarrier();

	// TODO: second frame is broken
	// TODO: minimize barriers
	setupInputData(ctx).check();
	uploadInstances(ctx).check();
	cmds.barrier(VPipeStage::transfer, VPipeStage::compute_shader, VAccess::transfer_write,
				 VAccess::memory_read | VAccess::memory_write);
	quadSetup(ctx);
	cmds.barrier(VPipeStage::compute_shader, VPipeStage::compute_shader, VAccess::memory_write,
				 VAccess::memory_read | VAccess::memory_write);
	computeBins(ctx);
	cmds.barrier(VPipeStage::compute_shader, VPipeStage::compute_shader, VAccess::memory_write,
				 VAccess::memory_read | VAccess::memory_write);
	rasterLow(ctx);
	//rasterHigh(ctx);

	cmds.barrier(VPipeStage::compute_shader, VPipeStage::all_graphics, VAccess::memory_write,
				 VAccess::memory_read | VAccess::memory_write);

	compose(ctx);

	cmds.barrier(VPipeStage::compute_shader, VPipeStage::transfer, VAccess::memory_write,
				 VAccess::transfer_read);
	if(auto result = cmds.download(m_info, "info", 16); result && *result) {
		m_last_info = move(*result);
		m_last_info_updated = true;
	}
	cmds.barrier(VPipeStage::transfer, VPipeStage::all_commands, VAccess::transfer_read,
				 VAccess::memory_write);
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
		if(!dc.num_quads)
			continue;

		auto opts = dc.opts;
		u32 color = u32(mat_colors[dc.material_id]);
		if(color != 0xffffffff)
			opts |= DrawCallOpt::has_inst_color;

		InstanceData out;
		out.index_offset = dc.quad_offset * 4;
		out.vertex_offset = 0;
		out.num_quads = dc.num_quads;
		out.flags = (uint(opts.bits) & 0xffff);
		float4 uv_rect(dc.uv_rect.x(), dc.uv_rect.y(), dc.uv_rect.width(), dc.uv_rect.height());
		m_num_quads += dc.num_quads;

		if(dc.num_quads <= max_instance_quads) {
			instances.emplace_back(out);
			uv_rects.emplace_back(uv_rect);
			colors.emplace_back(color);
		} else {
			for(int i = 0; i < dc.num_quads; i += max_instance_quads) {
				out.index_offset = dc.quad_offset * 4 + i * 4;
				out.num_quads = min(max_instance_quads, dc.num_quads - i);
				instances.emplace_back(out);
				uv_rects.emplace_back(uv_rect);
				colors.emplace_back(color);
			}
		}
	}

	if(instances.size() > max_instances) {
		static int prev_instance_count = 0;
		if(instances.size() != prev_instance_count) {
			prev_instance_count = instances.size();
			print("Instance limit reached! count:% max:%\n", instances.size(), max_instances);
		}
		instances.resize(max_instances);
	}

	auto usage = VBufferUsage::storage_buffer | VBufferUsage::transfer_dst;
	auto mem_usage = VMemoryUsage::frame;

	auto instance_data = m_frame_instance_data[cmds.frameIndex() % num_frames];
	m_instances = instance_data.reinterpret<shader::InstanceData>().subSpan(0, max_instances);
	uint offset = sizeof(shader::InstanceData) * max_instances;
	m_instance_colors =
		instance_data.subSpan(offset, offset + max_instances * sizeof(u32)).reinterpret<u32>();
	offset += sizeof(u32) * max_instances;
	m_instance_uv_rects = instance_data.subSpan(offset).reinterpret<float4>();

	EXPECT(cmds.upload(m_instances, instances));
	EXPECT(cmds.upload(m_instance_colors, colors));
	EXPECT(cmds.upload(m_instance_uv_rects, uv_rects));

	m_num_instances = instances.size();
	int max_dispatches = m_max_dispatches / 2; // TODO: tweak this properly...
	m_instance_packet_size = clamp(m_num_instances / max_dispatches, 1, 2);

	return {};
}

Ex<> LucidRenderer::setupInputData(const Context &ctx) {
	auto &cmds = ctx.device.cmdQueue();
	PERF_GPU_SCOPE(cmds);

	uint bin_counters_size = LUCID_INFO_SIZE + m_bin_count * 10;
	auto info_usage = VBufferUsage::storage_buffer | VBufferUsage::transfer_src |
					  VBufferUsage::transfer_dst | VBufferUsage::indirect_buffer;
	auto config_usage = VBufferUsage::uniform_buffer | VBufferUsage::transfer_dst;
	auto mem_usage = VMemoryUsage::device;
	auto frame_index = cmds.frameIndex() % num_frames;
	m_info = m_frame_info[frame_index];
	cmds.fill(m_info.subSpan(0, LUCID_INFO_SIZE + m_bin_count * 6), 0);

	shader::LucidConfig config;
	config.frustum = FrustumInfo(ctx.camera);
	config.view_proj_matrix = ctx.camera.matrix();
	config.lighting = ctx.lighting;
	config.background_color = (float4)FColor(ctx.config.background_color);
	config.num_instances = m_num_instances;
	config.enable_backface_culling = ctx.config.backface_culling;
	config.instance_packet_size = m_instance_packet_size;
	mem_usage = VMemoryUsage::frame;
	m_config = m_frame_config[frame_index];
	EXPECT(cmds.upload(m_config, cspan(&config, 1)));

	return {};
}

void LucidRenderer::quadSetup(const Context &ctx) {
	// TODO: co robić z trójkątami, które są na tyle małe, że wogóle ich nie widać nawet w pełnej rozdziałce?
	// TODO: backface-culling ?

	auto &cmds = ctx.device.cmdQueue();

	PERF_GPU_SCOPE(cmds);
	cmds.bind(p_quad_setup);

	// TODO: descriptor set may be optimized out, what should we do in such a case?
	auto ds = cmds.bindDS(0);
	ds.set(0, m_info);
	ds.set(1, VDescriptorType::uniform_buffer, m_config);
	ds = cmds.bindDS(1);
	ds.set(0, m_instances, ctx.quads_ib, ctx.verts.pos, ctx.verts.tex, ctx.verts.col, ctx.verts.nrm,
		   m_scratch_64, m_uvec4_storage, m_uint_storage);
	if(m_opts & Opt::debug_quad_setup) {
		ds.set(9, m_debug_buffer);
		shaderDebugInitBuffer(cmds, m_debug_buffer);
	}

	int num_workgroups = (m_num_instances + m_instance_packet_size - 1) / m_instance_packet_size;
	cmds.dispatchCompute({num_workgroups, 1, 1});

	if(m_opts & Opt::debug_bin_dispatcher)
		getDebugData(ctx, m_debug_buffer, "quad_setup_debug");
}

void LucidRenderer::computeBins(const Context &ctx) {
	auto &cmds = ctx.device.cmdQueue();
	PERF_GPU_SCOPE(cmds);

	cmds.bind(p_bin_dispatcher);
	auto ds = cmds.bindDS(0);
	ds.set(0, m_info);
	ds.set(1, VDescriptorType::uniform_buffer, m_config);
	ds = cmds.bindDS(1);
	ds.set(0, m_scratch_64, m_bin_quads, m_bin_tris, m_scratch_32, m_uvec4_storage);
	if(m_opts & Opt::debug_bin_dispatcher) {
		ds.set(5, m_debug_buffer);
		shaderDebugInitBuffer(cmds, m_debug_buffer);
	}

	PERF_CHILD_SCOPE("dispatcher phase");

	cmds.barrier(VPipeStage::compute_shader, VPipeStage::draw_indirect, VAccess::memory_write,
				 VAccess::indirect_command_read);

	cmds.dispatchComputeIndirect(m_info, LUCID_INFO_MEMBER_OFFSET(num_binning_dispatches));
	if(m_opts & Opt::debug_bin_dispatcher)
		getDebugData(ctx, m_debug_buffer, "bin_dispatcher_debug");

	//cmds.dispatchCompute({16, 1, 1});
	//if(m_opts & Opt::debug_bin_dispatcher)
	//	debugProgram(p_bin_dispatcher, "bin_dispatcher");

	cmds.barrier(VPipeStage::compute_shader, VPipeStage::compute_shader, VAccess::memory_write,
				 VAccess::memory_read | VAccess::memory_read);

	PERF_SIBLING_SCOPE("categorizer phase");
	cmds.bind(p_bin_categorizer);
	ds = cmds.bindDS(1);
	ds.set(0, m_compose_quads);
	cmds.dispatchCompute({1, 1, 1});
}

void LucidRenderer::bindRaster(PVPipeline pipeline, const Context &ctx) {
	auto &cmds = ctx.device.cmdQueue();

	cmds.bind(pipeline);
	auto ds = cmds.bindDS(0);
	ds.set(0, m_info);
	ds.set(1, VDescriptorType::uniform_buffer, m_config);

	ds = cmds.bindDS(1);
	ds.set(0, m_bin_quads, m_bin_tris, m_scratch_32, m_scratch_64, m_instance_colors,
		   m_instance_uv_rects, m_uvec4_storage, m_uint_storage, m_raster_image);
	// TODO: why this needs to be separate?
	if(m_opts & Opt::debug_raster) {
		ds.set(11, m_debug_buffer);
		shaderDebugInitBuffer(cmds, m_debug_buffer);
	}

	auto sampler = ctx.device.getSampler(ctx.config.sampler_setup);
	ds.set(9, {{sampler, ctx.opaque_tex}});
	ds.set(10, {{sampler, ctx.trans_tex}});
}

void LucidRenderer::rasterLow(const Context &ctx) {
	auto &cmds = ctx.device.cmdQueue();
	PERF_GPU_SCOPE(cmds);

	cmds.barrier(VPipeStage::compute_shader, VPipeStage::draw_indirect, VAccess::memory_write,
				 VAccess::indirect_command_read);
	bindRaster(p_raster_low, ctx);
	//if(m_opts & Opt::debug_raster)
	//	shaderDebugUseBuffer(m_errors);
	cmds.dispatchComputeIndirect(m_info,
								 LUCID_INFO_MEMBER_OFFSET(bin_level_dispatches[BIN_LEVEL_LOW]));

	if(m_opts & Opt::debug_raster)
		getDebugData(ctx, m_debug_buffer, "raster_low_debug");
	//cmds.dispatchCompute({64, 1, 1});
	//if(m_opts & Opt::debug_raster)
	//	debugProgram(p_raster_low, "raster_low");
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

	cmds.barrier(VPipeStage::all_commands, VPipeStage::all_commands, VAccess::memory_write,
				 VAccess::memory_read | VAccess::memory_write);

	auto swap_chain = ctx.device.swapChain();
	auto framebuffer = ctx.device.getFramebuffer({swap_chain->acquiredImage()});
	cmds.beginRenderPass(framebuffer, m_render_pass, none, {FColor(0.0, 0.0, 0.2)});

	cmds.bind(p_compose);
	auto ds = cmds.bindDS(0);
	ds.set(0, m_raster_image);
	cmds.bindVertices(0, m_compose_quads);
	cmds.bindIndices(m_compose_ibuffer);
	cmds.setViewport(IRect(m_size));
	cmds.setScissor(none);
	cmds.drawIndexed(m_bin_counts.x * m_bin_counts.y * 6);

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

struct ScanInfo {
	float3 scan;
	int y_aabb;
	float3 scan_step;
	int signs;
};
static_assert(sizeof(ScanInfo) == sizeof(int4) * 2);

array<uint, 4> decodeAABB28(uint aabb) {
	return {aabb & 0x7fu, (aabb >> 7) & 0x7fu, (aabb >> 14) & 0x7fu, (aabb >> 21) & 0x7fu};
}

void LucidRenderer::verifyInfo() {
	if(!m_last_info_updated)
		return;
	m_last_info_updated = false;
	PERF_SCOPE();

	shader::LucidInfo info;
	memcpy(&info, m_last_info.data(), sizeof(info));
	auto bin_counters = cspan(m_last_info).subSpan(LUCID_INFO_SIZE);

	CSpan<uint> bin_quad_counts = cspan(bin_counters.data() + m_bin_count * 0, m_bin_count);
	CSpan<uint> bin_quad_offsets = cspan(bin_counters.data() + m_bin_count * 1, m_bin_count);
	CSpan<uint> bin_quad_offsets_temp = cspan(bin_counters.data() + m_bin_count * 2, m_bin_count);

	CSpan<uint> bin_tri_counts = cspan(bin_counters.data() + m_bin_count * 3, m_bin_count);
	CSpan<uint> bin_tri_offsets = cspan(bin_counters.data() + m_bin_count * 4, m_bin_count);
	CSpan<uint> bin_tri_offsets_temp = cspan(bin_counters.data() + m_bin_count * 5, m_bin_count);

	int num_errors[2] = {0, 0};
	int num_valid = 0;

	// Checking bins quad offsets
	for(uint i = 0; i < m_bin_count; i++) {
		int prev_value = i == 0 ? 0 : bin_quad_counts[i - 1];
		int prev_offset = i == 0 ? 0 : bin_quad_offsets[i - 1];
		int cur_value = bin_quad_counts[i];
		int cur_offset = bin_quad_offsets[i];
		int cur_offset_temp = bin_quad_offsets_temp[i];

		if(i > 0 && cur_offset != prev_offset + prev_value && num_errors[0]++ < 32)
			print("Invalid bin quad offset [%]: % != % (prev_offset:% + prev_count:%)\n", i,
				  cur_offset, prev_offset + prev_value, prev_offset, prev_value);
		if(cur_offset_temp != cur_offset + cur_value && num_errors[1]++ < 32)
			print("Invalid temp bin quad offset [%]: % != % (offset:% + count:%)\n", i,
				  cur_offset_temp, cur_offset + cur_value, cur_offset, cur_value);
	}

	num_errors[0] = num_errors[1] = 0;
	// Checking bins tris offsets
	for(uint i = 0; i < m_bin_count; i++) {
		int prev_value = i == 0 ? 0 : bin_tri_counts[i - 1];
		int prev_offset = i == 0 ? 0 : bin_tri_offsets[i - 1];
		int cur_value = bin_tri_counts[i];
		int cur_offset = bin_tri_offsets[i];
		int cur_offset_temp = bin_tri_offsets_temp[i];

		if(i > 0 && cur_offset != prev_offset + prev_value && num_errors[0]++ < 32)
			print("Invalid bin tri offset [%]: % != % (prev_offset:% + prev_count:%)\n", i,
				  cur_offset, prev_offset + prev_value, prev_offset, prev_value);
		if(cur_offset_temp != cur_offset + cur_value && num_errors[1]++ < 32)
			print("Invalid temp bin tri offset [%]: % != % (offset:% + count:%)\n", i,
				  cur_offset_temp, cur_offset + cur_value, cur_offset, cur_value);
	}
}

vector<StatsGroup> LucidRenderer::getStats() const {
	PERF_SCOPE();

	vector<StatsGroup> out;

	if(!m_last_info)
		return out;

	shader::LucidInfo info;
	memcpy(&info, m_last_info.data(), sizeof(info));
	auto bin_counters = cspan(m_last_info).subSpan(LUCID_INFO_SIZE);
	CSpan<uint> bin_quad_counts = cspan(bin_counters.data() + m_bin_count * 0, m_bin_count);
	CSpan<uint> bin_tri_counts = cspan(bin_counters.data() + m_bin_count * 3, m_bin_count);

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
	int num_bin_dispatcher_work_groups =
		min(max(info.a_bin_dispatcher_work_groups), arraySize(info.dispatcher_task_counts));
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

	// TODO: add better stats once rasterizer is working on all levels

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

template <class T>
Maybe<ShaderDebugInfo> LucidRenderer::getDebugData(const Context &ctx, VBufferSpan<T> src,
												   Str title) {
	auto &cmds = ctx.device.cmdQueue();
	cmds.barrier(VPipeStage::all_commands, VPipeStage::transfer, VAccess::memory_write,
				 VAccess::transfer_read);
	auto debug_data = cmds.download(m_debug_buffer, title, 32);
	cmds.barrier(VPipeStage::transfer, VPipeStage::all_commands);
	if(debug_data && *debug_data) {
		ShaderDebugInfo info(*debug_data);
		print("%: ----------------------------------------------------\n%", title, info);
		return info;
	}
	return none;
}
