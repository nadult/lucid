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
#include <fwk/vulkan/vulkan_image.h>
#include <fwk/vulkan/vulkan_instance.h>
#include <fwk/vulkan/vulkan_pipeline.h>
#include <fwk/vulkan/vulkan_swap_chain.h>

// TODO: why on desktop AMD on windows some shaders work 2-3x slower? balancedDispatcher & rasterizers
// TODO: rename WARP to SUBGROUP ?
// TODO: better stats (than num_half_blocks)
// TODO: ability to change options without recreating renderer
// TODO: re-enable bin_64 support?
// TODO: opisać różnego rodzaju definicje/nazwy używane w kodzie
// TODO: problemy na hairballu i powerplancie na AMD jak jest duzo trisow?
// Cały czas są jakieś dziwne błędy przy 1 i 2 klatce renderingu (też jak się zmienia rozdziałkę)
// Nie tylko dziwne rzeczy się wyświetlają, ale są też błędy w tri offsetach w dispatcherze
// TODO: opcja rekonstrukcji ucid renderera, żeby nie trzeba było tworzyć od nowa buforów
// TODO: properly handling cases with more visible quads than is allowed
// Raz się zdarzyło, że na teapocie były jakieś dziwne glitche na NVIDII, po włączeniu wcześniej kilku dużych scen
// TODO: maybe it would be possible to remove some code from quad_setup, because we use 3D rasterization ?
// TODO: better way to stora data in scratch memory? we could do it in more optimal way in some cases (_32 instead of _64)
// TODO: raz zdarzył się crash przy przełączeniu sceny z powrplant na san miguel
// TODO: zjebane teksturki w chestnut_tree01 jak sie poprzelacza pomiedzy scenami; w trybie debug...
// TODO: jakies dziwne bugi a AMD jak sie wlaczy timery? problemy z mazaniem po pamięci?
// TODO: crash przy zmianie rozdziałki kilka razy
// TODO: coś się pierdzieli na NVIDII też, jak się załaduje kilka scen w tym duże i jest włączona visualizacja błędów

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

ShaderConfig getShaderConfig(VulkanDevice &device) {
	ShaderConfig out;

	auto warp_size = device.physInfo().subgroup_props.subgroupSize;
	auto &pinfo = device.physInfo();
	out.predefined_macros.emplace_back("WARP_SIZE", toString(warp_size));
	out.predefined_macros.emplace_back("WARP_SHIFT", toString(log2(warp_size)));
	out.predefined_macros.emplace_back(format("VENDOR_%", toUpper(toString(pinfo.vendor_id))), "1");

	out.build_name = format("%_%", pinfo.vendor_id, warp_size);

	return out;
}

void LucidRenderer::addShaderDefs(VulkanDevice &device, ShaderCompiler &compiler,
								  const ShaderConfig &shader_config) {
	vector<Pair<string>> debug_macros = {{"DEBUG_ENABLED", ""}};
	vector<Pair<string>> timers_macros = {{"TIMERS_ENABLED", ""}};
	auto base_macros = shader_config.predefined_macros;
	insertBack(debug_macros, base_macros);
	insertBack(timers_macros, base_macros);

	auto add_defs = [&](ZStr name, ZStr file_name, bool debuggable = true,
						bool with_timers = true) {
		compiler.add({name, VShaderStage::compute, file_name, base_macros});
		if(debuggable) {
			auto dbg_name = format("%_debug", name);
			compiler.add({dbg_name, VShaderStage::compute, file_name, debug_macros});
		}
		if(with_timers) {
			compiler.add(
				{format("%_timers", name), VShaderStage::compute, file_name, timers_macros});
		}
	};

	add_defs("quad_setup", "quad_setup.glsl");
	add_defs("bin_counter", "bin_counter.glsl");
	add_defs("bin_dispatcher", "bin_dispatcher.glsl");
	add_defs("bin_categorizer", "bin_categorizer.glsl", false, false);
	add_defs("raster_low", "raster_low.glsl");
	add_defs("raster_high", "raster_high.glsl");

	base_macros.emplace_back("ALPHA_THRESHOLD", "1");
	add_defs("raster_low_alpha_threshold", "raster_low.glsl");
	add_defs("raster_high_alpha_threshold", "raster_high.glsl");

	base_macros.back().first = "ADDITIVE_BLENDING";
	add_defs("raster_low_additive_blend", "raster_low.glsl");
	add_defs("raster_high_additive_blend", "raster_high.glsl");

	base_macros.back().first = "VISUALIZE_ERRORS";
	add_defs("raster_low_vis_errors", "raster_low.glsl");
	add_defs("raster_high_vis_errors", "raster_high.glsl");
}

Ex<void> LucidRenderer::exConstruct(VulkanDevice &device, ShaderCompiler &compiler,
									VColorAttachment color_att, Opts opts, int2 view_size) {
	print("Constructing LucidRenderer (flags:% res:%):\n", opts, view_size);
	auto time = getTime();
	m_bin_size = 32; //opts & Opt::bin_size_64 ? 64 : 32;
	m_block_size = 8;

	m_bin_counts = (view_size + int2(m_bin_size - 1)) / m_bin_size;
	m_bin_count = m_bin_counts.x * m_bin_counts.y;

	// TODO: Why adding more on intel causes problems?
	// TODO: properly get number of compute units (use opencl?)
	// TODO: max dispatches should also depend on lsize
	// https://tinyurl.com/o7s9ph3
	auto phys_info = device.physInfo();
	auto mem_size_mb = phys_info.deviceLocalMemorySize() / (1024 * 1024);
	m_max_dispatches = mem_size_mb >= 4000 ? 256 : mem_size_mb >= 2000 ? 128 : 64;
	uint max_visible_quads = min(1024ull * 1024 * 1024 / 224, mem_size_mb * 1024);
	m_max_visible_quads = max_visible_quads;
	DASSERT(m_max_dispatches <= LUCID_INFO_MAX_DISPATCHES);

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
	consts.BIN_CATEGORIZER_LSIZE = 1024;

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
	m_normals_storage = EX_PASS(VulkanBuffer::create<u32>(device, max_visible_tris, usage));

	uint scratch_32_size = 128 * 1024 * m_max_dispatches * sizeof(u32);
	uint scratch_64_size = (128 * 1024) * m_max_dispatches * sizeof(u64);
	scratch_64_size = max<uint>(scratch_64_size, max_visible_quads * sizeof(u32));

	// TODO: control size of scratch mem
	m_scratch_32 = EX_PASS(VulkanBuffer::create(device, scratch_32_size, usage));
	m_scratch_64 =
		EX_PASS(VulkanBuffer::create(device, scratch_64_size, usage | VBufferUsage::transfer_src));
	if(m_opts & (Opt::debug_bin_dispatcher | Opt::debug_raster))
		m_errors = EX_PASS(VulkanBuffer::create<u32>(device, 1024 * 1024, usage_copyable));

	auto make_compute_pipe = [&](string name, Opts debug_option,
								 bool has_timers) -> Ex<PVPipeline> {
		if(opts & debug_option)
			name = name + "_debug";
		else if(has_timers)
			name = name + "_timers";

		auto time = getTime();
		VComputePipelineSetup setup;
		auto def_id = *compiler.find(name);
		m_shader_def_ids.emplace_back(def_id);
		setup.compute_module = EX_PASS(compiler.createShaderModule(device.ref(), def_id));
		setup.spec_constants.emplace_back(consts, 0u);
		auto result = VulkanPipeline::create(device.ref(), setup);
		print("Compute pipeline '%': % ms\n", name, int((getTime() - time) * 1000));
		return result;
	};

	bool has_timers = m_opts & Opt::timers;
	p_quad_setup = EX_PASS(make_compute_pipe("quad_setup", Opt::debug_quad_setup, has_timers));
	p_bin_dispatcher =
		EX_PASS(make_compute_pipe("bin_dispatcher", Opt::debug_bin_dispatcher, has_timers));
	p_bin_counter = EX_PASS(make_compute_pipe("bin_counter", Opt::debug_bin_counter, has_timers));
	p_bin_categorizer = EX_PASS(make_compute_pipe("bin_categorizer", none, false));

	auto raster_suffix = m_opts & Opt::visualize_errors	 ? "_vis_errors" :
						 m_opts & Opt::additive_blending ? "_additive_blend" :
						 m_opts & Opt::alpha_threshold	 ? "_alpha_threshold" :
														   "";
	p_raster_low = EX_PASS(
		make_compute_pipe(format("raster_low%", raster_suffix), Opt::debug_raster, has_timers));
	p_raster_high = EX_PASS(
		make_compute_pipe(format("raster_high%", raster_suffix), Opt::debug_raster, has_timers));

	if(opts & (Opt::debug_bin_dispatcher | Opt::debug_quad_setup | Opt::debug_raster |
			   Opt::debug_bin_counter)) {
		auto usage =
			VBufferUsage::storage_buffer | VBufferUsage::transfer_dst | VBufferUsage::transfer_src;
		auto mem_usage = VMemoryUsage::temporary;
		m_debug_buffer = EX_PASS(VulkanBuffer::create<u32>(device, 1024 * 1024, usage, mem_usage));
	}

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

	print("Total build time: % ms\n\n", int((getTime() - time) * 1000.0));
	return {};
}

void LucidRenderer::render(const Context &ctx) {
	auto &cmds = ctx.device.cmdQueue();
	PERF_GPU_SCOPE(cmds);

	cmds.fullBarrier();

	// TODO: second frame is broken
	setupInputData(ctx).check();
	uploadInstances(ctx).check();
	quadSetup(ctx);
	computeBins(ctx);
	rasterLow(ctx);
	rasterHigh(ctx);

	cmds.barrier(VPipeStage::compute_shader, VPipeStage::transfer, VAccess::shader_write,
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
		   m_scratch_64, m_uvec4_storage, m_normals_storage);
	if(m_opts & Opt::debug_quad_setup) {
		ds.set(9, m_debug_buffer);
		shaderDebugInitBuffer(cmds, m_debug_buffer);
	}

	cmds.barrier(VPipeStage::transfer, VPipeStage::compute_shader, VAccess::transfer_write,
				 VAccess::shader_read);

	int num_workgroups = (m_num_instances + m_instance_packet_size - 1) / m_instance_packet_size;
	cmds.dispatchCompute({num_workgroups, 1, 1});

	if(m_opts & Opt::debug_bin_dispatcher)
		getDebugData(ctx, m_debug_buffer, "quad_setup_debug");
}

void LucidRenderer::computeBins(const Context &ctx) {
	auto &cmds = ctx.device.cmdQueue();
	PERF_GPU_SCOPE(cmds);

	PERF_CHILD_SCOPE("counter phase");
	cmds.barrier(VPipeStage::compute_shader, VPipeStage::compute_shader | VPipeStage::draw_indirect,
				 VAccess::shader_write, VAccess::shader_read | VAccess::indirect_command_read);

	cmds.bind(p_bin_counter);
	auto ds = cmds.bindDS(0);
	ds.set(0, m_info);
	ds.set(1, VDescriptorType::uniform_buffer, m_config);
	ds = cmds.bindDS(1);
	ds.set(0, m_scratch_64, m_bin_quads, m_bin_tris, m_scratch_32, m_uvec4_storage);
	if(m_opts & Opt::debug_bin_counter) {
		ds.set(5, m_debug_buffer);
		shaderDebugInitBuffer(cmds, m_debug_buffer);
	}
	cmds.dispatchComputeIndirect(m_info, LUCID_INFO_MEMBER_OFFSET(num_binning_dispatches));
	if((m_opts & Opt::debug_bin_counter))
		getDebugData(ctx, m_debug_buffer, "bin_counter_debug");

	cmds.barrier(VPipeStage::compute_shader, VPipeStage::compute_shader, VAccess::shader_write,
				 VAccess::shader_read);

	PERF_SIBLING_SCOPE("categorizer phase");
	cmds.bind(p_bin_categorizer);
	cmds.dispatchCompute({2, 1, 1});

	PERF_SIBLING_SCOPE("dispatcher phase");
	cmds.barrier(VPipeStage::compute_shader, VPipeStage::compute_shader, VAccess::shader_write,
				 VAccess::shader_read);

	cmds.bind(p_bin_dispatcher);
	ds = cmds.bindDS(1);
	ds.set(0, m_scratch_64, m_bin_quads, m_bin_tris, m_scratch_32, m_uvec4_storage);
	if(m_opts & Opt::debug_bin_dispatcher) {
		ds.set(5, m_debug_buffer);
		shaderDebugInitBuffer(cmds, m_debug_buffer);
	}

	cmds.dispatchComputeIndirect(m_info, LUCID_INFO_MEMBER_OFFSET(num_binning_dispatches));
	if((m_opts & Opt::debug_bin_dispatcher))
		getDebugData(ctx, m_debug_buffer, "bin_dispatcher_debug");
}

void LucidRenderer::bindRaster(PVPipeline pipeline, const Context &ctx) {
	auto &cmds = ctx.device.cmdQueue();

	cmds.bind(pipeline);
	auto ds = cmds.bindDS(0);
	ds.set(0, m_info);
	ds.set(1, VDescriptorType::uniform_buffer, m_config);

	ds = cmds.bindDS(1);
	ds.set(0, m_bin_quads, m_bin_tris, m_scratch_32, m_scratch_64, m_instance_colors,
		   m_instance_uv_rects, m_uvec4_storage, m_normals_storage);

	auto swap_chain = ctx.device.swapChain();
	auto raster_image = swap_chain->acquiredImage();
	ds.setStorageImage(8, raster_image, VImageLayout::general);
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

	cmds.barrier(VPipeStage::compute_shader, VPipeStage::draw_indirect | VPipeStage::compute_shader,
				 VAccess::shader_write, VAccess::indirect_command_read | VAccess::shader_read);

	bindRaster(p_raster_low, ctx);
	cmds.dispatchComputeIndirect(m_info,
								 LUCID_INFO_MEMBER_OFFSET(bin_level_dispatches[BIN_LEVEL_LOW]));
	if(m_opts & Opt::debug_raster)
		getDebugData(ctx, m_debug_buffer, "raster_low_debug");
}

void LucidRenderer::rasterHigh(const Context &ctx) {
	auto &cmds = ctx.device.cmdQueue();
	PERF_GPU_SCOPE(cmds);
	cmds.barrier(VPipeStage::compute_shader, VPipeStage::draw_indirect | VPipeStage::compute_shader,
				 VAccess::shader_write, VAccess::indirect_command_read | VAccess::shader_read);

	bindRaster(p_raster_high, ctx);
	cmds.dispatchComputeIndirect(m_info,
								 LUCID_INFO_MEMBER_OFFSET(bin_level_dispatches[BIN_LEVEL_HIGH]));
	if(m_opts & Opt::debug_raster)
		getDebugData(ctx, m_debug_buffer, "raster_high_debug");
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
		processTimers(info.bin_dispatcher_timers, {"count small quads", "count large tris",
												   "dispatch small quads", "dispatch large tris"});

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

	vector<int> dispatcher_task_counts;
	for(int i : intRange(info.dispatcher_num_batches[0])) {
		int count = info.dispatcher_num_batches[0][i] + info.dispatcher_num_batches[1][i];
		if(count > 0)
			dispatcher_task_counts.emplace_back(count);
	}

	uint num_fragments = info.stats[0];
	uint num_rblocks = info.stats[1];

	auto fragment_info = stdFormat("%.3f avg fragments / pixel\n%.3f avg fragments / rblock",
								   double(num_fragments) / (m_size.x * m_size.y),
								   double(num_fragments) / num_rblocks);

	vector<StatsRow> basic_rows = {
		{"input instances", formatLarge(m_num_instances)},
		{"bin dispatcher work-groups", toString(dispatcher_task_counts.size()),
		 toString(dispatcher_task_counts)},
		{"input quads", formatLarge(info.num_input_quads)},
		{"visible quads", visible_info, visible_details},
		{"rejected quads", rejected_info, rejection_details},
		{"bin quads", formatLarge(num_bin_quads), "Total per-bin quads"},
		{"bin tris", formatLarge(num_bin_tris), "Total per-bin tris"},
		{"max small quads / bin", formatLarge(max_quads_per_bin)},
		{"max large tris / bin", formatLarge(max_tris_per_bin)},
		{"render-blocks", formatLarge(num_rblocks)},
		{"fragments", formatLarge(num_fragments), fragment_info},
	};

	if(!allOf(info.temp, 0)) {
		int last = arraySize(info.temp) - 1;
		while(last > 0 && info.temp[last] == 0)
			last--;
		vector<int> temps = cspan(info.temp, last + 1);
		basic_rows.emplace_back("temps", toString(temps));
	}

	vector<StatsRow> limit_rows = {{"max visible quads", formatLarge(m_max_visible_quads)},
								   {"max_dispatches", formatLarge(m_max_dispatches)}};

	// TODO: add better stats once rasterizer is working on all levels

	if(setup_timers)
		out.emplace_back(move(setup_timers), "quad_setup timers", 130);
	if(bin_dispatcher_timers)
		out.emplace_back(move(bin_dispatcher_timers), "bin_dispatcher timers", 130);
	if(raster_timers)
		out.emplace_back(move(raster_timers), "raster_low & raster_high timers", 130);

	out.emplace_back(move(bin_level_rows), "Bins categorized by quad density levels:", 130);
	out.emplace_back(move(basic_rows), "", 130);
	out.emplace_back(move(limit_rows), "LucidRenderer limits", 130);

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
