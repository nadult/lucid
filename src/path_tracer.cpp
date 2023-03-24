#include "path_tracer.h"

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

PathTracer::PathTracer() = default;
FWK_MOVABLE_CLASS_IMPL(PathTracer)

void PathTracer::addShaderDefs(VulkanDevice &device, ShaderCompiler &compiler,
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

	add_defs("trace", "trace.glsl");
}

Ex<void> PathTracer::exConstruct(VulkanDevice &device, ShaderCompiler &compiler,
								 VColorAttachment color_att, Opts opts, int2 view_size) {
	print("Constructing PathTracer (flags:% res:%):\n", opts, view_size);
	auto time = getTime();
	m_bin_size = 32; //opts & Opt::bin_size_64 ? 64 : 32;

	m_bin_counts = (view_size + int2(m_bin_size - 1)) / m_bin_size;
	m_bin_count = m_bin_counts.x * m_bin_counts.y;
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
	consts.RENDER_OPTIONS = m_opts.bits;

	int bin_dispatcher_lsize = m_bin_size == 64 ? 512 : 1024;
	consts.BIN_DISPATCHER_LSIZE = bin_dispatcher_lsize;
	consts.BIN_DISPATCHER_LSHIFT = log2(bin_dispatcher_lsize);
	consts.BIN_CATEGORIZER_LSIZE = 1024;

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
	p_trace = EX_PASS(make_compute_pipe("trace", Opt::debug, has_timers));

	if(opts & Opt::debug) {
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
	}

	print("Total build time: % ms\n\n", int((getTime() - time) * 1000.0));
	return {};
}

void PathTracer::render(const Context &ctx) {
	auto &cmds = ctx.device.cmdQueue();
	PERF_GPU_SCOPE(cmds);

	cmds.fullBarrier();

	// TODO: second frame is broken
	setupInputData(ctx).check();
	cmds.bind(p_trace);
	auto ds = cmds.bindDS(0);
	ds.set(0, m_info);
	ds.set(1, VDescriptorType::uniform_buffer, m_config);
	auto sampler = ctx.device.getSampler(ctx.config.sampler_setup);
	ds.set(3, {{sampler, ctx.opaque_tex}});
	ds.set(4, {{sampler, ctx.trans_tex}});

	auto swap_chain = ctx.device.swapChain();
	auto raster_image = swap_chain->acquiredImage();
	ds.setStorageImage(2, raster_image, VImageLayout::general);
	if(m_opts & Opt::debug) {
		ds.set(10, m_debug_buffer);
		shaderDebugInitBuffer(cmds, m_debug_buffer);
	}

	cmds.dispatchCompute({m_bin_counts.x, m_bin_counts.y, 1});
	if(m_opts & Opt::debug)
		getDebugData(ctx, m_debug_buffer, "raster_low_debug");

	cmds.fullBarrier();
}

Ex<> PathTracer::setupInputData(const Context &ctx) {
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
	config.num_instances = 0;
	config.enable_backface_culling = ctx.config.backface_culling;
	config.instance_packet_size = 0;
	m_config = m_frame_config[frame_index];
	EXPECT(cmds.upload(m_config, cspan(&config, 1)));

	return {};
}

template <class T>
Maybe<ShaderDebugInfo> PathTracer::getDebugData(const Context &ctx, VBufferSpan<T> src, Str title) {
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
