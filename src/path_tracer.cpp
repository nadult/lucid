// Copyright (C) Krzysztof Jakubowski <nadult@fastmail.fm>
// This file is part of LucidRaster. See license.txt for details.

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

#include <fwk/vulkan/vulkan_ray_tracing.h>

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

Ex<void> PathTracer::exConstruct(VulkanDevice &device, ShaderCompiler &compiler, Opts opts,
								 int2 view_size) {
	print("Constructing PathTracer (flags:% res:%):\n", opts, view_size);
	auto time = getTime();

	m_bin_size = 32;
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
		setup.compute_module = EX_PASS(compiler.createShaderModule(device, def_id));
		setup.spec_constants.emplace_back(consts, 0u);
		auto result = VulkanPipeline::create(device, setup);
		print("Compute pipeline '%': % ms\n", name, int((getTime() - time) * 1000));
		return result;
	};

	bool has_timers = m_opts & Opt::timers;
	p_trace = EX_PASS(make_compute_pipe("trace", Opt::debug, has_timers));

	if(opts & Opt::debug)
		m_debug_buffer = EX_PASS(shaderDebugBuffer(device));

	for(int i : intRange(num_frames)) {
		auto info_usage = VBufferUsage::storage | VBufferUsage::transfer_src |
						  VBufferUsage::transfer_dst | VBufferUsage::indirect;
		auto config_usage = VBufferUsage::uniform | VBufferUsage::transfer_dst;
		auto mem_usage = VMemoryUsage::temporary;
		m_frame_info[i] =
			EX_PASS(VulkanBuffer::create<shader::PathTracerInfo>(device, 1, info_usage, mem_usage));
		m_frame_config[i] = EX_PASS(
			VulkanBuffer::create<shader::PathTracerConfig>(device, 1, config_usage, mem_usage));
	}

	print("Total build time: % ms\n\n", int((getTime() - time) * 1000.0));
	return {};
}

Ex<> PathTracer::updateScene(VulkanDevice &device, Scene &scene) {
	m_scene_id = scene.id;

	// TODO: multi-material support

	auto blas =
		EX_PASS(VulkanAccelStruct::buildBottom(device, scene.verts.positions, scene.tris_ib));
	VAccelStructInstance instance{blas, Matrix4::identity()};
	m_accel_struct = EX_PASS(VulkanAccelStruct::buildTop(device, {instance}));
	m_indices = scene.tris_ib;
	m_vertices = scene.verts.positions;
	m_tex_coords = scene.verts.tex_coords;
	// TODO: wait until AS is built?

	return {};
}

void PathTracer::render(const Context &ctx) {
	auto &cmds = ctx.device.cmdQueue();
	PERF_GPU_SCOPE(cmds);

	if(ctx.scene.id != m_scene_id)
		updateScene(ctx.device, ctx.scene).check();

	cmds.fullBarrier();

	// TODO: second frame is broken
	setupInputData(ctx).check();

	cmds.bind(p_trace);
	auto ds = cmds.bindDS(0);
	ds.set(0, m_info);
	ds.set(1, VDescriptorType::uniform_buffer, m_config);

	auto swap_chain = ctx.device.swapChain();
	auto raster_image = swap_chain->acquiredImage();
	ds.setStorageImage(2, raster_image, VImageLayout::general);

	ds.set(3, m_indices, m_vertices, m_tex_coords);
	ds.set(6, m_accel_struct);

	auto sampler = ctx.device.getSampler(ctx.config.sampler_setup);
	//ds.set(10, {{sampler, ctx.opaque_tex}});
	//ds.set(11, {{sampler, ctx.trans_tex}});

	DASSERT(ctx.scene.materials);
	auto &material = ctx.scene.materials.front();
	// TODO: different default textures for different map types
	auto &albedo_map = material.maps[SceneMapType::albedo];
	auto &normal_map = material.maps[SceneMapType::normal];
	auto &pbr_map = material.maps[SceneMapType::pbr];
	ds.set(10, {{sampler, albedo_map.vk_image},
				{sampler, normal_map.vk_image},
				{sampler, pbr_map.vk_image},
				{sampler, ctx.lighting.env_map}});

	if(m_opts & Opt::debug) {
		ds.set(12, m_debug_buffer);
		shaderDebugResetBuffer(cmds, m_debug_buffer);
	}

	cmds.dispatchCompute({m_bin_counts.x, m_bin_counts.y, 1});
	if(m_opts & Opt::debug)
		printDebugData(cmds, m_debug_buffer, "raster_low_debug");
	cmds.fullBarrier();
}

Ex<> PathTracer::setupInputData(const Context &ctx) {
	auto &cmds = ctx.device.cmdQueue();
	PERF_GPU_SCOPE(cmds);

	auto frame_index = cmds.frameIndex() % num_frames;

	shader::PathTracerConfig config;
	config.frustum = FrustumInfo(ctx.camera);
	config.view_proj_matrix = ctx.camera.matrix();
	config.lighting = ctx.lighting;
	config.background_color = (float4)FColor(ctx.config.background_color);
	m_config = m_frame_config[frame_index];
	EXPECT(m_config.upload(cspan(&config, 1)));
	return {};
}
