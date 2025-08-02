// Copyright (C) Krzysztof Jakubowski <nadult@fastmail.fm>
// This file is part of LucidRaster. See license.txt for details.

#pragma once

#include "lucid_base.h"
#include <fwk/gfx/color.h>
#include <fwk/gfx/shader_compiler.h>

DEFINE_ENUM(PathTracerOpt, timers, debug);
using PathTracerOpts = EnumFlags<PathTracerOpt>;

namespace shader {
struct PathTracerConfig;
struct PathTracerInfo;
}

class PathTracer {
  public:
	using Opt = PathTracerOpt;
	using Opts = PathTracerOpts;
	using Context = RenderContext;

	PathTracer();
	FWK_MOVABLE_CLASS(PathTracer)

	static void addShaderDefs(VulkanDevice &, ShaderCompiler &, const ShaderConfig &);
	CSpan<ShaderDefId> shaderDefIds() const { return m_shader_def_ids; }
	Ex<void> exConstruct(VulkanDevice &, ShaderCompiler &, Opts, int2 view_size);
	void render(const Context &);

  private:
	Ex<> setupInputData(const Context &);
	Ex<> updateScene(VulkanDevice &, Scene &);

	Opts m_opts;

	vector<ShaderDefId> m_shader_def_ids;
	PVPipeline p_trace;

	VBufferSpan<shader::PathTracerConfig> m_config;
	VBufferSpan<u32> m_info;

	string m_scene_id;

	static constexpr int num_frames = 2;
	VBufferSpan<shader::PathTracerInfo> m_frame_info[num_frames];
	VBufferSpan<shader::PathTracerConfig> m_frame_config[num_frames];
	VBufferSpan<u32> m_debug_buffer;

	VBufferSpan<u32> m_indices;
	VBufferSpan<float3> m_vertices;
	VBufferSpan<float2> m_tex_coords;
	PVAccelStruct m_accel_struct;

	int2 m_bin_counts;
	int m_bin_count, m_bin_size;
	int2 m_size; // TODO: rename
};
