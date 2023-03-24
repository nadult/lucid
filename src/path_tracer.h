#pragma once

#include "lucid_base.h"
#include <fwk/gfx/color.h>
#include <fwk/gfx/shader_compiler.h>

namespace fwk {
struct ShaderDebugInfo;
}

DEFINE_ENUM(PathTracerOpt, timers, debug);
using PathTracerOpts = EnumFlags<PathTracerOpt>;

namespace shader {
struct LucidConfig;
struct LucidInfo;
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
	Ex<void> exConstruct(VulkanDevice &, ShaderCompiler &, VColorAttachment, Opts, int2 view_size);
	void render(const Context &);

  private:
	Ex<> setupInputData(const Context &);

	template <class T>
	Maybe<ShaderDebugInfo> getDebugData(const Context &, VBufferSpan<T>, Str title);

	Opts m_opts;

	vector<ShaderDefId> m_shader_def_ids;
	PVPipeline p_trace;

	VBufferSpan<shader::LucidConfig> m_config;
	VBufferSpan<u32> m_info;

	static constexpr int num_frames = 2;
	VBufferSpan<> m_frame_instance_data[num_frames];
	VBufferSpan<u32> m_frame_info[num_frames];
	VBufferSpan<shader::LucidConfig> m_frame_config[num_frames];
	VBufferSpan<u32> m_debug_buffer;

	int2 m_bin_counts;
	int m_bin_count, m_bin_size;

	int2 m_size; // TODO: rename
};
