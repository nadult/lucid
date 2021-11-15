#include "program.h"

#include <fwk/gfx/camera.h>
#include <fwk/gfx/gl_program.h>
#include <fwk/gfx/gl_shader.h>
#include <fwk/gfx/opengl.h>
#include <fwk/gfx/shader_debug.h>
#include <fwk/math/frustum.h>
#include <fwk/math/ray.h>
#include <fwk/sys/on_fail.h>

static Dynamic<ShaderCombiner> s_combiner;

// TODO: move ti libfwk
static array<Ray3F, 4> frustumRays(Matrix4 matrix) {
	// Order of rays is the same as vertices in drawFullscreenRect
	Frustum cfrustum(matrix);
	return {{*cfrustum[FrustumPlaneId::down].isect(cfrustum[FrustumPlaneId::left]),
			 *cfrustum[FrustumPlaneId::left].isect(cfrustum[FrustumPlaneId::up]),
			 *cfrustum[FrustumPlaneId::up].isect(cfrustum[FrustumPlaneId::right]),
			 *cfrustum[FrustumPlaneId::right].isect(cfrustum[FrustumPlaneId::down])}};
}

FrustumRays::FrustumRays(const Camera &camera) {
	auto params = camera.params();
	auto iview = inverseOrZero(camera.viewMatrix());
	auto rays = frustumRays(camera.projectionMatrix());

	for(int n : intRange(dirs)) {
		origins[n] = rays[n].origin();
		dirs[n] = rays[n].dir();
		origins[n] = mulPoint(iview, origins[n]);
		dirs[n] = mulNormal(iview, dirs[n]);
	}
	origin0 = origins[0];
	dir0 = dirs[0];
	dirx = (dirs[3] - dirs[0]) * (1.0f / params.viewport.width());
	diry = (dirs[1] - dirs[0]) * (1.0f / params.viewport.height());
}

// SHADER IMPROVEMENTS TODO:
// - mapowanie indeksów różnego rodzaju buforów, tekstur, etc. między kodem C++ a shaderem
//   (tak, żeby nie trzeba było się martwić o indeksy)

void initShaderCombiner() {
	string prefix = dataPath("shaders/pieces") + "/";
	string suffix = ".shader";

	vector<string> names;
	Vector<FilePath> paths;

	for(auto entry : findFiles(prefix, suffix)) {
		names.emplace_back(entry);
		paths.emplace_back(string(prefix) + entry + suffix);
	}

	s_combiner.emplace(names, paths);
}

Ex<void> loadShaderPieces() { return s_combiner->loadPieces(); }

Program::Program(PProgram ref, ShaderPieceSet pieces, Opts opts, vector<pair<string, int>> ranges)
	: m_ref(move(ref)), m_pieces(pieces), m_opts(opts), m_source_ranges(move(ranges)) {}

FWK_MOVABLE_CLASS_IMPL(Program)

Ex<Program> Program::make(Str name, string defs, vector<string> locations, Opts opts) {
	PERF_SCOPE(); // TODO: info on what program it is
	auto path = dataPath(format("shaders/%.shader", name));
	auto source = EX_PASS(s_combiner->loadShader(path));

	TextFormatter header;
	header << "#version 430\n";
	header << "#extension GL_ARB_shader_draw_parameters: enable\n";
	if(!defs.empty())
		header << defs << "\n";
	header << shaderDebugDefs(opts & ProgramOpt::debug);
	source.defs = header.text();
	source.name = name;

	auto combined = s_combiner->combine(source);
	auto gl_program = EX_PASS(combined.compileAndLink(locations));
	return Program{gl_program, source.pieces, opts, move(combined.labels)};
}

Ex<Program> Program::makeCompute(Str name, string defs, Opts opts) {
	PERF_SCOPE(); // TODO: info on what program it is
	auto path = dataPath(format("shaders/%.shader", name));
	auto source = EX_PASS(s_combiner->loadShader(path));

	TextFormatter header;
	header << "#version 430\n";
	header << "#extension GL_ARB_shader_draw_parameters: enable\n";
	header << "#extension GL_ARB_shader_group_vote : enable\n";
	if(gl_info->vendor == GlVendor::nvidia) {
		// TODO: check if shuffle is actually available
		header << "#extension GL_NV_shader_thread_shuffle : enable\n";
	}
	if(gl_info->features & GlFeature::shader_ballot)
		header << "#extension GL_ARB_shader_ballot : enable\n#define BALLOT_ENABLED\n";
	if(gl_info->vendor == GlVendor::nvidia)
		header << "#define VENDOR_NVIDIA\n";
	if(!defs.empty())
		header << defs << "\n";
	header << shaderDebugDefs(opts & ProgramOpt::debug);
	source.defs = header.text();
	source.name = name;

	auto combined = s_combiner->combine(source, ShaderType::compute);
	auto gl_program = EX_PASS(combined.compileAndLink());
	return Program{gl_program, source.pieces, opts, move(combined.labels)};
}

void Program::use() { m_ref->use(); }

// -------------------------------------------------------------------------------------------
// ---  Program: functions for different shader pieces  --------------------------------------

#include <fwk/gfx/camera.h>
#include <fwk/gfx/gl_program.h>
#include <fwk/math/ray.h>

void Program::setViewport(const Camera &cam, int2 viewport_size) {
	// TODO: add view_matrix & view_proj_matrix ?
	m_ref["viewport.proj_matrix"] = cam.projectionMatrix();
	m_ref["viewport.near_plane"] = cam.params().depth.min;
	m_ref["viewport.far_plane"] = cam.params().depth.max;
	m_ref["viewport.inv_far_plane"] = 1.0f / cam.params().depth.max;
	m_ref["viewport.size"] = float2(viewport_size);
	m_ref["viewport.inv_size"] = vinv(float2(viewport_size));
}

void Program::setRect(FRect rect) {
	m_ref["rect.pos"] = rect.min();
	m_ref["rect.size"] = rect.size();
	m_ref["rect.min_uv"] = (rect.min() + float2(1.0f, 1.0f)) * 0.5f;
	m_ref["rect.max_uv"] = (rect.max() + float2(1.0f, 1.0f)) * 0.5f;
}

void Program::setFullscreenRect() { setRect(FRect(-1, -1, 1, 1)); }

void Program::setShadows(Matrix4 matrix, bool is_enabled) {
	m_ref["shadows.is_enabled"] = is_enabled ? 1 : 0;
	if(is_enabled)
		m_ref["shadows.matrix"] = matrix;
}

void Program::setFrustum(const Camera &camera) {
	// Computing coords of world-space frustum corner rays
	auto rays = frustumRays(camera.projectionMatrix());
	FrustumRays frays(camera);
	m_ref["frustum.ws_origin[0]"] = frays.origins;
	m_ref["frustum.ws_dir[0]"] = frays.dirs;
	m_ref["frustum.ws_shared_origin"] = frays.origin0;
	m_ref["frustum.ws_dir0"] = frays.dir0;
	m_ref["frustum.ws_dirx"] = frays.dirx;
	m_ref["frustum.ws_diry"] = frays.diry;

	// TODO: is the rest needed?
	float3 corner_dir[4];
	for(int n = 0; n < 4; n++) {
		corner_dir[n] = rays[n].at(camera.params().depth.max);
		corner_dir[n] /= corner_dir[n].z;
	}

	float2 diff(corner_dir[3].x - corner_dir[1].x, corner_dir[0].y - corner_dir[1].y);
	m_ref["frustum.vs_pos"] = corner_dir[1].xy();
	m_ref["frustum.vs_diff"] = diff;
}
