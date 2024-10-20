// Copyright (C) Krzysztof Jakubowski <nadult@fastmail.fm>
// This file is part of LucidRaster. See license.txt for details.

#pragma once

#include "lucid_pch.h"
#include <fwk/gfx/color.h>
#include <fwk/gfx_base.h>
#include <fwk/io/file_system.h>
#include <fwk/light_tuple.h>
#include <fwk/math_base.h>
#include <fwk/sys/expected.h>

using namespace fwk;

#define ORDER_BY FWK_ORDER_BY

FilePath mainPath();
string dataPath(string file_name);

struct RenderConfig {
	float scene_opacity = 1.0f;
	IColor background_color = IColor(0, 30, 30);
	VSamplerSetup sampler_setup = VSamplerSetup(VTexFilter::linear, VTexFilter::linear,
												VTexFilter::linear, VTexAddress::repeat, 16);
	bool backface_culling = false;
	bool additive_blending = false;
	bool pbr_mode = false;
};

struct ShaderConfig {
	string build_name;
	vector<Pair<string>> predefined_macros;
};

ShaderConfig getShaderConfig(VulkanDevice &);

DEFINE_ENUM(DrawCallOpt, has_vertex_colors, has_vertex_tex_coords, has_vertex_normals, is_opaque,
			tex_opaque, has_uv_rect, has_albedo_tex, has_normal_tex, has_pbr_tex, has_inst_color);
using DrawCallOpts = EnumFlags<DrawCallOpt>;

struct SceneDrawCall {
	FBox bbox;
	int material_id = -1; // -1 means no material assigned
	int num_tris = 0, tri_offset = 0;
	int num_quads = 0, quad_offset = 0;
	DrawCallOpts opts = none;
};

// TODO: cleanup in structures
struct Scene;
struct SceneMaterial;
struct SceneDrawCall;
struct SceneLighting;

struct VertexArray {
	static void getDefs(VPipelineSetup &, bool with_tangents);

	VBufferSpan<float3> positions;
	VBufferSpan<IColor> colors;
	VBufferSpan<float2> tex_coords;
	VBufferSpan<u32> normals;
	VBufferSpan<u32> tangents; // optional
};

struct RenderContext {
	Scene &scene;
	VulkanDevice &device;
	RenderConfig config;
	VertexArray verts;
	VBufferSpan<u32> tris_ib, quads_ib;
	vector<SceneDrawCall> dcs;
	vector<SceneMaterial> materials;
	PVImageView opaque_tex, trans_tex;
	const SceneLighting &lighting;
	const Frustum &frustum;
	const Camera &camera;
};

struct StatsRow {
	string label;
	string value;
	string tooltip = {};
};

struct StatsGroup {
	vector<StatsRow> rows;
	string title;
	int label_width = 100;
};
