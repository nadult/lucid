#pragma once

#include "lucid_pch.h"
#include <fwk/gfx/color.h>
#include <fwk/gfx/gl_ref.h>
#include <fwk/gfx_base.h>
#include <fwk/io/file_system.h>
#include <fwk/light_tuple.h>
#include <fwk/math_base.h>
#include <fwk/sys/expected.h>

#define ORDER_BY FWK_ORDER_BY

string dataPath(string file_name);

struct RenderConfig {
	float scene_opacity = 1.0f;
	float3 scene_color = float3(1);
	IColor background_color = IColor(0, 30, 30);
	bool backface_culling = false;
	bool additive_blending = false;
};

struct ShadowContext {
	PTexture map;
	Matrix4 matrix;
	bool enable = false;
};

DEFINE_ENUM(DrawCallOpt, has_vertex_colors, has_vertex_tex_coords, has_vertex_normals, is_opaque,
			tex_opaque, has_uv_rect, has_texture, has_inst_color);
using DrawCallOpts = EnumFlags<DrawCallOpt>;

struct SceneDrawCall {
	FBox bbox;
	FRect uv_rect;
	int material_id = -1; // -1 means no material assigned
	int num_tris = 0, tri_offset = 0;
	int num_quads = 0, quad_offset = 0;
	DrawCallOpts opts = none;
};

struct FrustumRays {
	FrustumRays() = default;
	FrustumRays(const Camera &);

	array<float3, 4> origins;
	array<float3, 4> dirs;

	float3 dir0, origin0;
	float3 dirx, diry;
};

// TODO: cleanup in structures
struct SceneMaterial;
struct SceneDrawCall;
struct SceneLighting;

struct RenderContext {
	VulkanDevice &device;
	RenderConfig config;
	PVertexArray vao;
	PBuffer quads_ib;
	vector<SceneDrawCall> dcs;
	vector<SceneMaterial> materials;
	PTexture opaque_tex, trans_tex;
	const SceneLighting &lighting;
	const Frustum &frustum;
	const Camera &camera;

	PFramebuffer out_fbo;
	PTexture depth_buffer;
	ShadowContext shadows;
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
