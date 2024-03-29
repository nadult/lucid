// Copyright (C) Krzysztof Jakubowski <nadult@fastmail.fm>
// This file is part of LucidRaster. See license.txt for details.

#ifndef _DEFINITIONS_GLSL_
#define _DEFINITIONS_GLSL_

#ifdef __cplusplus
// Thread about using vec3 in glsl buffers:
// https://stackoverflow.com/questions/38172696

#include <fwk/math/matrix4.h>
#include <fwk/math_base.h>

struct alignas(8) vec2 {
	vec2() : x(0.0f), y(0.0f) {}
	vec2(const fwk::float2 &rhs) : x(rhs.x), y(rhs.y) {}
	operator fwk::float2() const { return {x, y}; }

	float x, y;
};

struct alignas(16) vec4 {
	vec4() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}
	vec4(const fwk::float4 &rhs) : x(rhs.x), y(rhs.y), z(rhs.z), w(rhs.w) {}
	vec4(const fwk::float3 &rhs) : x(rhs.x), y(rhs.y), z(rhs.z), w(0.0f) {}
	operator fwk::float4() const { return {x, y, z, w}; }
	operator fwk::float3() const { return {x, y, z}; }

	float x, y, z, w;
};

struct alignas(16) mat4 {
	mat4() {}
	mat4(const fwk::Matrix4 &mat) : col{mat[0], mat[1], mat[2], mat[3]} {}

	vec4 col[4];
};
#endif

// TODO: locase names
#ifdef __cplusplus
#define CONSTANT(id, name, default_value) int name = default_value;
struct SpecializationConstants {
#else
#define CONSTANT(id, name, default_value) layout(constant_id = id) const int name = default_value;
#endif
	CONSTANT(0, VIEWPORT_SIZE_X, 1280)
	CONSTANT(1, VIEWPORT_SIZE_Y, 720)
	CONSTANT(2, BIN_COUNT, 880)
	CONSTANT(3, BIN_COUNT_X, 40)
	CONSTANT(4, BIN_COUNT_Y, 22)
	CONSTANT(5, BIN_SIZE, 32)
	CONSTANT(6, BIN_SHIFT, 5)
	CONSTANT(7, MAX_VISIBLE_QUADS, 1024 * 1024)
	CONSTANT(8, MAX_VISIBLE_QUADS_SHIFT, 20)
	CONSTANT(9, MAX_VISIBLE_TRIS, 2 * 1024 * 1024)
	CONSTANT(10, MAX_DISPATCHES, 128)
	CONSTANT(11, RENDER_OPTIONS, 0)

	CONSTANT(12, BIN_DISPATCHER_LSHIFT, 10)

#define BIN_DISPATCHER_LSIZE_ID 13
#define BIN_CATEGORIZER_LSIZE_ID 14
	CONSTANT(BIN_DISPATCHER_LSIZE_ID, BIN_DISPATCHER_LSIZE, 1024)
	CONSTANT(BIN_CATEGORIZER_LSIZE_ID, BIN_CATEGORIZER_LSIZE, 512)

#ifdef __cplusplus
};
#endif
#undef CONSTANT

// clang-format off
#define BIN_LEVELS_COUNT		5
#define REJECTION_TYPE_COUNT	4
#define TIMERS_COUNT			8
#define STATS_COUNT				4

#define BIN_LEVEL_EMPTY		0
#define BIN_LEVEL_MICRO		1
#define BIN_LEVEL_LOW		2
#define BIN_LEVEL_MEDIUM	3
#define BIN_LEVEL_HIGH		4

// These map directly to DrawCallOpts (lucid_base.h)
#define INST_HAS_VERTEX_COLORS		0x001
#define INST_HAS_VERTEX_TEX_COORDS	0x002
#define INST_HAS_VERTEX_NORMALS		0x004
#define INST_IS_OPAQUE				0x008
#define INST_TEX_OPAQUE				0x010
#define INST_HAS_UV_RECT			0x020
#define INST_HAS_ALBEDO_TEXTURE		0x040
#define INST_HAS_NORMAL_TEXTURE		0x080
#define INST_HAS_PBR_TEXTURE		0x100
#define INST_HAS_COLOR				0x200

// Different reasons for rejection of triangles/quads during setup
#define REJECTION_TYPE_OTHER			0
#define REJECTION_TYPE_BACKFACE			1
#define REJECTION_TYPE_FRUSTUM			2
#define REJECTION_TYPE_BETWEEN_SAMPLES	3

#ifndef __cplusplus
#define LIX		gl_LocalInvocationIndex
#define LID		gl_LocalInvocationID
#define WGID	gl_WorkGroupID

#if !defined(SUBGROUP_SIZE) || !defined(SUBGROUP_SHIFT)
#error "SUBGROUP_SIZE and SUBGROUP_SHIFT must be defined"
#endif
#define SUBGROUP_MASK (SUBGROUP_SIZE - 1)

bool renderOptSet(uint bit) {
	return (RENDER_OPTIONS & bit) != 0u;
}

// Per-bin number of quad counts, offsets, etc.
#define BIN_QUAD_COUNTS(idx)		g_counts[BIN_COUNT * 0 + (idx)]
#define BIN_QUAD_OFFSETS(idx)		g_counts[BIN_COUNT * 1 + (idx)]
#define BIN_QUAD_OFFSETS_TEMP(idx)	g_counts[BIN_COUNT * 2 + (idx)]

#define BIN_TRI_COUNTS(idx)			g_counts[BIN_COUNT * 3 + (idx)]
#define BIN_TRI_OFFSETS(idx)		g_counts[BIN_COUNT * 4 + (idx)]
#define BIN_TRI_OFFSETS_TEMP(idx)	g_counts[BIN_COUNT * 5 + (idx)]

// Lists of bins of different quad density levels
#define MICRO_LEVEL_BINS(idx)		g_counts[BIN_COUNT * 6 + (idx)]
#define LOW_LEVEL_BINS(idx)			g_counts[BIN_COUNT * 7 + (idx)]
#define MEDIUM_LEVEL_BINS(idx)		g_counts[BIN_COUNT * 8 + (idx)]
#define HIGH_LEVEL_BINS(idx)		g_counts[BIN_COUNT * 9 + (idx)]

// Macros useful when accessing storage
#define STORAGE_TRI_BARY_OFFSET		0
#define STORAGE_TRI_SCAN_OFFSET		(MAX_VISIBLE_QUADS * 4)
#define STORAGE_TRI_DEPTH_OFFSET	(MAX_VISIBLE_QUADS * 8)
#define STORAGE_QUAD_COLOR_OFFSET	(MAX_VISIBLE_QUADS * 10)
#define STORAGE_QUAD_NORMAL_OFFSET	(MAX_VISIBLE_QUADS * 11)
#define STORAGE_QUAD_TEXTURE_OFFSET	(MAX_VISIBLE_QUADS * 12)
#endif

#endif