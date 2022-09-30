#ifndef _SHADING_GLSL_
#define _SHADING_GLSL_

#include "funcs.glsl"
#include "structures.glsl"

#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_shader_subgroup_shuffle : require

#if WARP_SIZE == 32
#define WARP_BITMASK uint
#else
#define WARP_BITMASK uvec2
#endif

coherent layout(std430, binding = 0) buffer lucid_info_ {
	LucidInfo g_info;
	int g_counts[];
};
layout(binding = 1) uniform lucid_config_ { LucidConfig u_config; };

layout(std430, set = 1, binding = 0) readonly restrict buffer buf0_ { uint g_bin_quads[]; };
layout(std430, set = 1, binding = 1) readonly restrict buffer buf1_ { uint g_bin_tris[]; };
layout(std430, set = 1, binding = 2) restrict buffer buf2_ { uint g_scratch_32[]; };
layout(std430, set = 1, binding = 3) restrict buffer buf3_ { uvec2 g_scratch_64[]; };
layout(std430, set = 1, binding = 4) readonly restrict buffer buf4_ { uint g_instance_colors[]; };
layout(std430, set = 1, binding = 5) readonly restrict buffer buf5_ { vec4 g_instance_uv_rects[]; };
layout(std430, set = 1, binding = 6) readonly restrict buffer buf6_ { uvec4 g_uvec4_storage[]; };
layout(std430, set = 1, binding = 7) readonly restrict buffer buf7_ { uint g_normals_storage[]; };
layout(set = 1, binding = 8, rgba8) uniform image2D g_raster_image;
layout(set = 1, binding = 9) uniform sampler2D opaque_texture;
layout(set = 1, binding = 10) uniform sampler2D transparent_texture;

// TODO: separate opaque and transparent objects, draw opaque objects first to texture
// then read it and use depth to optimize drawing

shared ivec2 s_bin_pos;

void outputPixel(ivec2 pixel_pos, vec4 color) {
	//color = tintColor(color, vec3(0.2, 0.3, 0.4), 0.8);
	imageStore(g_raster_image, s_bin_pos + pixel_pos, color);
}

const float alpha_threshold = 1.0 / 128.0;

void getTriangleParams(uint tri_idx, out vec3 depth_eq, out vec2 bary_params, out vec3 edge0,
					   out vec3 edge1, out uint instance_id, out uint instance_flags) {
	uint bary_offset = STORAGE_TRI_BARY_OFFSET + tri_idx * 2;
	uvec4 val0 = g_uvec4_storage[STORAGE_TRI_DEPTH_OFFSET + tri_idx];
	uvec4 val1 = g_uvec4_storage[bary_offset + 0];
	uvec4 val2 = g_uvec4_storage[bary_offset + 1];
	depth_eq = uintBitsToFloat(val0.xyz);
	bary_params = uintBitsToFloat(uvec2(val1.w, val2.w));
	instance_flags = val0[3] & 0xffff;
	instance_id = val0[3] >> 16;
	edge0 = uintBitsToFloat(val1.xyz);
	edge1 = uintBitsToFloat(val2.xyz);
}

void getTriangleVertexColors(uint tri_idx, out vec4 color0, out vec4 color1, out vec4 color2) {
	uint quad_idx = tri_idx >> 1;
	uint second_tri = tri_idx & 1;
	uvec4 colors = g_uvec4_storage[STORAGE_QUAD_COLOR_OFFSET + quad_idx];
	color0 = decodeRGBA8(colors[0]);
	color1 = decodeRGBA8(colors[1 + second_tri]);
	color2 = decodeRGBA8(colors[2 + second_tri]);
}

void getTriangleVertexNormals(uint tri_idx, out vec3 normal0, out vec3 normal1, out vec3 normal2) {
	uint quad_idx = tri_idx >> 1;
	uint second_tri = tri_idx & 1;
	uvec4 normals = g_uvec4_storage[STORAGE_QUAD_NORMAL_OFFSET + quad_idx];
	normal0 = decodeNormalUint(normals[0]);
	normal1 = decodeNormalUint(normals[1 + second_tri]);
	normal2 = decodeNormalUint(normals[2 + second_tri]);
}

void getTriangleVertexTexCoords(uint tri_idx, out vec2 tex0, out vec2 tex1, out vec2 tex2) {
	uint quad_idx = tri_idx >> 1;
	uint second_tri = tri_idx & 1;
	uint tex_offset = STORAGE_QUAD_TEXTURE_OFFSET + quad_idx * 2;
	uvec4 tex_coords0 = g_uvec4_storage[tex_offset + 0];
	uvec4 tex_coords1 = g_uvec4_storage[tex_offset + 1];
	tex0 = uintBitsToFloat(tex_coords0.xy);
	tex1 = uintBitsToFloat(second_tri == 0 ? tex_coords0.zw : tex_coords1.xy);
	tex2 = uintBitsToFloat(second_tri == 0 ? tex_coords1.xy : tex_coords1.zw);
}

uint shadeSample(ivec2 pixel_pos, uint tri_idx, out float out_depth) {
	float px = float(pixel_pos.x), py = float(pixel_pos.y);

	vec3 depth_eq, edge0_eq, edge1_eq;
	uint instance_id, instance_flags;
	vec2 bary_params;
	getTriangleParams(tri_idx, depth_eq, bary_params, edge0_eq, edge1_eq, instance_id,
					  instance_flags);

	float inv_ray_pos = depth_eq.x * px + (depth_eq.y * py + depth_eq.z);
	out_depth = inv_ray_pos;
	float ray_pos = 1.0 / inv_ray_pos;

	float e0 = edge0_eq.x * px + (edge0_eq.y * py + edge0_eq.z);
	float e1 = edge1_eq.x * px + (edge1_eq.y * py + edge1_eq.z);
	vec2 bary = vec2(e0, e1) * ray_pos;

	vec2 bary_dx, bary_dy;
	if((instance_flags & INST_HAS_TEXTURE) != 0) {
		float ray_posx = 1.0 / (inv_ray_pos + depth_eq.x);
		float ray_posy = 1.0 / (inv_ray_pos + depth_eq.y);

		bary_dx = vec2(e0 + edge0_eq.x, e1 + edge1_eq.x) * ray_posx - bary;
		bary_dy = vec2(e0 + edge0_eq.y, e1 + edge1_eq.y) * ray_posy - bary;
	}
	// TODO: compute bary only if we use vertex attributes? That would be all scenes...
	bary -= bary_params;

	vec4 color = (instance_flags & INST_HAS_COLOR) != 0 ?
					 decodeRGBA8(g_instance_colors[instance_id]) :
					 vec4(1.0);

	if((instance_flags & INST_HAS_TEXTURE) != 0) {
		vec2 tex0, tex1, tex2;
		getTriangleVertexTexCoords(tri_idx, tex0, tex1, tex2);

		vec2 tex_coord = bary[0] * tex1 + (bary[1] * tex2 + tex0);
		vec2 tex_dx = bary_dx[0] * tex1 + bary_dx[1] * tex2;
		vec2 tex_dy = bary_dy[0] * tex1 + bary_dy[1] * tex2;

		if((instance_flags & INST_HAS_UV_RECT) != 0) {
			vec4 uv_rect = g_instance_uv_rects[instance_id];
			tex_coord = uv_rect.zw * fract(tex_coord) + uv_rect.xy;
			tex_dx *= uv_rect.zw, tex_dy *= uv_rect.zw;
		}

		vec4 tex_col;
		if((instance_flags & INST_TEX_OPAQUE) != 0)
			tex_col = vec4(textureGrad(opaque_texture, tex_coord, tex_dx, tex_dy).xyz, 1.0);
		else
			tex_col = textureGrad(transparent_texture, tex_coord, tex_dx, tex_dy);
		color *= tex_col;
	}

	if((instance_flags & INST_HAS_VERTEX_COLORS) != 0) {
		vec4 col0, col1, col2;
		getTriangleVertexColors(tri_idx, col0, col1, col2);
		color *= (1.0 - bary[0] - bary[1]) * col0 + (bary[0] * col1 + bary[1] * col2);
	}

	if(color.a == 0.0)
		return 0;

	vec3 normal;
	if((instance_flags & INST_HAS_VERTEX_NORMALS) != 0) {
		vec3 nrm0, nrm1, nrm2;
		getTriangleVertexNormals(tri_idx, nrm0, nrm1, nrm2);
		nrm1 -= nrm0;
		nrm2 -= nrm0;
		normal = bary[0] * nrm1 + (bary[1] * nrm2 + nrm0);
	} else {
		normal = decodeNormalUint(g_normals_storage[tri_idx]);
	}

	float light_value = max(0.0, dot(-u_config.lighting.sun_dir.xyz, normal) * 0.7 + 0.3);
	color.rgb = SATURATE(finalShading(u_config.lighting, color.rgb, light_value));
	return encodeRGBA8(color);
}

#define RC_COLOR_SIZE 3
#ifdef VISUALIZE_ERRORS
#define RC_DEPTH_SIZE (RC_COLOR_SIZE + 1)
#else
#define RC_DEPTH_SIZE RC_COLOR_SIZE
#endif

struct ReductionContext {
// FFS: for some reason vectors produce faster code than arrays on integrated AMDs
#if RC_DEPTH_SIZE == 3
	vec3 prev_depths;
#elif RC_DEPTH_SIZE == 4
	vec4 prev_depths;
#else
	float prev_depths[RC_DEPTH_SIZE];
#endif

#if RC_COLOR_SIZE == 3
	uvec3 prev_colors;
#elif RC_COLOR_SIZE == 4
	uvec4 prev_colors;
#else
	uint prev_colors[RC_COLOR_SIZE];
#endif

	float out_trans;
	vec3 out_color;
};

void swap(inout ReductionContext ctx, int idx0, int idx1) {
	swap(ctx.prev_colors[idx0], ctx.prev_colors[idx1]);
	swap(ctx.prev_depths[idx0], ctx.prev_depths[idx1]);
}

void initReduceSamples(out ReductionContext ctx) {
	for(int i = 0; i < RC_DEPTH_SIZE; i++)
		ctx.prev_depths[i] = 999999999.0;
	for(int i = 0; i < RC_COLOR_SIZE; i++)
		ctx.prev_colors[i] = 0;
	ctx.out_color = vec3(0.0);
	ctx.out_trans = 1.0;
}

bool reduceSample(inout ReductionContext ctx, inout vec3 out_color, uvec2 sample_s,
				  WARP_BITMASK pixel_bitmask) {
#if WARP_SIZE == 32
	int num_samples = bitCount(pixel_bitmask);
#else
	int num_samples = bitCount(pixel_bitmask.x) + bitCount(pixel_bitmask.y);
#endif

	while(subgroupAny(num_samples > 0)) {
#if WARP_SIZE == 32
		int bit = int(findLSB(pixel_bitmask));
		pixel_bitmask &= ~(1u << bit);
#else
		int bitmask_index = pixel_bitmask.x == 0 ? 1 : 0;
		int bit = findLSB(pixel_bitmask[bitmask_index]);
		pixel_bitmask[bitmask_index] &= ~(1u << bit);
		bit += bitmask_index << 5;
#endif
		uvec2 value = subgroupShuffle(sample_s, bit);
		uint color = value.x;
		float depth = uintBitsToFloat(value.y);

		if(num_samples <= 0)
			continue;
		num_samples--;

		if(depth > ctx.prev_depths[0]) {
			swap(color, ctx.prev_colors[0]);
			swap(depth, ctx.prev_depths[0]);
			if(ctx.prev_depths[0] > ctx.prev_depths[1]) {
				swap(ctx, 0, 1);
				if(ctx.prev_depths[1] > ctx.prev_depths[2]) {
					swap(ctx, 1, 2);
					int i = 3;
					for(; i < RC_DEPTH_SIZE && ctx.prev_depths[i - 1] > ctx.prev_depths[i]; i++)
						swap(ctx, i - 1, i);
#ifdef VISUALIZE_ERRORS
					if(i == RC_DEPTH_SIZE) {
						out_color = vec3(1.0, 0.0, 0.0);
						ctx.out_trans = 0.0;
						continue;
					}
#endif
				}
			}
		}

		for(int i = RC_DEPTH_SIZE - 1; i > 0; i--)
			ctx.prev_depths[i] = ctx.prev_depths[i - 1];
		ctx.prev_depths[0] = depth;

		if(ctx.prev_colors[RC_COLOR_SIZE - 1] != 0) {
			vec4 cur_color = decodeRGBA8(ctx.prev_colors[RC_COLOR_SIZE - 1]);
#ifdef ADDITIVE_BLENDING
			out_color += cur_color.rgb * cur_color.a;
#else
			out_color += cur_color.rgb * cur_color.a * ctx.out_trans;
			ctx.out_trans *= 1.0 - cur_color.a;

#ifdef ALPHA_THRESHOLD
			if(subgroupAll(ctx.out_trans < alpha_threshold))
				num_samples = 0;
#endif
#endif
		}

		for(int i = RC_COLOR_SIZE - 1; i > 0; i--)
			ctx.prev_colors[i] = ctx.prev_colors[i - 1];
		ctx.prev_colors[0] = color;
	}

	return false;
}

vec4 finishReduceSamples(ReductionContext ctx) {
	vec3 out_color = ctx.out_color;

	for(int i = RC_COLOR_SIZE - 1; i >= 0; i--)
		if(ctx.prev_colors[i] != 0) {
			vec4 cur_color = decodeRGBA8(ctx.prev_colors[i]);
			float cur_transparency = 1.0 - cur_color.a;
#ifdef ADDITIVE_BLENDING
			out_color += cur_color.rgb * cur_color.a;
#else
			out_color += cur_color.rgb * cur_color.a * ctx.out_trans;
			ctx.out_trans *= 1.0 - cur_color.a;
#endif
		}

	out_color += ctx.out_trans * u_config.background_color.xyz;
	return vec4(SATURATE(out_color), 1.0);
}

// Basic rasterization statistics

shared int s_stat_fragments;
shared int s_stat_hblocks;

void updateStats(int num_fragments, int num_hblocks) {
	atomicAdd(s_stat_fragments, num_fragments);
	atomicAdd(s_stat_hblocks, num_hblocks);
}

void initStats() {
	if(LIX == 0) {
		s_stat_fragments = 0;
		s_stat_hblocks = 0;
	}
}

void commitStats() {
	if(LIX == 0) {
		atomicAdd(g_info.num_fragments, s_stat_fragments);
		atomicAdd(g_info.num_half_blocks, s_stat_hblocks);
	}
}

#endif
