// $$include structures

#define LIX gl_LocalInvocationIndex
#define LID gl_LocalInvocationID

layout(std430, binding = 1) readonly restrict buffer buf1_ { uint g_bin_quads[]; };
layout(std430, binding = 2) readonly restrict buffer buf2_ { uint g_bin_tris[]; };

layout(std430, binding = 3) coherent restrict buffer buf3_ { uint g_scratch_32[]; };
layout(std430, binding = 4) coherent restrict buffer buf4_ { uvec2 g_scratch_64[]; };
layout(std430, binding = 5) readonly restrict buffer buf5_ { uint g_instance_colors[]; };
layout(std430, binding = 6) readonly restrict buffer buf6_ { vec4 g_instance_uv_rects[]; };
layout(std430, binding = 7) readonly restrict buffer buf7_ { uvec4 g_uvec4_storage[]; };
layout(std430, binding = 8) readonly restrict buffer buf8_ { uint g_uint_storage[]; };

layout(std430, binding = 9) writeonly restrict buffer buf9_ { uint g_raster_image[]; };

layout(binding = 0) uniform sampler2D opaque_texture;
layout(binding = 1) uniform sampler2D transparent_texture;

// TODO: separate opaque and transparent objects, draw opaque objects first to texture
// then read it and use depth to optimize drawing
uniform uint background_color;

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
		normal = decodeNormalUint(g_uint_storage[tri_idx]);
	}

	float light_value = max(0.0, dot(-lighting.sun_dir, normal) * 0.7 + 0.3);
	color.rgb = SATURATE(finalShading(color.rgb, light_value));
	return encodeRGBA8(color);
}

struct ReductionContext {
#ifdef VISUALIZE_ERRORS
	vec4 prev_depths;
#else
	vec3 prev_depths;
#endif
	uvec3 prev_colors;
	float out_trans;
	uint out_color;
};

void initReduceSamples(out ReductionContext ctx) {
#ifdef VISUALIZE_ERRORS
	ctx.prev_depths = vec4(999999999.0);
#else
	ctx.prev_depths = vec3(999999999.0);
#endif
	ctx.prev_colors = uvec3(0);
	ctx.out_color = 0;
	ctx.out_trans = 1.0;
}

bool reduceSample(inout ReductionContext ctx, inout vec3 out_color, uvec2 sample_s,
				  uint pixel_bitmask) {
	int j = findLSB(pixel_bitmask);

	while(anyInvocationARB(j != -1)) {
		uvec2 value = shuffleNV(sample_s, j, 32);
		uint color = value.x;
		float depth = uintBitsToFloat(value.y);

		if(j == -1)
			continue;
		pixel_bitmask &= ~(1 << j);
		j = findLSB(pixel_bitmask);

		if(depth > ctx.prev_depths[0]) {
			SWAP_UINT(color, ctx.prev_colors[0]);
			SWAP_FLOAT(depth, ctx.prev_depths[0]);
			if(ctx.prev_depths[0] > ctx.prev_depths[1]) {
				SWAP_UINT(ctx.prev_colors[1], ctx.prev_colors[0]);
				SWAP_FLOAT(ctx.prev_depths[1], ctx.prev_depths[0]);
				if(ctx.prev_depths[1] > ctx.prev_depths[2]) {
					SWAP_UINT(ctx.prev_colors[2], ctx.prev_colors[1]);
					SWAP_FLOAT(ctx.prev_depths[2], ctx.prev_depths[1]);

#ifdef VISUALIZE_ERRORS
					if(ctx.prev_depths[2] > ctx.prev_depths[3]) {
						ctx.prev_colors[2] = 0xff0000ff;
						i = sample_count;
						break;
					}
#endif
				}
			}
		}

#ifdef VISUALIZE_ERRORS
		ctx.prev_depths[3] = ctx.prev_depths[2];
#endif
		ctx.prev_depths[2] = ctx.prev_depths[1];
		ctx.prev_depths[1] = ctx.prev_depths[0];
		ctx.prev_depths[0] = depth;

		if(ctx.prev_colors[2] != 0) {
			vec4 cur_color = decodeRGBA8(ctx.prev_colors[2]);
#ifdef ADDITIVE_BLENDING
			out_color += cur_color.rgb * cur_color.a;
#else
			out_color += cur_color.rgb * cur_color.a * ctx.out_trans;
			ctx.out_trans *= 1.0 - cur_color.a;

#ifdef ALPHA_THRESHOLD
			if(allInvocationsARB(ctx.out_trans < 1.0 / 255.0))
				return true;
#endif
#endif
		}

		ctx.prev_colors[2] = ctx.prev_colors[1];
		ctx.prev_colors[1] = ctx.prev_colors[0];
		ctx.prev_colors[0] = color;
	}

	return false;
}

uint finishReduceSamples(ReductionContext ctx) {
	vec3 out_color = decodeRGB10(ctx.out_color);

	for(int i = 2; i >= 0; i--)
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

	out_color += ctx.out_trans * decodeRGB8(background_color);
	return encodeRGB8(SATURATE(out_color)); // TODO: 10 bit
}