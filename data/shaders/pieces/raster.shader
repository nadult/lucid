// $$include structures

#define LIX gl_LocalInvocationIndex
#define LID gl_LocalInvocationID

layout(std430, binding = 1) readonly restrict buffer buf1_ { uint g_bin_quads[]; };
layout(std430, binding = 2) writeonly restrict buffer buf2_ { uint g_raster_image[]; };

layout(std430, binding = 3) coherent restrict buffer buf3_ { uint g_scratch_32[]; };
layout(std430, binding = 4) coherent restrict buffer buf4_ { uvec2 g_scratch_64[]; };
layout(std430, binding = 5) readonly restrict buffer buf5_ { uint g_instance_colors[]; };
layout(std430, binding = 6) readonly restrict buffer buf6_ { vec4 g_instance_uv_rects[]; };
layout(std430, binding = 7) readonly restrict buffer buf7_ { uvec2 g_tri_storage[]; };
layout(std430, binding = 8) readonly restrict buffer buf8_ { uvec4 g_quad_storage[]; };
layout(std430, binding = 9) readonly restrict buffer buf9_ { uvec4 g_scan_storage[]; };
layout(std430, binding = 10) readonly restrict buffer buf10_ { uint g_uint_storage[]; };

layout(binding = 0) uniform sampler2D opaque_texture;
layout(binding = 1) uniform sampler2D transparent_texture;

// TODO: names
#define TRI_SCRATCH(var_idx) g_tri_storage[scratch_tri_idx * 4 + var_idx]
#define QUAD_SCRATCH(var_idx) g_quad_storage[scratch_quad_idx + var_idx * MAX_VISIBLE_QUADS]
#define QUAD_TEX_SCRATCH(var_idx)                                                                  \
	g_quad_storage[scratch_quad_idx * 2 + MAX_VISIBLE_QUADS * 2 + var_idx]
#define SCAN_SCRATCH(var_idx) g_scan_storage[scratch_tri_idx * 2 + var_idx]
#define DEPTH_SCRATCH() g_scan_storage[MAX_VISIBLE_QUADS * 4 + scratch_tri_idx]
#define UINT_SCRATCH() g_uint_storage[scratch_tri_idx]

// TODO: separate opaque and transparent objects, draw opaque objects first to texture
// then read it and use depth to optimize drawing
uniform uint background_color;

#ifdef ENABLE_TIMERS
shared uint s_timers[TIMERS_COUNT];
#define INIT_CLOCK() uint64_t clock0 = clockARB();
#define UPDATE_CLOCK(idx)                                                                          \
	if((LIX & 31) == 0) {                                                                          \
		uint64_t clock = clockARB();                                                               \
		atomicAdd(s_timers[idx], uint(clock - clock0) >> 4);                                       \
		clock0 = clock;                                                                            \
	}

void initTimers() {
	if(LIX < TIMERS_COUNT)
		s_timers[LIX] = 0;
}
void commitTimers() {
	if(LIX < TIMERS_COUNT)
		atomicAdd(g_info.timers[LIX], s_timers[LIX]);
}

#else
#define INIT_CLOCK()
#define UPDATE_CLOCK(idx)

void initTimers() {}
void commitTimers() {}
#endif

// Algorithm inspired by Nanite scanline rasterizer
void computeScanlineParams(vec3 tri0, vec3 tri1, vec3 tri2, vec2 start, out vec3 scan_min,
						   out vec3 scan_max, out vec3 scan_step) {
	vec3 nrm0 = cross(tri2, tri1 - tri2);
	vec3 nrm1 = cross(tri0, tri2 - tri0);
	vec3 nrm2 = cross(tri1, tri0 - tri1);
	float volume = dot(tri0, nrm0);
	if(volume < 0)
		nrm0 = -nrm0, nrm1 = -nrm1, nrm2 = -nrm2;

	vec3 edges[3] = {
		vec3(dot(nrm0, frustum.ws_dirx), dot(nrm0, frustum.ws_diry), dot(nrm0, frustum.ws_dir0)),
		vec3(dot(nrm1, frustum.ws_dirx), dot(nrm1, frustum.ws_diry), dot(nrm1, frustum.ws_dir0)),
		vec3(dot(nrm2, frustum.ws_dirx), dot(nrm2, frustum.ws_diry), dot(nrm2, frustum.ws_dir0)),
	};

	float inv_ex[3] = {1.0 / edges[0].x, 1.0 / edges[1].x, 1.0 / edges[2].x};
	vec3 scan_base = -vec3(edges[0].z * inv_ex[0], edges[1].z * inv_ex[1], edges[2].z * inv_ex[2]);
	scan_step = -vec3(edges[0].y * inv_ex[0], edges[1].y * inv_ex[1], edges[2].y * inv_ex[2]);

	bool sign0 = edges[0].x < 0.0;
	bool sign1 = edges[1].x < 0.0;
	bool sign2 = edges[2].x < 0.0;

	vec3 scan = scan_step * start.y + scan_base - vec3(start.x);
	scan_min = vec3(sign0 ? -1.0 / 0.0 : scan[0], sign1 ? -1.0 / 0.0 : scan[1],
					sign2 ? -1.0 / 0.0 : scan[2]);
	scan_max =
		vec3(sign0 ? scan[0] : 1.0 / 0.0, sign1 ? scan[1] : 1.0 / 0.0, sign2 ? scan[2] : 1.0 / 0.0);
}

void getTriangleParams(uint scratch_tri_idx, out vec3 depth_eq, out vec2 bary_params,
					   out vec3 edge0, out vec3 edge1, out uint instance_id,
					   out uint instance_flags) {
	{
		uvec4 val0 = DEPTH_SCRATCH();
		uvec2 val2 = TRI_SCRATCH(0);
		depth_eq =
			vec3(uintBitsToFloat(val0[0]), uintBitsToFloat(val0[1]), uintBitsToFloat(val0[2]));
		bary_params = vec2(uintBitsToFloat(val2[0]), uintBitsToFloat(val2[1]));
		instance_flags = val0[3] & 0xffff;
		instance_id = val0[3] >> 16;
	}
	{
		uvec2 val0 = TRI_SCRATCH(1), val1 = TRI_SCRATCH(2), val2 = TRI_SCRATCH(3);
		edge0 = vec3(uintBitsToFloat(val0[0]), uintBitsToFloat(val0[1]), uintBitsToFloat(val1[0]));
		edge1 = vec3(uintBitsToFloat(val1[1]), uintBitsToFloat(val2[0]), uintBitsToFloat(val2[1]));
	}
}

void getTriangleVertexColors(uint scratch_tri_idx, out vec4 color0, out vec4 color1,
							 out vec4 color2) {
	uint scratch_quad_idx = scratch_tri_idx >> 1;
	uint second_tri = scratch_tri_idx & 1;
	uvec4 colors = QUAD_SCRATCH(0);
	color0 = decodeRGBA8(colors[0]);
	color1 = decodeRGBA8(colors[1 + second_tri]);
	color2 = decodeRGBA8(colors[2 + second_tri]);
}

void getTriangleVertexNormals(uint scratch_tri_idx, out vec3 normal0, out vec3 normal1,
							  out vec3 normal2) {
	uint scratch_quad_idx = scratch_tri_idx >> 1;
	uint second_tri = scratch_tri_idx & 1;
	uvec4 normals = QUAD_SCRATCH(1);
	normal0 = decodeNormalUint(normals[0]);
	normal1 = decodeNormalUint(normals[1 + second_tri]);
	normal2 = decodeNormalUint(normals[2 + second_tri]);
}

void getTriangleVertexTexCoords(uint scratch_tri_idx, out vec2 tex0, out vec2 tex1, out vec2 tex2) {
	uint scratch_quad_idx = scratch_tri_idx >> 1;
	uint second_tri = scratch_tri_idx & 1;
	uvec4 tex_coords0 = QUAD_TEX_SCRATCH(0);
	uvec4 tex_coords1 = QUAD_TEX_SCRATCH(1);
	tex0 = uintBitsToFloat(tex_coords0.xy);
	tex1 = uintBitsToFloat(second_tri == 0 ? tex_coords0.zw : tex_coords1.xy);
	tex2 = uintBitsToFloat(second_tri == 0 ? tex_coords1.xy : tex_coords1.zw);
}

uint shadeSample(ivec2 pixel_pos, uint scratch_tri_idx, out float out_depth) {
	float px = float(pixel_pos.x), py = float(pixel_pos.y);

	vec3 depth_eq, edge0_eq, edge1_eq;
	uint instance_id, instance_flags;
	vec2 bary_params;
	getTriangleParams(scratch_tri_idx, depth_eq, bary_params, edge0_eq, edge1_eq, instance_id,
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
		getTriangleVertexTexCoords(scratch_tri_idx, tex0, tex1, tex2);

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
		getTriangleVertexColors(scratch_tri_idx, col0, col1, col2);
		color *= (1.0 - bary[0] - bary[1]) * col0 + (bary[0] * col1 + bary[1] * col2);
	}

	if(color.a == 0.0)
		return 0;

	vec3 normal;
	if((instance_flags & INST_HAS_VERTEX_NORMALS) != 0) {
		vec3 nrm0, nrm1, nrm2;
		getTriangleVertexNormals(scratch_tri_idx, nrm0, nrm1, nrm2);
		nrm1 -= nrm0;
		nrm2 -= nrm0;
		normal = bary[0] * nrm1 + (bary[1] * nrm2 + nrm0);
	} else {
		normal = decodeNormalUint(UINT_SCRATCH());
	}

	float light_value = max(0.0, dot(-lighting.sun_dir, normal) * 0.7 + 0.3);
	color.rgb = SATURATE(finalShading(color.rgb, light_value));
	return encodeRGBA8(color);
}