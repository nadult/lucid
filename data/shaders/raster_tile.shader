// $$include funcs lighting frustum viewport data

#define LIX gl_LocalInvocationIndex
#define LID gl_LocalInvocationID

#define LSIZE 256
#define LSHIFT 8

// - generacja rzędów    512: 0.77  256: 0.5
// - sumowanie pikseli   512: 0.09  256: 0.04
// - generacja sampli    512: 0.56  256: 0.4
// - cieniowanie sampli  512: 3.2   256: 2.23
// - redukcja sampli     512: 0.3   256: 0.19
// 
// total 256: 3.36
// total 512: 4.92
//
// TODO: keep all triangle data in one place (like in rtracer actually?)

#define WORKGROUP_SCRATCH_SIZE	(64 * 1024)
#define MAX_GROUP_TRIS (4 * 1024)
#define MAX_SCRATCH_TRIS (6 * 1024)
#define SCRATCH_TRI_OFFSET (MAX_GROUP_TRIS * 4)

#define SAMPLES_PER_THREAD 4
#define MAX_SAMPLES (LSIZE * SAMPLES_PER_THREAD)

#define GROUP_SIZE 8

// TODO: when storing rows, store each row independently as a triple: (tri_id, row, row_id)
// this way we can easily bin them into buckets of different sample sizes
//
// TODO: dziwna przycinka na san-miguel (kamera w kierunku kolumn)

layout(local_size_x = LSIZE) in;
layout(binding = 0, r32ui) uniform uimage2D final_raster;

layout(std430, binding = 0) buffer buf0_ { uvec4 g_tri_aabbs[]; };
layout(std430, binding = 1) buffer buf1_ { uint g_quad_indices[]; };

layout(std430, binding = 2) readonly buffer buf2_ { float g_verts[]; };
layout(std430, binding = 3) readonly buffer buf3_ { vec2 g_tex_coords[]; };
layout(std430, binding = 4) readonly buffer buf4_ { uint g_colors[]; };
layout(std430, binding = 5) readonly buffer buf5_ { uint g_normals[]; };

layout(std430, binding = 6) buffer buf6_ { BinCounters g_bins; };
layout(std430, binding = 7) buffer buf7_ { TileCounters g_tiles; };

layout(std430, binding = 8) buffer buf8_  { uint g_tile_tris[]; };
layout(std430, binding = 9) coherent buffer buf9_ { uvec2 g_scratch[]; };

layout(std430, binding = 10) readonly buffer buf10_ { InstanceData g_instances[]; };
layout(std430, binding = 11) readonly buffer buf11_ { vec4 g_uv_rects[]; };

layout(binding = 0) uniform sampler2D opaque_texture;
layout(binding = 1) uniform sampler2D transparent_texture;

shared int s_tile_tri_counts [TILES_PER_BIN];
shared int s_tile_tri_offsets[TILES_PER_BIN];
shared vec3 s_tile_ray_dirs0[TILES_PER_BIN];
shared int s_tile_tri_count, s_tile_tri_offset;
shared ivec2 s_bin_pos, s_tile_pos;
shared vec3 s_tile_ray_dir0;

//  low 16 bits: counts
// high 16 bits: offsets
// TODO: make it 1D array, indexing could be simpler, we could save some regs
shared uint s_pixel_counts[GROUP_SIZE * GROUP_SIZE];
shared int s_group_tri_counts[4], s_group_fragment_counts[4]; // TODO: merge counters

shared int s_buffer_count, s_sample_count;
shared uint s_buffer[MAX_SAMPLES + 1];
shared float s_fbuffer[MAX_SAMPLES + 1];

shared uint s_mini_buffer[32];

uint computeScanlineParams(vec3 tri0, vec3 tri1, vec3 tri2, out vec3 scan_base, out vec3 scan_step) {
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

	float inv_ex[3] = { 1.0 / edges[0].x, 1.0 / edges[1].x, 1.0 / edges[2].x };
	scan_base = -vec3(edges[0].z * inv_ex[0], edges[1].z * inv_ex[1], edges[2].z * inv_ex[2]);
	scan_step = -vec3(edges[0].y * inv_ex[0], edges[1].y * inv_ex[1], edges[2].y * inv_ex[2]);
	return (edges[0].x < 0.0? 1 : 0) | (edges[1].x < 0.0? 2 : 0) | (edges[2].x < 0.0? 4 : 0);
}

// TODO: można by tutaj użyć algorytmu bazującego na liniach
void generateTriGroups(uint tri_idx, vec3 tri0, vec3 tri1, vec3 tri2, int min_by, int max_by) {
	// Inspired by Nanite scanline rasterizer
	vec3 scan_min, scan_max, scan_step;
	{
		float sx = s_tile_pos.x - 0.5f; // TODO: why -0.5? it's correct though
		float sy = s_tile_pos.y + 0 * 4 + 0.5f;

		vec3 scan_base;
		uint sign_mask = computeScanlineParams(tri0, tri1, tri2, scan_base, scan_step);

		vec3 scan = scan_step * sy + scan_base - vec3(sx, sx, sx);
		scan_min = vec3(
				(sign_mask & 1) == 0? scan[0] : -1.0 / 0.0,
				(sign_mask & 2) == 0? scan[1] : -1.0 / 0.0,
				(sign_mask & 4) == 0? scan[2] : -1.0 / 0.0);
		scan_max = vec3(
				(sign_mask & 1) != 0? scan[0] : 1.0 / 0.0,
				(sign_mask & 2) != 0? scan[1] : 1.0 / 0.0,
				(sign_mask & 4) != 0? scan[2] : 1.0 / 0.0);
	}

	// TODO: min/max by
	uint soffset = gl_WorkGroupID.x * WORKGROUP_SCRATCH_SIZE;
	for(int gy = 0; gy < 2; gy++) {
		uint group_ranges[4] = {0, 0, 0, 0};
		int group_counts = 0; // TODO: uint?

		for(int iy = 0; iy < 2; iy++) for(int y = 0; y < 4; y++) {
			float xmin = max(max(scan_min[0], scan_min[1]), max(scan_min[2], 0.0));
			float xmax = min(min(scan_max[0], scan_max[1]), min(scan_max[2], TILE_SIZE));

			scan_min += scan_step;
			scan_max += scan_step;
			
			// TODO: use floor/ceil?
			int imin = int(xmin), imax = int(xmax) - 1;
			if(imin <= imax) {
				int w0 = imin < 8? min(imax, 7) - imin + 1 : 0;
				int w1 = imax > 7? imax - max(imin, 8) + 1 : 0;
				group_counts += w0 + (w1 << 16);

				uint g0 = imin < 8? uint(imin) | (uint(min(imax, 7)) << 3) : 0x7;
				uint g1 = imax > 7? uint(max(imin - 8, 0)) | (uint(imax - 8) << 3) : 0x7;
				group_ranges[0 + iy] |= g0 << (6 * y);
				group_ranges[2 + iy] |= g1 << (6 * y);
			}
			else {
				group_ranges[0 + iy] |= 0x7 << (6 * y);
				group_ranges[2 + iy] |= 0x7 << (6 * y);
			}
		}

		// 0x1c71c7 is equal to 4 empty rows (each row encoded in 2 * 3 bits)
		if((group_counts & 0xffff) != 0) {
			uint roffset = atomicAdd(s_group_tri_counts[gy * 2 + 0], 1);
			if(roffset < MAX_GROUP_TRIS) // TODO
				g_scratch[soffset + roffset] = uvec2(group_ranges[0] | (tri_idx << 24),
													 group_ranges[1] | ((tri_idx & 0xff00) << 16));
			atomicAdd(s_group_fragment_counts[gy * 2 + 0], group_counts & 0xffff);
		}
		if((group_counts & 0xffff0000) != 0) {
			uint roffset = atomicAdd(s_group_tri_counts[gy * 2 + 1], 1);
			if(roffset < MAX_GROUP_TRIS) // TODO
				g_scratch[soffset + roffset + MAX_GROUP_TRIS] = uvec2(group_ranges[2] | (tri_idx << 24),
																	  group_ranges[3] | ((tri_idx & 0xff00) << 16));
			atomicAdd(s_group_fragment_counts[gy * 2 + 1], group_counts >> 16);
		}
		soffset += MAX_GROUP_TRIS * 2;
	}
}

// TODO: problem: we cannot use tile triangle indices, because there would be too many...
// we have to remap triangles to those which are actually used in currently rasterized tile segment
// We would have to store those indices (32-bit) in scratch probably
//
// TODO: don't store triangles which generate very small number of samples in scratch,
// instead precompute them directly when sampling; We would have to somehow group those triangles together
void storeTriangle(uint local_tri_idx, vec3 tri0, vec3 tri1, vec3 tri2, uint v0, uint v1, uint v2, uint instance_id)
{
	if(local_tri_idx >= MAX_SCRATCH_TRIS)
		return;
	uint toffset = gl_WorkGroupID.x * WORKGROUP_SCRATCH_SIZE + SCRATCH_TRI_OFFSET + local_tri_idx * 8;

	vec3 normal = cross(tri0 - tri2, tri1 - tri0);
	float multiplier = 1.0 / length(normal);
	normal *= multiplier;

	vec3 edge0 = (tri0 - tri2) * multiplier;
	vec3 edge1 = (tri1 - tri0) * multiplier;
	
	float plane_dist = dot(normal, tri0);
	float param0 = dot(cross(edge0, tri0), normal);
	float param1 = dot(cross(edge1, tri0), normal);

	// Nice optimization for barycentric computations:
	// dot(cross(edge, dir), normal) == dot(dir, cross(normal, edge))
	edge0 = cross(normal, edge0);
	edge1 = cross(normal, edge1);

	g_scratch[toffset + 0] = uvec2(floatBitsToUint(normal.x), floatBitsToUint(normal.y));
	g_scratch[toffset + 1] = uvec2(floatBitsToUint(normal.z), floatBitsToUint(plane_dist));
	g_scratch[toffset + 2] = uvec2(floatBitsToUint(param0), floatBitsToUint(param1));
	g_scratch[toffset + 3] = uvec2(floatBitsToUint(edge0.x), floatBitsToUint(edge0.y));
	g_scratch[toffset + 4] = uvec2(floatBitsToUint(edge0.z), floatBitsToUint(edge1.x));
	g_scratch[toffset + 5] = uvec2(floatBitsToUint(edge1.y), floatBitsToUint(edge1.z));
	g_scratch[toffset + 6] = uvec2(v0, v1);
	g_scratch[toffset + 7] = uvec2(v2, instance_id);
}

void getTriangleParams(uint local_tri_idx, out vec3 normal, out vec3 params, out vec3 edge0, out vec3 edge1, out uint instance_id) {
	uint toffset = gl_WorkGroupID.x * WORKGROUP_SCRATCH_SIZE + SCRATCH_TRI_OFFSET + local_tri_idx * 8;

	{
		uvec2 val0 = g_scratch[toffset + 0], val1 = g_scratch[toffset + 1], val2 = g_scratch[toffset + 2];
		normal = vec3(uintBitsToFloat(val0[0]), uintBitsToFloat(val0[1]), uintBitsToFloat(val1[0]));
		params = vec3(uintBitsToFloat(val1[1]), uintBitsToFloat(val2[0]), uintBitsToFloat(val2[1]));
	}
	{
		uvec2 val0 = g_scratch[toffset + 3], val1 = g_scratch[toffset + 4], val2 = g_scratch[toffset + 5];
		edge0 = vec3(uintBitsToFloat(val0[0]), uintBitsToFloat(val0[1]), uintBitsToFloat(val1[0]));
		edge1 = vec3(uintBitsToFloat(val1[1]), uintBitsToFloat(val2[0]), uintBitsToFloat(val2[1]));
	}

	instance_id = g_scratch[toffset + 7].y;
}

void getTriangleVerts(uint local_tri_idx, out uint v0, out uint v1, out uint v2) {
	uint toffset = gl_WorkGroupID.x * WORKGROUP_SCRATCH_SIZE + SCRATCH_TRI_OFFSET + local_tri_idx * 8;
	uvec2 val0 = g_scratch[toffset + 6];
	v0 = val0[0], v1 = val0[1], v2 = g_scratch[toffset + 7].x;
}

void generateRows() {
	// TODO: optimization: in many cases all rows may very well fit in SMEM,
	// maybe it would be worth it not to use scratch at all then?
	//
	// TODO: check if using SMEM var here makes sense
	for(uint i = 0; i < s_tile_tri_count; i += LSIZE) {
		uint tile_tri_idx = i + LIX;
		if(tile_tri_idx < s_tile_tri_count) {
			uint tri_idx = g_tile_tris[s_tile_tri_offset + tile_tri_idx];
			uint second_tri = tri_idx >> 31;
			uint masked_idx = tri_idx & 0x7fffffff;
			
			uvec4 aabb = g_tri_aabbs[masked_idx];
			aabb = decodeAABB(second_tri != 0? aabb.zw : aabb.xy);
			int min_by = clamp(int(aabb[1]) - s_tile_pos.y, 0, 15) >> 2;
			int max_by = clamp(int(aabb[3]) - s_tile_pos.y, 0, 15) >> 2;

			uint verts[4] = { g_quad_indices[masked_idx * 4 + 0], g_quad_indices[masked_idx * 4 + 1],
							  g_quad_indices[masked_idx * 4 + 2], g_quad_indices[masked_idx * 4 + 3] };
			uint instance_id = (verts[0] >> 26) | ((verts[1] >> 20) & 0xfc0) | ((verts[2] >> 14) & 0x3f000);
			uint v0 = verts[0] & 0x03ffffff;
			uint v1 = verts[1 + second_tri] & 0x03ffffff;
			uint v2 = verts[2 + second_tri] & 0x03ffffff;

			vec3 tri0 = vec3(g_verts[v0 * 3 + 0], g_verts[v0 * 3 + 1], g_verts[v0 * 3 + 2]) - frustum.ws_shared_origin;
			vec3 tri1 = vec3(g_verts[v1 * 3 + 0], g_verts[v1 * 3 + 1], g_verts[v1 * 3 + 2]) - frustum.ws_shared_origin;
			vec3 tri2 = vec3(g_verts[v2 * 3 + 0], g_verts[v2 * 3 + 1], g_verts[v2 * 3 + 2]) - frustum.ws_shared_origin;
			
			// TODO: store only if samples were generated
			// TODO: do triangle storing later
			storeTriangle(tile_tri_idx, tri0, tri1, tri2, v0, v1, v2, instance_id);
			generateTriGroups(tile_tri_idx, tri0, tri1, tri2, min_by, max_by);
		}
	}
}

// TODO: optimize
void loadGroupSamples(uint group_id) {
	uint soffset = gl_WorkGroupID.x * WORKGROUP_SCRATCH_SIZE + group_id * MAX_GROUP_TRIS;
	ivec2 group_pos = ivec2((group_id & 1) << 3, (group_id & 2) << 2);
	uint y = LIX & 7, shift = (y & 3) * 6;
	uint tri_count = s_group_tri_counts[group_id];
	uint group_value = ((group_pos.y + y) << 4) | group_pos.x;

	for(uint i = LIX >> 3; i < tri_count; i += LSIZE / 8) {
		uvec2 group = g_scratch[soffset + i];
		int row_range = int(((y > 3? group.y : group.x) >> shift) & 0x3f);
		if(row_range == 0x7)
			continue;

		uint tile_tri_idx = (group.x >> 24) | ((group.y & 0xff000000) >> 16);
		int minx = row_range & 0x7, maxx = (row_range >> 3) & 0x7;
		int num_samples = maxx - minx + 1;
		uint sample_value = (group_value + minx) | (tile_tri_idx << 16);
		// Note: we're assuming that all samples will fit in s_buffer
		int sample_offset = atomicAdd(s_sample_count, num_samples);
		for(int j = 0; j < num_samples; j++) {
			atomicAdd(s_pixel_counts[y * 8 + minx + j], 1);
			s_buffer[sample_offset++] = sample_value++;
		}
	}
}

void computePixelOffsets()
{
#ifdef VENDOR_NVIDIA
	if(LIX < GROUP_SIZE * GROUP_SIZE) {
		uint value = s_pixel_counts[LIX], temp;
		value += (value << 16);
		temp = shuffleUpNV(value,  1, 32); if((LIX & 31) >=  1) value += temp & 0xffff0000;
		temp = shuffleUpNV(value,  2, 32); if((LIX & 31) >=  2) value += temp & 0xffff0000;
		temp = shuffleUpNV(value,  4, 32); if((LIX & 31) >=  4) value += temp & 0xffff0000;
		temp = shuffleUpNV(value,  8, 32); if((LIX & 31) >=  8) value += temp & 0xffff0000;
		temp = shuffleUpNV(value, 16, 32); if((LIX & 31) >= 16) value += temp & 0xffff0000;
		s_pixel_counts[LIX] = value;
	}
	barrier();
	if(LIX < 32) {
		uint half_offset = s_pixel_counts[31] & 0xffff0000;
		uint value0 = s_pixel_counts[LIX];
		uint value1 = s_pixel_counts[LIX + 32];
		s_pixel_counts[LIX     ] = value0 - (value0 << 16);
		s_pixel_counts[LIX + 32] = value1 - (value1 << 16) + half_offset;
	}
#else
//#error write me please
#endif
}

// Shading 2 samples at once didn't help:
// - decreased computation cost is not worth it because of increased register pressure
// - it seems that it does not help at all with loading vertex attribs; it makes sense:
//   if they are in the cache then it's not a problem...
//
// Can we improve speed of loading vertex data?
void shadeSample(ivec2 tile_pixel_pos, uint local_tri_idx, out uint out_color, out float out_depth)
{
	vec3 ray_dir = s_tile_ray_dir0 + frustum.ws_dirx * tile_pixel_pos.x
								   + frustum.ws_diry * tile_pixel_pos.y;

	vec3 normal, params, edge0, edge1;
	uint instance_id, instance_flags;
	getTriangleParams(local_tri_idx, normal, params, edge0, edge1, instance_id);
	instance_flags = g_instances[instance_id].flags;

	float ray_pos = params[0] / dot(normal, ray_dir);
	vec2 bary = vec2(dot(edge0, ray_dir), dot(edge1, ray_dir)) * ray_pos;

	vec2 bary_dx, bary_dy;
	if((instance_flags & INST_HAS_TEXTURE) != 0) {
		vec3 ray_dirx = ray_dir + frustum.ws_dirx;
		vec3 ray_diry = ray_dir + frustum.ws_diry;

		float ray_posx = params[0] / dot(normal, ray_dirx);
		float ray_posy = params[0] / dot(normal, ray_diry);

		bary_dx = vec2(dot(edge0, ray_dirx), dot(edge1, ray_dirx)) * ray_posx - bary;
		bary_dy = vec2(dot(edge0, ray_diry), dot(edge1, ray_diry)) * ray_posy - bary;
	}
	bary -= vec2(params[1], params[2]);
	// params, edge0 & edge1 no longer needed!

	uint v0, v1, v2;
	if((instance_flags & (INST_HAS_VERTEX_COLORS | INST_HAS_TEXTURE | INST_HAS_VERTEX_NORMALS)) != 0)
		getTriangleVerts(local_tri_idx, v0, v1, v2);

	// 0.5ms on Sponza!!!
	vec4 color = decodeRGBA8(g_instances[instance_id].color);
	if((instance_flags & INST_HAS_VERTEX_COLORS) != 0) {
		vec4 col0 = decodeRGBA8(g_colors[v0]);
		vec4 col1 = decodeRGBA8(g_colors[v1]);
		vec4 col2 = decodeRGBA8(g_colors[v2]);
		color *= (1.0 - bary[0] - bary[1]) * col0 + bary[0] * col1 + bary[1] * col2;
	}

	// ~2ms on Sponza (mostly loading g_ data)
	if((instance_flags & INST_HAS_TEXTURE) != 0) {
		vec2 tex0 = g_tex_coords[v0];
		vec2 tex1 = g_tex_coords[v1];
		vec2 tex2 = g_tex_coords[v2];
		tex1 -= tex0, tex2 -= tex0;

		vec2 tex_coord = tex0 + bary[0] * tex1 + bary[1] * tex2;
		vec2 tex_dx = bary_dx[0] * tex1 + bary_dx[1] * tex2;
		vec2 tex_dy = bary_dy[0] * tex1 + bary_dy[1] * tex2;

		// 0.8ms to load uv_rect on Sponza
		if((instance_flags & INST_HAS_UV_RECT) != 0) {
			vec4 uv_rect = g_uv_rects[instance_id];
			tex_coord = uv_rect.xy + uv_rect.zw * fract(tex_coord);
			tex_dx *= uv_rect.zw, tex_dy *= uv_rect.zw;
		}

		if((instance_flags & INST_TEX_OPAQUE) != 0)
			color.xyz *= textureGrad(opaque_texture, tex_coord, tex_dx, tex_dy).xyz;
		else
			color *= textureGrad(transparent_texture, tex_coord, tex_dx, tex_dy);
	}

	// 0.75 ms on Sponza
	if((instance_flags & INST_HAS_VERTEX_NORMALS) != 0) {
		vec3 nrm0 = decodeNormalUint(g_normals[v0]);
		vec3 nrm1 = decodeNormalUint(g_normals[v1]);
		vec3 nrm2 = decodeNormalUint(g_normals[v2]);
		nrm1 -= nrm0; nrm2 -= nrm0;
		normal = nrm0 + bary[0] * nrm1 + bary[1] * nrm2;
	}

	float light_value = max(0.0, dot(-lighting.sun_dir, normal) * 0.7 + 0.3);
	color.rgb = min(finalShading(color.rgb, light_value), vec3(1.0, 1.0, 1.0));

	out_depth = ray_pos;
	out_color = encodeRGBA8(color);
}

void reduceGroupSamples(int group_id)
{
	// TODO: optimize
	if(LIX < GROUP_SIZE * GROUP_SIZE) {
		ivec2 group_pos = s_tile_pos + ivec2((group_id & 1) << 3, (group_id & 2) << 2);

		uint pixel_counter = s_pixel_counts[LIX];
		int num_samples = int(pixel_counter & 0xffff);
		int sample_offset = int(pixel_counter >> 16) - num_samples;
		
		uint enc_color = 0;
		float depth = 1.0 / 0.0;
		for(int i = 0; i < num_samples; i++) {
			float sdepth = s_fbuffer[sample_offset + i];
			if(sdepth < depth) {
				depth = sdepth;
				enc_color = s_buffer[sample_offset + i];
			}
		}

		//uint enc_color = encodeRGBA8(vec4(vec3(float(sample_offset) / MAX_SAMPLES), 1.0));
		imageStore(final_raster, group_pos + ivec2(LIX & 7, LIX >> 3), uvec4(enc_color, 0, 0, 0));
	}
}


void rasterPixelCounts(int group_id)
{
	if(LIX < GROUP_SIZE * GROUP_SIZE) {
		ivec2 group_pos = s_tile_pos + ivec2((group_id & 1) << 3, (group_id & 2) << 2);
		ivec2 pixel_pos = ivec2(LIX & 7, LIX >> 3);


		uint group_tri_count = s_group_tri_counts[group_id];
		uint group_frag_count = s_group_fragment_counts[group_id];
		//vec4 color = vec4(float(group_tri_count) / 64.0, float(group_frag_count) / 1024.0, 0.0, 1.0);

		uint pix_count  = s_pixel_counts[LIX] & 0xffff;
		uint pix_offset = s_pixel_counts[LIX] >> 16;
		vec4 color = vec4(float(pix_count) / 1024.0, float(pix_count) / 16.0, float(pix_count) / 128.0, pix_count == 0? 0.0 : 1.0);
		//vec4 color = vec4(float(pix_count) / 255.0, float(pix_count) / 63.0, float(pix_count) / 255.0, pix_count == 0? 0.0 : 1.0);
		//vec4 color = vec4(vec3(float(pix_offset) / MAX_SAMPLES), pix_offset == 0? 0.0 : 1.0);
		//if(pix_offset >= MAX_SAMPLES)
		//	color.rgb = vec3(1.0, 0.0, 0.0);

		color = min(color, vec4(1.0));
		uint enc_col = encodeRGBA8(color);
		imageStore(final_raster, group_pos + pixel_pos, uvec4(enc_col, 0, 0, 0));
	}
}

void rasterInvalidTile(vec3 color)
{
	for(uint i = LIX; i < TILE_SIZE * TILE_SIZE; i += LSIZE) {
		ivec2 pixel_pos = ivec2(i & (TILE_SIZE - 1), i >> TILE_SHIFT);
		vec4 color = vec4(color, 1.0);
		uint enc_col = encodeRGBA8(color);
		imageStore(final_raster, s_tile_pos + pixel_pos, uvec4(enc_col, 0, 0, 0));
	}
}

void rasterInvalidGroup(int gid, vec3 color)
{
	ivec2 group_pos = s_tile_pos + ivec2((gid & 1) << 3, (gid & 2) << 2);
	uint enc_col = encodeRGBA8(vec4(color, 1.0));

	for(uint i = LIX; i < 8 * 8; i += LSIZE) {
		ivec2 pixel_pos = ivec2(i & 7, i >> 3);
		imageStore(final_raster, group_pos + pixel_pos, uvec4(enc_col, 0, 0, 0));
	}
}

void rasterBins(int bin_id) {
	if(LIX < TILES_PER_BIN) {
		s_tile_tri_counts [LIX] = int(g_tiles.tile_tri_counts[bin_id][LIX]);
		s_tile_tri_offsets[LIX] = int(g_tiles.tile_tri_offsets[bin_id][LIX]);
		ivec2 bin_pos = ivec2(bin_id % BIN_COUNT_X, bin_id / BIN_COUNT_X) * BIN_SIZE;
		ivec2 tile_pos = bin_pos + ivec2(LIX & 3, LIX >> 2) * TILE_SIZE;
		s_tile_ray_dirs0[LIX] = frustum.ws_dir0 + frustum.ws_dirx * (tile_pos.x + 0.5)
												+ frustum.ws_diry * (tile_pos.y + 0.5);
		if(LIX == 0)
			s_bin_pos = bin_pos;
	}
	barrier();

	for(int tile_id = 0; tile_id < TILES_PER_BIN; tile_id++) {
		barrier();
		if(LIX < GROUP_SIZE * GROUP_SIZE) {
			s_pixel_counts[LIX] = 0;
			if(LIX < 4) {
				if(LIX == 0) {
					s_tile_tri_count  = s_tile_tri_counts[tile_id];
					s_tile_tri_offset = s_tile_tri_offsets[tile_id];
					s_tile_pos = s_bin_pos + ivec2(tile_id & 3, tile_id >> 2) * TILE_SIZE;
					s_tile_ray_dir0 = s_tile_ray_dirs0[tile_id];
					s_sample_count = 0;
				}
				s_group_tri_counts[LIX] = 0;
				s_group_fragment_counts[LIX] = 0;
			}
		}
		barrier();
		generateRows();
		groupMemoryBarrier();
		barrier();
		
		if(s_tile_tri_count > MAX_SCRATCH_TRIS) {
			rasterInvalidTile(vec3(1.0, 0.0, 0.5));
			continue;
		}
		
		rasterInvalidTile(vec3(0.5, 0.5, 0.5));
		continue;

		for(int gid = 0; gid < 4; gid++) {
			uint sample_count = s_group_fragment_counts[gid];
			if(sample_count > MAX_SAMPLES) {
				rasterInvalidGroup(gid, vec3(1.0, 0.0, 0.0));
				continue;
			}

			loadGroupSamples(gid);
			barrier();
			computePixelOffsets();
			// Selecting samples
			uint samples[SAMPLES_PER_THREAD];
			for(int i = 0; i < SAMPLES_PER_THREAD; i++) {
				uint sample_idx = LSIZE * i + LIX;
				samples[i] = sample_idx < s_sample_count? s_buffer[sample_idx] : ~0u;
			}
			barrier();

			// Shading samples & storing them ordered by pixel pos
			for(int i = 0; i < SAMPLES_PER_THREAD; i++)
				if(samples[i] != ~0u) {
					ivec2 tile_pixel_pos = ivec2(samples[i] & 15, (samples[i] >> 4) & 15);
					uint tri_idx = samples[i] >> 16;
					int pixel_id = ((tile_pixel_pos.y & 7) << 3) + (tile_pixel_pos.x & 7);

					uint sample_color;
					float sample_depth;
					shadeSample(tile_pixel_pos, tri_idx, sample_color, sample_depth);
					uint sample_idx = atomicAdd(s_pixel_counts[pixel_id], 0x10000) >> 16;
					if(sample_idx < MAX_SAMPLES) {
						s_buffer[sample_idx] = sample_color;
						s_fbuffer[sample_idx] = sample_depth;
					}
				}

			barrier();
			reduceGroupSamples(gid);
			barrier();
			if(LIX < GROUP_SIZE * GROUP_SIZE) {
				s_pixel_counts[LIX] = 0;
				s_sample_count = 0;
			}
			barrier();
		}
	}
}

shared int s_num_bins, s_bin_id;

int loadNextBin() {
	if(LIX == 0) {
		uint bin_idx = atomicAdd(g_tiles.medium_bin_counter, 1);
		s_bin_id = bin_idx < s_num_bins? g_bins.medium_bins[bin_idx] : -1;
	}
	barrier();
	return s_bin_id;
}

void main() {
	if(LIX == 0)
		s_num_bins = g_bins.num_medium_bins;
	int bin_id = loadNextBin();
	while(bin_id != -1) {
		barrier();
		rasterBins(bin_id);
		bin_id = loadNextBin();
	}
}
