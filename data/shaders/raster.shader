// $$include funcs lighting frustum viewport data

#define LIX gl_LocalInvocationIndex
#define LID gl_LocalInvocationID

#define LSIZE 512

// TODO: when storing rows, store each row independently as a triple: (tri_id, row, row_id)
// this way we can easily bin them into buckets of different sample sizes

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

layout(binding = 0) uniform sampler2D opaque_texture;
layout(binding = 1) uniform sampler2D transparent_texture;

// TODO: enforce this somehow
// TODO: scratch too big?
// TODO: stratedy for large number of rows
//       first divide rows in 4 different sets
//       then divide each row in 4 blocks if necessary
#define MAX_TRIS (LSIZE / 2)
#define MAX_SAMPLES (LSIZE * 4)
#define MAX_ROW_TRIS (LSIZE * 4)
#define SAMPLES_PER_THREAD 4

shared int s_tile_tri_counts [TILES_PER_BIN];
shared int s_tile_tri_offsets[TILES_PER_BIN];
shared int s_tile_tri_count, s_tile_tri_offset;
shared ivec2 s_bin_pos, s_tile_pos;

// TODO: 16 bit
//  low 16 bits: counts
// high 16 bits: offsets
shared uint s_pixel_counts[TILE_SIZE][TILE_SIZE];

shared int s_buffer_count, s_tile_rowtri_count, s_tile_sample_count;
shared uint s_buffer[LSIZE * 4 + 1];
shared float s_fbuffer[LSIZE * 4 + 1];

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
void generateTriScanlines(uint local_tri_idx, vec3 tri0, vec3 tri1, vec3 tri2, int min_by, int max_by) {
	// Inspired by Nanite scanline rasterizer
	vec3 scan_min, scan_max, scan_step;
	{
		float sx = s_tile_pos.x - 0.5f; // TODO: why -0.5? it's correct though
		float sy = s_tile_pos.y + min_by * 4 + 0.5f;

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

	uint soffset = gl_WorkGroupID.x * MAX_ROW_TRIS;
	for(int by = min_by; by <= max_by; by++) {
		uint row_ranges = 0;

		for(int y = 0; y < 4; y++) {
			float xmin = max(max(scan_min[0], scan_min[1]), max(scan_min[2], 0.0));
			float xmax = min(min(scan_max[0], scan_max[1]), min(scan_max[2], TILE_SIZE));

			scan_min += scan_step;
			scan_max += scan_step;
			
			// TODO: use floor/ceil?
			int imin = int(xmin), imax = int(xmax) - 1;
			uint shift = (y & 3) * 8;
			uint enc_value = imin <= imax? uint(imin) | (uint(imax) << 4) : 0x0f;
			if(imin <= imax) {
				atomicAdd(s_pixel_counts[by * 4 + y][imin], 1);
				if(imax < 15)
					atomicAdd(s_pixel_counts[by * 4 + y][imax + 1], -1);

				int num_samples = imax - imin + 1;
				int sample_offset = atomicAdd(s_tile_sample_count, num_samples);
				uint sample_value = (local_tri_idx << 16) | ((by * 4 + y) << 4) | imin;
				num_samples = min(num_samples, MAX_SAMPLES - sample_offset);

				// TODO: this is slow
				for(int j = 0; j < num_samples; j++)
					s_buffer[sample_offset++] = sample_value++;
			}
			row_ranges |= enc_value << ((y & 3) << 3);
		}

		/*
		if(row_ranges != 0x0f0f0f0f) {
			uint roffset = atomicAdd(s_tile_rowtri_count, 1);
			if(roffset < MAX_ROW_TRIS) // TODO: handle this properly
				g_scratch[soffset + roffset] = uvec2(row_ranges, local_tri_idx | uint(by << 20));
		}*/
	}
}

void generateTileMasks() {
	// TODO: check if using SMEM var here makes sense
	for(uint i = 0; i < s_tile_tri_count; i += LSIZE) {
		uint local_tri_idx = i + LIX;
		if(local_tri_idx < s_tile_tri_count) {
			uint tri_idx = g_tile_tris[s_tile_tri_offset + local_tri_idx];

			bool second_tri = (tri_idx & 0x80000000) != 0;
			tri_idx &= 0x7fffffff;
			
			uvec4 aabb = g_tri_aabbs[tri_idx];
			aabb = decodeAABB(second_tri? aabb.zw : aabb.xy);
			int min_by = clamp(int(aabb[1]) - s_tile_pos.y, 0, 15) >> 2;
			int max_by = clamp(int(aabb[3]) - s_tile_pos.y, 0, 15) >> 2;

			uint v0 = g_quad_indices[tri_idx * 4 + 0] & 0x03ffffff;
			uint v1 = g_quad_indices[tri_idx * 4 + (second_tri? 2 : 1)] & 0x03ffffff;
			uint v2 = g_quad_indices[tri_idx * 4 + (second_tri? 3 : 2)] & 0x03ffffff;

			vec3 tri0 = vec3(g_verts[v0 * 3 + 0], g_verts[v0 * 3 + 1], g_verts[v0 * 3 + 2]) - frustum.ws_shared_origin;
			vec3 tri1 = vec3(g_verts[v1 * 3 + 0], g_verts[v1 * 3 + 1], g_verts[v1 * 3 + 2]) - frustum.ws_shared_origin;
			vec3 tri2 = vec3(g_verts[v2 * 3 + 0], g_verts[v2 * 3 + 1], g_verts[v2 * 3 + 2]) - frustum.ws_shared_origin;
			generateTriScanlines(local_tri_idx, tri0, tri1, tri2, min_by, max_by);
		}
	}
}

void sumPixelCounts()
{
#ifdef VENDOR_NVIDIA
	if(LIX < TILE_SIZE * TILE_SIZE) {
		uint col_id = LIX & (TILE_SIZE - 1);
		// Computing initial counts
		uint value = s_pixel_counts[LIX >> TILE_SHIFT][col_id], temp;
		temp = shuffleUpNV(value, 1, 16); if(col_id >= 1) value += temp;
		temp = shuffleUpNV(value, 2, 16); if(col_id >= 2) value += temp;
		temp = shuffleUpNV(value, 4, 16); if(col_id >= 4) value += temp;
		temp = shuffleUpNV(value, 8, 16); if(col_id >= 8) value += temp;

		// Computing pixel offsets per row
		value += (value << 16);
		temp = shuffleUpNV(value, 1, 16); if(col_id >= 1) value += temp & 0xffff0000;
		temp = shuffleUpNV(value, 2, 16); if(col_id >= 2) value += temp & 0xffff0000;
		temp = shuffleUpNV(value, 4, 16); if(col_id >= 4) value += temp & 0xffff0000;
		temp = shuffleUpNV(value, 8, 16); if(col_id >= 8) value += temp & 0xffff0000;
		s_pixel_counts[LIX >> TILE_SHIFT][col_id] = value;
	}
	barrier();
	// Computing pixel row offsets
	if(LIX < TILE_SIZE) {
		uint row_value = s_pixel_counts[LIX][15] >> 16, temp;
		temp = shuffleUpNV(row_value, 1, 16); if(LIX >= 1) row_value += temp;
		temp = shuffleUpNV(row_value, 2, 16); if(LIX >= 2) row_value += temp;
		temp = shuffleUpNV(row_value, 4, 16); if(LIX >= 4) row_value += temp;
		temp = shuffleUpNV(row_value, 8, 16); if(LIX >= 8) row_value += temp;
		s_mini_buffer[LIX] = row_value << 16;
	}
	barrier();
	if(LIX < TILE_SIZE * TILE_SIZE) {
		uint row_id = LIX >> TILE_SHIFT;
		uint col_id = LIX & (TILE_SIZE - 1);
		uint row_offset = row_id == 0? 0 : s_mini_buffer[row_id - 1];
		uint value = s_pixel_counts[row_id][col_id];
		value = value + row_offset - (value << 16);
		s_pixel_counts[row_id][col_id] = value;
	}
#else
#error write me please
#endif
}

uint getNumSamples()
{
	uint value = s_pixel_counts[TILE_SIZE - 1][TILE_SIZE - 1];
	return (value & 0xffff) + (value >> 16);
}

void rasterPixelCounts()
{
	if(LIX < TILE_SIZE * TILE_SIZE) {
		ivec2 pixel_pos = ivec2(LIX & (TILE_SIZE - 1), LIX >> TILE_SHIFT);

		uint pix_count  = s_pixel_counts[pixel_pos.y][pixel_pos.x] & 0xffff;
		uint pix_offset = s_pixel_counts[pixel_pos.y][pixel_pos.x] >> 16;
		pix_offset = getNumSamples();

		//vec4 color = vec4(float(pix_count) / 1024.0, float(pix_count) / 16.0, float(pix_count) / 128.0, pix_count == 0? 0.0 : 1.0);
		//vec4 color = vec4(float(pix_count) / 255.0, float(pix_count) / 63.0, float(pix_count) / 255.0, pix_count == 0? 0.0 : 1.0);
		vec4 color = vec4(vec3(float(pix_offset) / MAX_SAMPLES), pix_offset == 0? 0.0 : 1.0);
		if(pix_offset >= MAX_SAMPLES)
			color.rgb = vec3(1.0, 0.0, 0.0);
		color = min(color, vec4(1.0));
		uint enc_col = encodeRGBA8(color);
		imageStore(final_raster, s_tile_pos + pixel_pos, uvec4(enc_col, 0, 0, 0));
	}
}

void rasterInvalidTile()
{
	if(LIX < TILE_SIZE * TILE_SIZE) {
		ivec2 pixel_pos = ivec2(LIX & (TILE_SIZE - 1), LIX >> TILE_SHIFT);
		vec4 color = vec4(1.0, 0.0, 0.0, 1.0);
		uint enc_col = encodeRGBA8(color);
		imageStore(final_raster, s_tile_pos + pixel_pos, uvec4(enc_col, 0, 0, 0));
	}
}

// TODO: problem: we cannot use tile triangle indices, because there would be too many...
// we have to remap triangles to those which are actually used in currently rasterized tile segment
// We would have to store those indices (32-bit) in scratch probably
void loadTriangles()
{
	for(uint i = LIX; i < s_tile_tri_count; i += LSIZE) {
		uint tri_idx = g_tile_tris[s_tile_tri_offset + i];
		uint second_tri = tri_idx >> 31;
		uint verts[4] = { g_quad_indices[tri_idx * 4 + 0], g_quad_indices[tri_idx * 4 + 1],
						  g_quad_indices[tri_idx * 4 + 2], g_quad_indices[tri_idx * 4 + 3] };
		uint instance_id = (verts[0] >> 26) | ((verts[1] >> 20) & 0xfc0) | ((verts[2] >> 14) & 0x3f000);
		uint v0 = verts[0] & 0x03ffffff;
		uint v1 = verts[1 + second_tri] & 0x03ffffff;
		uint v2 = verts[2 + second_tri] & 0x03ffffff;

		// TODO: subtract ray origin to decreas number of ops ?
		vec3 tri0 = vec3(g_verts[v0 * 3 + 0], g_verts[v0 * 3 + 1], g_verts[v0 * 3 + 2]);
		vec3 tri1 = vec3(g_verts[v1 * 3 + 0], g_verts[v1 * 3 + 1], g_verts[v1 * 3 + 2]);
		vec3 tri2 = vec3(g_verts[v2 * 3 + 0], g_verts[v2 * 3 + 1], g_verts[v2 * 3 + 2]);
		
		vec3 normal = cross(tri0 - tri2, tri1 - tri0);
		float multiplier = 1.0 / length(normal);
		normal *= multiplier;

		vec3 edge0 = (tri0 - tri2) * multiplier;
		vec3 edge1 = (tri1 - tri0) * multiplier;

		s_buffer[MAX_TRIS * 0 + i] = floatBitsToUint(normal.x);
		s_buffer[MAX_TRIS * 1 + i] = floatBitsToUint(normal.y);
		s_buffer[MAX_TRIS * 2 + i] = floatBitsToUint(normal.z);
		
		s_buffer[MAX_TRIS * 3 + i] = floatBitsToUint(tri0.x);
		s_buffer[MAX_TRIS * 4 + i] = floatBitsToUint(tri0.y);
		s_buffer[MAX_TRIS * 5 + i] = floatBitsToUint(tri0.z);
	
		s_buffer[MAX_TRIS * 6 + i] = v0;
		s_buffer[MAX_TRIS * 7 + i] = v1;
	
		s_fbuffer[MAX_TRIS * 0 + i] = edge0.x;
		s_fbuffer[MAX_TRIS * 1 + i] = edge0.y;
		s_fbuffer[MAX_TRIS * 2 + i] = edge0.z;
		
		s_fbuffer[MAX_TRIS * 3 + i] = edge1.x;
		s_fbuffer[MAX_TRIS * 4 + i] = edge1.y;
		s_fbuffer[MAX_TRIS * 5 + i] = edge1.z;

		s_fbuffer[MAX_TRIS * 6 + i] = v2;
		s_fbuffer[MAX_TRIS * 7 + i] = instance_id;
	}
}

vec3 getTriangleNormal(uint local_tri_idx) {
	return vec3(
		uintBitsToFloat(s_buffer[MAX_TRIS * 0 + local_tri_idx]),
		uintBitsToFloat(s_buffer[MAX_TRIS * 1 + local_tri_idx]),
		uintBitsToFloat(s_buffer[MAX_TRIS * 2 + local_tri_idx]));
}

vec3 getTriangleV0(uint local_tri_idx) {
	return vec3(
		uintBitsToFloat(s_buffer[MAX_TRIS * 3 + local_tri_idx]),
		uintBitsToFloat(s_buffer[MAX_TRIS * 4 + local_tri_idx]),
		uintBitsToFloat(s_buffer[MAX_TRIS * 5 + local_tri_idx]));
}

vec3 getTriangleEdge(uint local_tri_idx, uint edge_idx) {
	uint eoffset = edge_idx == 1? MAX_TRIS * 3 : 0;
	return vec3(
		s_fbuffer[eoffset + MAX_TRIS * 0 + local_tri_idx],
		s_fbuffer[eoffset + MAX_TRIS * 1 + local_tri_idx],
		s_fbuffer[eoffset + MAX_TRIS * 2 + local_tri_idx]);
}

uvec3 getTriangleVerts(uint local_tri_idx) {
	return uvec3(
		s_buffer[MAX_TRIS * 6 + local_tri_idx],
		s_buffer[MAX_TRIS * 7 + local_tri_idx],
		floatBitsToUint(s_fbuffer[MAX_TRIS * 6 + local_tri_idx]));
}

uint getTriangleInstanceId(uint local_tri_idx) {
	return floatBitsToUint(s_fbuffer[MAX_TRIS * 7 + local_tri_idx]);
}

uvec2 shadeSample(uint sample_id)
{
	ivec2 pixel_pos = ivec2(sample_id & 15, (sample_id >> 4) & 15) + s_tile_pos;
	uint local_tri_idx = sample_id >> 16;

	vec3 ray_dir = frustum.ws_dir0 + frustum.ws_dirx * (pixel_pos.x + 0.5)
								   + frustum.ws_diry * (pixel_pos.y + 0.5);

	vec3 tri0   = getTriangleV0(local_tri_idx);
	vec3 normal = getTriangleNormal(local_tri_idx);
	vec3 edge0 = getTriangleEdge(local_tri_idx, 0);
	vec3 edge1 = getTriangleEdge(local_tri_idx, 1);
	float plane_dist = dot(normal, tri0);

	float ray_pos0 = -(dot(frustum.ws_shared_origin, normal) - plane_dist);
	float ray_pos = ray_pos0 / dot(normal, ray_dir);
	vec3 hitpoint = frustum.ws_shared_origin + ray_pos * ray_dir;

	vec3 diff = hitpoint - tri0;
	float v = dot(cross(edge0, diff), normal);
	float w = dot(cross(edge1, diff), normal);
	vec3 bary = vec3(1.0 - v - w, v, w);

	uint instance_id = getTriangleInstanceId(local_tri_idx);
	uint instance_flags = g_instances[instance_id].flags;

	uint v0, v1, v2;
	if((instance_flags & (INST_HAS_VERTEX_COLORS | INST_HAS_TEXTURE | INST_HAS_VERTEX_NORMALS)) != 0) {
		uvec3 verts = getTriangleVerts(local_tri_idx);
		v0 = verts[0], v1 = verts[1], v2 = verts[2];
	}

	// TODO: how to compute derivative?
	vec4 color = decodeRGBA8(g_instances[instance_id].color);
	if((instance_flags & INST_HAS_VERTEX_COLORS) != 0) {
		vec4 col0 = decodeRGBA8(g_colors[v0]);
		vec4 col1 = decodeRGBA8(g_colors[v1]);
		vec4 col2 = decodeRGBA8(g_colors[v2]);
		color *= bary[0] * col0 + bary[1] * col1 + bary[2] * col2;
	}

	// TODO: too many registers used?
	/*if((instance_flags & INST_HAS_TEXTURE) != 0) {
		vec2 tex0 = g_tex_coords[v0];
		vec2 tex1 = g_tex_coords[v1];
		vec2 tex2 = g_tex_coords[v2];
		tex1 -= tex0;
		tex2 -= tex0;

		vec3 ray_dirx = ray_dir + frustum.ws_dirx;
		float ray_posx = ray_pos0 / dot(normal, ray_dirx);
		vec3 hitpointx = frustum.ws_shared_origin + ray_posx * ray_dirx;
		vec3 diffx = hitpointx - tri0;
		
		vec3 ray_diry = ray_dir + frustum.ws_diry;
		float ray_posy = ray_pos0 / dot(normal, ray_diry);
		vec3 hitpointy = frustum.ws_shared_origin + ray_posy * ray_diry;
		vec3 diffy = hitpointy - tri0;

		vec2 bary_dx = vec2(dot(cross(edge0, diffx), normal), dot(cross(edge1, diffx), normal));
		vec2 bary_dy = vec2(dot(cross(edge0, diffy), normal), dot(cross(edge1, diffy), normal));

		vec2 tex_coord = bary[1] * tex1 + bary[2] * tex2;
		vec2 tex_dx = (bary_dx.x * tex1 + bary_dx.y * tex2 - tex_coord);
		vec2 tex_dy = bary_dy.x * tex1 + bary_dy.y * tex2 - tex_coord;
		tex_coord += tex0;

		if((instance_flags & INST_HAS_UV_RECT) != 0) {
			vec2 uv_rect_pos = vec2(g_instances[instance_id].uv_rect[0], g_instances[instance_id].uv_rect[1]);
			vec2 uv_rect_size = vec2(g_instances[instance_id].uv_rect[2], g_instances[instance_id].uv_rect[3]);
			tex_coord = uv_rect_pos + uv_rect_size * fract(tex_coord);
			tex_dx *= uv_rect_size;
			tex_dy *= uv_rect_size;
		}

		if((instance_flags & INST_TEX_OPAQUE) != 0)
			color.xyz *= textureGrad(opaque_texture, tex_coord, tex_dx, tex_dy).xyz;
		else
			color *= textureGrad(transparent_texture, tex_coord, tex_dx, tex_dy);
	}*/

	if((instance_flags & INST_HAS_VERTEX_NORMALS) != 0) {
		vec3 nrm0 = decodeNormalUint(g_normals[v0]);
		vec3 nrm1 = decodeNormalUint(g_normals[v1]);
		vec3 nrm2 = decodeNormalUint(g_normals[v2]);
		nrm1 -= nrm0; nrm2 -= nrm0;
		normal = nrm0 + bary[1] * nrm1 + bary[2] * nrm2;
	}

	float light_value = max(0.0, dot(-lighting.sun_dir, normal) * 0.7 + 0.3);
	color.rgb = min(finalShading(color.rgb, light_value), vec3(1.0, 1.0, 1.0));

	return uvec2(floatBitsToUint(ray_pos), encodeRGBA8(color));
}

void assignSamples_()
{
	// Dla każdego rzędu liczymy ile ma trójkątów
	// prefix sum: w s_buffer mamy ilości 
	// następnie binarnie każdy wątek znajduje sobie sample do rasteryzacji

	uint soffset = gl_WorkGroupID.x * MAX_ROW_TRIS;
	// Computing number of samples for each row-tri (offseted by 1 index)
	for(uint i = LIX; i < s_tile_rowtri_count; i += LSIZE) {
		uint rows = g_scratch[soffset + i].x;
		uint num_pixels = 0;
		for(int j = 0; j < 4; j++) {
			uint minx = rows & 0xf, maxx = (rows >> 4) & 0xf;
			num_pixels += max(0, maxx - minx + 1);
			rows >>= 8;
		}
		s_buffer[i + 1] = num_pixels;
	}
	if(LIX == 0)
		s_buffer[0] = 0;
	barrier();

	// Computing prefix sum of samples for each row-tri
	// TODO: optimize
	for(uint shift = 1; shift < s_tile_rowtri_count; shift *= 4) {
		for(uint i = LIX; i < s_tile_rowtri_count; i += LSIZE) {
			uint v0 = s_buffer[i], v1 = i >= shift? s_buffer[i - shift] : 0;
			uint value = (v0 & 0xffff) + (v1 & 0xffff);
			atomicAnd(s_buffer[i], 0xffff);
			atomicAdd(s_buffer[i], value << 16);
		}
		barrier();
		for(uint i = LIX; i < s_tile_rowtri_count; i += LSIZE) {
			uint v0 = s_buffer[i], v1 = i >= shift? s_buffer[i - shift] : 0;
			uint value = (v0 >> 16) + (v1 >> 16);
			atomicAnd(s_buffer[i], 0xffff);
			atomicAdd(s_buffer[i], value << 16);
		}
		barrier();
	}

	// Assigning samples to threads
	// TODO: fix & optimize (vectorized search)
	uint min_pos = 0, max_pos = s_tile_rowtri_count - 1;
	uint target_value = LIX * 4;
	int steps = 0;
	while(min_pos + 1 < max_pos && steps < 20) {
		uint mid_pos = (max_pos + min_pos) >> 1;
		uint mid_value = s_buffer[mid_pos] & 0xffff;
		if(mid_value <= target_value)
			min_pos = mid_pos;
		else
			max_pos = mid_pos - 1;
		steps++;
	}

	s_buffer[LIX] = min_pos;
}

// TODO: może tile dispatcher też jest zbędny? moglibyśmy od razu iterować po trójkątach z bina?
//       nawet jeśli to jest to pomysł na później jak już wszystko inne będzie szybciej działać
void generateBinMasks(int bin_id) {
	if(LIX < TILES_PER_BIN) {
		s_tile_tri_counts [LIX] = int(g_tiles.tile_tri_counts[bin_id][LIX]);
		s_tile_tri_offsets[LIX] = int(g_tiles.tile_tri_offsets[bin_id][LIX]);
		if(LIX == 0)
			s_bin_pos = ivec2(bin_id % BIN_COUNT_X, bin_id / BIN_COUNT_X) * BIN_SIZE;
	}
	barrier();

	for(int tile_id = 0; tile_id < TILES_PER_BIN; tile_id++) {
		barrier();
		if(LIX < TILE_SIZE * TILE_SIZE) {
			if(LIX == 0) {
				s_tile_tri_count  = s_tile_tri_counts[tile_id];
				s_tile_tri_offset = s_tile_tri_offsets[tile_id];
				s_tile_pos = s_bin_pos + ivec2(tile_id & 3, tile_id >> 2) * TILE_SIZE;
				s_tile_rowtri_count = 0;
				s_tile_sample_count = 0;
			}
			s_pixel_counts[LIX >> TILE_SHIFT][LIX & (TILE_SIZE - 1)] = 0;
		}
		barrier();
		generateTileMasks();
		barrier();
		if(s_tile_sample_count > MAX_ROW_TRIS || s_tile_tri_count > MAX_TRIS) {
			rasterInvalidTile();
			continue;
		}
		sumPixelCounts();
		barrier();

		uint samples[4];
		for(int i = 0; i < 4; i++) {
			uint sample_id = LIX * 4 + i;
			samples[i] = sample_id < s_tile_sample_count? s_buffer[i] : ~0u;
		}
		barrier();
		loadTriangles();
		barrier();

		for(int i = 0; i < 4; i++) {
			if(samples[i] == ~0u)
				break;
			uvec2 value = shadeSample(samples[i]);
			s_fbuffer[LIX * 4 + i] = uintBitsToFloat(value.x);
			s_buffer[LIX * 4 + i] = value.y;
		}
		
		rasterPixelCounts();
	}
}

shared int s_bin_id;

// TODO: some bins require a lot more computation than others
int loadNextBin() {
	if(LIX == 0) {
		s_bin_id = int(atomicAdd(g_tiles.mask_raster_bin_counter, 1));
	}
	barrier();
	return s_bin_id;
}

void main() {
	// TODO: remove this variable
	int bin_id = loadNextBin();
	while(bin_id < BIN_COUNT) {
		barrier();
		generateBinMasks(bin_id);
		bin_id = loadNextBin();
	}
}
