// $$include funcs lighting frustum viewport data
// clang-format off

#define LIX gl_LocalInvocationIndex
#define LID gl_LocalInvocationID
#define LSIZE (TILE_SIZE * TILE_SIZE)

#define XBLOCKS_PER_BIN  ( BIN_SIZE / BLOCK_SIZE)
#define XBLOCKS_PER_TILE (TILE_SIZE / BLOCK_SIZE)

// TODO: invalid pixels visible on dragon (white dots)
// TODO: robust rasterization should decrease invalid pixels even further

// TODO: problem with uneven amounts of data in different tiles
// TODO: jak dobrze obsługiwać różnego rodzaju dystrybucje trójkątów ?
// Raczej na pewno chcemy rzucić więcej wątków na bin ?

// TODO: don't we want to resolve multi-samples here ?
#if MSAA_SAMPLES > 1
#define DEPTH_BUFFER_SAMPLER sampler2DMS
#else
#define DEPTH_BUFFER_SAMPLER sampler2D
#endif

uniform float fog_multiplier;

layout(local_size_x = TILE_SIZE, local_size_y = TILE_SIZE) in;
//layout(binding = 0) uniform DEPTH_BUFFER_SAMPLER depth_buffer;

layout(std430, binding = 0) readonly buffer buf0_ { InstanceData g_instances[]; };
layout(std430, binding = 1) readonly buffer buf1_ { uint g_quad_indices[]; };
layout(std430, binding = 2) readonly buffer buf2_ { float g_verts[]; };
layout(std430, binding = 3) readonly buffer buf3_ { vec2 g_tex_coords[]; };
layout(std430, binding = 4) readonly buffer buf4_ { uint g_colors[]; };
layout(std430, binding = 5) readonly buffer buf5_ { uint g_normals[]; };

TILE_COUNTERS_BUFFER(6);
layout(std430, binding = 7) buffer buf7_ { uint g_block_counts[]; }; // TODO: 16-bits?
layout(std430, binding = 8) buffer buf8_ { uint g_block_offsets[]; };

layout(std430, binding = 9) readonly buffer buf9_ { uint g_tile_tris[]; };
layout(std430, binding = 10) readonly buffer buf10_ { uint g_block_tris[]; };
layout(std430, binding = 11) readonly buffer buf11_ { vec4 g_uv_rects[]; };
layout(std430, binding = 12) writeonly buffer buf12_ { uint g_raster_image[]; };

shared int s_tile_tri_counts [TILES_PER_BIN];
shared int s_tile_tri_offsets[TILES_PER_BIN];
shared int s_tile_raster_offsets[TILES_PER_BIN];
shared int s_tile_tri_count, s_tile_tri_offset, s_tile_raster_offset;

shared int s_block_tri_counts [BLOCKS_PER_TILE];
shared int s_block_tri_offsets[BLOCKS_PER_TILE];
shared int s_block_max_tri_count;

shared uint s_invalid_block_mask;
shared uint s_fragment_count;
shared uint s_total_fragment_count, s_max_fragment_count;
shared uint s_max_fragment_count_per_pixel;

layout(binding = 0) uniform sampler2D opaque_texture;
layout(binding = 1) uniform sampler2D transparent_texture;

// TODO: separate opaque and transparent objects, draw opaque objects first to texture
// then read it and use depth to optimize drawing
uniform uint background_color;

void outputPixel(ivec2 pixel_pos, uint color) {
	g_raster_image[s_tile_raster_offset + pixel_pos.x + (pixel_pos.y << BIN_SHIFT)] = color;
}

// TODO: better name
// TODO: unrolled filter is a bit faster
#define DEPTH_FILTER_SIZE 6

vec3 result;
float min_depth;
float old_depths[DEPTH_FILTER_SIZE + 1];
uint old_colors[DEPTH_FILTER_SIZE];

void resetPixel() {
	for(int i = 0; i <= DEPTH_FILTER_SIZE; i++)
		old_depths[i] = 1000000000.0;
	result = decodeRGB8(background_color);
	min_depth = viewport.near_plane;
}

void markPixelInvalid() {
	old_depths[0] = -100000000.0;
	min_depth = 1000000000.0;
	result = vec3(1.0, 0.0, 0.0);
	atomicAdd(g_tiles.num_invalid_pixels, 1);
	atomicOr(s_invalid_block_mask, 1 << LID.y);
}

void rasterPixel(vec4 color, float cur_d) {
	if(cur_d < min_depth)
		return;

	uint icolor = encodeRGBA8(color);
	result = result * (1.0 - color.a) + color.rgb * color.a;
	
	if(cur_d > old_depths[0]) {
		vec4 old_color = decodeRGBA8(old_colors[0]);
		result += (old_color.rgb - color.rgb) * color.a * old_color.a;
		SWAP_FLOAT(cur_d, old_depths[0]);
		SWAP_UINT(icolor, old_colors[0]);
		float mul = 1.0 - old_color.a;

		// Note: it should be automatically unrolled, but older version is a bit faster...
		for(int i = 1; i < DEPTH_FILTER_SIZE; i++) {
			if(old_depths[i - 1] <= old_depths[i])
				break; // TODO: expected

			old_color = decodeRGBA8(old_colors[i]);
			result += (old_color.rgb - color.rgb) * color.a * old_color.a;
			SWAP_FLOAT(old_depths[i - 1], old_depths[i]);
			SWAP_UINT(old_colors[i - 1], old_colors[i]);
			mul *= 1.0 - old_color.a;
		}

		if(old_depths[DEPTH_FILTER_SIZE - 1] > old_depths[DEPTH_FILTER_SIZE]) {
			markPixelInvalid();
			return;
		}
	}

	// TODO: is this costly ?
	old_depths[DEPTH_FILTER_SIZE] = old_depths[DEPTH_FILTER_SIZE - 1];
	for(int i = DEPTH_FILTER_SIZE - 1; i > 0; i--) {
		old_depths[i] = old_depths[i - 1];
		old_colors[i] = old_colors[i - 1];
	}
	old_depths[0] = cur_d;
	old_colors[0] = icolor;
}

// TODO: try different values on different platforms
#define MAX_MASK_GROUPS 4
#define MAX_MASKS (BLOCKS_PER_TILE * MAX_MASK_GROUPS)

shared uint s_masks[BLOCKS_PER_TILE * MAX_MASKS]; // TODO: 16-bit
shared uint s_vertex_ids[BLOCKS_PER_TILE * MAX_MASKS][3];
shared uint s_instance_ids[BLOCKS_PER_TILE * MAX_MASKS];

// On scene _ takes ~% time of whole final raster:
// Bunny:  34%
// Dragon: 30%
// Sponza: 75%
// San Miguel: 65%
// Hairball: 29%
// Power plant: 47%
// White Oak: 75%
// Teapot: 47%
void rasterizeTri(const ivec2 pixel_pos, uint tri_idx) {
	vec2 screen_pos = (vec2(pixel_pos) + vec2(0.5, 0.5)) * vec2(1.0 / float(VIEWPORT_SIZE_X), 1.0 / float(VIEWPORT_SIZE_Y));
	vec3 ray_dir = frustum.ws_dir0 + frustum.ws_dirx * (pixel_pos.x + 0.5)
								   + frustum.ws_diry * (pixel_pos.y + 0.5);

	uint v0 = s_vertex_ids[tri_idx][0];
	uint v1 = s_vertex_ids[tri_idx][1];
	uint v2 = s_vertex_ids[tri_idx][2];

	// TODO: subtract ray origin to decreas number of ops ?
	vec3 tri0 = vec3(g_verts[v0 * 3 + 0], g_verts[v0 * 3 + 1], g_verts[v0 * 3 + 2]);
	vec3 tri1 = vec3(g_verts[v1 * 3 + 0], g_verts[v1 * 3 + 1], g_verts[v1 * 3 + 2]);
	vec3 tri2 = vec3(g_verts[v2 * 3 + 0], g_verts[v2 * 3 + 1], g_verts[v2 * 3 + 2]);

	vec3 normal = cross(tri0 - tri2, tri1 - tri0);
	float multiplier = 1.0 / length(normal);
	normal *= multiplier;
	float plane_dist = dot(normal, tri0);

	vec3 edge0 = cross(normal, (tri0 - tri2) * multiplier);
	vec3 edge1 = cross(normal, (tri1 - tri0) * multiplier);

	float ray_pos0 = -(dot(frustum.ws_shared_origin, normal) - plane_dist);
	float ray_pos = ray_pos0 / dot(normal, ray_dir);
	vec3 hitpoint = frustum.ws_shared_origin + ray_pos * ray_dir;

	vec3 diff = hitpoint - tri0;
	float v = dot(edge0, diff);
	float w = dot(edge1, diff);
	vec3 bary = vec3(1.0 - v - w, v, w);

	uint instance_id = s_instance_ids[tri_idx];
	uint instance_flags = g_instances[instance_id].flags;

	// TODO: how to compute derivative?
	vec4 color = decodeRGBA8(g_instances[instance_id].color);
	if((instance_flags & INST_HAS_VERTEX_COLORS) != 0) {
		vec4 col0 = decodeRGBA8(g_colors[v0]);
		vec4 col1 = decodeRGBA8(g_colors[v1]);
		vec4 col2 = decodeRGBA8(g_colors[v2]);
		color *= bary[0] * col0 + bary[1] * col1 + bary[2] * col2;
	}

	if((instance_flags & INST_HAS_TEXTURE) != 0) {
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

		vec2 bary_dx = vec2(dot(edge0, diffx), dot(edge1, diffx));
		vec2 bary_dy = vec2(dot(edge0, diffy), dot(edge1, diffy));

		vec2 tex_coord = bary[1] * tex1 + bary[2] * tex2;
		vec2 tex_dx = (bary_dx.x * tex1 + bary_dx.y * tex2 - tex_coord);
		vec2 tex_dy = bary_dy.x * tex1 + bary_dy.y * tex2 - tex_coord;
		tex_coord += tex0;

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

	if((instance_flags & INST_HAS_VERTEX_NORMALS) != 0) {
		vec3 nrm0 = decodeNormalUint(g_normals[v0]);
		vec3 nrm1 = decodeNormalUint(g_normals[v1]);
		vec3 nrm2 = decodeNormalUint(g_normals[v2]);
		nrm1 -= nrm0; nrm2 -= nrm0;
		normal = nrm0 + bary[1] * nrm1 + bary[2] * nrm2;
	}

	float light_value = max(0.0, dot(-lighting.sun_dir, normal) * 0.7 + 0.3);
	color.rgb = min(finalShading(color.rgb, light_value), vec3(1.0, 1.0, 1.0));
	rasterPixel(color, ray_pos);
}

// Times:      total / only loading & iterating (shading %)
// powerplant: 12.63 /  6.14  (51%)
// san-miguel: 13.43 /  4.66  (65%)
//     dragon:  0.80 /  0.59  (26%)
//     sponza:  7.75 /  1.70  (78%)
//   hairball: 19.56 / 11.39  (41%)
//    gallery:  2.77 /  1.32  (52%)
void rasterizeBlocks(ivec2 tile_pos) {
	int tile_block_idx = int(LID.y);
	const ivec2 block_pos = ivec2(tile_block_idx & 3, tile_block_idx >> 2) * BLOCK_SIZE;
	const ivec2 pixel_pos = ivec2(LID.x & 3, LID.x >> 2) + tile_pos + block_pos;

	int num_block_tris = s_block_tri_counts[tile_block_idx];
	uint block_offset = s_block_tri_offsets[tile_block_idx];
		
	resetPixel();
	int cur_tri_count = 0;

	for(uint i = 0; i < s_block_max_tri_count; i += MAX_MASKS) {
		barrier();

		for(int j = 0; j < MAX_MASK_GROUPS; j++) {
			uint mask_id = BLOCKS_PER_TILE * j + LID.x;
			if(i + mask_id < num_block_tris) {
				uint block_tri = g_block_tris[block_offset + i + mask_id];
				uint mask = block_tri & 0xffff;
				atomicAdd(s_fragment_count, bitCount(mask));
				mask_id += LID.y * MAX_MASKS;
				s_masks[mask_id] = mask;
				uint tri_idx = g_tile_tris[s_tile_tri_offset + ((block_tri >> 16) & 0xffff)];
				uint second_tri = tri_idx >> 31;
				uint verts[4] = { g_quad_indices[tri_idx * 4 + 0], g_quad_indices[tri_idx * 4 + 1],
								  g_quad_indices[tri_idx * 4 + 2], g_quad_indices[tri_idx * 4 + 3] };
				uint instance_id = (verts[0] >> 26) | ((verts[1] >> 20) & 0xfc0) | ((verts[2] >> 14) & 0x3f000);
				s_vertex_ids[mask_id][0] = verts[0] & 0x03ffffff;
				s_vertex_ids[mask_id][1] = verts[1 + second_tri] & 0x03ffffff;
				s_vertex_ids[mask_id][2] = verts[2 + second_tri] & 0x03ffffff;
				s_instance_ids[mask_id] = instance_id;
			}
		}

		barrier();
		if(i >= num_block_tris)
			continue;

		uint num_loaded_tris = min(MAX_MASKS, num_block_tris - i);
		uint mask_bit = 1 << LID.x;
		uint cur_tri = 0;

		while(true) {
			while(cur_tri < num_loaded_tris) {
				uint mask = s_masks[LID.y * MAX_MASKS + cur_tri];
				if((mask & mask_bit) != 0)
					break;
				cur_tri++;
			}
			if(cur_tri >= num_loaded_tris)
				break;

			rasterizeTri(pixel_pos, LID.y * MAX_MASKS + cur_tri);
			cur_tri_count++;
			cur_tri++;
		}
	}
	
	atomicMax(s_max_fragment_count_per_pixel, cur_tri_count);
	//result = vec3(float(cur_tri_count) / 4, float(cur_tri_count) / 32, float(cur_tri_count) / 256);

	result = min(result, vec3(1.0));

	barrier();
	if((s_invalid_block_mask & (1 << LID.y)) != 0)
		result = result * 0.5 + vec3(0.5, 0.0, 0.0);
	uint col = encodeRGB8(result);
	outputPixel(pixel_pos - tile_pos, col);
}

void rasterizeBin(int bin_id) {
	const ivec2 bin_pos = ivec2(bin_id % BIN_COUNT_X, bin_id / BIN_COUNT_X);
	if(LIX < TILES_PER_BIN) {
		s_tile_tri_counts [LIX] = int(TILE_TRI_COUNTS(bin_id, LIX));
		s_tile_tri_offsets[LIX] = int(TILE_TRI_OFFSETS(bin_id, LIX));
		ivec2 tile_pos = ivec2(LIX & 3, LIX >> 2) * TILE_SIZE;
		s_tile_raster_offsets[LIX] = (bin_id << (BIN_SHIFT * 2)) + tile_pos.x + (tile_pos.y << BIN_SHIFT);
	}
	barrier();

	for(int tile_id = 0; tile_id < TILES_PER_BIN; tile_id++) {
		if(LIX == 0) {
			s_tile_raster_offset = s_tile_raster_offsets[tile_id];
			s_block_max_tri_count = 0;
			s_invalid_block_mask = 0;
			s_fragment_count = 0;
		}
		barrier();
		if(LIX < BLOCKS_PER_TILE) {
			uint block_id = (bin_id * TILES_PER_BIN + tile_id) * BLOCKS_PER_TILE + LIX;
			s_block_tri_counts[LIX]  = int(g_block_counts [block_id]);
			s_block_tri_offsets[LIX] = int(g_block_offsets[block_id]);
			atomicMax(s_block_max_tri_count, s_block_tri_counts[LIX]);
			if(LIX == 0) {
				s_tile_tri_offset = s_tile_tri_offsets[tile_id];
				s_tile_tri_count = s_tile_tri_counts[tile_id];
			}
		}
		barrier();
		const ivec2 tile_pos = bin_pos * BIN_SIZE + ivec2(tile_id & 3, tile_id >> 2) * TILE_SIZE;
		rasterizeBlocks(tile_pos);
		barrier();
		if(LIX == 0 && s_invalid_block_mask != 0) {
			atomicAdd(g_tiles.num_invalid_blocks, bitCount(s_invalid_block_mask));
			atomicAdd(g_tiles.num_invalid_tiles, 1);
		}
		if(LIX == 0) {
			atomicAdd(s_total_fragment_count, s_fragment_count);
			atomicMax(s_max_fragment_count, s_fragment_count);
		}
	}
}

shared int s_bin_id;

// Note: iterating over tiles directly is slower (170us vs 84us)
int loadNextBin() {
	if(LIX == 0)
		s_bin_id = int(atomicAdd(g_tiles.final_raster_bin_counter, 1));
	barrier();
	return s_bin_id;
}

void main() {
	if(LIX == 0) {
		s_total_fragment_count = 0;
		s_max_fragment_count = 0;
		s_max_fragment_count_per_pixel = 0;
	}
	int bin_id = loadNextBin();
	while(bin_id < BIN_COUNT) {
		barrier();
		rasterizeBin(bin_id);
		bin_id = loadNextBin();
	}
	if(LIX == 0) {
		atomicAdd(g_tiles.num_fragments, s_total_fragment_count);
		atomicMax(g_tiles.max_fragments_per_tile, s_max_fragment_count);
		atomicMax(g_tiles.max_fragments_per_pixel, s_max_fragment_count_per_pixel);
	}

}
