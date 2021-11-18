// $$include funcs frustum viewport data

#define LIX gl_LocalInvocationIndex
#define LID gl_LocalInvocationID

#define LSIZE 1024

// TODO: problem with uneven amounts of data in different tiles
// TODO: jak dobrze obsługiwać różnego rodzaju dystrybucje trójkątów ?
//
// TODO: replace 16 to BLOCKS_PER_TILE ? it's not that simple...
//
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

// TODO: enforce this somehow
// TODO: scratch too big?
// TODO: stratedy for large number of rows
//       first divide rows in 4 different sets
//       then divide each row in 4 blocks if necessary
#define MAX_TILE_TRIS 64 * 1024
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

shared int s_buffer_count, s_tile_rowtri_count;
shared uint s_buffer[LSIZE * 4 + 1];

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
			}
			row_ranges |= enc_value << ((y & 3) << 3);
		}

		if(row_ranges != 0x0f0f0f0f) {
			uint roffset = atomicAdd(s_tile_rowtri_count, 1);
			if(roffset < MAX_ROW_TRIS) // TODO: handle this properly
				g_scratch[soffset + roffset] = uvec2(row_ranges, local_tri_idx | uint(by << 20));
		}
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
			}
			s_pixel_counts[LIX >> TILE_SHIFT][LIX & (TILE_SIZE - 1)] = 0;
		}
		barrier();
		generateTileMasks();
		barrier();
		if(s_tile_rowtri_count > MAX_ROW_TRIS) {
			rasterInvalidTile();
			continue;
		}
		sumPixelCounts();
		barrier();
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
