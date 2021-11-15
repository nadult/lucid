// $$include funcs frustum viewport data

#define LIX gl_LocalInvocationIndex
#define LID gl_LocalInvocationID

// Different sizes are optimal for different number of tris/bin
// 512 for hairball, 256 for other scenes?
#define LSIZE 256

// TODO: problem with uneven amounts of data in different tiles
// TODO: jak dobrze obsługiwać różnego rodzaju dystrybucje trójkątów ?

layout(local_size_x = LSIZE) in;

layout(std430, binding =  2) buffer buf2_ { uvec4 g_tri_aabbs[]; };
layout(std430, binding =  3) buffer buf3_ { float g_verts[]; };
layout(std430, binding =  4) buffer buf4_ { uint g_quad_indices[]; };

layout(std430, binding =  5) buffer buf5_ { BinCounters g_bins; };
layout(std430, binding =  6) buffer buf6_ { TileCounters g_tiles; };
layout(std430, binding =  7) buffer buf7_ { uint g_block_counts[]; }; // TODO: 16-bits?
layout(std430, binding =  8) buffer buf8_ { uint g_block_offsets[]; }; // TODO: keep count|index<<16 in single value

layout(std430, binding =  9) buffer buf9_  { uint g_tile_tris[]; };
layout(std430, binding = 10) buffer buf10_ { uint g_block_tris[]; };
layout(std430, binding = 11) buffer buf11_ { uint g_block_keys[]; };
layout(std430, binding = 12) coherent buffer buf12_ { uint g_scratch[]; };

// TODO: w przypadku dużych ilości trójkątów per-tile możemy robić sortowanie każdego bloku niezależnie
//       łączenie ma sens w przypadku małych ilości instancji per blok

// TODO: enforce this somehow
// TODO: scratch too big?
#define MAX_TILE_TRIS 64 * 1024
#define MAX_TILE_BLOCK_ROWS 32 * 1024

shared int s_tile_tri_counts [TILES_PER_BIN];
shared int s_tile_tri_offsets[TILES_PER_BIN];
shared int s_tile_tri_count, s_tile_tri_offset;
shared ivec2 s_bin_pos, s_tile_pos;

shared uint s_block_tri_counts[BLOCKS_PER_TILE];
shared uint s_block_tri_offsets[BLOCKS_PER_TILE];
shared uint s_tile_blocktri_offset, s_tile_base_blocktri_offset;
shared uint s_tile_blocktri_count, s_tile_rowtri_count;
shared uint s_empty_tri_count;

// Sum of sample posititions in 4x4 block (multiplied by 2):
// bit 0-7:   X value for rows 0123
// bit 8-15:  Y value for rows 01
// bit 16-23: Y value for rows 23
uniform uint mask_centroids[256];

// Note: inlining it makes code run a bit faster...
vec2 computeCentroid4x4(vec2 base_pos, uint mask) {
	vec2 cpoint = base_pos + vec2(2.0, 2.0);
	bool quick_centroid = false;
	bool costly_centroid = true;
		
	// If all pixels in the middle are empty, lets try different center points:
	if(quick_centroid && (mask & ((1 << 5) | (1 << 6) | (1 << 9) | (1 << 10))) == 0) {
		cpoint.x += (mask & 0x3333) != 0? -1.0 : 1.0;
		cpoint.y += (mask & 0xff) != 0? -1.0 : 1.0;
	}

	int num_bits = bitCount(mask);
	if(costly_centroid && num_bits < 16) {
		uint lo_bits = mask_centroids[mask & 0xff];
		uint hi_bits = mask_centroids[mask >> 8];
		float cx = float((lo_bits & 0xff) + (hi_bits & 0xff));
		float cy = float(((lo_bits & 0xff00) >> 8) + (hi_bits >> 16));
		float scale = 0.5 / float(num_bits);

		// Slower version:
		// cx = 0.5 * bitCount(mask & 0x1111) + 1.5 * bitCount(mask & 0x2222) +
		//      2.5 * bitCount(mask & 0x4444) + 3.5 * bitCount(mask & 0x8888);
		// cy = 0.5 * bitCount(mask & 0x000f) + 1.5 * bitCount(mask & 0x00f0) +
		//      2.5 * bitCount(mask & 0x0f00) + 3.5 * bitCount(mask & 0xf000);
		// scale = 1.0 / float(num_bits);

		cpoint = base_pos + vec2(cx, cy) * scale;
	}

	return cpoint;
}

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

uint rangeToBits16(uint enc_range) {
	uint imin = enc_range & 0xf, imax = (enc_range >> 4);
	return ((0xffff << imin) & (0xffff >> (15 - imax)));
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

	uint num_blocktris = 0;
	uint soffset = gl_WorkGroupID.x * MAX_TILE_BLOCK_ROWS * 2;

	for(int by = min_by; by <= max_by; by++) {
		uint row_ranges = 0;
		uint block_bits = 0;

		for(int y = 0; y < 4; y++) {
			float xmin = max(max(scan_min[0], scan_min[1]), max(scan_min[2], 0.0));
			float xmax = min(min(scan_max[0], scan_max[1]), min(scan_max[2], TILE_SIZE));

			scan_min += scan_step;
			scan_max += scan_step;
			
			// TODO: use floor/ceil?
			int imin = int(xmin), imax = int(xmax) - 1;
			if(imin <= imax) {
				// TODO: this is costly
				int bmin = imin >> 2, bmax = imax >> 2;
				block_bits |= (0xf << bmin) & (0xf >> (3 - bmax));
			}
			uint shift = (y & 3) * 8;
			uint enc_value = imin <= imax? uint(imin) | (uint(imax) << 4) : 0x0f;
			row_ranges |= enc_value << ((y & 3) << 3);
		}

		if(row_ranges != 0x0f0f0f0f) {
			num_blocktris += bitCount(block_bits);
			uint roffset = atomicAdd(s_tile_rowtri_count, 1);
			if(roffset < MAX_TILE_BLOCK_ROWS) { // TODO: mark tile red?
				g_scratch[soffset + roffset] = row_ranges;
				g_scratch[soffset + roffset + MAX_TILE_BLOCK_ROWS] = local_tri_idx | uint(by << 20);
			}
		}
	}

	// TODO: accumulate over whole loop?
	atomicAdd(s_tile_blocktri_count, num_blocktris);
	if(num_blocktris == 0)
		atomicAdd(s_empty_tri_count, 1);
}

void generateTriMasks(uint local_tri_idx, uint row_ranges, int by, vec3 plane_normal) {
	int min_bx = 0, max_bx = 3;
	uint row0 = rangeToBits16(row_ranges & 0xff) >> (min_bx * 4);
	uint row1 = rangeToBits16((row_ranges >> 8) & 0xff) >> (min_bx * 4);
	uint row2 = rangeToBits16((row_ranges >> 16) & 0xff) >> (min_bx * 4);
	uint row3 = rangeToBits16(row_ranges >> 24) >> (min_bx * 4);

	for(int bx = min_bx; bx <= max_bx; bx++) {
		uint mask = (row0 & 0x000f) | ((row1 << 4) & 0x00f0) | ((row2 << 8) & 0x0f00) | ((row3 << 12) & 0xf000);
		row0 >>= 4, row1 >>= 4, row2 >>= 4, row3 >>= 4;
		if(mask == 0)
			continue;

		vec2 cpoint = computeCentroid4x4(s_tile_pos + vec2(bx * 4, by * 4), mask);

		vec3 ray_dir = frustum.ws_dir0 + frustum.ws_dirx * cpoint.x + frustum.ws_diry * cpoint.y;
		float ray_pos = 1.0 / dot(plane_normal, ray_dir);
		float depth = (1 << 28) / (1.0 + ray_pos); // 24 bits is enough

		// TODO: count this in smarter way?
		uint block_id = by * 4 + bx;
		atomicAdd(s_block_tri_counts[block_id], 1);

		// TODO: single add per row should be enough
		uint cur_offset = atomicAdd(s_tile_blocktri_offset, 1);
		g_block_tris[cur_offset] = (local_tri_idx << 16) | mask;
		g_block_keys[cur_offset] = (block_id << 28) | (uint(depth) & 0xfffffff);
	}
}

void generateTileMasks() {
	// TODO: check if using SMEM var here makes sense
	for(uint i = LIX; i < s_tile_tri_count; i += LSIZE) {
		uint tri_idx = g_tile_tris[s_tile_tri_offset + i];

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
		generateTriScanlines(i, tri0, tri1, tri2, min_by, max_by);
	}

	groupMemoryBarrier();
	barrier();

	if(LIX == 0) {
		// TODO: properly allocate mem for this?
		uint blocktri_offset = atomicAdd(g_tiles.num_block_tris, s_tile_blocktri_count);
		s_tile_base_blocktri_offset = blocktri_offset;
		s_tile_blocktri_offset = blocktri_offset;
		atomicAdd(g_tiles.num_processed_block_rows, s_tile_rowtri_count);
		atomicMax(g_tiles.max_block_rows_per_tile, s_tile_rowtri_count);
	}

	barrier();

	uint soffset = gl_WorkGroupID.x * MAX_TILE_BLOCK_ROWS * 2;
	for(uint i = LIX; i < s_tile_rowtri_count; i += LSIZE) {
		uint row_ranges = g_scratch[soffset + i];
		uint local_tri_idx = g_scratch[soffset + i + MAX_TILE_BLOCK_ROWS];
		int by = int((local_tri_idx >> 20) & 0x3);
		local_tri_idx &= 0xfffff;

		uint tri_idx = g_tile_tris[s_tile_tri_offset + local_tri_idx];
		bool second_tri = (tri_idx & 0x80000000) != 0;
		tri_idx &= 0x7fffffff;
		local_tri_idx |= (second_tri? 0x8000 : 0);

		uint v0 = g_quad_indices[tri_idx * 4 + 0] & 0x03ffffff;
		uint v1 = g_quad_indices[tri_idx * 4 + (second_tri? 2 : 1)] & 0x03ffffff;
		uint v2 = g_quad_indices[tri_idx * 4 + (second_tri? 3 : 2)] & 0x03ffffff;

		vec3 tri0 = vec3(g_verts[v0 * 3 + 0], g_verts[v0 * 3 + 1], g_verts[v0 * 3 + 2]) - frustum.ws_shared_origin;
		vec3 tri1 = vec3(g_verts[v1 * 3 + 0], g_verts[v1 * 3 + 1], g_verts[v1 * 3 + 2]) - frustum.ws_shared_origin;
		vec3 tri2 = vec3(g_verts[v2 * 3 + 0], g_verts[v2 * 3 + 1], g_verts[v2 * 3 + 2]) - frustum.ws_shared_origin;

		// TODO: this has to be procemputed, makes no sense to compute it multiple times for different blocks!
		vec3 plane_normal = normalize(cross(tri0 - tri2, tri1 - tri0));
		float plane_dist = dot(plane_normal, tri0);
		plane_normal *= (1.0 / plane_dist);

		generateTriMasks(local_tri_idx, row_ranges, by, plane_normal);
	}
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
		if(LIX < BLOCKS_PER_TILE) {
			if(LIX == 0) {
				s_tile_tri_count  = s_tile_tri_counts[tile_id];
				s_tile_tri_offset = s_tile_tri_offsets[tile_id];
				s_tile_pos = s_bin_pos + ivec2(tile_id & 3, tile_id >> 2) * TILE_SIZE;
				s_tile_blocktri_count = 0;
				s_tile_rowtri_count = 0;
			}
			s_block_tri_counts[LIX] = 0;
		}
		barrier();

		generateTileMasks();

		barrier();
		if(LIX < 16) {
			// TODO: make sure it works when warp size < 16
			s_block_tri_offsets[LIX] = LIX == 0? 0 : s_block_tri_counts[LIX - 1];
			if(LIX == 0) {
				// TODO: write to smem first?
				g_tiles.tile_block_tri_counts[bin_id][tile_id] = s_tile_blocktri_count;
				g_tiles.tile_block_tri_offsets[bin_id][tile_id] = s_tile_base_blocktri_offset;
			}
		}

		barrier(); // TODO: why is this barrier needed?

		// TODO: cleanup small parallel scans
		if(LIX < 16) {
			if(LIX >= 1) s_block_tri_offsets[LIX] += s_block_tri_offsets[LIX - 1];
			if(LIX >= 2) s_block_tri_offsets[LIX] += s_block_tri_offsets[LIX - 2];
			if(LIX >= 4) s_block_tri_offsets[LIX] += s_block_tri_offsets[LIX - 4];
			if(LIX >= 8) s_block_tri_offsets[LIX] += s_block_tri_offsets[LIX - 8];
		}

		barrier();
		if(LIX < 16) {
			// Sanity checks
			if(LIX >= 1 && s_block_tri_offsets[LIX - 1] > s_block_tri_offsets[LIX])
				RECORD(s_block_tri_offsets[LIX - 1], s_block_tri_offsets[LIX], s_block_tri_counts[LIX - 1], s_block_tri_counts[LIX]);
			if(LIX == 0 && s_block_tri_offsets[15] + s_block_tri_counts[15] != s_tile_blocktri_count)
				RECORD(s_block_tri_offsets[15], s_block_tri_counts[15], s_tile_blocktri_count, 0);
			if(LIX == 0 && s_tile_blocktri_offset - s_tile_base_blocktri_offset != s_tile_blocktri_count)
				RECORD(s_tile_blocktri_count, s_tile_base_blocktri_offset, s_tile_blocktri_offset, 0);

			uint block_id = (bin_id * TILES_PER_BIN + tile_id) * BLOCKS_PER_TILE + LIX;
			g_block_counts[block_id] = s_block_tri_counts[LIX];
			g_block_offsets[block_id] = s_block_tri_offsets[LIX] + s_tile_base_blocktri_offset;
			atomicMax(g_tiles.max_tris_per_block, s_block_tri_counts[LIX]);
		}
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
	if(LIX == 0)
		s_empty_tri_count = 0;
	// TODO: remove this variable
	int bin_id = loadNextBin();
	while(bin_id < BIN_COUNT) {
		barrier();
		generateBinMasks(bin_id);
		bin_id = loadNextBin();
	}
	if(LIX == 0)
		atomicAdd(g_tiles.num_tile_tris_with_no_blocks, s_empty_tri_count);
}
