// $$include funcs frustum viewport declarations
// clang-format off

#define LIX gl_LocalInvocationIndex
#define LID gl_LocalInvocationID

// Different sizes are optimal for different number of tris/bin
// 512 for hairball, 256 for other scenes?
#define LSIZE 512
#define LSHIFT 9

// TODO: problem with uneven amounts of data in different tiles
// TODO: jak dobrze obsługiwać różnego rodzaju dystrybucje trójkątów ?
//
// TODO: replace 16 to BLOCKS_PER_TILE ? it's not that simple...

layout(local_size_x = LSIZE) in;

layout(std430, binding =  2) buffer buf2_ { uvec4 g_tri_aabbs[]; };
layout(std430, binding =  3) buffer buf3_ { float g_verts[]; };
layout(std430, binding =  4) buffer buf4_ { uint g_quad_indices[]; };

TILE_COUNTERS_BUFFER(6);
layout(std430, binding =  7) buffer buf7_ { uint g_block_counts[]; }; // TODO: 16-bits?
layout(std430, binding =  8) buffer buf8_ { uint g_block_offsets[]; }; // TODO: keep count|index<<16 in single value

layout(std430, binding =  9) buffer buf9_  { uint g_tile_tris[]; };
layout(std430, binding = 10) buffer buf10_ { uint g_block_tris[]; };
layout(std430, binding = 11) buffer buf11_ { uint g_block_keys[]; };
layout(std430, binding = 12) coherent buffer buf12_ { uvec2 g_scratch[]; };

// TODO: w przypadku dużych ilości trójkątów per-tile możemy robić sortowanie każdego bloku niezależnie
//       łączenie ma sens w przypadku małych ilości instancji per blok
//
// TODO: problem: możemy jednak chcieć oddzielić sortowanie od generacji masek bo:
// - możliwe że do generacji masek optymalnie jest użyć małej ilości wątków na kafel (np. 64)
//   natomiast do sortowania musimy już miejscami użyć 512...
//
// generacja:
// - mogą być duże różnice w ilości tróəkątów / tri-bloków między kaflami
// - często trzeba robić synchronizację między wszystkimi wątkami: wtedy pierwsze 16 wątków coś liczy a reszta czeka...

// TODO: enforce this somehow
// TODO: scratch too big?
#define MAX_TILE_TRIS 64 * 1024
#define MAX_BLOCK_TRIS 2048

shared int s_tile_tri_counts [TILES_PER_BIN];
shared int s_tile_tri_offsets[TILES_PER_BIN];
shared int s_tile_tri_count, s_tile_tri_offset;
shared ivec2 s_bin_pos, s_tile_pos;

shared uint s_block_tri_counts[BLOCKS_PER_TILE];
shared uint s_block_tri_offsets[BLOCKS_PER_TILE];
shared uint s_tile_blocktri_offset;
shared uint s_tile_blocktri_count, s_tile_rowtri_count;
shared uint s_total_rowtri_count, s_max_rowtri_count;
shared uint s_max_blocktri_count, s_empty_tri_count;

// TODO: possible opt: keep values & keys in separate buffers?
shared uvec2 s_buffer[LSIZE * 4];
shared uint s_buffer_size;

void computeOffsets()
{
#ifdef VENDOR_NVIDIA
	if(LIX < 16) {
		uint off = LIX == 0? 0 : s_block_tri_counts[LIX - 1], temp;
		temp = shuffleUpNV(off, 1, 16); if(LIX >= 1) off += temp;
		temp = shuffleUpNV(off, 2, 16); if(LIX >= 2) off += temp;
		temp = shuffleUpNV(off, 4, 16); if(LIX >= 4) off += temp;
		temp = shuffleUpNV(off, 8, 16); if(LIX >= 8) off += temp;
		s_block_tri_offsets[LIX] = off;
	}
#else
	if(LIX < 16)
		s_block_tri_offsets[LIX] = LIX == 0? 0 : s_block_tri_counts[LIX - 1];
	barrier(); // TODO: why is this barrier needed?
	// TODO: cleanup small parallel scans
	// TODO: this is probably wrong on intel because warp size is 8 (need more barriers)
	if(LIX < 16) {
		if(LIX >= 1) s_block_tri_offsets[LIX] += s_block_tri_offsets[LIX - 1];
		if(LIX >= 2) s_block_tri_offsets[LIX] += s_block_tri_offsets[LIX - 2];
		if(LIX >= 4) s_block_tri_offsets[LIX] += s_block_tri_offsets[LIX - 4];
		if(LIX >= 8) s_block_tri_offsets[LIX] += s_block_tri_offsets[LIX - 8];
	}
#endif
}

#ifdef VENDOR_NVIDIA
uvec2 swap(uvec2 x, int mask, uint dir)
{
	uvec2 y = shuffleXorNV(x, mask, 32);
	return x.x != y.x && (x.x < y.x) == (dir != 0) ? y : x;
}

uint bitExtract(uint value, int boffset) 
{
	return (value >> boffset) & 1;
}

uint xorBits(uint value, int bit0, int bit1)
{
	return ((value >> bit0) ^ (value >> bit1)) & 1;
}
#endif

void sortBuffer(uint N)
{
	uint TN = max(32, (N & (N - 1)) == 0? N : (2 << findMSB(N)));
	for(uint i = LIX + N; i < TN; i += LSIZE)
		s_buffer[i].x = 0xffffffff;
	barrier();

#ifdef VENDOR_NVIDIA
	for(uint i = LIX; i < TN; i += LSIZE) {
		uvec2 value = s_buffer[i];
		value = swap(value, 0x01, xorBits(LIX, 1, 0)); // K = 2
		value = swap(value, 0x02, xorBits(LIX, 2, 1)); // K = 4
		value = swap(value, 0x01, xorBits(LIX, 2, 0));
		value = swap(value, 0x04, xorBits(LIX, 3, 2)); // K = 8
		value = swap(value, 0x02, xorBits(LIX, 3, 1));
		value = swap(value, 0x01, xorBits(LIX, 3, 0));
		value = swap(value, 0x08, xorBits(LIX, 4, 3)); // K = 16
		value = swap(value, 0x04, xorBits(LIX, 4, 2));
		value = swap(value, 0x02, xorBits(LIX, 4, 1));
		value = swap(value, 0x01, xorBits(LIX, 4, 0));
		value = swap(value, 0x10, xorBits(LIX, 5, 4)); // K = 32
		value = swap(value, 0x08, xorBits(LIX, 5, 3));
		value = swap(value, 0x04, xorBits(LIX, 5, 2));
		value = swap(value, 0x02, xorBits(LIX, 5, 1));
		value = swap(value, 0x01, xorBits(LIX, 5, 0));
		s_buffer[i] = value;
	}
	barrier();
	int start_k = 64, end_j = 32;
#else
	int start_k = 2, end_j = 1;
#endif

	for(uint k = start_k; k <= TN; k = 2 * k) {
		for(uint j = k >> 1; j >= end_j; j = j >> 1) {
			for(uint i = LIX; i < TN; i += LSIZE * 2) {
				uint idx = (i & j) != 0? i + LSIZE - j : i;
				uvec2 lvalue = s_buffer[idx];
				uvec2 rvalue = s_buffer[idx + j];
				if( ((idx & k) != 0) == (lvalue.x < rvalue.x) ) {
					s_buffer[idx] = rvalue;
					s_buffer[idx + j] = lvalue;
				}
			}
			barrier();
		}
#ifdef VENDOR_NVIDIA
		for(uint i = LIX; i < TN; i += LSIZE) {
			uint bit = (i & k) == 0? 0 : 1;
			uvec2 value = s_buffer[i];
			value = swap(value, 0x10, bit ^ bitExtract(LIX, 4));
			value = swap(value, 0x08, bit ^ bitExtract(LIX, 3));
			value = swap(value, 0x04, bit ^ bitExtract(LIX, 2));
			value = swap(value, 0x02, bit ^ bitExtract(LIX, 1));
			value = swap(value, 0x01, bit ^ bitExtract(LIX, 0));
			s_buffer[i] = value;
		}
		barrier();
#endif
	}
}

// Sum of sample posititions in 4x4 block (multiplied by 2):
// bit 0-7:   X value for rows 0123
// bit 8-15:  Y value for rows 01
// bit 16-23: Y value for rows 23
uniform uint mask_centroids[256];

// Note: inlining makes this code run a bit faster...
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
			uint toffset = atomicAdd(s_buffer_size, 1);
			s_buffer[toffset] = uvec2(row_ranges, local_tri_idx | uint(by << 20));
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
		// TODO: single add per row should be enough
		uint block_id = by * 4 + bx;
		uint boffset = atomicAdd(s_block_tri_counts[block_id], 1);
		uint soffset = gl_WorkGroupID.x * MAX_BLOCK_TRIS * 16 + block_id * MAX_BLOCK_TRIS;
		if(boffset < MAX_BLOCK_TRIS) {
			uint value = (local_tri_idx << 16) | mask;
			uint key = (block_id << 28) | (uint(depth) & 0xfffffff);
			g_scratch[soffset + boffset] = uvec2(value, key);
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
			aabb = decodeAABB64(second_tri? aabb.zw : aabb.xy);
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
		barrier();
		for(uint j = LIX; j < s_buffer_size; j += LSIZE) {
			uint row_ranges = s_buffer[j].x;
			uint local_tri_idx = s_buffer[j].y;
			int by = int((local_tri_idx >> 20) & 0x3);
			local_tri_idx &= 0xfffff;

			uint tri_idx = g_tile_tris[s_tile_tri_offset + local_tri_idx];
			bool second_tri = (tri_idx & 0x80000000) != 0;
			tri_idx &= 0x7fffffff;

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

		barrier();
		if(LIX == 0) {
			atomicAdd(s_tile_rowtri_count, s_buffer_size);
			s_buffer_size = 0;
		}
		barrier();
	}
}

// TODO: może tile dispatcher też jest zbędny? moglibyśmy od razu iterować po trójkątach z bina?
//       nawet jeśli to jest to pomysł na później jak już wszystko inne będzie szybciej działać
void generateBinMasks(int bin_id) {
	if(LIX < TILES_PER_BIN) {
		s_tile_tri_counts [LIX] = int(TILE_TRI_COUNTS(bin_id, LIX));
		s_tile_tri_offsets[LIX] = int(TILE_TRI_OFFSETS(bin_id, LIX));
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
			if(LIX == 0) {
				// TODO: properly allocate mem for this?
				uint blocktri_count = s_tile_blocktri_count;
				uint blocktri_offset = atomicAdd(g_tiles.num_block_tris, blocktri_count);
				s_tile_blocktri_offset = blocktri_offset;
				// TODO: write to smem first? then use all threads to write all in one go
				TILE_BLOCK_TRI_COUNTS(bin_id, tile_id) = blocktri_count;
				TILE_BLOCK_TRI_OFFSETS(bin_id, tile_id) = blocktri_offset;
			}
			// TODO: make sure it works when warp size < 16
			s_block_tri_counts[LIX] = min(s_block_tri_counts[LIX], MAX_BLOCK_TRIS);
		}
		
		computeOffsets();

		groupMemoryBarrier();
		barrier();

		if(s_tile_blocktri_count < MAX_BLOCK_TRIS) {
			uint soffset = gl_WorkGroupID.x * MAX_BLOCK_TRIS * 16;
			uint b = LIX >> (LSHIFT - 4);
			uint boffset = soffset + b * MAX_BLOCK_TRIS;
			uint bcount = s_block_tri_counts[b];
			for(uint block_idx = LIX & (LSIZE / 16 - 1); block_idx < bcount; block_idx += LSIZE / 16) {
				uvec2 value_key = g_scratch[boffset + block_idx];
				s_buffer[s_block_tri_offsets[b] + block_idx] = uvec2(value_key.y, value_key.x);
			}
			barrier();
			sortBuffer(s_tile_blocktri_count);
			barrier();
			uint roffset = s_tile_blocktri_offset + s_block_tri_offsets[b];
			for(uint block_idx = LIX & (LSIZE / 16 - 1); block_idx < bcount; block_idx += LSIZE / 16)
				g_block_tris[roffset + block_idx] = s_buffer[s_block_tri_offsets[b] + block_idx].y;
		}
		else {
			// TODO: possible optimization: merge more groups together
			uint soffset = gl_WorkGroupID.x * MAX_BLOCK_TRIS * 16;
			for(int b = 0; b < BLOCKS_PER_TILE; b++) {
				uint boffset = soffset + b * MAX_BLOCK_TRIS;
				uint roffset = s_tile_blocktri_offset + s_block_tri_offsets[b];
				uint bcount = s_block_tri_counts[b];

				for(uint block_idx = LIX; block_idx < bcount; block_idx += LSIZE) {
					uvec2 value_key = g_scratch[boffset + block_idx];
					s_buffer[block_idx] = uvec2(value_key.y, value_key.x);
				}
				barrier();
				sortBuffer(bcount);
				barrier();
				for(uint block_idx = LIX; block_idx < bcount; block_idx += LSIZE)
					g_block_tris[roffset + block_idx] = s_buffer[block_idx].y;
			}
		}

		if(LIX < 16) {
			// Sanity checks
			if(LIX >= 1 && s_block_tri_offsets[LIX - 1] > s_block_tri_offsets[LIX])
				RECORD(s_block_tri_offsets[LIX - 1], s_block_tri_offsets[LIX], s_block_tri_counts[LIX - 1], s_block_tri_counts[LIX]);
			if(LIX == 0 && s_block_tri_offsets[15] + s_block_tri_counts[15] != s_tile_blocktri_count)
				RECORD(s_block_tri_offsets[15], s_block_tri_counts[15], s_tile_blocktri_count, 0);

			uint block_id = (bin_id * TILES_PER_BIN + tile_id) * BLOCKS_PER_TILE + LIX;
			// TODO: more efficient update? for whole tile?
			g_block_counts[block_id] = s_block_tri_counts[LIX];
			g_block_offsets[block_id] = s_block_tri_offsets[LIX] + s_tile_blocktri_offset;
			atomicMax(g_tiles.max_tris_per_block, s_block_tri_counts[LIX]);
			if(LIX == 0) {
				s_total_rowtri_count += s_tile_rowtri_count;
				s_max_rowtri_count = max(s_max_rowtri_count, s_tile_rowtri_count);
				s_max_blocktri_count = max(s_max_blocktri_count, s_tile_blocktri_count);
			}
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
	if(LIX == 0) {
		s_empty_tri_count = 0;
		s_total_rowtri_count = 0;
		s_max_rowtri_count = 0;
		s_max_blocktri_count = 0;
		s_buffer_size = 0;
	}
	// TODO: remove this variable
	int bin_id = loadNextBin();
	while(bin_id < BIN_COUNT) {
		barrier();
		generateBinMasks(bin_id);
		bin_id = loadNextBin();
	}
	if(LIX == 0) {
		atomicAdd(g_tiles.num_tile_tris_with_no_blocks, s_empty_tri_count);
		atomicAdd(g_tiles.num_processed_block_rows, s_total_rowtri_count);
		atomicMax(g_tiles.max_row_tris_per_tile, s_max_rowtri_count);
		atomicMax(g_tiles.max_block_tris_per_tile, s_max_blocktri_count);
	}
}
