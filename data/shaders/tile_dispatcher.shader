// $$include funcs frustum declarations

#define LID gl_LocalInvocationID
#define LIX gl_LocalInvocationIndex

#define LSIZE 512

// TODO: with small number of bins we might have poor paralellization. maybe we spawn more threads per each bin?
// There is definitely space for optimization here

layout(local_size_x = LSIZE) in;

BIN_COUNTERS_BUFFER(1);
TILE_COUNTERS_BUFFER(2);
layout(std430, binding = 3) buffer buf3_ { uint g_bin_quads[]; };
layout(std430, binding = 4) buffer buf4_ { uint g_tile_tris[]; };
layout(std430, binding = 5) buffer buf5_ { float g_verts[]; };
layout(std430, binding = 6) buffer buf6_ { uint g_quad_indices[]; };
layout(std430, binding = 7) buffer buf7_ { uvec4 g_tri_aabbs[]; };

shared uint s_bin_offset, s_total_count;
shared uint s_tile_counts[TILES_PER_BIN];
shared uint s_tile_buffer[TILES_PER_BIN];
shared uint s_tile_offsets[TILES_PER_BIN], s_tile_base_offsets[TILES_PER_BIN];

shared vec2 s_bin_pos;
shared ivec2 s_bin_ipos;

uint computeScanlineParams(vec3 tri0, vec3 tri1, vec3 tri2, out vec3 scan_base,
						   out vec3 scan_step) {
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
	scan_base = -vec3(edges[0].z * inv_ex[0], edges[1].z * inv_ex[1], edges[2].z * inv_ex[2]);
	scan_step = -vec3(edges[0].y * inv_ex[0], edges[1].y * inv_ex[1], edges[2].y * inv_ex[2]);
	uint xsigns =
		(edges[0].x < 0.0 ? 1 : 0) | (edges[1].x < 0.0 ? 2 : 0) | (edges[2].x < 0.0 ? 4 : 0);
	uint ysigns =
		(edges[0].y < 0.0 ? 8 : 0) | (edges[1].y < 0.0 ? 16 : 0) | (edges[2].y < 0.0 ? 32 : 0);
	return xsigns | ysigns;
}

uint testTriangle(vec3 tri0, vec3 tri1, vec3 tri2, int tsx, int tsy, int tex, int tey) {
	tsx = max(tsx, 0);
	tsy = max(tsy, 0);
	tex = min(tex, 3);
	tey = min(tey, 3);

	vec3 scan_min, scan_max, scan_step;
	{
		float sx = s_bin_pos.x + 0.49;
		float sy = s_bin_pos.y + tsy * TILE_SIZE + 0.49;

		vec3 scan_base;
		uint sign_mask = computeScanlineParams(tri0, tri1, tri2, scan_base, scan_step);

		float tile_offset = TILE_SIZE - 0.989;
		vec3 scan = vec3(scan_step[0] * (sy + ((sign_mask & 8) == 0 ? tile_offset : 0.0)) +
							 scan_base[0] - (sx + ((sign_mask & 1) == 0 ? tile_offset : 0.0)),
						 scan_step[1] * (sy + ((sign_mask & 16) == 0 ? tile_offset : 0.0)) +
							 scan_base[1] - (sx + ((sign_mask & 2) == 0 ? tile_offset : 0.0)),
						 scan_step[2] * (sy + ((sign_mask & 32) == 0 ? tile_offset : 0.0)) +
							 scan_base[2] - (sx + ((sign_mask & 4) == 0 ? tile_offset : 0.0)));
		scan_min = vec3((sign_mask & 1) == 0 ? scan[0] : -1.0 / 0.0,
						(sign_mask & 2) == 0 ? scan[1] : -1.0 / 0.0,
						(sign_mask & 4) == 0 ? scan[2] : -1.0 / 0.0);
		scan_max = vec3((sign_mask & 1) != 0 ? scan[0] : 1.0 / 0.0,
						(sign_mask & 2) != 0 ? scan[1] : 1.0 / 0.0,
						(sign_mask & 4) != 0 ? scan[2] : 1.0 / 0.0);
	}

	// Trivial reject test
	uint mask = 0;
	for(int ty = tsy; ty <= tey; ty++) {
		float xmin = max(max(scan_min[0], scan_min[1]), max(scan_min[2], 0.0));
		float xmax = min(min(scan_max[0], scan_max[1]), min(scan_max[2], BIN_SIZE));

		scan_min += scan_step * TILE_SIZE;
		scan_max += scan_step * TILE_SIZE;
		for(int tx = tsx; tx <= tex; tx++)
			if(tx * TILE_SIZE >= xmin && tx * TILE_SIZE <= xmax)
				mask |= 1 << (ty * 4 + tx);
	}

	// TODO: trivial accept
	return mask;
}

// TODO: this algorithm is slower when we have small number of bins with a lot of triangles
// increasing LSIZE helps (at least 1024 for san miguel for example)
//
// TODO: do it in two phases: first estimate, then dispatch, just like for bins;
// This way we could split work more evenly?
// TODO: don't dispatch to tiles which are not visible
void dispatchBinTris(int bin_id) {
	int bin_sy = int(bin_id) / BIN_COUNT_X;
	int bin_sx = int(bin_id) - bin_sy * BIN_COUNT_X;

	if(LIX < TILES_PER_BIN) {
		s_tile_counts[LIX] = 0;
		if(LIX == 0) {
			s_total_count = 0;
			s_bin_ipos = ivec2(bin_sx, bin_sy) * BIN_SIZE;
			s_bin_pos = vec2(bin_sx, bin_sy) * BIN_SIZE;
		}
	}

	int tri_count = BIN_QUAD_COUNTS(bin_id);
	int tri_offset = BIN_QUAD_OFFSETS(bin_id);
	barrier();

	bin_sx *= BIN_SIZE / TILE_SIZE;
	bin_sy *= BIN_SIZE / TILE_SIZE;

	for(int i = 0; i < tri_count; i += LSIZE) {
		if(i + LIX >= tri_count)
			continue;
		uint tile_ranges = g_bin_quads[tri_offset + i + LIX] >> 24;
		uint tsx = (tile_ranges)&0x3, tsy = (tile_ranges >> 2) & 0x3;
		uint tex = (tile_ranges >> 4) & 0x3, tey = (tile_ranges >> 6);

		for(uint ty = tsy; ty <= tey; ty++)
			for(uint tx = tsx; tx <= tex; tx++) {
				uint tile_id = tx + ty * XTILES_PER_BIN;
				// Note: computing in registers first (8 is enough) is slower than with atomics + SMEM
				atomicAdd(s_tile_counts[tile_id], 2);
			}
	}

	// TODO: most of these barriers are needed only on platforms with warp size < 16
	// TODO: use 8 uints instead of 16 (16-bit per tile is enough)
	barrier();
	if(LIX < TILES_PER_BIN)
		atomicAdd(s_total_count, s_tile_counts[LIX]);
	if(LIX < TILES_PER_BIN)
		s_tile_buffer[LIX] = s_tile_counts[LIX] + (LIX >= 1 ? s_tile_counts[LIX - 1] : 0);
	barrier();
	if(LIX < TILES_PER_BIN)
		s_tile_offsets[LIX] = s_tile_buffer[LIX] + (LIX >= 2 ? s_tile_buffer[LIX - 2] : 0);
	barrier();
	if(LIX < TILES_PER_BIN)
		s_tile_buffer[LIX] = s_tile_offsets[LIX] + (LIX >= 4 ? s_tile_offsets[LIX - 4] : 0);
	barrier();
	if(LIX < TILES_PER_BIN)
		s_tile_offsets[LIX] = s_tile_buffer[LIX] + (LIX >= 8 ? s_tile_buffer[LIX - 8] : 0);
	if(LIX == 0)
		s_bin_offset = atomicAdd(g_tiles.num_tile_tris, s_total_count);
	barrier();
	if(LIX < TILES_PER_BIN)
		s_tile_offsets[LIX] += s_bin_offset - s_tile_counts[LIX];
	barrier();
	if(LIX < TILES_PER_BIN) {
		s_tile_base_offsets[LIX] = s_tile_offsets[LIX];
		TILE_TRI_OFFSETS(bin_id, LIX) = s_tile_offsets[LIX];
	}
	barrier();

	for(uint i = LIX; i < tri_count; i += LSIZE) {
		uint tri_idx = g_bin_quads[tri_offset + i];
		uint tile_ranges = tri_idx >> 24;
		uint tsx = (tile_ranges)&0x3, tsy = (tile_ranges >> 2) & 0x3;
		uint tex = (tile_ranges >> 4) & 0x3, tey = (tile_ranges >> 6);
		tri_idx &= 0xffffff;

		// TODO: mark 2nd triangle as empty earlier?
		uint v0 = g_quad_indices[tri_idx * 4 + 0] & 0x03ffffff;
		uint v1 = g_quad_indices[tri_idx * 4 + 1] & 0x03ffffff;
		uint v2 = g_quad_indices[tri_idx * 4 + 2] & 0x03ffffff;
		uint v3 = g_quad_indices[tri_idx * 4 + 3] & 0x03ffffff;

		uint mask0 = 0xffff, mask1 = v2 == v3 ? 0 : 0xffff;
		// TODO: thread divergence here; can we decrease it somehow?
		// Best way to add separate pass for small triangles?
		vec3 quad0 = vec3(g_verts[v0 * 3 + 0], g_verts[v0 * 3 + 1], g_verts[v0 * 3 + 2]) -
					 frustum.ws_shared_origin;
		vec3 quad1 = vec3(g_verts[v1 * 3 + 0], g_verts[v1 * 3 + 1], g_verts[v1 * 3 + 2]) -
					 frustum.ws_shared_origin;
		vec3 quad2 = vec3(g_verts[v2 * 3 + 0], g_verts[v2 * 3 + 1], g_verts[v2 * 3 + 2]) -
					 frustum.ws_shared_origin;
		vec3 quad3 = vec3(g_verts[v3 * 3 + 0], g_verts[v3 * 3 + 1], g_verts[v3 * 3 + 2]) -
					 frustum.ws_shared_origin;

		// AABB here helps only on hairball
		uvec4 aabb = g_tri_aabbs[tri_idx];
		ivec4 aabb0 = ivec4(decodeAABB64(aabb.xy)) - ivec4(s_bin_ipos, s_bin_ipos);
		ivec4 aabb1 = ivec4(decodeAABB64(aabb.zw)) - ivec4(s_bin_ipos, s_bin_ipos);
		for(int i = 0; i < 4; i++) {
			aabb0[i] >>= TILE_SHIFT;
			aabb1[i] >>= TILE_SHIFT;
		}

		mask0 = testTriangle(quad0, quad1, quad2, aabb0[0], aabb0[1], aabb0[2], aabb0[3]);
		if(v2 != v3)
			mask1 = testTriangle(quad0, quad2, quad3, aabb1[0], aabb1[1], aabb1[2], aabb1[3]);

		if((mask0 | mask1) != 0)
			for(uint ty = tsy; ty <= tey; ty++)
				for(uint tx = tsx; tx <= tex; tx++) {
					uint tile_id = tx + ty * XTILES_PER_BIN;
					if((mask0 & (1 << tile_id)) != 0)
						g_tile_tris[atomicAdd(s_tile_offsets[tile_id], 1)] = tri_idx;
					if((mask1 & (1 << tile_id)) != 0)
						g_tile_tris[atomicAdd(s_tile_offsets[tile_id], 1)] = tri_idx | 0x80000000;
				}
	}
	barrier();
	if(LIX < TILES_PER_BIN) {
		uint num_tris = s_tile_offsets[LIX] - s_tile_base_offsets[LIX];
		TILE_TRI_COUNTS(bin_id, LIX) = num_tris;
		atomicMax(g_tiles.max_tris_per_tile, num_tris);
	}
}

shared int s_num_bins, s_bin_id;

int loadNextBin() {
	if(LIX == 0) {
		uint bin_idx = atomicAdd(g_tiles.tiled_bin_counter, 1);
		s_bin_id = bin_idx < s_num_bins ? BIN_TILED_BINS(bin_idx) : -1;
	}
	barrier();
	return s_bin_id;
}

void main() {
	if(LIX == 0)
		s_num_bins = g_bins.num_tiled_bins;
	int bin_id = loadNextBin();
	while(bin_id != -1) {
		barrier();
		dispatchBinTris(bin_id);
		bin_id = loadNextBin();
	}
}
