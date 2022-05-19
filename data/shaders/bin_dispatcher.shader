// $$include data

#define LID gl_LocalInvocationID
#define LIX gl_LocalInvocationIndex

#define LSIZE 512

#define TRIS_PER_THREAD 2
#define MAX_GROUP_QUADS (LSIZE * TRIS_PER_THREAD)

layout(local_size_x = LSIZE) in;
layout(std430, binding = 0) buffer buf0_ { uint g_quad_aabbs[]; };
BIN_COUNTERS_BUFFER(1);
layout(std430, binding = 3) buffer buf2_ { uint g_bin_quads[]; };

shared int s_num_quads;
shared int s_size_type;
shared int s_quads_offset;

// TODO: treat 1x1, 1x2, 2x1 & 2x2 cases separately
// TODO: filter big tris here (with bin-rasterization similarily to Laine's?)
// TODO: optimize barriers
// TODO: there are surely more ways to optimize this
// TODO: rename tris to quads

// Offsets have to be 32-bit...

shared uint s_offsets[BIN_COUNT];
shared uint s_counts[BIN_COUNT];

int getWorkItem() {
	if(LIX == 0) {
		int size_type = s_size_type;
		int quads_offset =
			atomicAdd(g_bins.num_dispatched_visible_quads[size_type], MAX_GROUP_QUADS);
		if(size_type == 0 && quads_offset >= s_num_quads) {
			quads_offset = atomicAdd(g_bins.num_dispatched_visible_quads[1], MAX_GROUP_QUADS);
			s_num_quads = g_bins.num_visible_quads[1];
			s_size_type = 1;
		}
		s_quads_offset = quads_offset;
	}
	barrier();
	return s_quads_offset;
}

void main() {
	if(LIX == 0) {
		s_num_quads = g_bins.num_visible_quads[0];
		s_size_type = 0;
	}
	barrier();

	while(true) {
		barrier();
		int quads_offset = getWorkItem();
		int num_quads = s_num_quads;
		int size_type = s_size_type;
		if(quads_offset >= num_quads)
			break;

		for(uint i = LIX; i < BIN_COUNT; i += LSIZE)
			s_counts[i] = 0;
		barrier();

		uint lbins[TRIS_PER_THREAD];
		for(uint i = 0; i < TRIS_PER_THREAD; i++)
			lbins[i] = ~0u;
		// TODO: pojedyncze trojkaty w ramach bin-a to nie te same co w ramach tile-a

		for(uint i = 0; i < TRIS_PER_THREAD; i++) {
			uint quad_idx = quads_offset + LSIZE * i + LIX;
			if(quad_idx >= num_quads)
				break;
			if(size_type == 1)
				quad_idx = (MAX_QUADS - 1) - quad_idx;

			uint aabb = g_quad_aabbs[quad_idx];
			int tsx = int(aabb & 0xff), tsy = int((aabb >> 8) & 0xff);
			int tex = int((aabb >> 16) & 0xff), tey = int((aabb >> 24));
			int bsx = tsx >> 2, bsy = tsy >> 2;
			int bex = tex >> 2, bey = tey >> 2;
			// Encodes tile ranges within a single bin
			uint tile_ranges =
				(tsx & 0x3) | ((tsy & 0x3) << 2) | ((tex & 0x3) << 4) | ((tey & 0x3) << 6);

			// 6 bits per bin dimension = max 64 bins in X/Y
			// 6 * 4 = 24 bits total per whole bin dimension
			// last 6 bits for first bin tile id
			lbins[i] = bsx | (bsy << 6) | (bex << 12) | (bey << 18) | (tile_ranges << 24);

			for(int by = bsy; by <= bey; by++)
				for(int bx = bsx; bx <= bex; bx++) {
					int bin_id = bx + by * BIN_COUNT_X;
					atomicAdd(s_counts[bin_id], 1);
				}
		}

		barrier();
		for(uint i = LIX; i < BIN_COUNT; i += LSIZE)
			if(s_counts[i] > 0)
				s_offsets[i] = atomicAdd(BIN_QUAD_OFFSETS_TEMP(i), int(s_counts[i]));
		barrier();

		for(uint i = 0; i < TRIS_PER_THREAD; i++) {
			if(lbins[i] == ~0u)
				continue;
			uint quad_idx = quads_offset + LSIZE * i + LIX;
			if(size_type == 1)
				quad_idx = (MAX_QUADS - 1) - quad_idx;

			int bsx = int((lbins[i]) & 0x3f);
			int bsy = int((lbins[i] >> 6) & 0x3f);
			int bex = int((lbins[i] >> 12) & 0x3f);
			int bey = int((lbins[i] >> 18) & 0x3f);
			uint tile_ranges = lbins[i] >> 24;

			for(int by = bsy; by <= bey; by++) {
				uint tile_ranges_cury =
					(tile_ranges | (by < bey ? 0xc0 : 0)) & (by > bsy ? 0xf3 : 0xff);
				for(int bx = bsx; bx <= bex; bx++) {
					int bin_id = bx + by * BIN_COUNT_X;
					uint quad_offset = atomicAdd(s_offsets[bin_id], 1);
					uint tile_ranges_cur =
						(tile_ranges_cury | (bx < bex ? 0x30 : 0)) & (bx > bsx ? 0xfc : 0xff);
					g_bin_quads[quad_offset] = (quad_idx & 0xffffff) | (tile_ranges_cur << 24);
				}
			}
		}
	}
}
