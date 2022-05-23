// $$include funcs declarations

#define LID gl_LocalInvocationID
#define LIX gl_LocalInvocationIndex
#define WGID gl_WorkGroupID

#define LSIZE BINNING_LSIZE

layout(local_size_x = LSIZE) in;
layout(std430, binding = 0) buffer buf0_ { uint g_quad_aabbs[]; };
BIN_COUNTERS_BUFFER(1);
layout(std430, binding = 3) buffer buf2_ { uint g_bin_quads[]; };

shared int s_num_quads[2], s_num_cur_quads;
shared int s_size_type;
shared int s_quads_offset;
shared int s_item_id, s_num_items;

// TODO: treat 1x1, 1x2, 2x1 & 2x2 cases separately
// TODO: filter big tris here (with bin-rasterization similarily to Laine's?)
// TODO: optimize barriers
// TODO: there are surely more ways to optimize this
// TODO: rename tris to quads

// Offsets have to be 32-bit...

shared int s_offsets[BIN_COUNT];

int getWorkItem() {
	if(LIX == 0) {
		int quads_offset = BIN_WORKGROUP_ITEMS(WGID.x, s_item_id++);
		int size_type = quads_offset < 0 ? 1 : 0;
		s_num_cur_quads = s_num_quads[size_type];
		s_quads_offset = quads_offset < 0 ? -quads_offset - 1 : quads_offset;
		s_size_type = size_type;
	}
	barrier();
	return s_quads_offset;
}

void main() {
	if(LIX < 2) {
		if(LIX == 0) {
			s_num_items = BIN_WORKGROUP_ITEMS(WGID.x, MAX_BIN_WORKGROUP_ITEMS - 1);
			s_item_id = 0;
			s_num_quads[LIX] = g_bins.num_visible_quads[LIX];
		}
		s_num_quads[LIX] = g_bins.num_visible_quads[LIX];
	}
	for(uint i = LIX; i < BIN_COUNT; i += LSIZE) {
		int workgroup_bin_count = BIN_WORKGROUP_COUNTS(WGID.x, i);
		if(workgroup_bin_count > 0)
			s_offsets[i] = atomicAdd(BIN_QUAD_OFFSETS_TEMP(i), workgroup_bin_count);
	}
	barrier();

	while(s_item_id < s_num_items) {
		barrier();
		int quads_offset = getWorkItem();
		int num_quads = s_num_cur_quads;
		int quad_idx = quads_offset + int(LIX);
		if(quad_idx >= num_quads)
			continue;

		if(s_size_type == 1)
			quad_idx = (MAX_QUADS - 1) - quad_idx;

		uvec4 aabb = decodeAABB32(g_quad_aabbs[quad_idx]);
#if BIN_SIZE == 64
		// Encodes tile ranges within a single bin
		uint tile_ranges = (aabb[0] & 0x3) | ((aabb[1] & 0x3) << 2) | ((aabb[2] & 0x3) << 4) |
						   ((aabb[3] & 0x3) << 6);
		aabb >>= BIN_SHIFT - TILE_SHIFT;
#endif
		uint bsx = aabb[0], bsy = aabb[1], bex = aabb[2], bey = aabb[3];

		for(uint by = bsy; by <= bey; by++) {
#if BIN_SIZE == 64
			uint tile_ranges_cury =
				(tile_ranges | (by < bey ? 0xc0 : 0)) & (by > bsy ? 0xf3 : 0xff);
#endif
			for(uint bx = bsx; bx <= bex; bx++) {
				uint bin_id = bx + by * BIN_COUNT_X;
				uint quad_offset = atomicAdd(s_offsets[bin_id], 1);
#if BIN_SIZE == 64
				uint tile_ranges_cur =
					(tile_ranges_cury | (bx < bex ? 0x30 : 0)) & (bx > bsx ? 0xfc : 0xff);
				g_bin_quads[quad_offset] = (quad_idx & 0xffffff) | (tile_ranges_cur << 24);
#else
				g_bin_quads[quad_offset] = quad_idx;
#endif
			}
		}
	}

	barrier();
}
