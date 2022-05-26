// $$include funcs declarations

// TODO: don't run too many groups if we have small amount of data (indirect dispatch)

#define LID gl_LocalInvocationID
#define LIX gl_LocalInvocationIndex
#define WGID gl_WorkGroupID

#define LSIZE BINNING_LSIZE
#define LSHIFT BINNING_LSHIFT

#define ITEM_STEPS 1
#define ITEM_SIZE (LSIZE * ITEM_STEPS)

layout(local_size_x = LSIZE) in;
layout(std430, binding = 0) buffer buf0_ { uint g_quad_aabbs[]; };
coherent BIN_COUNTERS_BUFFER(1);
layout(std430, binding = 3) buffer buf3_ { uint g_bin_quads[]; };
layout(std430, binding = 4) buffer buf4_ { uint g_bin_row_quads[]; };

shared int s_bins[BIN_COUNT];
shared int s_rows[BIN_COUNT_Y + 1];

// Constants useful for efficient processing of bin counts
const int xbin_warps = LSIZE / BIN_COUNT_X, ybin_warps = LSIZE / BIN_COUNT_Y;
const int xbin_step = int(log2(xbin_warps)), ybin_step = int(log2(ybin_warps));
const int xbin_count = 1 << xbin_step, ybin_count = 1 << ybin_step;

int prefixSum32(int accum) {
	int temp;
	uint sub_id = LIX & 31;
	temp = shuffleUpNV(accum, 1, 32), accum += sub_id >= 1 ? temp : 0;
	temp = shuffleUpNV(accum, 2, 32), accum += sub_id >= 2 ? temp : 0;
	temp = shuffleUpNV(accum, 4, 32), accum += sub_id >= 4 ? temp : 0;
	temp = shuffleUpNV(accum, 8, 32), accum += sub_id >= 8 ? temp : 0;
	temp = shuffleUpNV(accum, 16, 32), accum += sub_id >= 16 ? temp : 0;
	return accum;
}

shared int s_segments[LSIZE];

shared int s_num_row_quads;
shared int s_quads_offset;

void main() {
	if(LIX == 0)
		s_num_row_quads = g_bins.num_bin_rows_quads;
	for(uint i = LIX; i < BIN_COUNT; i += LSIZE)
		s_bins[i] = 0;
	if(LIX < BIN_COUNT_Y)
		s_rows[LIX] = 0;
	barrier();

	// Jak równomiernie roz³o¿yæ pracê ?
	// Ka¿dy quad-row generuje ileœ tam bloków w rzêdzie
	// Mamy w sumie ileœ tam bloków do przetworzenia, jak je roz³o¿yæ równomiernie ?
	// Róbmy równomiernie w ramach warpa ?
	//
	// Generujemy sample ? aby to robiæ efektywnie musielibyœmy mieæ segmenty
	// To mo¿e od razu przy generowaniu rzêdów generujmy segmenty? zapisujmy do segmentu
	//
	// Wiemy ile ma byæ docelowo bin-quadów, wiêc wiemy ile ma byæ segmentów
	// Segment ma np. 1024 bin-quady

	// Czy da siê bez segmentów ?
	// Mo¿e móg³bym ew. wyznaczyæ segmenty tutaj ?

	// Dispatching large quads from rows to bins
	int row_num_quads = s_num_row_quads;
	while(true) {
		if(LIX == 0)
			s_quads_offset = atomicAdd(g_bins.num_dispatched_visible_quads[2], LSIZE);
		barrier();

		int row_quad_offset = s_quads_offset;
		if(row_quad_offset >= row_num_quads)
			break;

		// TODO: only clear selected rows
		for(uint i = LIX; i < BIN_COUNT; i += LSIZE)
			s_bins[i] = 0;
		barrier();

		int row_quad_idx = row_quad_offset + int(LIX);
		uint quad_idx;

		uint by, bsx, bex;
		int num_samples = 0;

		if(row_quad_idx < row_num_quads) {
			quad_idx = g_bin_row_quads[row_quad_idx];
			by = quad_idx >> 24;
			quad_idx &= 0xffffff;

			const uint shift = BIN_SIZE == 64 ? BIN_SHIFT - TILE_SHIFT : 0;
			uvec4 aabb = decodeAABB32(g_quad_aabbs[quad_idx]) >> shift;
			bsx = aabb[0], bex = aabb[2];
			num_samples = int(bex - bsx + 1);
			//for(uint bx = bsx; bx <= bex; bx++)
			//	atomicAdd(s_bins[bx + by * BIN_COUNT_X], 1);

			atomicAdd(s_bins[bsx + by * BIN_COUNT_X], 1);
			if(bex + 1 < BIN_COUNT_X)
				atomicAdd(s_bins[bex + 1 + by * BIN_COUNT_X], -1);
		}

		int warp_offset = int(LIX & ~31), warp_sub_id = int(LIX & 31);
		int sample_offset = prefixSum32(num_samples);
		int warp_num_samples = shuffleNV(sample_offset, 31, 32);
		sample_offset -= num_samples;

		// TODO: use float division somehow? or uint division at least
		int segment_size = (warp_num_samples + 31) / 32;
		int segment_id = sample_offset / segment_size;
		int segment_offset = sample_offset - segment_id * segment_size;
		s_segments[LIX] = warp_sub_id;
		if(segment_offset == 0)
			s_segments[warp_offset + segment_id] = warp_sub_id;
		// TODO: limit number of tested forward segments? (depends on BIN_COUNT_X)
		for(int k = 1; segment_offset + num_samples > segment_size * k; k++)
			s_segments[warp_offset + segment_id + k] = warp_sub_id;

		barrier();

		// Accumulating large quad counts across rows
		if(LIX < BIN_COUNT_Y * ybin_count) {
			uint by = LIX >> ybin_step;
			int accum = 0;
			for(uint bx = 0, subx = LIX & (ybin_count - 1); bx < BIN_COUNT_X; bx += ybin_count) {
				uint idx = bx + subx + by * BIN_COUNT_X;
				int cur = bx + subx < BIN_COUNT_X ? s_bins[idx] : 0, temp;
				for(int i = 1; i < ybin_count; i <<= 1) {
					temp = shuffleUpNV(cur, i, ybin_count);
					cur += subx >= i ? temp : 0;
				}
				cur += accum;
				if(bx + subx < BIN_COUNT_X)
					s_bins[idx] = cur;
				accum = shuffleNV(cur, ybin_count - 1, ybin_count);
			}
		}
		barrier();
		// Reserving space for quad indices in bins
		for(uint i = LIX; i < BIN_COUNT; i += LSIZE)
			if(s_bins[i] > 0)
				s_bins[i] = atomicAdd(BIN_QUAD_OFFSETS_TEMP(i), s_bins[i]);
		barrier();

		int first_quad = s_segments[LIX];
		int first_sample_id = warp_sub_id * segment_size;
		int cur_offset = first_sample_id - shuffleNV(sample_offset, first_quad, 32);
		int cur_samples = min(segment_size, warp_num_samples - first_sample_id);

		for(int i = 0; i < segment_size; i++) {
			uvec4 quad_data = shuffleNV(uvec4(quad_idx, by, bsx, bex), first_quad, 32);
			if(i < cur_samples) {
				uint pos = quad_data[2] + cur_offset;
				uint quad_offset = atomicAdd(s_bins[pos + quad_data[1] * BIN_COUNT_X], 1);
				// TODO: tile ranges
				g_bin_quads[quad_offset] = quad_data[0];
				if(pos == quad_data[3]) // last sample
					first_quad++, cur_offset = 0;
				else
					cur_offset++;
			}
		}

		/*
		// Dispatching quad indices to bins
		// TODO: this work can be more fairly divided among threads
		if(row_quad_idx < row_num_quads)
			for(uint bx = bsx; bx <= bex; bx++) {
				uint quad_offset = atomicAdd(s_bins[bx + by * BIN_COUNT_X], 1);
				// TODO: tile ranges
				g_bin_quads[quad_offset] = quad_idx;
			}*/
		barrier();
	}
}
