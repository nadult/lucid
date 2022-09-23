#include "shared/compute_funcs.glsl"
#include "shared/funcs.glsl"
#include "shared/scanline.glsl"
#include "shared/structures.glsl"
#include "shared/timers.glsl"

#include "%shader_debug"

// TODO: use gl_SubgroupSize, gl_SubgroupId
// TODO: simplify offsets generation; we don't have to do this in 2D

#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_shuffle_relative : require
#extension GL_ARB_shader_group_vote : require

#define LSIZE BIN_DISPATCHER_LSIZE
#define LSHIFT BIN_DISPATCHER_LSHIFT

#define SMALL_BATCH_STEPS 4
#define SMALL_BATCH_SHIFT (LSHIFT + 2)
#define SMALL_BATCH_SIZE (LSIZE * SMALL_BATCH_STEPS)

#define LARGE_BATCH_SHIFT (LSHIFT - 1)
#define LARGE_BATCH_SIZE (LSIZE / 2)

layout(local_size_x = 1024, local_size_x_id = BIN_DISPATCHER_LSIZE_ID) in;

layout(std430, binding = 0) coherent buffer lucid_info_ {
	LucidInfo g_info;
	int g_counts[];
};
layout(binding = 1) uniform lucid_config_ { LucidConfig u_config; };

layout(std430, set = 1, binding = 0) restrict readonly buffer buf1_ { uint g_quad_aabbs[]; };
layout(std430, set = 1, binding = 1) restrict writeonly buffer buf2_ { uint g_bin_quads[]; };
layout(std430, set = 1, binding = 2) restrict writeonly buffer buf3_ { uint g_bin_tris[]; };
layout(std430, set = 1, binding = 3) restrict buffer buf4_ { int g_bin_batches[]; };
layout(std430, set = 1, binding = 4) restrict readonly buffer buf6_ { uvec4 g_uvec4_storage[]; };
DEBUG_SETUP(1, 5)

#define MAX_SMALL_BATCHES (MAX_VISIBLE_QUADS / SMALL_BATCH_SIZE + 256)
#define MAX_LARGE_BATCHES (MAX_VISIBLE_QUADS / LARGE_BATCH_SIZE + 256)

#define SMALL_BATCHES(idx) g_bin_batches[(MAX_DISPATCHES * BIN_COUNT * 2) + idx]
#define LARGE_BATCHES(idx) g_bin_batches[(MAX_DISPATCHES * BIN_COUNT * 2 + MAX_SMALL_BATCHES) + idx]

shared int s_bins[BIN_COUNT];

void scanlineStep(in out ScanlineParams params, out int bmin, out int bmax) {
	float xmin = max(max(params.min[0], params.min[1]), params.min[2]);
	float xmax = min(min(params.max[0], params.max[1]), params.max[2]);
	params.min += params.step;
	params.max += params.step;

	// There can be holes between two tris, should we exploit this? Maybe it's not worth it?
	bmin = int(xmin + 1.0) >> BIN_SHIFT;
	bmax = int(xmax) >> BIN_SHIFT;
}

void countSmallQuadBins(uint quad_idx) {
	ivec4 aabb = decodeAABB28(g_quad_aabbs[quad_idx]);
	int bsx = aabb[0], bsy = aabb[1], bex = aabb[2], bey = aabb[3];
	int area = (bex - bsx + 1) * (bey - bsy + 1);

	for(int by = bsy; by <= bey; by++)
		for(int bx = bsx; bx <= bex; bx++)
			atomicAdd(s_bins[by * BIN_COUNT_X + bx], 1);

	// Handling only tris with bin area 1 to 3:
	/*atomicAdd(s_bins[bsy * BIN_COUNT_X + bsx], 1);
	if(bex != bsx || bey != bsy)
		atomicAdd(s_bins[bey * BIN_COUNT_X + bex], 1);
	int bmx = (bsx + bex) >> 1, bmy = (bsy + bey) >> 1;
	if(bmx > bsx || bmy > bsy)
		atomicAdd(s_bins[bmy * BIN_COUNT_X + bmx], 1);*/
}

void accumulateLargeTriCountsAcrossRows() {
	// Accumulating large quad counts across rows
	for(uint by = LIX >> WARP_SHIFT; by < BIN_COUNT_Y; by += LSIZE / WARP_SIZE) {
		int prev_accum = 0;
		for(uint bx = LIX & WARP_MASK; bx < BIN_COUNT_X; bx += WARP_SIZE) {
			uint idx = bx + by * BIN_COUNT_X;
			int value = s_bins[idx];
			int accum = prev_accum + subgroupInclusiveAddFast(value);
			s_bins[idx] = accum;
			prev_accum = subgroupShuffle(accum, WARP_MASK);
		}
	}
	/*if(LIX < BIN_COUNT_Y) { // Slow version
		uint by = LIX;
		int accum = 0;
		for(uint bx = 0; bx < BIN_COUNT_X; bx++) {
			uint idx = bx + by * BIN_COUNT_X;
			accum += s_bins[idx];
			s_bins[idx] = accum;
		}
	}*/
}

ScanlineParams loadScanlineParamsBin(uint tri_idx, out int bsy, out int bey) {
	uint scan_offset = STORAGE_TRI_SCAN_OFFSET + tri_idx * 2;
	uvec4 val0 = g_uvec4_storage[scan_offset + 0];
	uvec4 val1 = g_uvec4_storage[scan_offset + 1];
	return loadScanlineParamsBin(val0, val1, bsy, bey);
}

void countLargeTriBins(int quad_idx, int second_tri) {
	uint enc_aabb = g_quad_aabbs[quad_idx];
	uint cull_flag = (enc_aabb >> (30 + second_tri)) & 1;
	if(cull_flag == 1)
		return;

	uint tri_idx = quad_idx * 2 + second_tri;
	ivec4 aabb = decodeAABB28(enc_aabb);
	int bsx = aabb[0], bex = aabb[2], bsy, bey;
	ScanlineParams params = loadScanlineParamsBin(tri_idx, bsy, bey);

	for(int by = bsy; by <= bey; by++) {
		int bmin, bmax;
		scanlineStep(params, bmin, bmax);
		bmin = max(bmin, bsx), bmax = min(bmax, bex);

		if(bmax >= bmin) {
			atomicAdd(s_bins[bmin + by * BIN_COUNT_X], 1);
			if(bmax + 1 < BIN_COUNT_X)
				atomicAdd(s_bins[bmax + 1 + by * BIN_COUNT_X], -1);
		}
	}
}

shared int s_num_quads[2];
shared int s_quads_offset, s_active_work_group_id;
shared int s_first_batch[2], s_last_batch[2], s_num_batches[2];
shared int s_num_finished_batches[2], s_num_all_batches[2];

void countSmallQuads() {
	START_TIMER();
	for(uint i = LIX; i < BIN_COUNT; i += LSIZE)
		s_bins[i] = 0;
	barrier();

	// Computing small quads bin coverage
	int num_quads = s_num_quads[0];
	while(true) {
		if(LIX == 0) {
			int quads_offset = atomicAdd(g_info.num_counted_quads[0], SMALL_BATCH_SIZE);
			if(quads_offset < num_quads) {
				int batch = quads_offset >> SMALL_BATCH_SHIFT;
				int last_batch = s_last_batch[0];
				if(last_batch == -1)
					s_first_batch[0] = batch;
				else
					SMALL_BATCHES(last_batch) = batch;
				s_last_batch[0] = batch;
				s_num_batches[0]++;
			}
			s_quads_offset = quads_offset;
		}
		barrier();

		int quad_offset = s_quads_offset;
		if(quad_offset >= num_quads)
			break;

		for(int s = 0; s < SMALL_BATCH_STEPS; s++) {
			int quad_idx = quad_offset + (LSIZE * s) + int(LIX);
			if(quad_idx >= num_quads)
				break;
			countSmallQuadBins(quad_idx);
		}
		barrier();
	}

	UPDATE_TIMER(0);
	barrier();
	if(s_num_batches[0] == 0)
		return;

	// Copying bin counters to global memory buffer
	uint wg_offset = WGID.x * BIN_COUNT * 2;
	for(uint i = LIX; i < BIN_COUNT; i += LSIZE) {
		g_bin_batches[wg_offset + i] = s_bins[i];
		if(s_bins[i] > 0)
			atomicAdd(BIN_QUAD_COUNTS(i), s_bins[i]);
	}
}

void countLargeTris() {
	START_TIMER();
	// Computing large quads bin coverage
	int num_quads = s_num_quads[1];
	while(true) {
		if(LIX == 0) {
			int quads_offset = atomicAdd(g_info.num_counted_quads[1], LARGE_BATCH_SIZE);
			if(quads_offset < num_quads) {
				int batch = quads_offset >> LARGE_BATCH_SHIFT;
				int last_batch = s_last_batch[1];
				if(last_batch == -1)
					s_first_batch[1] = batch;
				else
					LARGE_BATCHES(last_batch) = batch;
				s_last_batch[1] = batch;
				s_num_batches[1]++;
			}
			s_quads_offset = quads_offset;
		}
		barrier();

		int large_quads_offset = s_quads_offset;
		if(large_quads_offset >= num_quads)
			break;

		int large_quad_idx = large_quads_offset + int(LIX >> 1);
		if(large_quad_idx < num_quads)
			countLargeTriBins((MAX_VISIBLE_QUADS - 1) - large_quad_idx, int(LIX & 1));

		barrier();
	}

	barrier();
	UPDATE_TIMER(1);

	// Thread groups which didn't do any estimation can quit early:
	// they won't participate in dispatching either
	if(s_num_batches[1] == 0)
		return;
	accumulateLargeTriCountsAcrossRows();
	barrier();

	// Copying bin counters to global memory buffer
	uint wg_offset = WGID.x * BIN_COUNT * 2 + BIN_COUNT;
	for(uint i = LIX; i < BIN_COUNT; i += LSIZE) {
		g_bin_batches[wg_offset + i] = s_bins[i];
		if(s_bins[i] > 0)
			atomicAdd(BIN_TRI_COUNTS(i), s_bins[i]);
	}
}

void main() {
	INIT_TIMERS();
	if(LIX < 2) {
		int num_quads = g_info.num_visible_quads[LIX];
		s_num_quads[LIX] = num_quads;
		int batch_size = LIX == 0 ? SMALL_BATCH_SIZE : LARGE_BATCH_SIZE;
		int batch_shift = LIX == 0 ? SMALL_BATCH_SHIFT : LARGE_BATCH_SHIFT;
		s_num_all_batches[LIX] = (num_quads + (batch_size - 1)) >> batch_shift;
		s_first_batch[LIX] = -1;
		s_last_batch[LIX] = -1;
		s_num_batches[LIX] = 0;
	}
	barrier();

	if(s_num_all_batches[0] > 0) {
		for(uint i = LIX; i < BIN_COUNT; i += LSIZE)
			s_bins[i] = 0;
		barrier();
		countSmallQuads();
	}

	if(s_num_all_batches[1] > 0) {
		barrier();
		for(uint i = LIX; i < BIN_COUNT; i += LSIZE)
			s_bins[i] = 0;
		barrier();
		countLargeTris();
	}

	barrier();
	if(LIX < 2) {
		g_info.dispatcher_first_batch[LIX][WGID.x] = s_first_batch[LIX];
		g_info.dispatcher_num_batches[LIX][WGID.x] = s_num_batches[LIX];
	}

	COMMIT_TIMERS(g_info.bin_dispatcher_timers);
}
