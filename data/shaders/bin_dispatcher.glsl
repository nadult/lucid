// Copyright (C) Krzysztof Jakubowski <nadult@fastmail.fm>
// This file is part of LucidRaster. See license.txt for details.

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
#extension GL_KHR_shader_subgroup_vote : require

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
shared int s_temp[LSIZE], s_temp2[SUBGROUP_SIZE];

void scanlineStep(in out ScanlineParams params, out int bmin, out int bmax) {
	float xmin = max(max(params.min[0], params.min[1]), params.min[2]);
	float xmax = min(min(params.max[0], params.max[1]), params.max[2]);
	params.min += params.step;
	params.max += params.step;

	// There can be holes between two tris, should we exploit this? Maybe it's not worth it?
	bmin = int(xmin + 1.0) >> BIN_SHIFT;
	bmax = int(xmax) >> BIN_SHIFT;
}

void dispatchQuad(int quad_idx) {
	uint enc_aabb = g_quad_aabbs[quad_idx];
	uint cull_flags = enc_aabb & 0xf0000000;
	uint bin_quad_idx = uint(quad_idx) | cull_flags;
	ivec4 aabb = decodeAABB28(enc_aabb);
	int bsx = aabb[0], bsy = aabb[1], bex = aabb[2], bey = aabb[3];

	for(int by = bsy; by <= bey; by++) {
		for(int bx = bsx; bx <= bex; bx++) {
			uint bin_id = bx + by * BIN_COUNT_X;
			uint quad_offset = atomicAdd(s_bins[bin_id], 1);
			g_bin_quads[quad_offset] = bin_quad_idx;
		}
	}
}

ScanlineParams loadScanlineParamsBin(uint tri_idx, out int bsy, out int bey) {
	uint scan_offset = STORAGE_TRI_SCAN_OFFSET + tri_idx * 2;
	uvec4 val0 = g_uvec4_storage[scan_offset + 0];
	uvec4 val1 = g_uvec4_storage[scan_offset + 1];
	return loadScanlineParamsBin(val0, val1, bsy, bey);
}

void dispatchLargeTriSimple(int large_quad_idx, int second_tri, int num_quads) {
	if(large_quad_idx >= num_quads)
		return;
	int quad_idx = (MAX_VISIBLE_QUADS - 1) - large_quad_idx;
	int tri_idx = quad_idx * 2 + second_tri;

	uint enc_aabb = g_quad_aabbs[quad_idx];
	uint cull_flag = (enc_aabb >> (30 + second_tri)) & 1;
	if(cull_flag == 1)
		return;

	ivec4 aabb = decodeAABB28(enc_aabb);
	int bsx = aabb[0], bex = aabb[2], bsy, bey;
	ScanlineParams params = loadScanlineParamsBin(tri_idx, bsy, bey);

	for(int by = bsy; by <= bey; by++) {
		int bmin, bmax;
		scanlineStep(params, bmin, bmax);
		bmin = max(bmin, bsx), bmax = min(bmax, bex);

		for(int bx = bmin; bx <= bmax; bx++) {
			uint bin_id = bx + by * BIN_COUNT_X;
			g_bin_tris[atomicAdd(s_bins[bin_id], 1)] = tri_idx;
		}
	}
}

// This is an optimized tri dispatcher which is more work efficient. It is especially
// useful if there is a large variation in quad sizes and for large tris in general.
//
// Work balancing happens at each bin row. First we find out how many bins do we have
// to write to and then we divide this work equally across all threads within a subgroup.
// We do this by dividing those items into SUBGROUP_SIZE segments and then assigning 1 segment
// to each thread.
void dispatchLargeTriBalanced(int large_quad_idx, int second_tri, int num_quads) {
	bool is_valid = large_quad_idx < num_quads;
	if(subgroupAll(!is_valid))
		return;

	ScanlineParams params;
	uint tri_idx;
	int bsy = 0, bey = -1, bsx, bex;

	if(is_valid) {
		int quad_idx = (MAX_VISIBLE_QUADS - 1) - large_quad_idx;
		uint enc_aabb = g_quad_aabbs[quad_idx];
		uint cull_flag = (enc_aabb >> (30 + second_tri)) & 1;
		ivec4 aabb = decodeAABB28(enc_aabb);
		bsx = aabb[0], bex = aabb[2];
		if(cull_flag == 0) {
			tri_idx = quad_idx * 2 + second_tri;
			params = loadScanlineParamsBin(tri_idx, bsy, bey);
		}
	}

	for(int by = bsy; subgroupAny(by <= bey); by++) {
		int bmin = 0, bmax = -1;
		if(by <= bey) {
			scanlineStep(params, bmin, bmax);
			bmin = max(bmin, bsx), bmax = min(bmax, bex);
		}

		int num_samples = max(0, bmax - bmin + 1);
		if(subgroupAll(num_samples == 0))
			continue;

		int sample_offset = subgroupInclusiveAddFast(num_samples);
		int subgroup_num_samples = subgroupShuffle(sample_offset, SUBGROUP_MASK);
		sample_offset -= num_samples;

		int subgroup_offset = int(LIX & ~SUBGROUP_MASK), thread_id = int(LIX & SUBGROUP_MASK);
		int segment_size = (subgroup_num_samples + SUBGROUP_MASK) / SUBGROUP_SIZE;
		int segment_id = sample_offset / segment_size;
		int segment_offset = sample_offset - segment_id * segment_size;
		if(num_samples > 0) {
			if(segment_offset == 0)
				s_temp[subgroup_offset + segment_id] = thread_id;
			for(int k = 1; segment_offset + num_samples > segment_size * k; k++)
				s_temp[subgroup_offset + segment_id + k] = thread_id;
		}

		uint cur_src_thread_id = s_temp[LIX];
		int cur_sample_id = thread_id * segment_size;
		int cur_offset = cur_sample_id - subgroupShuffle(sample_offset, cur_src_thread_id);
		int cur_num_samples = min(subgroup_num_samples - cur_sample_id, segment_size);
		int base_bin_id = by * BIN_COUNT_X + bmin;

		int i = 0;
		while(subgroupAny(i < cur_num_samples)) {
			uint cur_tri_idx = subgroupShuffle(tri_idx, cur_src_thread_id);
			int cur_bin_id = subgroupShuffle(base_bin_id, cur_src_thread_id);
			int cur_width = subgroupShuffle(num_samples, cur_src_thread_id);

			if(cur_width == 0) {
				cur_src_thread_id++;
				continue;
			}
			if(i < cur_num_samples) {
				uint tri_offset = atomicAdd(s_bins[cur_bin_id + cur_offset], 1);
				g_bin_tris[tri_offset] = cur_tri_idx;
				cur_offset++;
				if(cur_offset == cur_width)
					cur_offset = 0, cur_src_thread_id++;
				i++;
			}
		}
	}
}

shared int s_num_quads[2], s_quads_offset;
shared int s_first_batch[2], s_num_batches[2];
shared int s_num_finished_batches[2];

void dispatchSmallQuads() {
	START_TIMER();
	uint wg_offset = WGID.x * BIN_COUNT * 2;
	for(uint i = LIX; i < BIN_COUNT; i += LSIZE) {
		s_bins[i] = g_bin_batches[wg_offset + i];
		if(s_bins[i] > 0)
			s_bins[i] = atomicAdd(BIN_QUAD_OFFSETS_TEMP(i), s_bins[i]);
	}
	barrier();

	int num_quads = s_num_quads[0];
	while(s_num_batches[0] > 0) {
		barrier();
		if(LIX == 0) {
			int batch = s_first_batch[0];
			s_first_batch[0] = SMALL_BATCHES(batch);
			s_num_batches[0]--;
			s_quads_offset = batch << SMALL_BATCH_SHIFT;
		}
		barrier();
		int quads_offset = s_quads_offset;
		for(int s = 0; s < SMALL_BATCH_STEPS; s++) {
			int quad_idx = quads_offset + LSIZE * s + int(LIX);
			if(quad_idx >= num_quads)
				break;
			dispatchQuad(quad_idx);
		}
	}
	barrier();
	UPDATE_TIMER(2);
}

void dispatchLargeTris() {
	START_TIMER();
	uint wg_offset = WGID.x * BIN_COUNT * 2 + BIN_COUNT;
	for(uint i = LIX; i < BIN_COUNT; i += LSIZE) {
		s_bins[i] = g_bin_batches[wg_offset + i];
		if(s_bins[i] > 0)
			s_bins[i] = atomicAdd(BIN_TRI_OFFSETS_TEMP(i), s_bins[i]);
	}
	barrier();

	int num_quads = s_num_quads[1];
	while(s_num_batches[1] > 0) {
		barrier();
		if(LIX == 0) {
			int batch = s_first_batch[1];
			s_first_batch[1] = LARGE_BATCHES(batch);
			s_num_batches[1]--;
			s_quads_offset = batch << LARGE_BATCH_SHIFT;
		}
		barrier();
		int large_quad_idx = s_quads_offset + int(LIX >> 1);
		dispatchLargeTriBalanced(large_quad_idx, int(LIX & 1), num_quads);
	}
	UPDATE_TIMER(3);
}

void main() {
	INIT_TIMERS();

	if(LIX < 2) {
		int num_quads = g_info.num_visible_quads[LIX];
		s_num_quads[LIX] = num_quads;
		s_first_batch[LIX] = g_info.dispatcher_first_batch[LIX][WGID.x];
		s_num_batches[LIX] = g_info.dispatcher_num_batches[LIX][WGID.x];
	}
	barrier();

	if(s_num_batches[0] > 0)
		dispatchSmallQuads();
	if(s_num_batches[1] > 0) {
		barrier();
		dispatchLargeTris();
	}

	COMMIT_TIMERS(g_info.bin_dispatcher_timers);
}
