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

#define SMALL_TASK_STEPS 4
#define SMALL_TASK_SHIFT (LSHIFT + 2)
#define SMALL_TASK_SIZE (LSIZE * SMALL_TASK_STEPS)

#define LARGE_TASK_SHIFT (LSHIFT - 1)
#define LARGE_TASK_SIZE (LSIZE / 2)

layout(local_size_x = 1024, local_size_x_id = BIN_DISPATCHER_LSIZE_ID) in;

layout(std430, binding = 0) coherent buffer lucid_info_ {
	LucidInfo g_info;
	int g_counts[];
};
layout(binding = 1) uniform lucid_config_ { LucidConfig u_config; };

layout(std430, set = 1, binding = 0) restrict readonly buffer buf1_ { uint g_quad_aabbs[]; };
layout(std430, set = 1, binding = 1) restrict writeonly buffer buf2_ { uint g_bin_quads[]; };
layout(std430, set = 1, binding = 2) restrict writeonly buffer buf3_ { uint g_bin_tris[]; };
layout(std430, set = 1, binding = 3) restrict buffer buf4_ { int g_tasks[]; };
layout(std430, set = 1, binding = 4) restrict readonly buffer buf5_ { uvec4 g_uvec4_storage[]; };
DEBUG_SETUP(1, 5)

shared int s_bins[BIN_COUNT];
shared int s_temp[LSIZE], s_temp2[WARP_SIZE];

void scanlineStep(in out ScanlineParams params, out int bmin, out int bmax) {
	float xmin = max(max(params.min[0], params.min[1]), params.min[2]);
	float xmax = min(min(params.max[0], params.max[1]), params.max[2]);
	params.min += params.step;
	params.max += params.step;

	// There can be holes between two tris, should we exploit this? Maybe it's not worth it?
	bmin = int(xmin + 1.0) >> BIN_SHIFT;
	bmax = int(xmax) >> BIN_SHIFT;
}

// Inclusive
// TODO: make portable
int prefixSum32(int accum) {
	int temp;
	uint thread_id = LIX & 31;
	temp = subgroupShuffleUp(accum, 1), accum += thread_id >= 1 ? temp : 0;
	temp = subgroupShuffleUp(accum, 2), accum += thread_id >= 2 ? temp : 0;
	temp = subgroupShuffleUp(accum, 4), accum += thread_id >= 4 ? temp : 0;
	temp = subgroupShuffleUp(accum, 8), accum += thread_id >= 8 ? temp : 0;
	temp = subgroupShuffleUp(accum, 16), accum += thread_id >= 16 ? temp : 0;
	return accum;
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

#define BIN_COUNT_WARPS (BIN_COUNT / WARP_SIZE)
#if BIN_COUNT_WARPS > LSIZE
#error "Fix me"
#endif

void accumulateLargeTriCountsAcrossRows() {
	// Accumulating large quad counts across rows
#if WARP_SIZE == 32
	for(uint by = LIX >> WARP_SHIFT; by < BIN_COUNT_Y; by += LSIZE / WARP_SIZE) {
		int prev_accum = 0;
		for(uint bx = LIX & WARP_MASK; bx < BIN_COUNT_X; bx += WARP_SIZE) {
			uint idx = bx + by * BIN_COUNT_X;
			int value = s_bins[idx];
			int accum = prev_accum + prefixSum32(value);
			s_bins[idx] = accum;
			prev_accum = subgroupShuffle(accum, WARP_MASK);
		}
	}
#else
	if(LIX < BIN_COUNT_Y) { // Slow version
		uint by = LIX;
		int accum = 0;
		for(uint bx = 0; bx < BIN_COUNT_X; bx++) {
			uint idx = bx + by * BIN_COUNT_X;
			accum += s_bins[idx];
			s_bins[idx] = accum;
		}
	}
#endif
}


void computeQuadOffsets() {
#if WARP_SIZE == 32
	for(uint idx = LIX; idx < BIN_COUNT; idx += LSIZE) {
		int value = BIN_QUAD_COUNTS(idx);
		int accum = prefixSum32(value);
		BIN_QUAD_OFFSETS(idx) = accum - value;
		if((idx & 31) == 31)
			s_temp[idx >> 5] = accum;
	}
	barrier();
	if(LIX < BIN_COUNT_WARPS)
		s_temp[LIX] = prefixSum32(s_temp[LIX]);
	barrier();
	if(LIX < BIN_COUNT_WARPS / WARP_SIZE)
		s_temp2[LIX] = prefixSum32(s_temp[(LIX << 5) + 31]);
	barrier();
	if(LIX < BIN_COUNT_WARPS && LIX >= WARP_SIZE)
		s_temp[LIX] += s_temp2[int(LIX >> 5) - 1];
	barrier();
	groupMemoryBarrier();
	for(uint idx = LIX; idx < BIN_COUNT; idx += LSIZE) {
		int widx = int(idx >> 5) - 1;
		int accum = BIN_QUAD_OFFSETS(idx);
		if(widx >= 0)
			accum += s_temp[widx];
		BIN_QUAD_OFFSETS(idx) = accum;
		BIN_QUAD_OFFSETS_TEMP(idx) = accum;
	}
#else
	// Slow version
	if(LIX < BIN_COUNT_Y) { 
		uint by = LIX;
		int accum = 0;
		for(uint bx = 0; bx < BIN_COUNT_X; bx++) {
			uint idx = bx + by * BIN_COUNT_X;
			accum += BIN_QUAD_COUNTS(idx);
			BIN_QUAD_OFFSETS(idx) = accum;
		}
	}
	barrier();
	// Storing accumulated bin quad counts for each row
	if(LIX < BIN_COUNT_Y)
		s_temp[LIX] = BIN_QUAD_OFFSETS((BIN_COUNT_X - 1) + LIX * BIN_COUNT_X);
	barrier();
	if(LIX < BIN_COUNT_X) {
		uint bx = LIX;
		int accum = 0;
		for(uint by = 0; by < BIN_COUNT_Y; by++) {
			uint idx = bx + by * BIN_COUNT_X;
			accum += by > 0 ? s_temp[by - 1] : 0;
			int value = BIN_QUAD_OFFSETS(idx) + accum - BIN_QUAD_COUNTS(idx);
			BIN_QUAD_OFFSETS(idx) = value;
			BIN_QUAD_OFFSETS_TEMP(idx) = value;
		}
	}
#endif
	barrier();
}

void computeTriOffsets() {
#if WARP_SIZE == 32
	for(uint idx = LIX; idx < BIN_COUNT; idx += LSIZE) {
		int value = BIN_TRI_COUNTS(idx);
		int accum = prefixSum32(value);
		BIN_TRI_OFFSETS(idx) = accum - value;
		if((idx & 31) == 31)
			s_temp[idx >> 5] = accum;
	}
	barrier();
	if(LIX < BIN_COUNT_WARPS)
		s_temp[LIX] = prefixSum32(s_temp[LIX]);
	barrier();
	if(LIX < BIN_COUNT_WARPS / WARP_SIZE)
		s_temp2[LIX] = prefixSum32(s_temp[(LIX << 5) + 31]);
	barrier();
	if(LIX < BIN_COUNT_WARPS && LIX >= WARP_SIZE)
		s_temp[LIX] += s_temp2[int(LIX >> 5) - 1];
	barrier();
	groupMemoryBarrier();
	for(uint idx = LIX; idx < BIN_COUNT; idx += LSIZE) {
		int widx = int(idx >> 5) - 1;
		int accum = BIN_TRI_OFFSETS(idx);
		if(widx >= 0)
			accum += s_temp[widx];
		BIN_TRI_OFFSETS(idx) = accum;
		BIN_TRI_OFFSETS_TEMP(idx) = accum;
	}
#else
	if(LIX < BIN_COUNT_Y) { // Slow version
		uint by = LIX;
		int accum = 0;
		for(uint bx = 0; bx < BIN_COUNT_X; bx++) {
			uint idx = bx + by * BIN_COUNT_X;
			accum += BIN_TRI_COUNTS(idx);
			BIN_TRI_OFFSETS(idx) = accum;
		}
	}
	barrier();
	// Storing accumulated bin quad counts for each row
	if(LIX < BIN_COUNT_Y)
		s_temp[LIX] = BIN_TRI_OFFSETS((BIN_COUNT_X - 1) + LIX * BIN_COUNT_X);
	barrier();
	if(LIX < BIN_COUNT_X) { // Slow version
		uint bx = LIX;
		int accum = 0;
		for(uint by = 0; by < BIN_COUNT_Y; by++) {
			uint idx = bx + by * BIN_COUNT_X;
			accum += by > 0 ? s_temp[by - 1] : 0;
			int value = BIN_TRI_OFFSETS(idx) + accum - BIN_TRI_COUNTS(idx);
			BIN_TRI_OFFSETS(idx) = value;
			BIN_TRI_OFFSETS_TEMP(idx) = value;
		}
	}
#endif
	barrier();
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
// to write to and then we divide this work equally across all threads within a warp.
// We do this by dividing those items into 32 segments and then assigning 1 segment
// to each thread.
void dispatchLargeTriBalanced(int large_quad_idx, int second_tri, int num_quads) {
	bool is_valid = large_quad_idx < num_quads;
	if(allInvocationsARB(!is_valid))
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

	for(int by = bsy; anyInvocationARB(by <= bey); by++) {
		int bmin = 0, bmax = -1;
		if(by <= bey) {
			scanlineStep(params, bmin, bmax);
			bmin = max(bmin, bsx), bmax = min(bmax, bex);
		}

		int num_samples = max(0, bmax - bmin + 1);
		if(allInvocationsARB(num_samples == 0))
			continue;

		int sample_offset = prefixSum32(num_samples);
		int warp_num_samples = subgroupShuffle(sample_offset, 31);
		sample_offset -= num_samples;

		int warp_offset = int(LIX & ~31), thread_id = int(LIX & 31);
		int segment_size = (warp_num_samples + 31) / 32;
		int segment_id = sample_offset / segment_size;
		int segment_offset = sample_offset - segment_id * segment_size;
		if(num_samples > 0) {
			if(segment_offset == 0)
				s_temp[warp_offset + segment_id] = thread_id;
			for(int k = 1; segment_offset + num_samples > segment_size * k; k++)
				s_temp[warp_offset + segment_id + k] = thread_id;
		}

		uint cur_src_thread_id = s_temp[LIX];
		int cur_sample_id = thread_id * segment_size;
		int cur_offset = cur_sample_id - subgroupShuffle(sample_offset, cur_src_thread_id);
		int cur_num_samples = min(warp_num_samples - cur_sample_id, segment_size);
		int base_bin_id = by * BIN_COUNT_X + bmin;

		int i = 0;
		while(anyInvocationARB(i < cur_num_samples)) {
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

shared int s_num_quads[2];
shared int s_quads_offset, s_active_work_group_id;

shared int s_first_task[2], s_last_task[2], s_num_tasks[2];
shared int s_num_finished_tasks[2], s_num_all_tasks[2];

#define MAX_SMALL_TASKS (MAX_VISIBLE_QUADS / SMALL_TASK_SIZE + 256)
#define MAX_LARGE_TASKS (MAX_VISIBLE_QUADS / LARGE_TASK_SIZE + 256)

#define LARGE_TASKS_OFFSET MAX_SMALL_TASKS

void waitForSecondPhase(int type_index) {
	// TODO: can we do something useful while waiting?
	// We could run bin categorizer here ! All it needs is bin quad counts
	if(LIX == 0)
		while(g_info.a_bin_dispatcher_phase[type_index] == 0)
			memoryBarrier();
	while(g_info.a_bin_dispatcher_phase[type_index] == 0)
		memoryBarrier();
	barrier();
}

void processSmallQuads() {
	START_TIMER();
	for(uint i = LIX; i < BIN_COUNT; i += LSIZE)
		s_bins[i] = 0;
	barrier();

	// Computing small quads bin coverage
	int num_quads = s_num_quads[0];
	while(true) {
		if(LIX == 0) {
			int quads_offset = atomicAdd(g_info.num_estimated_quads[0], SMALL_TASK_SIZE);
			if(quads_offset < num_quads) {
				int task = quads_offset >> SMALL_TASK_SHIFT;
				int last_task = s_last_task[0];
				if(last_task == -1)
					s_first_task[0] = task;
				else
					g_tasks[last_task] = task;
				s_last_task[0] = task;
				s_num_tasks[0]++;
			}
			s_quads_offset = quads_offset;
		}
		barrier();

		int quad_offset = s_quads_offset;
		if(quad_offset >= num_quads)
			break;

		for(int s = 0; s < SMALL_TASK_STEPS; s++) {
			int quad_idx = quad_offset + (LSIZE * s) + int(LIX);
			if(quad_idx >= num_quads)
				break;
			countSmallQuadBins(quad_idx);
		}
		barrier();
	}

	UPDATE_TIMER(0);
	barrier();

	// Thread groups which didn't do any estimation can quit early:
	// they won't participate in dispatching either
	if(s_num_tasks[0] == 0)
		return;

	// Copying bin counters to global memory buffer
	for(uint i = LIX; i < BIN_COUNT; i += LSIZE)
		if(s_bins[i] > 0)
			atomicAdd(BIN_QUAD_COUNTS(i), s_bins[i]);
	barrier();

	// Finishing estimation phase
	if(LIX == 0) {
		int num_tasks = s_num_tasks[0];
		s_active_work_group_id = atomicAdd(g_info.a_bin_dispatcher_work_groups[0], 1);
		s_num_finished_tasks[0] =
			atomicAdd(g_info.a_bin_dispatcher_items[0], num_tasks) + num_tasks;
		g_info.dispatcher_task_counts[s_active_work_group_id] = num_tasks;
	}
	barrier();

	// Last group is responsible for computing bin offsets
	if(s_num_finished_tasks[0] == s_num_all_tasks[0]) {
		memoryBarrier();
		computeQuadOffsets();
		memoryBarrier();
		if(LIX == 0)
			atomicExchange(g_info.a_bin_dispatcher_phase[0], 1);
	}

	// Waiting until all bin offsets are computed
	waitForSecondPhase(0);
	UPDATE_TIMER(1);

	// Reserving space for quad indices in bins
	for(uint i = LIX; i < BIN_COUNT; i += LSIZE)
		if(s_bins[i] > 0)
			s_bins[i] = atomicAdd(BIN_QUAD_OFFSETS_TEMP(i), s_bins[i]);
	barrier();

	// Dispatching small quads
	while(s_num_tasks[0] > 0) {
		barrier();
		if(LIX == 0) {
			int task = s_first_task[0];
			s_first_task[0] = g_tasks[task];
			s_num_tasks[0]--;
			s_quads_offset = task << SMALL_TASK_SHIFT;
		}
		barrier();
		int quads_offset = s_quads_offset;
		for(int s = 0; s < SMALL_TASK_STEPS; s++) {
			int quad_idx = quads_offset + LSIZE * s + int(LIX);
			if(quad_idx >= num_quads)
				break;
			dispatchQuad(quad_idx);
		}
	}
	barrier();
	UPDATE_TIMER(2);
}

void processLargeTris() {
	START_TIMER();
	// Computing large quads bin coverage
	int num_quads = s_num_quads[1];
	while(true) {
		if(LIX == 0) {
			int quads_offset = atomicAdd(g_info.num_estimated_quads[1], LARGE_TASK_SIZE);
			if(quads_offset < num_quads) {
				int task = quads_offset >> LARGE_TASK_SHIFT;
				int last_task = s_last_task[1];
				if(last_task == -1)
					s_first_task[1] = task;
				else
					g_tasks[LARGE_TASKS_OFFSET + last_task] = task;
				s_last_task[1] = task;
				s_num_tasks[1]++;
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
	UPDATE_TIMER(3);

	// Thread groups which didn't do any estimation can quit early:
	// they won't participate in dispatching either
	if(s_num_tasks[1] == 0)
		return;
	accumulateLargeTriCountsAcrossRows();
	barrier();

	// Copying bin counters to global memory buffer
	for(uint i = LIX; i < BIN_COUNT; i += LSIZE)
		if(s_bins[i] > 0)
			atomicAdd(BIN_TRI_COUNTS(i), s_bins[i]);
	barrier();

	// Finishing estimation phase
	if(LIX == 0) {
		int num_tasks = s_num_tasks[1];
		s_active_work_group_id = atomicAdd(g_info.a_bin_dispatcher_work_groups[1], 1);
		s_num_finished_tasks[1] =
			atomicAdd(g_info.a_bin_dispatcher_items[1], num_tasks) + num_tasks;
		g_info.dispatcher_task_counts[s_active_work_group_id] += num_tasks;
	}
	barrier();

	// Last group is responsible for computing bin offsets
	if(s_num_finished_tasks[1] == s_num_all_tasks[1]) {
		memoryBarrier();
		computeTriOffsets();
		memoryBarrier();
		if(LIX == 0)
			atomicExchange(g_info.a_bin_dispatcher_phase[1], 1);
	}

	// Waiting until all bin offsets are computed
	waitForSecondPhase(1);
	UPDATE_TIMER(4);

	// Reserving space for quad indices in bins
	for(uint i = LIX; i < BIN_COUNT; i += LSIZE)
		if(s_bins[i] > 0)
			s_bins[i] = atomicAdd(BIN_TRI_OFFSETS_TEMP(i), s_bins[i]);
	barrier();

	// Dispatching large quads
	while(s_num_tasks[1] > 0) {
		barrier();
		if(LIX == 0) {
			int task = s_first_task[1];
			s_first_task[1] = g_tasks[LARGE_TASKS_OFFSET + task];
			s_num_tasks[1]--;
			s_quads_offset = task << LARGE_TASK_SHIFT;
		}
		barrier();
		int large_quad_idx = s_quads_offset + int(LIX >> 1);
#if WARP_SIZE == 32
		dispatchLargeTriBalanced(large_quad_idx, int(LIX & 1), num_quads);
#else
		dispatchLargeTriSimple(large_quad_idx, int(LIX & 1), num_quads);
#endif
	}
	UPDATE_TIMER(5);
}

void main() {
	INIT_TIMERS();

	if(LIX < 2) {
		int num_quads = g_info.num_visible_quads[LIX];
		s_num_quads[LIX] = num_quads;
		int task_size = LIX == 0 ? SMALL_TASK_SIZE : LARGE_TASK_SIZE;
		int task_shift = LIX == 0 ? SMALL_TASK_SHIFT : LARGE_TASK_SHIFT;
		s_num_all_tasks[LIX] = (num_quads + (task_size - 1)) >> task_shift;
		s_first_task[LIX] = -1;
		s_last_task[LIX] = -1;
		s_num_tasks[LIX] = 0;
	}
	barrier();

	if(s_num_all_tasks[0] > 0) {
		for(uint i = LIX; i < BIN_COUNT; i += LSIZE)
			s_bins[i] = 0;
		barrier();
		processSmallQuads();
	}

	if(s_num_all_tasks[1] > 0) {
		barrier();
		for(uint i = LIX; i < BIN_COUNT; i += LSIZE)
			s_bins[i] = 0;
		barrier();
		processLargeTris();
	}

	COMMIT_TIMERS(g_info.bin_dispatcher_timers);
}
