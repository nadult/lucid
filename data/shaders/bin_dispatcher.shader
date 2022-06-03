// $$include funcs frustum structures

// TODO: we have to put workgroup items in GMEM: in case when there is
// only 1 active thread-group, all items won't fit in small SMEM array

// TODO: Do we want to perform rasterization here? Especially if we're dealing with slivers?

#define LID gl_LocalInvocationID
#define LIX gl_LocalInvocationIndex
#define WGID gl_WorkGroupID

#define LSIZE BINNING_LSIZE
#define LSHIFT BINNING_LSHIFT

#define ITEM_STEPS 4
#define ITEM_SIZE (LSIZE * ITEM_STEPS)

layout(local_size_x = LSIZE) in;

layout(std430, binding = 1) readonly buffer buf1_ { uint g_quad_aabbs[]; };
layout(std430, binding = 2) writeonly buffer buf2_ { uint g_bin_quads[]; };

layout(std430, binding = 3) readonly buffer buf3_ { uint g_quad_indices[]; };
layout(std430, binding = 4) readonly buffer buf4_ { float g_verts[]; };

shared int s_bins[BIN_COUNT];
shared int s_rows[BIN_COUNT_Y + 1];
shared int s_segments[LSIZE]; // TODO: merge rows and segments into single array
shared float s_inverses[BIN_COUNT_X + 1];

// Constants useful for efficient processing of bin counts
const int xbin_warps = LSIZE / BIN_COUNT_X, ybin_warps = LSIZE / BIN_COUNT_Y;
const int xbin_step = int(log2(xbin_warps)), ybin_step = int(log2(ybin_warps));
const int xbin_count = 1 << xbin_step, ybin_count = 1 << ybin_step;

struct QuadScanlineInfo {
	vec3 scan_min[2];
	vec3 scan_max[2];
	vec3 scan_step[2];
	bool second_empty;
};

void computeScanlineParams(vec3 tri0, vec3 tri1, vec3 tri2, vec2 start, out vec3 scan_min,
						   out vec3 scan_max, out vec3 scan_step) {
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
	vec3 scan_base = -vec3(edges[0].z * inv_ex[0], edges[1].z * inv_ex[1], edges[2].z * inv_ex[2]);
	scan_step = -vec3(edges[0].y * inv_ex[0], edges[1].y * inv_ex[1], edges[2].y * inv_ex[2]);

	bvec3 xsigns = bvec3(edges[0].x >= 0.0, edges[1].x >= 0.0, edges[2].x >= 0.0);

	float bin_offset = BIN_SIZE - 0.989;
	vec3 yoffset = vec3(edges[0].y >= 0.0 ? bin_offset : 0.0, edges[1].y >= 0.0 ? bin_offset : 0.0,
						edges[2].y >= 0.0 ? bin_offset : 0.0);
	vec3 xoffset = vec3(xsigns[0] ? bin_offset : 0.0, xsigns[1] ? bin_offset : 0.0,
						xsigns[2] ? bin_offset : 0.0);

	vec3 scan = scan_step * (yoffset + vec3(start.y)) + scan_base - (xoffset + vec3(start.x));
	const float inf = 1.0 / 0.0;
	scan_min =
		vec3(xsigns[0] ? scan[0] : -inf, xsigns[1] ? scan[1] : -inf, xsigns[2] ? scan[2] : -inf);
	scan_max =
		vec3(xsigns[0] ? inf : scan[0], xsigns[1] ? inf : scan[1], xsigns[2] ? inf : scan[2]);
	scan_step *= BIN_SIZE;
}

QuadScanlineInfo quadScanlineInfo(int quad_idx, int bsy) {
	QuadScanlineInfo info;

	// TODO: mark 2nd triangle as empty earlier?
	uint v0 = g_quad_indices[quad_idx * 4 + 0] & 0x03ffffff;
	uint v1 = g_quad_indices[quad_idx * 4 + 1] & 0x03ffffff;
	uint v2 = g_quad_indices[quad_idx * 4 + 2] & 0x03ffffff;
	uint v3 = g_quad_indices[quad_idx * 4 + 3] & 0x03ffffff;
	info.second_empty = v2 == v3;

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

	vec2 start = vec2(0.49, bsy * BIN_SIZE + 0.49);
	// TODO: merge these
	computeScanlineParams(quad0, quad1, quad2, start, info.scan_min[0], info.scan_max[0],
						  info.scan_step[0]);
	computeScanlineParams(quad0, quad2, quad3, start, info.scan_min[1], info.scan_max[1],
						  info.scan_step[1]);
	return info;
}

void quadScanStep(in out QuadScanlineInfo info, out int bmin, out int bmax) {
	float xmin[2] = {
		max(max(info.scan_min[0][0], info.scan_min[0][1]), max(info.scan_min[0][2], 0.0)),
		max(max(info.scan_min[1][0], info.scan_min[1][1]), max(info.scan_min[1][2], 0.0))};
	float xmax[2] = {min(min(info.scan_max[0][0], info.scan_max[0][1]),
						 min(info.scan_max[0][2], VIEWPORT_SIZE_X)),
					 min(min(info.scan_max[1][0], info.scan_max[1][1]),
						 min(info.scan_max[1][2], VIEWPORT_SIZE_X))};

	info.scan_min[0] += info.scan_step[0];
	info.scan_max[0] += info.scan_step[0];
	info.scan_min[1] += info.scan_step[1];
	info.scan_max[1] += info.scan_step[1];

	// There can be holes between two tris: exploit this? Maybe it's not worth it?
	bmin = int(xmin[0] + 1.0);
	bmax = int(xmax[0]);
	if(!info.second_empty) {
		bmin = min(bmin, int(xmin[1] + 1.0));
		bmax = max(bmax, int(xmax[1]));
	}
	bmin = bmin >> BIN_SHIFT;
	bmax = bmax >> BIN_SHIFT;
}

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

void countSmallQuadBins(uint quad_idx) {
	ivec4 aabb = decodeAABB32(g_quad_aabbs[quad_idx]);
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

void countLargeQuadBins(int quad_idx) {
	ivec4 aabb = decodeAABB32(g_quad_aabbs[quad_idx]);
	int bsx = aabb[0], bsy = aabb[1], bex = aabb[2], bey = aabb[3];
	QuadScanlineInfo info = quadScanlineInfo(quad_idx, bsy);

	for(int by = bsy; by <= bey; by++) {
		int bmin, bmax;
		quadScanStep(info, bmin, bmax);
		bmin = max(bmin, bsx), bmax = min(bmax, bex);

		if(bmax >= bmin) {
			atomicAdd(s_bins[bmin + by * BIN_COUNT_X], 1);
			if(bmax + 1 < BIN_COUNT_X)
				atomicAdd(s_bins[bmax + 1 + by * BIN_COUNT_X], -1);
		}
	}
}

void accumulateLargeQuadCounts() {
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
}

void computeOffsets() {
	// TODO: no need for groupMemoryBarrier ?
	// Computing bin quad counts offsets
	if(LIX < BIN_COUNT_Y * ybin_count) {
		uint by = LIX >> ybin_step;
		int accum = 0;
		for(uint bx = 0, subx = LIX & (ybin_count - 1); bx < BIN_COUNT_X; bx += ybin_count) {
			uint idx = bx + subx + by * BIN_COUNT_X;
			int cur = bx + subx < BIN_COUNT_X ? BIN_QUAD_COUNTS(idx) : 0, temp;
			for(int i = 1; i < ybin_count; i <<= 1) {
				temp = shuffleUpNV(cur, i, ybin_count);
				cur += subx >= i ? temp : 0;
			}
			cur += accum;
			if(bx + subx < BIN_COUNT_X)
				BIN_QUAD_OFFSETS(idx) = cur;
			accum = shuffleNV(cur, ybin_count - 1, ybin_count);
		}
	}
	barrier();
	// Storing accumulated bin quad counts for each row
	if(LIX < BIN_COUNT_Y)
		s_rows[LIX] = BIN_QUAD_OFFSETS((BIN_COUNT_X - 1) + LIX * BIN_COUNT_X);
	barrier();
	if(LIX < BIN_COUNT_X * xbin_count) {
		uint bx = LIX >> xbin_step;
		int accum = 0;
		for(uint by = 0, suby = LIX & (xbin_count - 1); by < BIN_COUNT_Y; by += xbin_count) {
			uint idx = by + suby;
			int cur = idx < BIN_COUNT_Y && idx > 0 ? s_rows[idx - 1] : 0, temp;
			for(int i = 1; i < xbin_count; i <<= 1) {
				temp = shuffleUpNV(cur, i, xbin_count);
				cur += suby >= i ? temp : 0;
			}
			cur += accum;
			if(idx < BIN_COUNT_Y) {
				uint bidx = bx + idx * BIN_COUNT_X;
				int value = BIN_QUAD_OFFSETS(bidx) + cur - BIN_QUAD_COUNTS(bidx);
				BIN_QUAD_OFFSETS(bidx) = value;
				BIN_QUAD_OFFSETS_TEMP(bidx) = value;
			}
			accum = shuffleNV(cur, xbin_count - 1, xbin_count);
		}
	}
	barrier();
}

void dispatchQuad(int quad_idx) {
	ivec4 aabb = decodeAABB32(g_quad_aabbs[quad_idx]);
	int bsx = aabb[0], bsy = aabb[1], bex = aabb[2], bey = aabb[3];

	for(int by = bsy; by <= bey; by++) {
		for(int bx = bsx; bx <= bex; bx++) {
			uint bin_id = bx + by * BIN_COUNT_X;
			uint quad_offset = atomicAdd(s_bins[bin_id], 1);
			g_bin_quads[quad_offset] = quad_idx;
		}
	}
}

// TODO: better desc
// This is an optimization of dispatchQuad which is more work efficient
// This is especially useful if there is a large variation in quad sizes
// and for large quads in general.
//
// Basic idea: within each warp we divide quad-rows into segments which contain
// more or less equal amount of bin-quad instances (samples in this context).
// Each thread processes single segment.
void dispatchLargeQuad(int large_quad_idx, int num_quads) {
	bool is_valid = large_quad_idx < num_quads;
	if(allInvocationsARB(!is_valid))
		return;

	QuadScanlineInfo quad_scan_info;
	int quad_idx = (MAX_QUADS - 1) - large_quad_idx;
	int bsx, bsy = 0, bex, bey = -1;

	if(is_valid) {
		ivec4 aabb = decodeAABB32(g_quad_aabbs[quad_idx]);
		bsx = aabb[0], bsy = aabb[1], bex = aabb[2], bey = aabb[3];
		quad_scan_info = quadScanlineInfo(quad_idx, bsy);
	}

	for(int by = bsy; anyInvocationARB(by <= bey); by++) {
		int bmin = 0, bmax = -1;
		if(by <= bey) {
			quadScanStep(quad_scan_info, bmin, bmax);
			bmin = max(bmin, bsx), bmax = min(bmax, bex);
		}

		int num_samples = max(0, bmax - bmin + 1);
		if(allInvocationsARB(num_samples == 0))
			continue;

		int sample_offset = prefixSum32(num_samples);
		int warp_num_samples = shuffleNV(sample_offset, 31, 32);
		sample_offset -= num_samples;

		int warp_offset = int(LIX & ~31), warp_sub_id = int(LIX & 31);
		int segment_size = (warp_num_samples + 31) / 32;
		int segment_id = sample_offset / segment_size;
		int segment_offset = sample_offset - segment_id * segment_size;
		if(num_samples > 0) {
			if(segment_offset == 0)
				s_segments[warp_offset + segment_id] = warp_sub_id;
			for(int k = 1; segment_offset + num_samples > segment_size * k; k++)
				s_segments[warp_offset + segment_id + k] = warp_sub_id;
		}

		int cur_sub_id = s_segments[LIX];
		int cur_sample_id = warp_sub_id * segment_size;
		int cur_offset = cur_sample_id - shuffleNV(sample_offset, cur_sub_id, 32);
		int cur_num_samples = min(warp_num_samples - cur_sample_id, segment_size);

		int i = 0;
		int base_bin_id = by * BIN_COUNT_X + bmin;
		while(anyInvocationARB(i < cur_num_samples)) {
			int cur_quad_idx = shuffleNV(quad_idx, cur_sub_id, 32);
			int cur_bin_id = shuffleNV(base_bin_id, cur_sub_id, 32);
			int cur_width = shuffleNV(num_samples, cur_sub_id, 32);

			if(cur_width == 0) {
				cur_sub_id++;
				continue;
			}
			if(i < cur_num_samples) {
				uint quad_offset = atomicAdd(s_bins[cur_bin_id + cur_offset], 1);
				g_bin_quads[quad_offset] = cur_quad_idx;
				cur_offset++;
				if(cur_offset == cur_width)
					cur_offset = 0, cur_sub_id++;
				i++;
			}
		}
	}
}

shared int s_num_quads[2], s_size_type;
shared int s_quads_offset, s_active_work_group_id;
shared int s_item_id, s_num_items;
shared int s_num_processed_items, s_num_all_items;

// TODO: Zamiast tego jedna tablica w gmem?
// TODO: 16 bits is enough for workgroup items ?
shared int s_workgroup_items[MAX_BIN_WORKGROUP_ITEMS];

void main() {
#ifdef ENABLE_TIMERS
	uint64_t clock0 = clockARB();
#endif

	if(LIX < 2) {
		s_num_quads[LIX] = g_info.num_visible_quads[LIX];
		// TODO: we can probably improve perf by dividing work more evenly
		// Two types of quads make it a bit tricky
		if(LIX == 0) {
			int num_small_quads = (s_num_quads[0] + ITEM_SIZE - 1) / ITEM_SIZE;
			int num_large_quads = (s_num_quads[1] + LSIZE - 1) / LSIZE;
			s_num_all_items = num_small_quads + num_large_quads;
			s_item_id = 0;
		}
	}

	if(LIX < BIN_COUNT_X + 1)
		s_inverses[LIX] = 1.0 / float(LIX);
	for(uint i = LIX; i < BIN_COUNT; i += LSIZE)
		s_bins[i] = 0;
	if(LIX < BIN_COUNT_Y)
		s_rows[LIX] = 0;
	barrier();

	// Computing large quads bin coverage
	int num_quads = s_num_quads[1];
	while(num_quads > 0 && s_item_id < MAX_BIN_WORKGROUP_ITEMS - 1) {
		if(LIX == 0) {
			int quads_offset = atomicAdd(g_info.num_estimated_quads[1], LSIZE);
			if(quads_offset < num_quads)
				s_workgroup_items[s_item_id++] = -quads_offset - 1;
			s_quads_offset = quads_offset;
		}
		barrier();

		int large_quads_offset = s_quads_offset;
		if(large_quads_offset >= num_quads)
			break;

		int large_quad_idx = large_quads_offset + int(LIX);
		if(large_quad_idx < num_quads)
			countLargeQuadBins((MAX_QUADS - 1) - large_quad_idx);

		barrier();
	}

	barrier();
	accumulateLargeQuadCounts();

	// Computing small quads bin coverage
	num_quads = s_num_quads[0];
	while(num_quads > 0 && s_item_id < MAX_BIN_WORKGROUP_ITEMS - 1) {
		if(LIX == 0) {
			int quads_offset = atomicAdd(g_info.num_estimated_quads[0], ITEM_SIZE);
			if(quads_offset < num_quads)
				s_workgroup_items[s_item_id++] = quads_offset;
			s_quads_offset = quads_offset;
		}
		barrier();

		int quad_offset = s_quads_offset;
		if(quad_offset >= num_quads)
			break;

		for(int s = 0; s < ITEM_STEPS; s++) {
			int quad_idx = quad_offset + (LSIZE * s) + int(LIX);
			if(quad_idx >= num_quads)
				break;
			countSmallQuadBins(quad_idx);
		}
		barrier();
	}

	barrier();

	// Thread groups which didn't do any estimation can quit early:
	// they won't participate in dispatching either
	if(s_item_id == 0)
		return;

	// Copying bin counters to global memory buffer
	for(uint i = LIX; i < BIN_COUNT; i += LSIZE)
		if(s_bins[i] > 0)
			atomicAdd(BIN_QUAD_COUNTS(i), s_bins[i]);
	barrier();

	// Finishing estimation phase
	if(LIX == 0) {
		s_active_work_group_id = atomicAdd(g_info.a_bin_dispatcher_work_groups, 1);
		s_num_processed_items = atomicAdd(g_info.a_bin_dispatcher_items, s_item_id) + s_item_id;
		s_num_items = s_item_id;
		g_info.dispatcher_item_counts[s_active_work_group_id] = s_num_items;
		s_item_id = 0;
	}
	barrier();

	// Last group is responsible for computing bin offsets
	if(s_num_processed_items == s_num_all_items) {
		groupMemoryBarrier();
		computeOffsets();
		if(LIX == 0)
			atomicExchange(g_info.a_bin_dispatcher_phase, 1);
	}

	// Waiting until all bin offsets are computed
	// TODO: can we do something useful while waiting?
	if(LIX == 0)
		while(g_info.a_bin_dispatcher_phase == 0)
			;
	barrier();

	// Reserving space for quad indices in bins
	for(uint i = LIX; i < BIN_COUNT; i += LSIZE)
		if(s_bins[i] > 0)
			s_bins[i] = atomicAdd(BIN_QUAD_OFFSETS_TEMP(i), s_bins[i]);
	barrier();

	// Dispatching quads
	while(s_item_id < s_num_items) {
		barrier();
		if(LIX == 0) {
			int quads_offset = s_workgroup_items[s_item_id++];
			s_size_type = quads_offset < 0 ? 1 : 0;
			s_quads_offset = quads_offset < 0 ? -quads_offset - 1 : quads_offset;
		}
		barrier();
		int quads_offset = s_quads_offset;
		int size_type = s_size_type;

		if(size_type == 0) { // Small quads
			int num_quads = s_num_quads[0];
			for(int s = 0; s < ITEM_STEPS; s++) {
				int quad_idx = quads_offset + LSIZE * s + int(LIX);
				if(quad_idx >= num_quads)
					break;
				dispatchQuad(quad_idx);
			}
		} else { // Large quads
			int num_quads = s_num_quads[1];
			int large_quad_idx = quads_offset + int(LIX);
			dispatchLargeQuad(large_quad_idx, num_quads);
		}
	}
	barrier();

#ifdef ENABLE_TIMERS
	g_info.dispatcher_timers[s_active_work_group_id] = int(clockARB() - clock0);
#endif
}
