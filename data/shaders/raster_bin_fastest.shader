// $$include funcs lighting frustum viewport data

// TODO: add synthetic test: 256 planes one after another
// TODO: cleanup in the beginning (group definitions)

// NOTE: converting integer multiplications to shifts does not increase perf

#define LIX gl_LocalInvocationIndex
#define LID gl_LocalInvocationID

// Acceptable values: 128, 256, 512
#define LSIZE 512
#define LSHIFT 9

#define NUM_WARPS (LSIZE / 32)
#define NUM_WARPS_SHIFT (LSHIFT - 5)

#define WARP_STEP 32
#define WARP_MASK 31

#define BUFFER_SIZE (LSIZE * 8)

#define MAX_BLOCK_ROW_TRIS 1024 // TODO: detect overflow
#define MAX_BLOCK_TRIS 256
#define MAX_BLOCK_TRIS_SHIFT 8

#define MAX_SCRATCH_TRIS 2048
#define MAX_SCRATCH_TRIS_SHIFT 11

#define SEGMENT_SIZE 128
#define SEGMENT_SHIFT 7

#define MAX_SEGMENTS 32
#define MAX_SEGMENTS_SHIFT 5

// TODO: half block?
#define MAX_BLOCK_SAMPLES (MAX_SEGMENTS * SEGMENT_SIZE - 1)

// TODO: opiz gdzie uzywamy pelne bloki a gdize polowki, jakie sa wymiary, etc.
#define NUM_BLOCK_COLS 8
#define NUM_BLOCK_ROWS 8

layout(local_size_x = LSIZE) in;

layout(std430, binding = 0) buffer buf0_ { uvec4 g_tri_aabbs[]; };
layout(std430, binding = 1) buffer buf1_ { uint g_quad_indices[]; };

layout(std430, binding = 2) readonly buffer buf2_ { float g_verts[]; };
layout(std430, binding = 3) readonly buffer buf3_ { vec2 g_tex_coords[]; };
layout(std430, binding = 4) readonly buffer buf4_ { uint g_colors[]; };
layout(std430, binding = 5) readonly buffer buf5_ { uint g_normals[]; };

layout(std430, binding = 6) buffer buf6_ { BinCounters g_bins; };
layout(std430, binding = 8) buffer buf8_ { uint g_bin_quads[]; };

layout(std430, binding = 9) coherent buffer buf9_ { uint g_scratch_32[]; };
layout(std430, binding = 10) coherent buffer buf10_ { uvec2 g_scratch_64[]; };

layout(std430, binding = 11) readonly buffer buf11_ { InstanceData g_instances[]; };
layout(std430, binding = 12) readonly buffer buf12_ { vec4 g_uv_rects[]; };
layout(std430, binding = 13) writeonly buffer buf13_ { uint g_raster_image[]; };

layout(binding = 0) uniform sampler2D opaque_texture;
layout(binding = 1) uniform sampler2D transparent_texture;

// TODO: separate opaque and transparent objects, draw opaque objects first to texture
// then read it and use depth to optimize drawing
uniform uint background_color;

#define SATURATE(val) clamp(val, 0.0, 1.0)

#define WORKGROUP_32_SCRATCH_SIZE (32 * 1024)
#define WORKGROUP_32_SCRATCH_SHIFT 15

#define WORKGROUP_64_SCRATCH_SIZE (64 * 1024)
#define WORKGROUP_64_SCRATCH_SHIFT 16

#define TRI_SCRATCH(var_idx) g_scratch_64[scratch_tri_offset + (var_idx << MAX_SCRATCH_TRIS_SHIFT)]

uint scratch32BlockTrisOffset(uint bx) {
	return (gl_WorkGroupID.x << WORKGROUP_32_SCRATCH_SHIFT) + bx * MAX_BLOCK_TRIS;
}

uint scratch64TriOffset(uint tri_idx) {
	return (gl_WorkGroupID.x << WORKGROUP_64_SCRATCH_SHIFT) + tri_idx;
}

uint scratch64BlockRowTrisOffset(uint by) {
	return (gl_WorkGroupID.x << WORKGROUP_64_SCRATCH_SHIFT) + 32 * 1024 +
		   by * (MAX_BLOCK_ROW_TRIS * 2);
}

uint scratch64BlockTrisOffset(uint bx) {
	return (gl_WorkGroupID.x << WORKGROUP_64_SCRATCH_SHIFT) + 48 * 1024 + bx * MAX_BLOCK_TRIS;
}

uint scratch64HalfBlockTrisOffset(uint bx) {
	return (gl_WorkGroupID.x << WORKGROUP_64_SCRATCH_SHIFT) + 52 * 1024 + bx * MAX_BLOCK_TRIS;
}

shared int s_num_bins, s_bin_id, s_bin_raster_offset;
shared uint s_bin_quad_count, s_bin_quad_offset;
shared ivec2 s_bin_pos;
shared vec3 s_bin_ray_dir0;

shared uint s_block_row_tri_count[NUM_BLOCK_ROWS];
shared uint s_block_tri_count[NUM_WARPS];
shared uint s_block_frag_count[NUM_WARPS];

shared uint s_buffer[BUFFER_SIZE + 1];
shared uint s_mini_buffer[LSIZE];
shared uint s_segments[NUM_WARPS * MAX_SEGMENTS * 2];
shared int s_raster_error;

// Only used when debugging
shared uint s_vis_pixels[LSIZE];

void outputPixel(ivec2 pixel_pos, uint color) {
	g_raster_image[s_bin_raster_offset + pixel_pos.x + (pixel_pos.y << BIN_SHIFT)] = color;
}

// Note: UPDATE_CLOCK should be called after a barrier
#ifdef ENABLE_TIMINGS
#define MAX_TIMERS 8
shared uint s_timings[MAX_TIMERS];
#define INIT_CLOCK() uint64_t clock0 = clockARB();
#define UPDATE_CLOCK(idx)                                                                          \
	if((LIX & 31) == 0) {                                                                          \
		uint64_t clock = clockARB();                                                               \
		atomicAdd(s_timings[idx], uint(clock - clock0) >> 4);                                      \
		clock0 = clock;                                                                            \
	}

void initTimers() {
	if(LIX < MAX_TIMERS)
		s_timings[LIX] = 0;
}
void commitTimers() {
	if(LIX < MAX_TIMERS)
		atomicAdd(g_bins.timings[LIX], s_timings[LIX]);
}

#else
#define INIT_CLOCK()
#define UPDATE_CLOCK(idx)

void initTimers() {}
void commitTimers() {}
#endif

// TODO: don't store triangles which generate very small number of samples in scratch,
// instead precompute them directly when sampling; We would have to somehow group those triangles together
//
// TODO: use scratch based on uints, not uvec2, maybe it will be a bit faster?

void storeTriangle(uint tri_idx, vec3 tri0, vec3 tri1, vec3 tri2, uint v0, uint v1, uint v2,
				   uint instance_id) {
	uint scratch_tri_offset = scratch64TriOffset(tri_idx);
	vec3 normal = cross(tri0 - tri2, tri1 - tri0);
	float multiplier = 1.0 / length(normal);
	normal *= multiplier;
	uint unormal = encodeNormalUint(normal);

	vec3 edge0 = (tri0 - tri2) * multiplier;
	vec3 edge1 = (tri1 - tri0) * multiplier;

	float plane_dist = dot(normal, tri0);
	vec3 nrm_tri0 = cross(tri0, normal);
	float param0 = dot(edge0, nrm_tri0);
	float param1 = dot(edge1, nrm_tri0);

	uint instance_flags = g_instances[instance_id].flags;
	uint instance_color = g_instances[instance_id].color;

	// Nice optimization for barycentric computations:
	// dot(cross(edge, dir), normal) == dot(dir, cross(normal, edge))
	edge0 = cross(normal, edge0);
	edge1 = cross(normal, edge1);

	edge0 =
		vec3(dot(edge0, frustum.ws_dirx), dot(edge0, frustum.ws_diry), dot(edge0, s_bin_ray_dir0));
	edge1 =
		vec3(dot(edge1, frustum.ws_dirx), dot(edge1, frustum.ws_diry), dot(edge1, s_bin_ray_dir0));

	vec3 pnormal = normal * (1.0 / plane_dist);
	vec3 depth_eq = vec3(dot(pnormal, frustum.ws_dirx), dot(pnormal, frustum.ws_diry),
						 dot(pnormal, s_bin_ray_dir0));

	TRI_SCRATCH(0) = uvec2(floatBitsToUint(depth_eq.x), floatBitsToUint(depth_eq.y));
	TRI_SCRATCH(1) = uvec2(floatBitsToUint(depth_eq.z), instance_flags | (instance_id << 16));
	TRI_SCRATCH(2) = uvec2(floatBitsToUint(param0), floatBitsToUint(param1));
	TRI_SCRATCH(3) = uvec2(floatBitsToUint(edge0.x), floatBitsToUint(edge0.y));
	TRI_SCRATCH(4) = uvec2(floatBitsToUint(edge0.z), floatBitsToUint(edge1.x));
	TRI_SCRATCH(5) = uvec2(floatBitsToUint(edge1.y), floatBitsToUint(edge1.z));

	TRI_SCRATCH(6) = uvec2(unormal, instance_color);

	uint vcolor2 = 0;
	if((instance_flags & INST_HAS_VERTEX_COLORS) != 0) {
		TRI_SCRATCH(7) = uvec2(g_colors[v0], g_colors[v1]);
		vcolor2 = g_colors[v2];
	}
	uint vnormal2 = 0;
	if((instance_flags & INST_HAS_VERTEX_NORMALS) != 0) {
		TRI_SCRATCH(9) = uvec2(g_normals[v0], g_normals[v1]);
		vnormal2 = g_normals[v2];
	}

	if((instance_flags & (INST_HAS_VERTEX_COLORS | INST_HAS_VERTEX_NORMALS)) != 0) {
		TRI_SCRATCH(8) = uvec2(vcolor2, vnormal2);
	}
	if((instance_flags & INST_HAS_TEXTURE) != 0) {
		vec2 tex0 = g_tex_coords[v0];
		vec2 tex1 = g_tex_coords[v1];
		vec2 tex2 = g_tex_coords[v2];
		tex1 -= tex0;
		tex2 -= tex0;
		TRI_SCRATCH(10) = floatBitsToUint(tex0);
		TRI_SCRATCH(11) = floatBitsToUint(tex1);
		TRI_SCRATCH(12) = floatBitsToUint(tex2);
	}
}

void getTriangleParams(uint scratch_tri_offset, out vec3 depth_eq, out vec2 bary_params,
					   out vec3 edge0, out vec3 edge1, out uint instance_id,
					   out uint instance_flags) {
	{
		uvec2 val0 = TRI_SCRATCH(0), val1 = TRI_SCRATCH(1), val2 = TRI_SCRATCH(2);
		depth_eq =
			vec3(uintBitsToFloat(val0[0]), uintBitsToFloat(val0[1]), uintBitsToFloat(val1[0]));
		bary_params = vec2(uintBitsToFloat(val2[0]), uintBitsToFloat(val2[1]));
		instance_flags = val1[1] & 0xffff;
		instance_id = val1[1] >> 16;
	}
	{
		uvec2 val0 = TRI_SCRATCH(3), val1 = TRI_SCRATCH(4), val2 = TRI_SCRATCH(5);
		edge0 = vec3(uintBitsToFloat(val0[0]), uintBitsToFloat(val0[1]), uintBitsToFloat(val1[0]));
		edge1 = vec3(uintBitsToFloat(val1[1]), uintBitsToFloat(val2[0]), uintBitsToFloat(val2[1]));
	}
}

void getTriangleSecondaryParams(uint scratch_tri_offset, out uint unormal,
								out uint instance_color) {
	uvec2 val0 = TRI_SCRATCH(6);
	unormal = val0.x;
	instance_color = val0.y;
}

void getTriangleVertexColors(uint scratch_tri_offset, out vec4 color0, out vec4 color1,
							 out vec4 color2) {
	uvec2 val0 = TRI_SCRATCH(7);
	uvec2 val1 = TRI_SCRATCH(8);
	color0 = decodeRGBA8(val0[0]);
	color1 = decodeRGBA8(val0[1]);
	color2 = decodeRGBA8(val1[0]);
}

void getTriangleVertexNormals(uint scratch_tri_offset, out vec3 normal0, out vec3 normal1,
							  out vec3 normal2) {
	uvec2 val0 = TRI_SCRATCH(9);
	uvec2 val1 = TRI_SCRATCH(8);
	normal0 = decodeNormalUint(val0[0]);
	normal1 = decodeNormalUint(val0[1]);
	normal2 = decodeNormalUint(val1[1]);
}

void getTriangleVertexTexCoords(uint scratch_tri_offset, out vec2 tex0, out vec2 tex1,
								out vec2 tex2) {
	uvec2 val0 = TRI_SCRATCH(10);
	uvec2 val1 = TRI_SCRATCH(11);
	uvec2 val2 = TRI_SCRATCH(12);
	tex0 = uintBitsToFloat(val0);
	tex1 = uintBitsToFloat(val1);
	tex2 = uintBitsToFloat(val2);
}

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
	return (edges[0].x < 0.0 ? 1 : 0) | (edges[1].x < 0.0 ? 2 : 0) | (edges[2].x < 0.0 ? 4 : 0);
}

void generateRowTris(uint tri_idx, vec3 tri0, vec3 tri1, vec3 tri2, int min_by, int max_by) {
	// Inspired by Nanite scanline rasterizer
	vec3 scan_min, scan_max, scan_step;

	{
		float sx = s_bin_pos.x - 0.5f;
		float sy = s_bin_pos.y + (float(min_by) * 8.0 + 0.5f);

		vec3 scan_base;
		uint sign_mask = computeScanlineParams(tri0, tri1, tri2, scan_base, scan_step);

		vec3 scan = scan_step * sy + scan_base - vec3(sx, sx, sx);
		scan_min = vec3((sign_mask & 1) == 0 ? scan[0] : -1.0 / 0.0,
						(sign_mask & 2) == 0 ? scan[1] : -1.0 / 0.0,
						(sign_mask & 4) == 0 ? scan[2] : -1.0 / 0.0);
		scan_max = vec3((sign_mask & 1) != 0 ? scan[0] : 1.0 / 0.0,
						(sign_mask & 2) != 0 ? scan[1] : 1.0 / 0.0,
						(sign_mask & 4) != 0 ? scan[2] : 1.0 / 0.0);
	}

	uint dst_offset_64 = scratch64BlockRowTrisOffset(0);

	// TODO: is it worth it to make this loop more work-efficient?
	for(int by = min_by; by <= max_by; by++) {
#define SCAN_STEP(id)                                                                              \
	int min##id = int(max(max(scan_min[0], scan_min[1]), max(scan_min[2], 0.0)));                  \
	int max##id = int(min(min(scan_max[0], scan_max[1]), min(scan_max[2], BIN_SIZE))) - 1;         \
	if(min##id > max##id)                                                                          \
		min##id = 63, max##id = 0;                                                                 \
	scan_min += scan_step, scan_max += scan_step;

		uint bx_mask;
		uvec2 min_bits, max_bits;
		{
			SCAN_STEP(0);
			SCAN_STEP(1);
			SCAN_STEP(2);
			SCAN_STEP(3);

			bx_mask = ((0xff << (min0 >> 3)) & (0xff >> (7 - (max0 >> 3)))) |
					  ((0xff << (min1 >> 3)) & (0xff >> (7 - (max1 >> 3)))) |
					  ((0xff << (min2 >> 3)) & (0xff >> (7 - (max2 >> 3)))) |
					  ((0xff << (min3 >> 3)) & (0xff >> (7 - (max3 >> 3))));
			min_bits.x = (min0 << 0) | (min1 << 6) | (min2 << 12) | (min3 << 18);
			max_bits.x = (max0 << 0) | (max1 << 6) | (max2 << 12) | (max3 << 18);
		}
		{
			SCAN_STEP(0);
			SCAN_STEP(1);
			SCAN_STEP(2);
			SCAN_STEP(3);

			bx_mask |= ((0xff << (min0 >> 3)) & (0xff >> (7 - (max0 >> 3)))) |
					   ((0xff << (min1 >> 3)) & (0xff >> (7 - (max1 >> 3)))) |
					   ((0xff << (min2 >> 3)) & (0xff >> (7 - (max2 >> 3)))) |
					   ((0xff << (min3 >> 3)) & (0xff >> (7 - (max3 >> 3))));
			min_bits.y = (min0 << 0) | (min1 << 6) | (min2 << 12) | (min3 << 18);
			max_bits.y = (max0 << 0) | (max1 << 6) | (max2 << 12) | (max3 << 18);
		}

#undef SCAN_STEP

		if(bx_mask == 0)
			continue;

		uint roffset = atomicAdd(s_block_row_tri_count[by], 1) + by * (MAX_BLOCK_ROW_TRIS * 2);
		g_scratch_64[dst_offset_64 + roffset] = uvec2(min_bits.x | (bx_mask << 24), min_bits.y);
		g_scratch_64[dst_offset_64 + roffset + MAX_BLOCK_ROW_TRIS] =
			uvec2(max_bits.x | ((tri_idx & 0xff) << 24), max_bits.y | ((tri_idx & 0xff00) << 16));
	}
}

void processQuads() {
	// TODO: optimization: in many cases all rows may very well fit in SMEM,
	// maybe it would be worth it not to use scratch at all then?
	// TODO: this loop is slooooow
	// TODO: divide big tris across different threads
	for(uint i = LIX >> 1; i < s_bin_quad_count; i += LSIZE / 2) {
		uint second_tri = LIX & 1;
		uint quad_idx = g_bin_quads[s_bin_quad_offset + i] & 0xffffff;

		uvec4 aabb = g_tri_aabbs[quad_idx];
		aabb = decodeAABB(second_tri != 0 ? aabb.zw : aabb.xy);
		int min_by = clamp(int(aabb[1]) - s_bin_pos.y, 0, 63) >> 3;
		int max_by = clamp(int(aabb[3]) - s_bin_pos.y, 0, 63) >> 3;

		uvec4 verts = uvec4(g_quad_indices[quad_idx * 4 + 0], g_quad_indices[quad_idx * 4 + 1],
							g_quad_indices[quad_idx * 4 + 2], g_quad_indices[quad_idx * 4 + 3]);
		uint instance_id =
			(verts[0] >> 26) | ((verts[1] >> 20) & 0xfc0) | ((verts[2] >> 14) & 0x3f000);
		uint v0 = verts[0] & 0x03ffffff;
		uint v1 = verts[1 + second_tri] & 0x03ffffff;
		uint v2 = verts[2 + second_tri] & 0x03ffffff;

		vec3 tri0 = vec3(g_verts[v0 * 3 + 0], g_verts[v0 * 3 + 1], g_verts[v0 * 3 + 2]) -
					frustum.ws_shared_origin;
		vec3 tri1 = vec3(g_verts[v1 * 3 + 0], g_verts[v1 * 3 + 1], g_verts[v1 * 3 + 2]) -
					frustum.ws_shared_origin;
		vec3 tri2 = vec3(g_verts[v2 * 3 + 0], g_verts[v2 * 3 + 1], g_verts[v2 * 3 + 2]) -
					frustum.ws_shared_origin;

		// TODO: store only if samples were generated
		// TODO: do triangle storing later
		uint tri_idx = i * 2 + (LIX & 1);
		storeTriangle(tri_idx, tri0, tri1, tri2, v0, v1, v2, instance_id);
		generateRowTris(tri_idx, tri0, tri1, tri2, min_by, max_by);
	}
}

shared uint s_sort_rcount[NUM_WARPS];

void prepareSortTris() {
	if(LIX < NUM_WARPS) {
		uint count = s_block_tri_count[LIX];
		// rcount: count rounded up to next power of 2, minimum: 32
		s_sort_rcount[LIX] = max(32, (count & (count - 1)) == 0 ? count : (2 << findMSB(count)));
	}
}

#ifdef VENDOR_NVIDIA
uint swap(uint x, int mask, uint dir) {
	uint y = shuffleXorNV(x, mask, 32);
	return uint(x < y) == dir ? y : x;
}
uint bitExtract(uint value, int boffset) { return (value >> boffset) & 1; }
uint xorBits(uint value, int bit0, int bit1) { return ((value >> bit0) ^ (value >> bit1)) & 1; }
#endif

void sortTris(uint lbid, uint count, uint buf_offset) {
	uint lid = LIX & WARP_MASK;
	uint rcount = s_sort_rcount[lbid];
	for(uint i = lid + count; i < rcount; i += WARP_STEP)
		s_buffer[buf_offset + i] = 0xffffffff;

#ifdef VENDOR_NVIDIA
	for(uint i = lid; i < rcount; i += WARP_STEP) {
		uint value = s_buffer[buf_offset + i];
		// TODO: register sort could be faster
		value = swap(value, 0x01, xorBits(lid, 1, 0)); // K = 2
		value = swap(value, 0x02, xorBits(lid, 2, 1)); // K = 4
		value = swap(value, 0x01, xorBits(lid, 2, 0));
		value = swap(value, 0x04, xorBits(lid, 3, 2)); // K = 8
		value = swap(value, 0x02, xorBits(lid, 3, 1));
		value = swap(value, 0x01, xorBits(lid, 3, 0));
		value = swap(value, 0x08, xorBits(lid, 4, 3)); // K = 16
		value = swap(value, 0x04, xorBits(lid, 4, 2));
		value = swap(value, 0x02, xorBits(lid, 4, 1));
		value = swap(value, 0x01, xorBits(lid, 4, 0));
		s_buffer[buf_offset + i] = value;
	}
	int start_k = 32, end_j = 32;
#else
	int start_k = 2, end_j = 1;
#endif
	for(uint k = start_k; k <= rcount; k = 2 * k) {
		for(uint j = k >> 1; j >= end_j; j = j >> 1) {
			for(uint i = lid; i < rcount; i += WARP_STEP * 2) {
				uint idx = (i & j) != 0 ? i + WARP_STEP - j : i;
				uint lvalue = s_buffer[buf_offset + idx];
				uint rvalue = s_buffer[buf_offset + idx + j];
				if(((idx & k) != 0) == (lvalue.x < rvalue.x)) {
					s_buffer[buf_offset + idx] = rvalue;
					s_buffer[buf_offset + idx + j] = lvalue;
				}
			}
		}
#ifdef VENDOR_NVIDIA
		for(uint i = lid; i < rcount; i += WARP_STEP) {
			uint bit = (i & k) == 0 ? 0 : 1;
			uint value = s_buffer[buf_offset + i];
			value = swap(value, 0x10, bit ^ bitExtract(lid, 4));
			value = swap(value, 0x08, bit ^ bitExtract(lid, 3));
			value = swap(value, 0x04, bit ^ bitExtract(lid, 2));
			value = swap(value, 0x02, bit ^ bitExtract(lid, 1));
			value = swap(value, 0x01, bit ^ bitExtract(lid, 0));
			s_buffer[buf_offset + i] = value;
		}
#endif
	}
}

// TODO: maybe process smaller amount of blocks at the same time?
// smaller chance that it will leave cache
void generateBlocks(uint bid) {
	int lbid = int(LIX >> 5);
	bid += lbid;
	uint by = bid >> 3, bx = bid & 7;

	uint src_offset_64 = scratch64BlockRowTrisOffset(by);
	uint tri_count = s_block_row_tri_count[by];
	uint buf_offset = lbid << MAX_BLOCK_TRIS_SHIFT;

#if MAX_SEGMENTS * NUM_WARPS != LSIZE
#error Segment initialization invalid
#endif
	s_segments[LIX] = 0;
	s_segments[LIX + LSIZE] = 0;

	{
		uint bx_bits_shift = 24 + bx;
		uint block_tri_count = 0;
		uint thread_bit_mask = ~(0xffffffffu << (LIX & 31));

		for(uint i = LIX & WARP_MASK; i < tri_count; i += WARP_STEP) {
			uint bx_bit = (g_scratch_64[src_offset_64 + i].x >> bx_bits_shift) & 1;
			uint bit_mask = uint(ballotARB(bx_bit != 0));
			if(bit_mask == 0)
				continue;

			uint warp_offset = bitCount(bit_mask & thread_bit_mask);
			if(bx_bit != 0) {
				uint tri_offset =
					(block_tri_count + warp_offset) & ((1 << MAX_BLOCK_TRIS_SHIFT) - 1);
				s_buffer[buf_offset + tri_offset] = i;
			}
			block_tri_count += bitCount(bit_mask);
		}

		if((LIX & 31) == 0) {
			s_block_tri_count[lbid] = block_tri_count;
			if(block_tri_count > MAX_BLOCK_TRIS)
				atomicOr(s_raster_error, 1 << lbid);
		}
		barrier();
		if(s_raster_error != 0)
			return;
	}
	prepareSortTris();

	uint dst_offset_64 = scratch64BlockTrisOffset(lbid);
	uint dst_offset_32 = scratch32BlockTrisOffset(lbid);
	tri_count = s_block_tri_count[lbid];
	int startx = int(bx << 3);

	for(uint i = LIX & WARP_MASK; i < tri_count; i += WARP_STEP) {
		uint row_idx = s_buffer[buf_offset + i];

		uvec2 tri_mins = g_scratch_64[src_offset_64 + row_idx];
		uvec2 tri_maxs = g_scratch_64[src_offset_64 + row_idx + MAX_BLOCK_ROW_TRIS];
		uint tri_idx = (tri_maxs.x >> 24) | ((tri_maxs.y >> 16) & 0xff00);

		uint min_bits, count_bits;
		uvec2 num_frags;
		vec2 cpos = vec2(0);

#define PROCESS_ROWS(e)                                                                            \
	int min0 = max(int((tri_mins.e >> 0) & 63) - startx, 0);                                       \
	int min1 = max(int((tri_mins.e >> 6) & 63) - startx, 0);                                       \
	int min2 = max(int((tri_mins.e >> 12) & 63) - startx, 0);                                      \
	int min3 = max(int((tri_mins.e >> 18) & 63) - startx, 0);                                      \
	int max0 = min(int((tri_maxs.e >> 0) & 63) - startx, 7);                                       \
	int max1 = min(int((tri_maxs.e >> 6) & 63) - startx, 7);                                       \
	int max2 = min(int((tri_maxs.e >> 12) & 63) - startx, 7);                                      \
	int max3 = min(int((tri_maxs.e >> 18) & 63) - startx, 7);                                      \
	int count0 = max(max0 - min0 + 1, 0);                                                          \
	int count1 = max(max1 - min1 + 1, 0);                                                          \
	int count2 = max(max2 - min2 + 1, 0);                                                          \
	int count3 = max(max3 - min3 + 1, 0);                                                          \
	cpos += vec2(float(count0 + (min0 << 1)), 1.0) * count0;                                       \
	cpos += vec2(float(count1 + (min1 << 1)), 3.0) * count1;                                       \
	cpos += vec2(float(count2 + (min2 << 1)), 5.0) * count2;                                       \
	cpos += vec2(float(count3 + (min3 << 1)), 7.0) * count3;

		{
			PROCESS_ROWS(x);
			min_bits = (min0 & 7) | ((min1 & 7) << 3) | ((min2 & 7) << 6) | ((min3 & 7) << 9);
			count_bits = count0 | (count1 << 4) | (count2 << 8) | (count3 << 12);
			num_frags.x = count0 + count1 + count2 + count3;
		}
		{
			PROCESS_ROWS(y);
			min_bits |=
				((min0 & 7) << 12) | ((min1 & 7) << 15) | ((min2 & 7) << 18) | ((min3 & 7) << 21);
			count_bits |= (count0 << 16) | (count1 << 20) | (count2 << 24) | (count3 << 28);
			num_frags.y = count0 + count1 + count2 + count3;
		}

		uint num_all_frags = num_frags.x + num_frags.y;
		cpos *= 0.5 / float(num_all_frags);
		cpos += vec2(bx << 3, by << 3);

		uint scratch_tri_offset = scratch64TriOffset(tri_idx);
		vec2 val0 = uintBitsToFloat(TRI_SCRATCH(0));
		vec2 val1 = uintBitsToFloat(TRI_SCRATCH(1));
		vec3 depth_eq = vec3(val0.x, val0.y, val1.x);
		float ray_pos = depth_eq.x * cpos.x + (depth_eq.y * cpos.y + depth_eq.z);
		float depth = 0xffffe * SATURATE(inversesqrt(ray_pos + 1)); // 20 bits

		if(num_all_frags == 0) // This means that bx_mask is invalid
			RECORD(0, 0, 0, 0);
		uint frag_bits = (num_frags.x << 20) | (num_frags.y << 26);
		g_scratch_64[dst_offset_64 + i] = uvec2(min_bits, count_bits);
		g_scratch_32[dst_offset_32 + i] = tri_idx | frag_bits;

		// 12 bits for tile-tri index, 20 bits for depth
		s_buffer[buf_offset + i] = i | (uint(depth) << 12);
	}
	barrier();

	// For 3 tris and less depth filter is enough, there is no need to sort
	if(tri_count > 3)
		sortTris(lbid, tri_count, buf_offset);

	barrier();
	groupMemoryBarrier();

#ifdef SHADER_DEBUG
	for(uint i = LIX & WARP_MASK; i < tri_count; i += WARP_STEP) {
		uint value = s_buffer[buf_offset + i];
		uint prev_value = i == 0 ? 0 : s_buffer[buf_offset + i - 1];
		if(value <= prev_value)
			RECORD(i, tri_count, prev_value, value);
	}
#endif

#define PREFIX_SUM_STEP(value, step)                                                               \
	{                                                                                              \
		uint temp = shuffleUpNV(value, step, 32);                                                  \
		if((LIX & 31) >= step)                                                                     \
			value += temp;                                                                         \
	}

	for(uint i = LIX & WARP_MASK; i < tri_count; i += WARP_STEP) {
		uint idx = s_buffer[buf_offset + i] & 0xff;
		uint counts = g_scratch_32[dst_offset_32 + idx] >> 20;
		uint num_frags1 = counts & 63, num_frags2 = (counts >> 6) & 63;
		uint num_frags = num_frags1 | (num_frags2 << 12);

		// Computing triangle-ordered sample offsets within each block
		PREFIX_SUM_STEP(num_frags, 1);
		PREFIX_SUM_STEP(num_frags, 2);
		PREFIX_SUM_STEP(num_frags, 4);
		PREFIX_SUM_STEP(num_frags, 8);
		PREFIX_SUM_STEP(num_frags, 16);
		s_buffer[buf_offset + i] = num_frags | (idx << 24);
	}
	barrier();

	// Computing prefix sum across whole blocks (at most 8 * 32 elements)
	if(LIX < 8 * NUM_WARPS) {
		uint lbid = LIX >> 3, warp_idx = LIX & 7, warp_offset = warp_idx << 5;
		uint buf_offset = lbid << MAX_BLOCK_TRIS_SHIFT;
		uint tri_count = s_block_tri_count[lbid];
		uint value = 0;

		if(warp_offset < tri_count) {
			uint tri_idx = min(warp_offset + 31, tri_count - 1);
			value = s_buffer[buf_offset + tri_idx];
		}
		value = (value & 0xfff) | ((value & 0xfff000) << 4);
		uint sum = value, temp;
		temp = shuffleUpNV(sum, 1, 8), sum += warp_idx >= 1 ? temp : 0;
		temp = shuffleUpNV(sum, 2, 8), sum += warp_idx >= 2 ? temp : 0;
		temp = shuffleUpNV(sum, 4, 8), sum += warp_idx >= 4 ? temp : 0;

		if(warp_idx == 7) {
			s_block_frag_count[lbid] = sum;
			if(max(sum & 0xffff, sum >> 16) > MAX_BLOCK_SAMPLES)
				atomicOr(s_raster_error, 0x10000 << lbid);
		}

		sum -= value;
		s_mini_buffer[LIX] = (sum & 0xffff) | ((sum & 0xffff0000) >> 4);
	}
	barrier();
	if(s_raster_error != 0)
		return;

	// Storing triangle fragment offsets to scratch mem
	// Also finding first triangle for each segment
	uint src_offset_32 = dst_offset_32;
	src_offset_64 = dst_offset_64;
	dst_offset_64 = scratch64HalfBlockTrisOffset(lbid << 1);

	uint seg_block1_offset = lbid << (MAX_SEGMENTS_SHIFT + 1);
	uint seg_block2_offset = seg_block1_offset + MAX_SEGMENTS;

	// TODO: split this loop into two
	for(uint i = LIX & WARP_MASK; i < tri_count; i += WARP_STEP) {
		uint tri_offset = 0;
		if(i > 0) {
			uint prev = i - 1;
			tri_offset = s_buffer[buf_offset + prev] & 0xffffff;
			tri_offset += s_mini_buffer[(lbid << 3) + (prev >> 5)];
		}

		uint tri_value = s_buffer[buf_offset + i] + s_mini_buffer[(lbid << 3) + (i >> 5)];
		uint tile_tri_idx = tri_value >> 24;
		tri_value = (tri_value & 0xffffff) - tri_offset;

		uint tri_offset0 = tri_offset & 0xfff, tri_offset1 = (tri_offset >> 12) & 0xfff;
		uint tri_value0 = tri_value & 0xfff, tri_value1 = (tri_value >> 12) & 0xfff;

#define FILL_SEGMENT(tri_offset, tri_value, base)                                                  \
	{                                                                                              \
		uint seg_id = tri_offset >> SEGMENT_SHIFT;                                                 \
		if(tri_value > 0) {                                                                        \
			uint seg_offset = tri_offset & (SEGMENT_SIZE - 1);                                     \
			uint value = i + 1;                                                                    \
			if(seg_offset == 0)                                                                    \
				s_segments[base + seg_id] = value;                                                 \
			else if(seg_offset + tri_value > SEGMENT_SIZE)                                         \
				s_segments[base + seg_id + 1] = value;                                             \
		}                                                                                          \
	}
		FILL_SEGMENT(tri_offset0, tri_value0, seg_block1_offset);
		FILL_SEGMENT(tri_offset1, tri_value1, seg_block2_offset);

#undef FILL_SEGMENT

		uint tri_idx = g_scratch_32[src_offset_32 + tile_tri_idx] & 0xffff;
		uvec2 tri_data = g_scratch_64[src_offset_64 + tile_tri_idx];

		// TODO: group min & count bits earlier
		// TODO: just store plain bits
		g_scratch_64[dst_offset_64 + i] = uvec2(
			tri_idx | (tri_offset0 << 16), (tri_data.x & 0xfff) | ((tri_data.y & 0xffff) << 16));
		g_scratch_64[dst_offset_64 + i + MAX_BLOCK_TRIS] =
			uvec2(tri_idx | (tri_offset1 << 16),
				  ((tri_data.x & 0xfff000) >> 12) | (tri_data.y & 0xffff0000));
	}
	barrier();

	{
		uint sbid = LIX >> 4, seg_group_offset = sbid << MAX_SEGMENTS_SHIFT;
		uint tri_count = s_block_tri_count[sbid >> 1];

		for(uint seg_id = LIX & 15; seg_id < MAX_SEGMENTS; seg_id += 16) {
			uint cur_value = s_segments[seg_group_offset + seg_id];
			if(cur_value == 0)
				break;

			uint next_value =
				seg_id + 1 == MAX_SEGMENTS ? 0 : s_segments[seg_group_offset + seg_id + 1];
			next_value = next_value == 0 ? tri_count : min(tri_count, next_value);
			cur_value = (cur_value - 1) | ((next_value - (cur_value - 1)) << 8);
			s_segments[seg_group_offset + seg_id] = cur_value;
		}
	}
}

void loadSamples(uint bid, uint hbid, int segment_id) {
	uint sbid = ((bid << 1) + hbid) & (NUM_WARPS * 2 - 1); // TODO: describe
	uint seg_group_offset = sbid << MAX_SEGMENTS_SHIFT;
	uint segment_data = s_segments[seg_group_offset + segment_id];
	uint tri_count = segment_data >> 8;
	uint first_tri = segment_data & 0xff;

	uint src_offset_64 = scratch64HalfBlockTrisOffset(sbid) + first_tri;
	uint buf_offset = (LIX >> 5) << SEGMENT_SHIFT;
	int first_offset = segment_id << SEGMENT_SHIFT;

	int y = int(LIX & 3);
	uint count_shift = 16 + (y << 2), min_shift = (y << 1) + y;

	// TODO: group differently for better memory accesses (and measure)
	for(uint i = (LIX & WARP_MASK) >> 2; i < tri_count; i += WARP_STEP / 4) {
		uvec2 tri_data = g_scratch_64[src_offset_64 + i];
		uint tri_idx = tri_data.x & 0xffff;
		int tri_offset = int(tri_data.x >> 16) - first_offset;

		int minx = int((tri_data.y >> min_shift) & 7);
		int countx = int((tri_data.y >> count_shift) & 15);

		int prevx = countx, temp;
		temp = shuffleUpNV(prevx, 1, 4), prevx += y >= 1 ? temp : 0;
		temp = shuffleUpNV(prevx, 2, 4), prevx += y >= 2 ? temp : 0;
		prevx -= countx;
		tri_offset += prevx;

		countx = min(countx, SEGMENT_SIZE - tri_offset);
		if(tri_offset < 0) {
			countx += tri_offset;
			minx -= tri_offset;
			tri_offset = 0;
		}
		if(countx <= 0)
			continue;

		uint scratch_tri_offset = scratch64TriOffset(tri_idx);
		uint pixel_id = (y << 3) | minx;
		uint value = pixel_id | (scratch_tri_offset << 8);

		for(int j = 0; j < countx; j++) {
			s_buffer[buf_offset + tri_offset] = value;
			tri_offset++;
			value++;
		}
	}
}

// TODO: Can we improve speed of loading vertex data?
uint shadeSample(ivec2 bin_pixel_pos, uint scratch_tri_offset, out float out_depth) {
	float px = float(bin_pixel_pos.x), py = float(bin_pixel_pos.y);

	vec3 depth_eq, edge0_eq, edge1_eq;
	uint instance_id, instance_flags;
	vec2 bary_params;
	getTriangleParams(scratch_tri_offset, depth_eq, bary_params, edge0_eq, edge1_eq, instance_id,
					  instance_flags);
	uint instance_color, unormal;
	getTriangleSecondaryParams(scratch_tri_offset, unormal, instance_color);

	float inv_ray_pos = depth_eq.x * px + (depth_eq.y * py + depth_eq.z);
	out_depth = inv_ray_pos;
	float ray_pos = 1.0 / inv_ray_pos;

	float e0 = edge0_eq.x * px + (edge0_eq.y * py + edge0_eq.z);
	float e1 = edge1_eq.x * px + (edge1_eq.y * py + edge1_eq.z);
	vec2 bary = vec2(e0, e1) * ray_pos;

	vec2 bary_dx, bary_dy;
	if((instance_flags & INST_HAS_TEXTURE) != 0) {
		float ray_posx = 1.0 / (inv_ray_pos + depth_eq.x);
		float ray_posy = 1.0 / (inv_ray_pos + depth_eq.y);

		bary_dx = vec2(e0 + edge0_eq.x, e1 + edge1_eq.x) * ray_posx - bary;
		bary_dy = vec2(e0 + edge0_eq.y, e1 + edge1_eq.y) * ray_posy - bary;
	}
	bary -= bary_params;

	vec4 color = decodeRGBA8(instance_color);
	if((instance_flags & INST_HAS_TEXTURE) != 0) {
		vec2 tex0, tex1, tex2;
		getTriangleVertexTexCoords(scratch_tri_offset, tex0, tex1, tex2);

		vec2 tex_coord = bary[0] * tex1 + (bary[1] * tex2 + tex0);
		vec2 tex_dx = bary_dx[0] * tex1 + bary_dx[1] * tex2;
		vec2 tex_dy = bary_dy[0] * tex1 + bary_dy[1] * tex2;

		if((instance_flags & INST_HAS_UV_RECT) != 0) {
			vec4 uv_rect = g_uv_rects[instance_id];
			tex_coord = uv_rect.zw * fract(tex_coord) + uv_rect.xy;
			tex_dx *= uv_rect.zw, tex_dy *= uv_rect.zw;
		}

		vec4 tex_col;
		if((instance_flags & INST_TEX_OPAQUE) != 0)
			tex_col = vec4(textureGrad(opaque_texture, tex_coord, tex_dx, tex_dy).xyz, 1.0);
		else
			tex_col = textureGrad(transparent_texture, tex_coord, tex_dx, tex_dy);
		color *= tex_col;
	}

	if((instance_flags & INST_HAS_VERTEX_COLORS) != 0) {
		vec4 col0, col1, col2;
		getTriangleVertexColors(scratch_tri_offset, col0, col1, col2);
		color *= (1.0 - bary[0] - bary[1]) * col0 + (bary[0] * col1 + bary[1] * col2);
	}

	if(color.a == 0.0)
		return 0;

	vec3 normal;
	if((instance_flags & INST_HAS_VERTEX_NORMALS) != 0) {
		vec3 nrm0, nrm1, nrm2;
		getTriangleVertexNormals(scratch_tri_offset, nrm0, nrm1, nrm2);
		nrm1 -= nrm0;
		nrm2 -= nrm0;
		normal = bary[0] * nrm1 + (bary[1] * nrm2 + nrm0);
	} else {
		normal = decodeNormalUint(unormal);
	}

	float light_value = max(0.0, dot(-lighting.sun_dir, normal) * 0.7 + 0.3);
	color.rgb = SATURATE(finalShading(color.rgb, light_value));
	return encodeRGBA8(color);
}

void shadeSamples(int bid, int hbid, uint sample_count) {
	// TODO: what's the best way to fix broken pixels?
	// full sort ? recreate full depth values and sort pairs?

	uint buf_offset = (LIX >> 5) << SEGMENT_SHIFT;
	ivec2 half_block_pos = ivec2((bid & 7) << 3, (bid & ~7) + (hbid << 2));

	for(uint i = LIX & WARP_MASK; i < sample_count; i += WARP_STEP) {
		uint value = s_buffer[buf_offset + i];
		uint pixel_id = value & 31;
		uint scratch_tri_offset = value >> 8;
		ivec2 pix_pos = half_block_pos + ivec2(pixel_id & 7, pixel_id >> 3);
		float depth;
		s_buffer[buf_offset + i] = shadeSample(pix_pos, scratch_tri_offset, depth);
		s_buffer[buf_offset + i + SEGMENT_SIZE * NUM_WARPS] =
			(floatBitsToUint(depth) & ~31) | pixel_id;
	}
}

struct ReductionContext {
#ifdef VISUALIZE_ERRORS
	vec4 prev_depths;
#else
	vec3 prev_depths;
#endif
	uvec3 prev_colors;
	float out_trans;
	uint out_color;
};

void initReduceSamples(out ReductionContext ctx) {
#ifdef VISUALIZE_ERRORS
	ctx.prev_depths = vec4(999999999.0);
#else
	ctx.prev_depths = vec3(999999999.0);
#endif
	ctx.prev_colors = uvec3(0);
	ctx.out_color = 0;
	ctx.out_trans = 1.0;
}

void reduceSamples(uint bid, uint sample_count, in out ReductionContext ctx) {
	uint buf_offset = (LIX >> 5) << SEGMENT_SHIFT;
	uint mini_offset = LIX & ~31;
	uint pixel_bit = 1u << (LIX & 31);
	uint pixel_id = LIX & 31;
	vec3 out_color = decodeRGB10(ctx.out_color);

	for(uint i = 0; i < sample_count; i += 32) {
		uint sample_offset = i + (LIX & 31);
		uint sample_pixel_id = s_buffer[buf_offset + sample_offset + SEGMENT_SIZE * NUM_WARPS] & 31;

		s_mini_buffer[LIX] = 0;
		if(sample_offset < sample_count)
			atomicOr(s_mini_buffer[mini_offset + sample_pixel_id], pixel_bit);

		uint bitmask = s_mini_buffer[mini_offset + pixel_id];
		int j = findLSB(bitmask);
		while(j != -1) {
			// TODO: pass through regs?
			uint value = s_buffer[buf_offset + i + j];
			float depth = uintBitsToFloat(s_buffer[buf_offset + i + j + SEGMENT_SIZE * NUM_WARPS]);

			bitmask &= ~(1 << j);
			j = findLSB(bitmask);

			if(depth > ctx.prev_depths[0]) {
				SWAP_UINT(value, ctx.prev_colors[0]);
				SWAP_FLOAT(depth, ctx.prev_depths[0]);
				if(ctx.prev_depths[0] > ctx.prev_depths[1]) {
					SWAP_UINT(ctx.prev_colors[1], ctx.prev_colors[0]);
					SWAP_FLOAT(ctx.prev_depths[1], ctx.prev_depths[0]);
					if(ctx.prev_depths[1] > ctx.prev_depths[2]) {
						SWAP_UINT(ctx.prev_colors[2], ctx.prev_colors[1]);
						SWAP_FLOAT(ctx.prev_depths[2], ctx.prev_depths[1]);

#ifdef VISUALIZE_ERRORS
						if(ctx.prev_depths[2] > ctx.prev_depths[3]) {
							ctx.prev_colors[2] = 0xff0000ff;
							i = sample_count;
							break;
						}
#endif
					}
				}
			}

#ifdef VISUALIZE_ERRORS
			ctx.prev_depths[3] = ctx.prev_depths[2];
#endif
			ctx.prev_depths[2] = ctx.prev_depths[1];
			ctx.prev_depths[1] = ctx.prev_depths[0];
			ctx.prev_depths[0] = depth;

			if(ctx.prev_colors[2] != 0) {
				vec4 cur_color = decodeRGBA8(ctx.prev_colors[2]);
#ifdef ADDITIVE_BLENDING
				out_color += cur_color.rgb * cur_color.a;
#else
				out_color += cur_color.rgb * cur_color.a * ctx.out_trans;
				ctx.out_trans *= 1.0 - cur_color.a;

#ifdef ALPHA_THRESHOLD
				if(allInvocationsARB(ctx.out_trans < 1.0 / 255.0)) {
					i = sample_count;
					break;
				}
#endif
#endif
			}

			ctx.prev_colors[2] = ctx.prev_colors[1];
			ctx.prev_colors[1] = ctx.prev_colors[0];
			ctx.prev_colors[0] = value;
		}
	}

	// TODO: check if encode+decode for out_color is really needed (to save 2 regs)
	ctx.out_color = encodeRGB10(out_color);
}

void finishReduceSamples(ivec2 pixel_pos, ReductionContext ctx) {
	vec3 out_color = decodeRGB10(ctx.out_color);

	for(int i = 2; i >= 0; i--)
		if(ctx.prev_colors[i] != 0) {
			vec4 cur_color = decodeRGBA8(ctx.prev_colors[i]);
			float cur_transparency = 1.0 - cur_color.a;
#ifdef ADDITIVE_BLENDING
			out_color += cur_color.rgb * cur_color.a;
#else
			out_color += cur_color.rgb * cur_color.a * ctx.out_trans;
			ctx.out_trans *= 1.0 - cur_color.a;
#endif
		}

	out_color += ctx.out_trans * decodeRGB8(background_color);
	uint enc_color = encodeRGB8(SATURATE(out_color)); // TODO: 10 bit
	outputPixel(pixel_pos, enc_color);
}

void initVisualizeSamples() { s_vis_pixels[LIX] = 0; }

void visualizeSamples(uint sample_count) {
	uint buf_offset = (LIX >> 5) << SEGMENT_SHIFT;
	for(uint i = LIX & 31; i < sample_count; i += 32) {
		uint pixel_id = s_buffer[buf_offset + i] & 31;
		atomicAdd(s_vis_pixels[(LIX & ~31) + pixel_id], 1);
	}
}

void finishVisualizeSamples(int bid, ivec2 pixel_pos) {
	uint pixel_id = (pixel_pos.x & 7) + ((pixel_pos.y & 3) << 3);
	vec3 color = vec3(s_vis_pixels[(LIX & ~31) + pixel_id]) / 32.0;
	uint enc_col = encodeRGBA8(vec4(SATURATE(color), 1.0));
	outputPixel(pixel_pos, enc_col);
}

void visualizeFragmentCounts(int bid, int hbid, ivec2 pixel_pos) {
	int lbid = bid & (NUM_WARPS - 1);
	uint count = (s_block_frag_count[lbid] >> (hbid << 4)) & 0xffff;
	vec4 color = vec4(SATURATE(vec3(count) / 512.0), 1.0);
	outputPixel(pixel_pos, encodeRGBA8(color));
}

void visualizeTriangleCounts(int bid, ivec2 pixel_pos) {
	uint count = s_block_tri_count[bid & (NUM_WARPS - 1)];
	//count = s_block_row_tri_count[bid >> 3];

	vec3 color;
	if(count == 0)
		color = vec3(0);
	else if(count < 16)
		color = vec3(count + 16, 0, 0) / 32.0;
	else if(count < 64)
		color = vec3(count + 32, count + 32, 0) / 96.0;
	else
		color = vec3(0, count, 0.0) / 256.0;
	outputPixel(pixel_pos, encodeRGBA8(vec4(SATURATE(color), 1.0)));
}

void visualizeSegments(int bid, int hbid, ivec2 pixel_pos) {
	barrier();
	if(LIX < NUM_WARPS * 2) {
		uint seg_group_offset = LIX << MAX_SEGMENTS_SHIFT;
		s_mini_buffer[LIX] = 0;

		int prev_tri = -1;
		for(int i = 0; i < MAX_SEGMENTS; i++) {
			uint segment = s_segments[seg_group_offset + i];
			if(segment == 0)
				break;

			int first_tri = int(segment & 0xff);
			uint tri_count = (segment >> 8) & 0xff;
			if(first_tri <= prev_tri)
				RECORD(bid, i, prev_tri, first_tri);
			prev_tri = first_tri;
			s_mini_buffer[LIX] += tri_count;
		}
	}
	barrier();
	int sbid = (bid * 2 + hbid) & (NUM_WARPS * 2 - 1);
	uint count = s_mini_buffer[sbid] * 8;
	vec3 color = vec3(count) / 512;
	outputPixel(pixel_pos, encodeRGBA8(vec4(SATURATE(color), 1.0)));
	barrier();
}

void visualizeErrors(int bid) {
	int lbid = int(LIX >> 5);
	bid += lbid;

	uint color = 0xff000031;
	if((s_raster_error & (1 << lbid)) != 0)
		color += 0x32;
	if((s_raster_error & (0x10000 << lbid)) != 0)
		color += 0x64;

	uint bx = bid & 7, by = bid >> 3;
	ivec2 pixel_pos = ivec2((LIX & 7) + (bx << 3), ((LIX >> 3) & 3) + (by << 3));
	outputPixel(pixel_pos, color);
	outputPixel(pixel_pos + ivec2(0, 4), color);
}

#define NUM_BIN_STEPS (64 / NUM_WARPS)

void rasterBin(int bin_id) {
	INIT_CLOCK();

	if(LIX < NUM_BLOCK_ROWS) {
		if(LIX == 0) {
			ivec2 bin_pos = ivec2(bin_id % BIN_COUNT_X, bin_id / BIN_COUNT_X) * BIN_SIZE;
			s_bin_pos = bin_pos;
			s_bin_quad_count = g_bins.bin_quad_counts[bin_id];
			s_bin_quad_offset = g_bins.bin_quad_offsets[bin_id];
			s_bin_ray_dir0 = frustum.ws_dir0 + frustum.ws_dirx * (bin_pos.x + 0.5) +
							 frustum.ws_diry * (bin_pos.y + 0.5);
			s_raster_error = 0;
		}

		s_block_row_tri_count[LIX] = 0;
	}
	barrier();
	processQuads();
	groupMemoryBarrier();
	barrier();
	UPDATE_CLOCK(0);

	//  bid: block (8x8) id; We have 8 x 8 = 64 blocks
	// fbid: first block id currently processed out of all warps
	// lbid: local block id (range: 0 up to NUM_WARPS - 1)
	// hbid: half block id; either 0 (upper half) or 1 (bottom half)
	// Each block has 64 pixels, so we need 2 warps to process one block
	for(int fbid = 0; fbid < 64; fbid += NUM_WARPS / 2) {
		barrier();
		if((fbid & NUM_WARPS / 2) == 0) {
			generateBlocks(fbid);
			groupMemoryBarrier();
			barrier();

			if(s_raster_error != 0) {
				visualizeErrors(fbid);
				fbid += NUM_WARPS;
				barrier();
				if(LIX == 0)
					s_raster_error = 0;
				continue;
			}
		}
		UPDATE_CLOCK(1);

		int bid = fbid + int(LIX >> 6), hbid = int((LIX >> 5) & 1);
		int frag_count = int((s_block_frag_count[bid & (NUM_WARPS - 1)] >> (hbid << 4)) & 0xffff);
		int segment_id = 0;

		ReductionContext context;
		initReduceSamples(context);
		//initVisualizeSamples();

		while(frag_count > 0) {
			int cur_frag_count = min(frag_count, SEGMENT_SIZE);
			frag_count -= SEGMENT_SIZE;

			loadSamples(bid, hbid, segment_id++);
			UPDATE_CLOCK(2);

			//visualizeSamples(cur_frag_count);
			shadeSamples(bid, hbid, cur_frag_count);
			UPDATE_CLOCK(3);

			reduceSamples(bid, cur_frag_count, context);
			UPDATE_CLOCK(5);

#ifdef ALPHA_THRESHOLD
			if(allInvocationsARB(context.out_trans < 1.0 / 255.0))
				break;
#endif
		}

		ivec2 pixel_pos =
			ivec2((LIX & 7) + ((bid & 7) << 3), ((LIX >> 3) & 3) + (bid & ~7) + (hbid << 2));
		finishReduceSamples(pixel_pos, context);
		UPDATE_CLOCK(6);

		//finishVisualizeSamples(bid, pixel_pos);
		//visualizeFragmentCounts(bid, hbid, pixel_pos);
		//visualizeTriangleCounts(bid, pixel_pos);
		//visualizeSegments(bid, hbid, pixel_pos);
		//UPDATE_CLOCK(7);
		barrier();
	}
}

int loadNextBin() {
	if(LIX == 0) {
		uint bin_idx = atomicAdd(g_bins.small_bin_counter, 1);
		s_bin_id = bin_idx < s_num_bins ? g_bins.small_bins[bin_idx] : -1;
		s_bin_raster_offset = s_bin_id << (BIN_SHIFT * 2);
	}
	barrier();
	return s_bin_id;
}

void main() {
	initTimers();
	if(LIX == 0)
		s_num_bins = g_bins.num_small_bins;

	int bin_id = loadNextBin();
	while(bin_id != -1) {
		barrier();
		rasterBin(bin_id);
		bin_id = loadNextBin();
	}
	commitTimers();
}
