// $$include funcs lighting frustum viewport data

#define LIX gl_LocalInvocationIndex
#define LID gl_LocalInvocationID

#define LSIZE 256
#define LSHIFT 8

layout(local_size_x = LSIZE) in;
layout(binding = 0, r32ui) uniform uimage2D final_raster;

layout(std430, binding = 0) buffer buf0_ { uvec4 g_tri_aabbs[]; };
layout(std430, binding = 1) buffer buf1_ { uint g_quad_indices[]; };

layout(std430, binding = 2) readonly buffer buf2_ { float g_verts[]; };
layout(std430, binding = 3) readonly buffer buf3_ { vec2 g_tex_coords[]; };
layout(std430, binding = 4) readonly buffer buf4_ { uint g_colors[]; };
layout(std430, binding = 5) readonly buffer buf5_ { uint g_normals[]; };

layout(std430, binding = 6) buffer buf6_ { BinCounters g_bins; };

layout(std430, binding = 8) buffer buf8_ { uint g_bin_quads[]; };
layout(std430, binding = 9) coherent buffer buf9_ { uvec2 g_scratch[]; };

layout(std430, binding = 10) readonly buffer buf10_ { InstanceData g_instances[]; };
layout(std430, binding = 11) readonly buffer buf11_ { vec4 g_uv_rects[]; };

layout(binding = 0) uniform sampler2D opaque_texture;
layout(binding = 1) uniform sampler2D transparent_texture;

#define WORKGROUP_SCRATCH_SIZE	(64 * 1024)
#define WORKGROUP_SCRATCH_SHIFT	16

#define SAMPLES_PER_THREAD		4
#define MAX_SAMPLES				(LSIZE * SAMPLES_PER_THREAD)

// TODO: there is no need to check that, no of input tris is <= this
#define MAX_BLOCK_ROW_TRIS		MAX_SAMPLES
#define MAX_SCRATCH_TRIS		1024
#define MAX_SCRATCH_TRIS_SHIFT  10

#define SCRATCH_TRI_OFFSET		(MAX_BLOCK_ROW_TRIS * BLOCK_ROW_COUNT)
#define BLOCK_ROW_COUNT			16

shared ivec2 s_bin_pos;
shared vec3 s_bin_ray_dir0;

shared uint s_block_row_tri_counts[BLOCK_ROW_COUNT];

// TODO: add protection from too big number of samples:
// maximum per row for raster_bin = min(4 * LSIZE, 32768) ?
// we have to somehow estimate max# of samples during categorization?

// TODO: 16-bit (it speeds up by about 1%...)
shared uint s_row_frag_counts[BIN_SIZE];
shared uint s_row2_frag_counts[BIN_SIZE / 2];
shared uint s_block_row_frag_counts[BLOCK_ROW_COUNT];
shared uint s_block_row_max_frag_counts[BLOCK_ROW_COUNT];

shared uint s_bin_quad_count, s_bin_quad_offset;

shared int s_pixel_counts[BIN_SIZE * BLOCK_SIZE];

shared int s_sample_count;
shared uvec2 s_buffer[MAX_SAMPLES + 1];

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

// TODO: add synthetic test: 256 planes one after another

void computeRowFragCounts() {
	if(LIX < BIN_SIZE) {
		uint value = s_row_frag_counts[LIX];
		atomicAdd(s_block_row_frag_counts[LIX >> 2], value);
		atomicMax(s_block_row_max_frag_counts[LIX >> 2], value);
		atomicAdd(s_row2_frag_counts[LIX >> 1], value);
	}
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

// TODO: można by tutaj użyć algorytmu bazującego na liniach
void generateTriGroups(uint tri_idx, vec3 tri0, vec3 tri1, vec3 tri2, int min_by, int max_by) {
	// Inspired by Nanite scanline rasterizer
	vec3 scan_min, scan_max, scan_step;
	{
		float sx = s_bin_pos.x - 0.5f; // TODO: why -0.5? it's correct though
		float sy = s_bin_pos.y + min_by * 4 + 0.5f;

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

	uint soffset = gl_WorkGroupID.x * WORKGROUP_SCRATCH_SIZE + min_by * MAX_BLOCK_ROW_TRIS;
	for(int by = min_by; by <= max_by; by++) {
		uint row_ranges[2] = {0, 0};

		for(int y = 0; y < 4; y++) {
			float xmin = max(max(scan_min[0], scan_min[1]), max(scan_min[2], 0.0));
			float xmax = min(min(scan_max[0], scan_max[1]), min(scan_max[2], BIN_SIZE));

			scan_min += scan_step;
			scan_max += scan_step;
			
			// TODO: use floor/ceil?
			int imin = int(xmin), imax = int(xmax) - 1;
			atomicAdd(s_row_frag_counts[by * 4 + y], uint(max(0, imax - imin + 1)));
			row_ranges[y >> 1] |= (imin <= imax? (uint(imin) | (uint(imax) << 6)) : 0x3f) << ((y & 1) * 12);
		}

		if(row_ranges[0] != 0x03f03f || row_ranges[1] != 0x03f03f) {
			uint roffset = atomicAdd(s_block_row_tri_counts[by], 1);
			g_scratch[soffset + roffset] = uvec2(row_ranges[0] | (tri_idx << 24),
												 row_ranges[1] | ((tri_idx & 0xff00) << 16));
		}
		soffset += MAX_BLOCK_ROW_TRIS;
	}
}

// TODO: don't store triangles which generate very small number of samples in scratch,
// instead precompute them directly when sampling; We would have to somehow group those triangles together
//
// TODO: use scratch based on uints, not uvec2, maybe it will be a bit faster?

#define TRI_SCRATCH(var_idx) \
	g_scratch[toffset + (var_idx << MAX_SCRATCH_TRIS_SHIFT)]

uint scratchTriOffset(uint tri_idx) {
	return (gl_WorkGroupID.x << WORKGROUP_SCRATCH_SHIFT) + SCRATCH_TRI_OFFSET + tri_idx;
}

void storeTriangle(uint tri_idx, vec3 tri0, vec3 tri1, vec3 tri2, uint v0, uint v1, uint v2, uint instance_id)
{
	uint toffset = scratchTriOffset(tri_idx);
	vec3 normal = cross(tri0 - tri2, tri1 - tri0);
	float multiplier = 1.0 / length(normal);
	normal *= multiplier;

	vec3 edge0 = (tri0 - tri2) * multiplier;
	vec3 edge1 = (tri1 - tri0) * multiplier;
	
	float plane_dist = dot(normal, tri0);
	float param0 = dot(cross(edge0, tri0), normal);
	float param1 = dot(cross(edge1, tri0), normal);

	// Nice optimization for barycentric computations:
	// dot(cross(edge, dir), normal) == dot(dir, cross(normal, edge))
	edge0 = cross(normal, edge0);
	edge1 = cross(normal, edge1);

	TRI_SCRATCH(0) = uvec2(floatBitsToUint(normal.x), floatBitsToUint(normal.y));
	TRI_SCRATCH(1) = uvec2(floatBitsToUint(normal.z), floatBitsToUint(plane_dist));
	TRI_SCRATCH(2) = uvec2(floatBitsToUint(param0), floatBitsToUint(param1));
	TRI_SCRATCH(3) = uvec2(floatBitsToUint(edge0.x), floatBitsToUint(edge0.y));
	TRI_SCRATCH(4) = uvec2(floatBitsToUint(edge0.z), floatBitsToUint(edge1.x));
	TRI_SCRATCH(5) = uvec2(floatBitsToUint(edge1.y), floatBitsToUint(edge1.z));

	uint instance_flags = g_instances[instance_id].flags;
	uint instance_color = g_instances[instance_id].color;
	
	TRI_SCRATCH(6) = uvec2(instance_flags, instance_color);
	TRI_SCRATCH(7) = uvec2(instance_id, 0);

	uint vcolor2 = 0;
	if((instance_flags & INST_HAS_VERTEX_COLORS) != 0) {
		TRI_SCRATCH(8) = uvec2(g_colors[v0], g_colors[v1]);
		vcolor2 = g_colors[v2];
	}
	uint vnormal2 = 0;
	if((instance_flags & INST_HAS_VERTEX_NORMALS) != 0) {
		TRI_SCRATCH(10) = uvec2(g_normals[v0], g_normals[v1]);
		vnormal2 = g_normals[v2];
	}
	
	if((instance_flags & (INST_HAS_VERTEX_COLORS | INST_HAS_VERTEX_NORMALS)) != 0) {
		TRI_SCRATCH(9) = uvec2(vcolor2, vnormal2);
	}
	if((instance_flags & INST_HAS_TEXTURE) != 0) {
		vec2 tex0 = g_tex_coords[v0];
		vec2 tex1 = g_tex_coords[v1];
		vec2 tex2 = g_tex_coords[v2];
		tex1 -= tex0; tex2 -= tex0;
		TRI_SCRATCH(11) = floatBitsToUint(tex0);
		TRI_SCRATCH(12) = floatBitsToUint(tex1);
		TRI_SCRATCH(13) = floatBitsToUint(tex2);
	}
}

void getTriangleParams(uint tri_idx, out vec3 normal, out vec3 params, out vec3 edge0, out vec3 edge1,
		out uint instance_id, out uint instance_flags, out uint instance_color) {
	uint toffset = scratchTriOffset(tri_idx);
	{
		uvec2 val0 = TRI_SCRATCH(0), val1 = TRI_SCRATCH(1), val2 = TRI_SCRATCH(2);
		normal = vec3(uintBitsToFloat(val0[0]), uintBitsToFloat(val0[1]), uintBitsToFloat(val1[0]));
		params = vec3(uintBitsToFloat(val1[1]), uintBitsToFloat(val2[0]), uintBitsToFloat(val2[1]));
	}
	{
		uvec2 val0 = TRI_SCRATCH(3), val1 = TRI_SCRATCH(4), val2 = TRI_SCRATCH(5);
		edge0 = vec3(uintBitsToFloat(val0[0]), uintBitsToFloat(val0[1]), uintBitsToFloat(val1[0]));
		edge1 = vec3(uintBitsToFloat(val1[1]), uintBitsToFloat(val2[0]), uintBitsToFloat(val2[1]));
	}

	{
		uvec2 val0 = TRI_SCRATCH(6);
		instance_flags = val0.x;
		instance_color = val0.y;
		instance_id = TRI_SCRATCH(7).x;
	}
}

void getTriangleVertexColors(uint tri_idx, out vec4 color0, out vec4 color1, out vec4 color2) {
	uint toffset = scratchTriOffset(tri_idx);
	uvec2 val0 = TRI_SCRATCH(8);
	uvec2 val1 = TRI_SCRATCH(9);
	color0 = decodeRGBA8(val0[0]);
	color1 = decodeRGBA8(val0[1]);
	color2 = decodeRGBA8(val1[0]);
}

void getTriangleVertexNormals(uint tri_idx, out vec3 normal0, out vec3 normal1, out vec3 normal2) {
	uint toffset = scratchTriOffset(tri_idx);
	uvec2 val0 = TRI_SCRATCH(10);
	uvec2 val1 = TRI_SCRATCH(9);
	normal0 = decodeNormalUint(val0[0]);
	normal1 = decodeNormalUint(val0[1]);
	normal2 = decodeNormalUint(val1[1]);
}

void getTriangleVertexTexCoords(uint tri_idx, out vec2 tex0, out vec2 tex1, out vec2 tex2) {
	uint toffset = scratchTriOffset(tri_idx);
	uvec2 val0 = TRI_SCRATCH(11);
	uvec2 val1 = TRI_SCRATCH(12);
	uvec2 val2 = TRI_SCRATCH(13);
	tex0 = uintBitsToFloat(val0);
	tex1 = uintBitsToFloat(val1);
	tex2 = uintBitsToFloat(val2);
}

void generateRows() {
	// TODO: optimization: in many cases all rows may very well fit in SMEM,
	// maybe it would be worth it not to use scratch at all then?
	// TODO: this loop is slooooow
	// TODO: divide big tris across different threads
	for(uint i = LIX >> 1; i < s_bin_quad_count; i += LSIZE / 2) {
		uint second_tri = LIX & 1;
		uint quad_idx = g_bin_quads[s_bin_quad_offset + i] & 0xffffff;
		
		uvec4 aabb = g_tri_aabbs[quad_idx];
		aabb = decodeAABB(second_tri != 0? aabb.zw : aabb.xy);
		int min_by = clamp(int(aabb[1]) - s_bin_pos.y, 0, 63) >> 2;
		int max_by = clamp(int(aabb[3]) - s_bin_pos.y, 0, 63) >> 2;

		uint verts[4] = { g_quad_indices[quad_idx * 4 + 0], g_quad_indices[quad_idx * 4 + 1],
						  g_quad_indices[quad_idx * 4 + 2], g_quad_indices[quad_idx * 4 + 3] };
		uint instance_id = (verts[0] >> 26) | ((verts[1] >> 20) & 0xfc0) | ((verts[2] >> 14) & 0x3f000);
		uint v0 = verts[0] & 0x03ffffff;
		uint v1 = verts[1 + second_tri] & 0x03ffffff;
		uint v2 = verts[2 + second_tri] & 0x03ffffff;

		vec3 tri0 = vec3(g_verts[v0 * 3 + 0], g_verts[v0 * 3 + 1], g_verts[v0 * 3 + 2]) - frustum.ws_shared_origin;
		vec3 tri1 = vec3(g_verts[v1 * 3 + 0], g_verts[v1 * 3 + 1], g_verts[v1 * 3 + 2]) - frustum.ws_shared_origin;
		vec3 tri2 = vec3(g_verts[v2 * 3 + 0], g_verts[v2 * 3 + 1], g_verts[v2 * 3 + 2]) - frustum.ws_shared_origin;
		
		// TODO: store only if samples were generated
		// TODO: do triangle storing later
		uint tri_idx = i * 2 + (LIX & 1);
		storeTriangle(tri_idx, tri0, tri1, tri2, v0, v1, v2, instance_id);
		generateTriGroups(tri_idx, tri0, tri1, tri2, min_by, max_by);
	}
}

// TODO: optimize
void loadRowSamples(int by, int y, int ystep) {
	uint soffset = gl_WorkGroupID.x * WORKGROUP_SCRATCH_SIZE + by * MAX_BLOCK_ROW_TRIS;
	uint row_count = s_block_row_tri_counts[by];
	int x = int(LIX & (BIN_SIZE - 1));
	int local_y = int(LIX >> BIN_SHIFT);
	// TODO: share pixels between threads for ystep < 4?
	if(local_y >= ystep)
		return;
	y += local_y;
	
	uint pixel_id = (y << 6) | x;
	uint pix_offset = s_pixel_counts[pixel_id] >> 16;
	int num_samples = 0;
	
#ifdef VENDOR_NVIDIA
	// TODO: WARP_SIZE?
	// TODO: more precomputation here
	vec3 ray_dir = s_bin_ray_dir0 + (by * 4 + y) * frustum.ws_diry + x * frustum.ws_dirx;

	// TODO: move depth generation out of here
	for(uint i = 0; i < row_count; i += 32) {
		uint sub_count = min(32, row_count - i);
		int sel_tri_minx = 31337, sel_tri_maxx;
		uint bitmask, sel_tri_idx = 0;
		vec3 sel_tri_normal;
		{
			bool in_range = false;
			if((LIX & 31) < sub_count) {
				uvec2 row = g_scratch[soffset + i + (LIX & 31)];
				sel_tri_idx = (row.x >> 24) | ((row.y & 0xff000000) >> 16);
				int row_range = int(row[y >> 1] >> ((y & 1) * 12));
				sel_tri_minx = row_range & 0x3f;
				sel_tri_maxx = (row_range >> 6) & 0x3f;

				uint toffset = scratchTriOffset(sel_tri_idx);
				vec2 val0 = uintBitsToFloat(TRI_SCRATCH(0));
				vec2 val1 = uintBitsToFloat(TRI_SCRATCH(1));
				vec3 normal = vec3(val0.x, val0.y, val1.x);
				float param0 = val1.y; //TODO: mul by normal
				sel_tri_normal = normal / param0;

				in_range = (LIX & 32) == 0? sel_tri_minx < 32 : sel_tri_maxx >= 32;
			}
			bitmask = uint(ballotARB(in_range));
		}

		// TODO: sometimes raster_bin (I guess) hangs; noticable when moving camera around sponza
		for(uint j = 0; j < sub_count; j++) {
			int bit_pos = findLSB(bitmask);
			if(bit_pos == -1)
				break;
			j += bit_pos;
			bitmask >>= bit_pos + 1;

			int minx = shuffleNV(sel_tri_minx, j, 32);
			int maxx = shuffleNV(sel_tri_maxx, j, 32);
			uint bin_tri_idx = shuffleNV(sel_tri_idx, j, 32);
			vec3 normal = shuffleNV(sel_tri_normal, j, 32);
			if(x < minx || x > maxx)
				continue;

			num_samples++;
			float ray_pos = 1.0 / dot(normal, ray_dir);
			float depth = float(0xfffffe) / (1.0 + ray_pos);

			uint key = uint(depth) | (pixel_id << 24);
			uint value = (bin_tri_idx << 16) | pix_offset;
			s_buffer[pix_offset++] = uvec2(key, value);
		}
	}

	atomicAdd(s_sample_count, num_samples);
#else
#error write me please
#endif
}
			
void postSortSamples(int by, int y, int ystep) {
	int sample_count = s_sample_count;
	for(uint i = LIX + 1; i < sample_count; i += LSIZE) {
		uint cur_key = s_buffer[i].x, prev_key = s_buffer[i - 1].x;
		if(cur_key >= prev_key)
			continue;

		uint cur_pixel = cur_key >> 24;
		uint cur_value = cur_key & 0xffffff;

		uvec2 temp = s_buffer[i];
		s_buffer[i] = s_buffer[i - 1];
		s_buffer[i - 1] = temp;
	}

	barrier();
	for(uint i = LIX + 1; i < sample_count; i += LSIZE) {
		uint cur_key = s_buffer[i].x;
		if(cur_key < s_buffer[i - 1].x)
			s_pixel_counts[cur_key >> 24] = int(~0u);
	}
}

void reduceRowSamples(int by, int y, int ystep)
{
	// TODO: optimize
	if(LIX < BIN_SIZE * ystep) {
		y += int(LIX >> BIN_SHIFT);
		uint x = LIX & (BIN_SIZE - 1);
		ivec2 pixel_pos = s_bin_pos + ivec2(x, by * 4 + y);

		uint pixel_counter = s_pixel_counts[y * BIN_SIZE + x];
		int num_samples = int(pixel_counter & 0xffff);
		int sample_offset = int(pixel_counter >> 16);
		
		vec3 color = vec3(0);
		float neg_alpha = 1.0;

		if(pixel_counter == ~0u || s_sample_count > MAX_SAMPLES) {
			color = vec3(1.0, 0.0, 0.0);
			neg_alpha = 0.0;
		}
		else {
			for(int i = 0; i < num_samples; i++) {
				vec4 cur_color = decodeRGBA8(s_buffer[sample_offset + i].x);
				float cur_neg_alpha = 1.0 - cur_color.a;
				color.rgb = color.rgb * cur_neg_alpha + cur_color.rgb * cur_color.a;
				neg_alpha *= cur_neg_alpha;
			}

			color = min(color, vec3(1.0));
		}

		uint enc_color = encodeRGBA8(vec4(color, 1.0 - neg_alpha));
		imageStore(final_raster, pixel_pos, uvec4(enc_color, 0, 0, 0));
	}
}

void sortBlockRow(int by)
{
	uint soffset = gl_WorkGroupID.x * WORKGROUP_SCRATCH_SIZE + by * MAX_BLOCK_ROW_TRIS;
	uint tri_count = s_block_row_tri_counts[by];

	for(uint i = LIX; i < tri_count; i += LSIZE) {
		uvec2 row = g_scratch[soffset + i];

		vec2 cpos = vec2(0, 0);
		float weight = 0.0;

		for(int y = 0; y < 4; y++) {
			uint row_range = row[y >> 1] >> ((y & 1) * 12);
			if(row_range == 0x3f)
				continue;
			int minx = int(row_range & 0x3f), maxx = int((row_range >> 6) & 0x3f);
			int count = maxx - minx + 1;
			cpos += vec2(float(maxx + minx + 1) * 0.5, y + 0.5) * count;
			weight += count;
		}

		cpos /= weight;
		cpos.y += by * 4;

		uint bin_tri_idx = (row.x >> 24) | ((row.y & 0xff000000) >> 16);

		uint toffset = scratchTriOffset(bin_tri_idx);
		vec2 val0 = uintBitsToFloat(TRI_SCRATCH(0));
		vec2 val1 = uintBitsToFloat(TRI_SCRATCH(1));
		vec3 normal = vec3(val0.x, val0.y, val1.x);
		float param0 = val1.y; //TODO: mul by normal

		vec3 ray_dir = s_bin_ray_dir0 + cpos.x * frustum.ws_dirx + cpos.y * frustum.ws_diry;
		float ray_pos = param0 / dot(normal, ray_dir);
		float depth = float(0x7ffffffe) / (1.0 + ray_pos);

		s_buffer[i] = uvec2(uint(depth), uint(i));
	}
	barrier();
	sortBuffer(tri_count);
	barrier();
	for(uint i = LIX; i < tri_count; i += LSIZE)
		s_buffer[i] = g_scratch[soffset + s_buffer[i].y];
	barrier();
	for(uint i = LIX; i < tri_count; i += LSIZE)
		g_scratch[soffset + i] = s_buffer[i];
}

// Computes pixel counts for all rows within block-row
// Computes offsets within rows
void computeBlockRowPixelCounts(uint by)
{
	uint soffset = gl_WorkGroupID.x * WORKGROUP_SCRATCH_SIZE + by * MAX_BLOCK_ROW_TRIS;
	uint tri_count = s_block_row_tri_counts[by];
	uint y = LIX & 3, shift = (y & 1) * 12;
		
	for(uint i = LIX; i < BIN_SIZE * BLOCK_SIZE; i += LSIZE)
		s_pixel_counts[LIX] = 0;
	barrier();

	for(uint i = LIX >> 2; i < tri_count; i += LSIZE / 4) {
		uint row_range = (g_scratch[soffset + i][y >> 1] >> shift) & 0xfff;
		if(row_range == 0x3f)
			continue;

		int minx = int(row_range & 0x3f), maxx = int((row_range >> 6) & 0x3f);
		// Storing +1 at the beginning and -1 after the end of each row
		atomicAdd(s_pixel_counts[y * 64 + minx], 1);
		if(maxx < 63)
			atomicAdd(s_pixel_counts[y * 64 + maxx + 1], -1);
	}
	barrier();

#ifdef VENDOR_NVIDIA
	// Computing actual pixel values
	if(LIX < BIN_SIZE * BLOCK_SIZE) {
		int value = s_pixel_counts[LIX], temp;
		temp = shuffleUpNV(value,  1, 32); if((LIX & 31) >=  1) value += temp;
		temp = shuffleUpNV(value,  2, 32); if((LIX & 31) >=  2) value += temp;
		temp = shuffleUpNV(value,  4, 32); if((LIX & 31) >=  4) value += temp;
		temp = shuffleUpNV(value,  8, 32); if((LIX & 31) >=  8) value += temp;
		temp = shuffleUpNV(value, 16, 32); if((LIX & 31) >= 16) value += temp;
		s_pixel_counts[LIX] = value;
	}
	barrier();
	if(LIX < BIN_SIZE * BLOCK_SIZE / 2) {
		uint y = LIX >> 5, x = LIX & 31;
		s_pixel_counts[y * 64 + x + 32] += s_pixel_counts[y * 64 + 31];
	}
	barrier();

	// Computing prefix sums for each row (storing in higher 16-bits)
	if(LIX < BIN_SIZE * BLOCK_SIZE) {
		int value = s_pixel_counts[LIX], temp;
		value += (value << 16);
		temp = shuffleUpNV(value,  1, 32); if((LIX & 31) >=  1) value += temp & 0xffff0000;
		temp = shuffleUpNV(value,  2, 32); if((LIX & 31) >=  2) value += temp & 0xffff0000;
		temp = shuffleUpNV(value,  4, 32); if((LIX & 31) >=  4) value += temp & 0xffff0000;
		temp = shuffleUpNV(value,  8, 32); if((LIX & 31) >=  8) value += temp & 0xffff0000;
		temp = shuffleUpNV(value, 16, 32); if((LIX & 31) >= 16) value += temp & 0xffff0000;
		s_pixel_counts[LIX] = value - (value << 16);
	}
	barrier();
	// Summing prefix sums from two 32-pixel columns
	if(LIX < BIN_SIZE * BLOCK_SIZE / 2) {
		uint y = LIX >> 5, x = LIX & 31;
		int value = s_pixel_counts[y * 64 + 31];
		s_pixel_counts[y * 64 + x + 32] += (value & 0xffff0000) + (value << 16);
	}
#else
#error write me please
#endif
}

// Shading 2 samples at once didn't help:
// - decreased computation cost is not worth it because of increased register pressure
// - it seems that it does not help at all with loading vertex attribs; it makes sense:
//   if they are in the cache then it's not a problem...
//
// Can we improve speed of loading vertex data?
uint shadeSample(ivec2 bin_pixel_pos, uint local_tri_idx)
{
	vec3 ray_dir = s_bin_ray_dir0 + frustum.ws_dirx * bin_pixel_pos.x
								  + frustum.ws_diry * bin_pixel_pos.y;

	vec3 normal, params, edge0, edge1;
	uint instance_id, instance_flags, instance_color;
	getTriangleParams(local_tri_idx, normal, params, edge0, edge1, instance_id, instance_flags, instance_color);

	float ray_pos = params[0] / dot(normal, ray_dir);
	vec2 bary = vec2(dot(edge0, ray_dir), dot(edge1, ray_dir)) * ray_pos;

	vec2 bary_dx, bary_dy;
	if((instance_flags & INST_HAS_TEXTURE) != 0) {
		vec3 ray_dirx = ray_dir + frustum.ws_dirx;
		vec3 ray_diry = ray_dir + frustum.ws_diry;

		float ray_posx = params[0] / dot(normal, ray_dirx);
		float ray_posy = params[0] / dot(normal, ray_diry);

		bary_dx = vec2(dot(edge0, ray_dirx), dot(edge1, ray_dirx)) * ray_posx - bary;
		bary_dy = vec2(dot(edge0, ray_diry), dot(edge1, ray_diry)) * ray_posy - bary;
	}
	bary -= vec2(params[1], params[2]);
	// params, edge0 & edge1 no longer needed!

	vec4 color = decodeRGBA8(instance_color);
	if((instance_flags & INST_HAS_VERTEX_COLORS) != 0) {
		vec4 col0, col1, col2;
		getTriangleVertexColors(local_tri_idx, col0, col1, col2);
		color *= (1.0 - bary[0] - bary[1]) * col0 + bary[0] * col1 + bary[1] * col2;
	}

	if((instance_flags & INST_HAS_TEXTURE) != 0) {
		vec2 tex0, tex1, tex2;
		getTriangleVertexTexCoords(local_tri_idx, tex0, tex1, tex2);

		vec2 tex_coord = tex0 + bary[0] * tex1 + bary[1] * tex2;
		vec2 tex_dx = bary_dx[0] * tex1 + bary_dx[1] * tex2;
		vec2 tex_dy = bary_dy[0] * tex1 + bary_dy[1] * tex2;

		if((instance_flags & INST_HAS_UV_RECT) != 0) {
			vec4 uv_rect = g_uv_rects[instance_id];
			tex_coord = uv_rect.xy + uv_rect.zw * fract(tex_coord);
			tex_dx *= uv_rect.zw, tex_dy *= uv_rect.zw;
		}

		if((instance_flags & INST_TEX_OPAQUE) != 0)
			color.xyz *= textureGrad(opaque_texture, tex_coord, tex_dx, tex_dy).xyz;
		else
			color *= textureGrad(transparent_texture, tex_coord, tex_dx, tex_dy);
	}

	if((instance_flags & INST_HAS_VERTEX_NORMALS) != 0) {
		vec3 nrm0, nrm1, nrm2;
		getTriangleVertexNormals(local_tri_idx, nrm0, nrm1, nrm2);
		nrm1 -= nrm0; nrm2 -= nrm0;
		normal = nrm0 + bary[0] * nrm1 + bary[1] * nrm2;
	}

	float light_value = max(0.0, dot(-lighting.sun_dir, normal) * 0.7 + 0.3);
	color.rgb = min(finalShading(color.rgb, light_value), vec3(1.0, 1.0, 1.0));
	return encodeRGBA8(color);
}

void rasterInvalidBlockRow(int by, vec3 color)
{
	for(uint i = LIX; i < BIN_SIZE * BLOCK_SIZE; i += LSIZE) {
		ivec2 pixel_pos = ivec2(i & (BIN_SIZE - 1), by * 4 + (i >> BIN_SHIFT));
		vec4 color = vec4(color, 1.0);
		uint enc_col = encodeRGBA8(color);
		imageStore(final_raster, s_bin_pos + pixel_pos, uvec4(enc_col, 0, 0, 0));
	}
}

void rasterFragmentCounts(int by)
{
	for(uint i = LIX; i < BIN_SIZE * BLOCK_SIZE; i += LSIZE) {
		ivec2 pixel_pos = ivec2(i & (BIN_SIZE - 1), by * 4 + (i >> BIN_SHIFT));
		uint count = s_pixel_counts[i] & 0xffff;

		vec4 color = vec4(vec3(float(count) / 32.0), 1.0);
		uint enc_col = encodeRGBA8(color);
		imageStore(final_raster, s_bin_pos + pixel_pos, uvec4(enc_col, 0, 0, 0));
	}
}

void rasterBin(int bin_id) {
	if(LIX < BIN_SIZE) {
		if(LIX < BLOCK_ROW_COUNT) {
			if(LIX == 0) {
				ivec2 bin_pos = ivec2(bin_id % BIN_COUNT_X, bin_id / BIN_COUNT_X) * BIN_SIZE;
				s_bin_pos = bin_pos;
				s_bin_quad_count = g_bins.bin_quad_counts[bin_id];
				s_bin_quad_offset = g_bins.bin_quad_offsets[bin_id];
				s_bin_ray_dir0 = frustum.ws_dir0 + frustum.ws_dirx * (bin_pos.x + 0.5)
												 + frustum.ws_diry * (bin_pos.y + 0.5);
			}

			s_block_row_tri_counts[LIX] = 0;
			s_block_row_max_frag_counts[LIX] = 0;
			s_block_row_frag_counts[LIX] = 0;
		}
		s_row_frag_counts[LIX] = 0;
		if(LIX < BIN_SIZE / 2)
			s_row2_frag_counts[LIX] = 0;
	}
	barrier();
	generateRows();
	groupMemoryBarrier();
	barrier();
	computeRowFragCounts();

	for(int by = 0; by < BLOCK_ROW_COUNT; by++) {
		barrier();
		if(s_block_row_tri_counts[by] > MAX_BLOCK_ROW_TRIS) {
			rasterInvalidBlockRow(by, vec3(0.5, 0.0, 0.0));
			continue;
		}
		if(s_block_row_max_frag_counts[by] > MAX_SAMPLES) {
			rasterInvalidBlockRow(by, vec3(1.0, 0.0, 0.0));
			continue;
		}

		computeBlockRowPixelCounts(by);
		sortBlockRow(by);
		barrier();

		uint frag_count01 = s_row2_frag_counts[by * 2 + 0];
		uint frag_count23 = s_row2_frag_counts[by * 2 + 1];

		// How many rows can we rasterize in single step?
		int ystep = frag_count01 + frag_count23 <= MAX_SAMPLES? 4 : max(frag_count01, frag_count23) <= MAX_SAMPLES? 2 : 1;

		// Accumulating prefix sums from 2 rows
		if(ystep >= 2) {
			if(LIX < BIN_SIZE * 2) {
				uint y = (LIX >> BIN_SHIFT) * 2 + 1, x = LIX & (BIN_SIZE - 1);
				s_pixel_counts[y * BIN_SIZE + x] += int(s_row_frag_counts[by * 4 + y - 1] << 16);
			}
		}
		barrier();
		// Accumulating prefix sums from 4 rows
		if(ystep == 4) {
			if(LIX < BIN_SIZE * 2) {
				uint y = 2 + (LIX >> BIN_SHIFT), x = LIX & (BIN_SIZE - 1);
				s_pixel_counts[y * BIN_SIZE + x] += int(frag_count01 << 16);
			}
		}
		barrier();

		for(int y = 0; y < 4; y += ystep) {
			barrier();
			if(LIX == 0)
				s_sample_count = 0;
			barrier();
			loadRowSamples(by, y, ystep);
			barrier();
			postSortSamples(by, y, ystep);
			//sortBuffer(s_sample_count);
			barrier();
			// TODO: reorder samples to increase coherency during shading

			// Shading samples & storing them ordered by pixel pos
			for(uint i = LIX; i < s_sample_count; i += LSIZE) {
				uvec2 value = s_buffer[i];
				if(value.x == ~0u)
					continue;

				uint pixel_id = value.x >> 24;
				uint bin_tri_idx = value.y >> 16;
				ivec2 bin_pixel_pos = ivec2(pixel_id & (BIN_SIZE - 1), (by * 4) + (pixel_id >> 6));
				s_buffer[i].x = shadeSample(bin_pixel_pos, bin_tri_idx);
			}
			barrier();
			reduceRowSamples(by, y, ystep);
			barrier();
		}
		//rasterFragmentCounts(by);
	}
}

shared int s_num_bins, s_bin_id;

int loadNextBin() {
	if(LIX == 0) {
		uint bin_idx = atomicAdd(g_bins.small_bin_counter, 1);
		s_bin_id = bin_idx < s_num_bins? g_bins.small_bins[bin_idx] : -1;
	}
	barrier();
	return s_bin_id;
}

void main() {
	if(LIX == 0)
		s_num_bins = g_bins.num_small_bins;
	int bin_id = loadNextBin();
	while(bin_id != -1) {
		barrier();
		rasterBin(bin_id);
		bin_id = loadNextBin();
	}
}
