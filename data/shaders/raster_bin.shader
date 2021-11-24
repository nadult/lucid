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
layout(std430, binding = 7) buffer buf7_ { TileCounters g_tiles; };

layout(std430, binding = 8) buffer buf8_ { uint g_bin_quads[]; };
layout(std430, binding = 9) coherent buffer buf9_ { uvec2 g_scratch[]; };

layout(std430, binding = 10) readonly buffer buf10_ { InstanceData g_instances[]; };
layout(std430, binding = 11) readonly buffer buf11_ { vec4 g_uv_rects[]; };

layout(binding = 0) uniform sampler2D opaque_texture;
layout(binding = 1) uniform sampler2D transparent_texture;

#define WORKGROUP_SCRATCH_SIZE	(64 * 1024)
#define MAX_BLOCK_ROW_TRIS		(4 * 1024)
#define MAX_SCRATCH_TRIS		(2 * 1024)
#define SCRATCH_TRI_OFFSET		(MAX_BLOCK_ROW_TRIS * 4)
#define BLOCK_ROW_COUNT			16

#define SAMPLES_PER_THREAD		4
#define MAX_SAMPLES				(LSIZE * SAMPLES_PER_THREAD)

shared ivec2 s_bin_pos;

shared uint s_block_row_tri_counts[BLOCK_ROW_COUNT];
shared uint s_block_row_frag_counts[BLOCK_ROW_COUNT];

shared uint s_bin_quad_count, s_bin_quad_offset;

shared int s_pixel_counts[BIN_SIZE * BLOCK_SIZE];
shared int s_sample_count;


// Ile wątków najlepiej? mamy 4K pikseli; max 64 linie na trójkąt;
// 12 bitów na linię: 48 bitów na 4 linie?
//
// Max 4K instancji dla 256 trójkątów?
// Rysujemy po 16 linii na raz?
//
// najpierw porządkujemy trojkaty po ilosci rzedow? potem
// 

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

	// TODO: min/max by
	uint soffset = gl_WorkGroupID.x * WORKGROUP_SCRATCH_SIZE + min_by * MAX_BLOCK_ROW_TRIS;
	for(int by = min_by; by <= max_by; by++) {
		uint row_ranges[2] = {0, 0};
		int frag_counts = 0; // TODO: uint?

		for(int y = 0; y < 4; y++) {
			float xmin = max(max(scan_min[0], scan_min[1]), max(scan_min[2], 0.0));
			float xmax = min(min(scan_max[0], scan_max[1]), min(scan_max[2], BIN_SIZE));

			scan_min += scan_step;
			scan_max += scan_step;
			
			// TODO: use floor/ceil?
			int imin = int(xmin), imax = int(xmax) - 1;
			frag_counts += max(0, imax - imin + 1);
			row_ranges[y >> 1] |= (imin <= imax? (uint(imin) | (uint(imax) << 6)) : 0x3f) << ((y & 1) * 12);
		}

		// 0x1c71c7 is equal to 4 empty rows (each row encoded in 2 * 3 bits)
		if(frag_counts > 0) {
			uint roffset = atomicAdd(s_block_row_tri_counts[by], 1);
			g_scratch[soffset + roffset] = uvec2(row_ranges[0] | (tri_idx << 24),
												 row_ranges[1] | ((tri_idx & 0xff00) << 16));
			atomicAdd(s_block_row_frag_counts[by], frag_counts);
		}
		soffset += MAX_BLOCK_ROW_TRIS;
	}
}

// TODO: problem: we cannot use tile triangle indices, because there would be too many...
// we have to remap triangles to those which are actually used in currently rasterized tile segment
// We would have to store those indices (32-bit) in scratch probably
//
// TODO: don't store triangles which generate very small number of samples in scratch,
// instead precompute them directly when sampling; We would have to somehow group those triangles together
void storeTriangle(uint local_tri_idx, vec3 tri0, vec3 tri1, vec3 tri2, uint v0, uint v1, uint v2, uint instance_id)
{
	if(local_tri_idx >= MAX_SCRATCH_TRIS)
		return;
	uint toffset = gl_WorkGroupID.x * WORKGROUP_SCRATCH_SIZE + SCRATCH_TRI_OFFSET + local_tri_idx * 8;

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

	g_scratch[toffset + 0] = uvec2(floatBitsToUint(normal.x), floatBitsToUint(normal.y));
	g_scratch[toffset + 1] = uvec2(floatBitsToUint(normal.z), floatBitsToUint(plane_dist));
	g_scratch[toffset + 2] = uvec2(floatBitsToUint(param0), floatBitsToUint(param1));
	g_scratch[toffset + 3] = uvec2(floatBitsToUint(edge0.x), floatBitsToUint(edge0.y));
	g_scratch[toffset + 4] = uvec2(floatBitsToUint(edge0.z), floatBitsToUint(edge1.x));
	g_scratch[toffset + 5] = uvec2(floatBitsToUint(edge1.y), floatBitsToUint(edge1.z));
	g_scratch[toffset + 6] = uvec2(v0, v1);
	g_scratch[toffset + 7] = uvec2(v2, instance_id);
}

void getTriangleParams(uint local_tri_idx, out vec3 normal, out vec3 params, out vec3 edge0, out vec3 edge1, out uint instance_id) {
	uint toffset = gl_WorkGroupID.x * WORKGROUP_SCRATCH_SIZE + SCRATCH_TRI_OFFSET + local_tri_idx * 8;

	{
		uvec2 val0 = g_scratch[toffset + 0], val1 = g_scratch[toffset + 1], val2 = g_scratch[toffset + 2];
		normal = vec3(uintBitsToFloat(val0[0]), uintBitsToFloat(val0[1]), uintBitsToFloat(val1[0]));
		params = vec3(uintBitsToFloat(val1[1]), uintBitsToFloat(val2[0]), uintBitsToFloat(val2[1]));
	}
	{
		uvec2 val0 = g_scratch[toffset + 3], val1 = g_scratch[toffset + 4], val2 = g_scratch[toffset + 5];
		edge0 = vec3(uintBitsToFloat(val0[0]), uintBitsToFloat(val0[1]), uintBitsToFloat(val1[0]));
		edge1 = vec3(uintBitsToFloat(val1[1]), uintBitsToFloat(val2[0]), uintBitsToFloat(val2[1]));
	}

	instance_id = g_scratch[toffset + 7].y;
}

void getTriangleVerts(uint local_tri_idx, out uint v0, out uint v1, out uint v2) {
	uint toffset = gl_WorkGroupID.x * WORKGROUP_SCRATCH_SIZE + SCRATCH_TRI_OFFSET + local_tri_idx * 8;
	uvec2 val0 = g_scratch[toffset + 6];
	v0 = val0[0], v1 = val0[1], v2 = g_scratch[toffset + 7].x;
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
		//storeTriangle(tile_tri_idx, tri0, tri1, tri2, v0, v1, v2, instance_id);
		generateTriGroups(LIX, tri0, tri1, tri2, min_by, max_by);
	}
}

// TODO: optimize
void loadBlockRowSamples(uint by) {
	uint soffset = gl_WorkGroupID.x * WORKGROUP_SCRATCH_SIZE + by * MAX_BLOCK_ROW_TRIS;
	uint tri_count = s_block_row_tri_counts[by];
	uint y = LIX & 3, shift = (y & 1) * 12;

	for(uint i = LIX >> 2; i < tri_count; i += LSIZE / 4) {
		uvec2 row = g_scratch[soffset + i];
		int row_range = int(((y > 1? row.y : row.x) >> shift) & 0xfff);
		if(row_range == 0x3f)
			continue;

		uint bin_tri_idx = (row.x >> 24) | ((row.y & 0xff000000) >> 16);
		int minx = row_range & 0x3f, maxx = (row_range >> 6) & 0x3f;
		int num_samples = maxx - minx + 1;
		uint sample_value = (y * 64 + minx) | (bin_tri_idx << 16);
		// Note: we're assuming that all samples will fit in s_buffer
		int sample_offset = atomicAdd(s_sample_count, num_samples);

		atomicAdd(s_pixel_counts[y * 64 + minx], 1);
		if(maxx < 63)
			atomicAdd(s_pixel_counts[y * 64 + maxx + 1], -1);
		//for(int j = 0; j < num_samples; j++)
			//s_buffer[sample_offset++] = sample_value++;
	}
}

void computePixelOffsets()
{
#ifdef VENDOR_NVIDIA
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
#else
#error write me please
#endif
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
		uint count = s_pixel_counts[i];
		vec4 color = vec4(vec3(float(count) / 64.0), 1.0);
		uint enc_col = encodeRGBA8(color);
		imageStore(final_raster, s_bin_pos + pixel_pos, uvec4(enc_col, 0, 0, 0));
	}
}

void rasterBin(int bin_id) {
	if(LIX < BLOCK_ROW_COUNT) {
		if(LIX == 0) {
			s_bin_pos = ivec2(bin_id % BIN_COUNT_X, bin_id / BIN_COUNT_X) * BIN_SIZE;
			s_bin_quad_count = g_bins.bin_quad_counts[bin_id];
			s_bin_quad_offset = g_bins.bin_quad_offsets[bin_id];
		}

		s_block_row_tri_counts[LIX] = 0;
		s_block_row_frag_counts[LIX] = 0;
	}
	barrier();
	generateRows();
	for(int by = 0; by < BLOCK_ROW_COUNT; by++) {
		barrier();
		if(s_block_row_frag_counts[by] > MAX_SAMPLES * 4) {
			rasterInvalidBlockRow(by, vec3(1.0, 0.0, 0.0));
			continue;
		}

		if(LIX == 0)
			s_sample_count = 0;
		for(uint i = LIX; i < BIN_SIZE * BLOCK_SIZE; i += LSIZE)
			s_pixel_counts[LIX] = 0;
		barrier();
		loadBlockRowSamples(by);
		barrier();
		computePixelOffsets();
		barrier();
		rasterFragmentCounts(by);
	}
}

shared int s_num_bins, s_bin_id;

int loadNextBin() {
	if(LIX == 0) {
		uint bin_idx = atomicAdd(g_tiles.small_bin_counter, 1);
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
