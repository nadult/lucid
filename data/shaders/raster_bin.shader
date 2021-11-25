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
#define MAX_BLOCK_ROW_TRIS		(2 * 1024)
#define MAX_SCRATCH_TRIS		(2 * 1024)
#define SCRATCH_TRI_OFFSET		(MAX_BLOCK_ROW_TRIS * BLOCK_ROW_COUNT)
#define BLOCK_ROW_COUNT			16

#define SAMPLES_PER_THREAD		4
#define MAX_SAMPLES				(LSIZE * SAMPLES_PER_THREAD)

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
shared uint s_buffer[MAX_SAMPLES + 1];
shared float s_fbuffer[MAX_SAMPLES + 1];

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

// TODO: problem: we cannot use tile triangle indices, because there would be too many...
// we have to remap triangles to those which are actually used in currently rasterized tile segment
// We would have to store those indices (32-bit) in scratch probably
//
// TODO: don't store triangles which generate very small number of samples in scratch,
// instead precompute them directly when sampling; We would have to somehow group those triangles together
//
// TODO: store attributes close together (SOA)
void storeTriangle(uint local_tri_idx, vec3 tri0, vec3 tri1, vec3 tri2, uint v0, uint v1, uint v2, uint instance_id)
{
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
		uint tri_idx = i * 2 + (LIX & 1);
		storeTriangle(tri_idx, tri0, tri1, tri2, v0, v1, v2, instance_id);
		generateTriGroups(tri_idx, tri0, tri1, tri2, min_by, max_by);
	}
}

// TODO: optimize
void loadRowSamples(int by, int y, int ystep) {
	ystep--;
	y += int(LIX & ystep);

	uint soffset = gl_WorkGroupID.x * WORKGROUP_SCRATCH_SIZE + by * MAX_BLOCK_ROW_TRIS;
	uint tri_count = s_block_row_tri_counts[by];
	int shift = (y & 1) * 12;
	uint ystep_shift = ystep == 3? 2 : ystep;

	for(uint i = LIX >> ystep_shift, istep = LSIZE >> ystep_shift; i < tri_count; i += istep) {
		uvec2 row = g_scratch[soffset + i];
		int row_range = int((row[y >> 1] >> shift) & 0xfff);
		if(row_range == 0x3f)
			continue;

		uint bin_tri_idx = (row.x >> 24) | ((row.y & 0xff000000) >> 16);
		int minx = row_range & 0x3f, maxx = (row_range >> 6) & 0x3f;
		int num_samples = maxx - minx + 1;
		uint sample_value = (bin_tri_idx << 16) | ((LIX & ystep) << 6) | minx;
		// Note: we're assuming that all samples will fit in s_buffer
		int sample_offset = atomicAdd(s_sample_count, num_samples);
		num_samples = min(num_samples, MAX_SAMPLES - sample_offset);

		// TODO: divide this further
		for(int j = 0; j < num_samples; j++)
			s_buffer[sample_offset++] = sample_value++;
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
		int sample_offset = int(pixel_counter >> 16) - num_samples * 2;
		
		uint enc_color = 0;
		float depth = 1.0 / 0.0;
		for(int i = 0; i < num_samples; i++) {
			float sdepth = s_fbuffer[sample_offset + i];
			if(sdepth < depth) {
				depth = sdepth;
				enc_color = s_buffer[sample_offset + i];
			}
		}

		if(s_sample_count > MAX_SAMPLES)
			enc_color = 0xff0000ff;
		imageStore(final_raster, pixel_pos, uvec4(enc_color, 0, 0, 0));
	}
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
		s_pixel_counts[LIX] = value;
	}
	barrier();
	if(LIX < BIN_SIZE * BLOCK_SIZE / 2) {
		uint y = LIX >> 5, x = LIX & 31;
		s_pixel_counts[y * 64 + x + 32] += s_pixel_counts[y * 64 + 31] & 0xffff0000;
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
void shadeSample(ivec2 bin_pixel_pos, uint local_tri_idx, out uint out_color, out float out_depth)
{
	vec3 ray_dir = s_bin_ray_dir0 + frustum.ws_dirx * bin_pixel_pos.x
								  + frustum.ws_diry * bin_pixel_pos.y;

	vec3 normal, params, edge0, edge1;
	uint instance_id, instance_flags;
	getTriangleParams(local_tri_idx, normal, params, edge0, edge1, instance_id);
	instance_flags = g_instances[instance_id].flags;

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

	uint v0, v1, v2;
	if((instance_flags & (INST_HAS_VERTEX_COLORS | INST_HAS_TEXTURE | INST_HAS_VERTEX_NORMALS)) != 0)
		getTriangleVerts(local_tri_idx, v0, v1, v2);

	// 0.5ms on Sponza!!!
	vec4 color = decodeRGBA8(g_instances[instance_id].color);
	if((instance_flags & INST_HAS_VERTEX_COLORS) != 0) {
		vec4 col0 = decodeRGBA8(g_colors[v0]);
		vec4 col1 = decodeRGBA8(g_colors[v1]);
		vec4 col2 = decodeRGBA8(g_colors[v2]);
		color *= (1.0 - bary[0] - bary[1]) * col0 + bary[0] * col1 + bary[1] * col2;
	}

	// ~2ms on Sponza (mostly loading g_ data)
	if((instance_flags & INST_HAS_TEXTURE) != 0) {
		vec2 tex0 = g_tex_coords[v0];
		vec2 tex1 = g_tex_coords[v1];
		vec2 tex2 = g_tex_coords[v2];
		tex1 -= tex0, tex2 -= tex0;

		vec2 tex_coord = tex0 + bary[0] * tex1 + bary[1] * tex2;
		vec2 tex_dx = bary_dx[0] * tex1 + bary_dx[1] * tex2;
		vec2 tex_dy = bary_dy[0] * tex1 + bary_dy[1] * tex2;

		// 0.8ms to load uv_rect on Sponza
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

	// 0.75 ms on Sponza
	if((instance_flags & INST_HAS_VERTEX_NORMALS) != 0) {
		vec3 nrm0 = decodeNormalUint(g_normals[v0]);
		vec3 nrm1 = decodeNormalUint(g_normals[v1]);
		vec3 nrm2 = decodeNormalUint(g_normals[v2]);
		nrm1 -= nrm0; nrm2 -= nrm0;
		normal = nrm0 + bary[0] * nrm1 + bary[1] * nrm2;
	}

	float light_value = max(0.0, dot(-lighting.sun_dir, normal) * 0.7 + 0.3);
	color.rgb = min(finalShading(color.rgb, light_value), vec3(1.0, 1.0, 1.0));

	out_depth = ray_pos;
	out_color = encodeRGBA8(color);
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
		computeBlockRowPixelCounts(by);
		// iterujemy po liniach?
		barrier();
		if(s_block_row_max_frag_counts[by] > MAX_SAMPLES) {
			rasterInvalidBlockRow(by, vec3(1.0, 0.0, 0.0));
			continue;
		}
		
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

		for(int y = 0; y < 4; y += ystep) {
			barrier();
			if(LIX == 0)
				s_sample_count = 0;
			barrier();
			loadRowSamples(by, y, ystep);
			barrier();
			// Selecting samples
			uint samples[SAMPLES_PER_THREAD];
			for(int i = 0; i < SAMPLES_PER_THREAD; i++) {
				uint sample_idx = LSIZE * i + LIX;
				samples[i] = sample_idx < s_sample_count? s_buffer[sample_idx] : ~0u;
			}
			barrier();
			// Shading samples & storing them ordered by pixel pos
			for(int i = 0; i < SAMPLES_PER_THREAD; i++)
				if(samples[i] != ~0u) {
					uint pixel_id = samples[i] & 0xfff;
					ivec2 bin_pixel_pos = ivec2(pixel_id & (BIN_SIZE - 1), (by * 4 + y) + (pixel_id >> 6));
					uint bin_tri_idx = samples[i] >> 16;

					uint sample_color;
					float sample_depth;
					shadeSample(bin_pixel_pos, bin_tri_idx, sample_color, sample_depth);
					uint pixel_counter = atomicAdd(s_pixel_counts[y * BIN_SIZE + pixel_id], 0x10000);
					uint sample_idx = (pixel_counter >> 16) - (pixel_counter & 0xffff);
					if(sample_idx < MAX_SAMPLES) {
						s_buffer[sample_idx] = sample_color;
						s_fbuffer[sample_idx] = sample_depth;
					}
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
