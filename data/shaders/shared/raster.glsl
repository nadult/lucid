#ifndef _RASTER_GLSL_
#define _RASTER_GLSL_

// TODO: better documentation
// TODO: move all docs to one place at the top of the shader
// TODO: properly handling scenes with too many visible triangles / quads
// TODO: bug when pipeline takes 800ms to finish: when closing app, cannot destroy vk objects which are in use
// TODO: add synthetic test: 256 planes one after another
// TODO: cleanup in the beginning (group definitions)
// NOTE: converting integer multiplications to shifts does not increase perf

#include "compute_funcs.glsl"
#include "funcs.glsl"
#include "scanline.glsl"
#include "shading.glsl"
#include "structures.glsl"
#include "timers.glsl"

// TODO: improve docs
// Block: 8x8 pixels
// Half-block: 8x4 pixels
// Render-block: depends on warp size: 32: half-block, 64: block

#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_shuffle_relative : require

#ifndef LSIZE
#error "LSIZE should be defined before including raster.glsl"
#endif

#define NUM_WARPS (LSIZE / WARP_SIZE)
#define NUM_WARPS_MASK (NUM_WARPS - 1)
#define NUM_WARPS_SHIFT (LSHIFT - WARP_SHIFT)

#define BIN_MASK (BIN_SIZE - 1)

#define BLOCK_SIZE 8
#define BLOCK_SHIFT 3

#define BLOCK_ROWS (BIN_SIZE / BLOCK_SIZE)
#define BLOCK_ROWS_SHIFT (BIN_SHIFT - BLOCK_SHIFT)
#define BLOCK_ROWS_MASK (BLOCK_ROWS - 1)

#define RBLOCK_WIDTH 8
#define RBLOCK_WIDTH_SHIFT 3

#if WARP_SIZE == 32
#define RBLOCK_HEIGHT 4
#define RBLOCK_HEIGHT_SHIFT 2
#define RBLOCK_COUNT (NUM_WARPS * 2)
#elif WARP_SIZE == 64
#define RBLOCK_HEIGHT 8
#define RBLOCK_HEIGHT_SHIFT 3
#define RBLOCK_COUNT NUM_WARPS
#else
#error "Currently only 32 & 64 warp size is supported"
#endif

#define RBLOCK_ROWS (BIN_SIZE / RBLOCK_HEIGHT)
#define RBLOCK_ROWS_SHIFT (BIN_SHIFT - RBLOCK_HEIGHT_SHIFT)
#define RBLOCK_ROWS_MASK (RBLOCK_ROWS - 1)

#define RBLOCK_COLS (BIN_SIZE / RBLOCK_WIDTH)
#define RBLOCK_COLS_SHIFT (BIN_SHIFT - RBLOCK_WIDTH_SHIFT)
#define RBLOCK_COLS_MASK (RBLOCK_COLS - 1)

#define SEGMENT_SIZE 256
#define SEGMENT_SHIFT 8

#define BASE_BUFFER_SIZE (LSIZE * 8)
#define FULL_BUFFER_SIZE (BASE_BUFFER_SIZE + LSIZE)

shared uint s_buffer[FULL_BUFFER_SIZE + 1];
#if WARP_SIZE == 64
shared uint s_mini_buffer[LSIZE * 2];
#endif

uvec3 rasterBinStep(inout ScanlineParams scan) {
#define SCAN_STEP(id)                                                                              \
	int min##id = int(max(max(scan.min[0], scan.min[1]), max(scan.min[2], 0.0)));                  \
	int max##id = int(min(min(scan.max[0], scan.max[1]), min(scan.max[2], BIN_SIZE))) - 1;         \
	if(min##id > max##id)                                                                          \
		min##id = BIN_MASK, max##id = 0;                                                           \
	scan.min += scan.step, scan.max += scan.step;

#define BX_MASK_ROW(id)                                                                            \
	((((1 << RBLOCK_COLS) - 1) << (min##id >> RBLOCK_WIDTH_SHIFT)) &                               \
	 (((1 << RBLOCK_COLS) - 1) >> (RBLOCK_COLS_MASK - (max##id >> RBLOCK_WIDTH_SHIFT))))
	SCAN_STEP(0);
	SCAN_STEP(1);
	SCAN_STEP(2);
	SCAN_STEP(3);

	uint bx_mask = BX_MASK_ROW(0) | BX_MASK_ROW(1) | BX_MASK_ROW(2) | BX_MASK_ROW(3);
	uint min_bits = (min0 << 0) | (min1 << 6) | (min2 << 12) | (min3 << 18);
	uint max_bits = (max0 << 0) | (max1 << 6) | (max2 << 12) | (max3 << 18);

#undef BX_MASK_ROW
#undef SCAN_STEP

	return uvec3(min_bits, max_bits, bx_mask);
}

uint rasterBlock(uint tri_mins, uint tri_maxs, int startx, out uint num_frags, inout vec2 cpos) {
	ivec4 xmin = max(((ivec4(tri_mins) >> ivec4(0, 6, 12, 18)) & BIN_MASK) - startx, 0);
	ivec4 xmax = min(((ivec4(tri_maxs) >> ivec4(0, 6, 12, 18)) & BIN_MASK) - startx, 7);
	ivec4 count = max(xmax - xmin + 1, 0);
	vec4 cpx = vec4(xmin * 2 + count) * count;
	vec4 cpy = vec4(1.0, 3.0, 5.0, 7.0) * count;
	cpos += vec2(cpx[0] + cpx[1] + cpx[2] + cpx[3], cpy[0] + cpy[1] + cpy[2] + cpy[3]);
	num_frags = count[0] + count[1] + count[2] + count[3];
	uint min_bits =
		(xmin[0] & 7) | ((xmin[1] & 7) << 3) | ((xmin[2] & 7) << 6) | ((xmin[3] & 7) << 9);
	uint count_bits = (count[0] << 12) | (count[1] << 16) | (count[2] << 20) | (count[3] << 24);
	return min_bits | count_bits;
}

uint rasterBlockDepth(vec2 cpos, uint tri_idx) {
	uint depth_offset = STORAGE_TRI_DEPTH_OFFSET + tri_idx;
	vec3 depth_eq = uintBitsToFloat(g_uvec4_storage[depth_offset].xyz);
	float ray_pos = depth_eq.x * cpos.x + (depth_eq.y * cpos.y + depth_eq.z);
	float depth = 0xffffe * SATURATE(inversesqrt(ray_pos + 1)); // 20 bits
	return uint(depth);
}

ivec2 rasterBlockPos(uint rbid) {
	uint rbx = rbid & RBLOCK_COLS_MASK;
	uint rby = rbid >> RBLOCK_COLS_SHIFT;
	return ivec2(rbx, rby);
}

uint blockIdFromRaster(uint rbid) {
#if RBLOCK_HEIGHT == BLOCK_HEIGHT
	return rbid;
#else
	ivec2 pos = rasterBlockPos(rbid);
	return pos.x + ((pos.y >> 1) << RBLOCK_COLS_SHIFT);
#endif
}

uint blockIdToRaster(uint bid) {
#if RBLOCK_HEIGHT == BLOCK_HEIGHT
	return bid;
#else
	uint bx = bid & RBLOCK_COLS_MASK;
	uint by = bid >> RBLOCK_COLS_SHIFT;
	return bx + ((by * 2) << RBLOCK_COLS_SHIFT);
#endif
}

uvec2 rasterBlockShift() { return uvec2(RBLOCK_WIDTH_SHIFT, RBLOCK_HEIGHT_SHIFT); }

ivec2 rasterBlockPixelPos(uint rbid) {
	ivec2 pix_pos =
		ivec2(LIX & (RBLOCK_WIDTH - 1), (LIX >> RBLOCK_WIDTH_SHIFT) & (RBLOCK_HEIGHT - 1));
	return (rasterBlockPos(rbid) << rasterBlockShift()) + pix_pos;
}

uint swap(uint x, int mask, bool dir) {
	uint y = subgroupShuffleXor(x, mask);
	return (x < y) == dir ? y : x;
}
bool bitExtract(uint value, int boffset) { return ((value >> boffset) & 1) != 0; }

void sortBuffer(uint lrbid, uint count, uint rcount, uint buf_offset, uint group_size, uint lid,
				bool with_barriers) {
	for(uint i = lid + count; i < rcount; i += group_size)
		s_buffer[buf_offset + i] = 0xffffffff;

	bool bit0 = ((lid >> 0) & 1) != 0, bit1 = ((lid >> 1) & 1) != 0;
	bool bit2 = ((lid >> 2) & 1) != 0, bit3 = ((lid >> 3) & 1) != 0;
	bool bit4 = ((lid >> 4) & 1) != 0, bit5 = ((lid >> 5) & 1) != 0;

	for(uint i = lid; i < rcount; i += group_size) {
		uint value = s_buffer[buf_offset + i];
		// TODO: register sort could be faster?
		value = swap(value, 0x01, bit1 != bit0); // K = 2
		value = swap(value, 0x02, bit2 != bit1); // K = 4
		value = swap(value, 0x01, bit2 != bit0);
		value = swap(value, 0x04, bit3 != bit2); // K = 8
		value = swap(value, 0x02, bit3 != bit1);
		value = swap(value, 0x01, bit3 != bit0);
		value = swap(value, 0x08, bit4 != bit3); // K = 16
		value = swap(value, 0x04, bit4 != bit2);
		value = swap(value, 0x02, bit4 != bit1);
		value = swap(value, 0x01, bit4 != bit0);
		if(group_size >= 64) {
			value = swap(value, 0x10, bit5 != bit4); // K = 32
			value = swap(value, 0x08, bit5 != bit3);
			value = swap(value, 0x04, bit5 != bit2);
			value = swap(value, 0x02, bit5 != bit1);
			value = swap(value, 0x01, bit5 != bit0);
		}
		s_buffer[buf_offset + i] = value;
	}
	int start_k = group_size >= 64 ? 64 : 32, end_j = 32;
	if(with_barriers)
		barrier();

	for(uint k = start_k; k <= rcount; k = 2 * k) {
		for(uint j = k >> 1; j >= end_j; j = j >> 1) {
			for(uint i = lid; i < rcount; i += group_size * 2) {
				uint idx = (i & j) != 0 ? i + group_size - j : i;
				uint lvalue = s_buffer[buf_offset + idx];
				uint rvalue = s_buffer[buf_offset + idx + j];
				if(((idx & k) != 0) == (lvalue.x < rvalue.x)) {
					s_buffer[buf_offset + idx] = rvalue;
					s_buffer[buf_offset + idx + j] = lvalue;
				}
			}
			if(with_barriers)
				barrier();
		}
		for(uint i = lid; i < rcount; i += group_size) {
			bool bit = (i & k) != 0;
			uint value = s_buffer[buf_offset + i];
			value = swap(value, 0x10, bit != bitExtract(lid, 4));
			value = swap(value, 0x08, bit != bitExtract(lid, 3));
			value = swap(value, 0x04, bit != bitExtract(lid, 2));
			value = swap(value, 0x02, bit != bitExtract(lid, 1));
			value = swap(value, 0x01, bit != bitExtract(lid, 0));
			s_buffer[buf_offset + i] = value;
		}
		if(with_barriers)
			barrier();
	}
}

uint blockRowsToBits(uint rows) {
	uvec4 countx = (uvec4(rows) >> uvec4(12, 16, 20, 24)) & 15;
	uvec4 minx = (uvec4(rows) >> uvec4(0, 3, 6, 9)) & 7;
	uint bits0 = uint((1 << countx[0]) - 1) << (minx[0] + 0);
	uint bits1 = uint((1 << countx[1]) - 1) << (minx[1] + 8);
	uint bits2 = uint((1 << countx[2]) - 1) << (minx[2] + 16);
	uint bits3 = uint((1 << countx[3]) - 1) << (minx[3] + 24);
	return bits0 | bits1 | bits2 | bits3;
}

bool highTriDensity(uint tri_count, uint frag_count) {
	return frag_count < (tri_count << 3) && tri_count >= 16;
}

void loadSamples(uint lrbid, inout uint cur_tri_idx, int segment_id, uint rblock_counts,
				 uint src_offset) {
	uint buf_offset = (LIX >> WARP_SHIFT) * (SEGMENT_SIZE + WARP_SIZE);
	{
		// Copying samples generated in previous round which may overlap into current segment
		uint prev_offset = buf_offset + gl_SubgroupInvocationID;
		s_buffer[prev_offset] = s_buffer[prev_offset + SEGMENT_SIZE];
	}
	subgroupMemoryBarrierShared();

	uint tri_count = rblock_counts & 0xffff, frag_count = rblock_counts >> 16;
	int segment_bits = segment_id & 15;
	uint tri_offset;

	// TODO: compute highTriDensity outside
	if(highTriDensity(tri_count, frag_count)) {
		uint i = cur_tri_idx == 0 ? LIX & WARP_MASK : cur_tri_idx;
		for(; i < tri_count; i += WARP_SIZE) {
			uvec2 tri_data = g_scratch_64[src_offset + i];
			uint tri_segment_bits = tri_data.y >> 28;
			if(tri_segment_bits != segment_bits)
				break;

			tri_offset = tri_data.x & 0xff;
			uint tri_idx = tri_data.x & 0xffffff00;
			uint bits = blockRowsToBits(tri_data.y);
			while(bits != 0) {
				uint pixel_id = findLSB(bits);
				bits &= ~(1u << pixel_id);
				s_buffer[buf_offset + tri_offset] = pixel_id | tri_idx;
				tri_offset++;
			}
		}
		cur_tri_idx = i;
	} else {
		uint y = LIX & (RBLOCK_HEIGHT - 1);
		uint minx_shift = (y & 3) * 3, countx_shift = 12 + (y & 3) * 4;
		int mask1 = y >= 1 ? ~0 : 0, mask2 = y >= 2 ? ~0 : 0, mask3 = y >= 4 ? ~0 : 0;
		uint i = cur_tri_idx == 0 ? (LIX & WARP_MASK) >> RBLOCK_HEIGHT_SHIFT : cur_tri_idx;
		for(; i < tri_count; i += WARP_SIZE / RBLOCK_HEIGHT) {
#if RBLOCK_HEIGHT == 8
			uint tri_data = g_scratch_64[src_offset + i][y >> 2];
			uint tri_info = g_scratch_32[src_offset + i];
#else
			uvec2 tri_block_data = g_scratch_64[src_offset + i];
			uint tri_data = tri_block_data.y, tri_info = tri_block_data.x;
#endif

			uint tri_segment_bits = tri_data >> 28;
			if(tri_segment_bits != segment_bits)
				break;

			tri_offset = tri_info & 0xff;
			uint tri_idx = tri_info & 0xffffff00;
			int minx = int((tri_data >> minx_shift) & 7);
			int countx = int((tri_data >> countx_shift) & 15);

			int prevx = countx + (subgroupShuffleUp(countx, 1) & mask1);
			prevx += (subgroupShuffleUp(prevx, 2) & mask2);
#if RBLOCK_HEIGHT == 8
			prevx += (subgroupShuffleUp(prevx, 4) & mask3);
#endif

			tri_offset += prevx - countx;
			uint pixel_id = (y << 3) | minx;
			uint value = pixel_id | tri_idx;
			for(int j = 0; j < countx; j++)
				s_buffer[buf_offset + tri_offset++] = value++;
		}

		cur_tri_idx = i;
	}

	subgroupMemoryBarrierShared();
}

void shadeAndReduceSamples(uint rbid, uint sample_count, in out ReductionContext ctx) {
	uint buf_offset = (LIX >> WARP_SHIFT) * (SEGMENT_SIZE + WARP_SIZE);
	uint mini_offset =
		WARP_SIZE == 64 ? (LIX & ~WARP_MASK) + ((LIX & 32) != 0 ? LSIZE : 0) : LIX & ~WARP_MASK;
	ivec2 rblock_pos = (rasterBlockPos(rbid) << rasterBlockShift()) + s_bin_pos;
	vec3 out_color = ctx.out_color;

	for(uint i = 0; i < sample_count; i += WARP_SIZE) {
		// TODO: load two values at once in 64-bit mode, mini_buffer won't be needed!
		uint value = s_buffer[buf_offset + gl_SubgroupInvocationID];
#if WARP_SIZE == 32
		s_buffer[buf_offset + gl_SubgroupInvocationID] = 0;
		subgroupMemoryBarrierShared();
#else
		s_mini_buffer[LIX] = 0;
		if(WARP_SIZE == 64)
			s_mini_buffer[LSIZE + LIX] = 0;
#endif
		uvec2 sample_s;

		uint sample_id = i + gl_SubgroupInvocationID;
		if(sample_id < sample_count) {
			uint sample_pixel_id = value & WARP_MASK;
			uint tri_idx = value >> 8;
			ivec2 pix_pos = rblock_pos + ivec2(sample_pixel_id & 7, sample_pixel_id >> 3);
			float sample_depth;
			uint sample_color = shadeSample(pix_pos, tri_idx, sample_depth);
			sample_s = uvec2(sample_color, floatBitsToUint(sample_depth));
#if WARP_SIZE == 32
			atomicOr(s_buffer[buf_offset + sample_pixel_id], gl_SubgroupEqMask.x);
#else
			const uint pixel_bit = gl_SubgroupEqMask.x | gl_SubgroupEqMask.y;
			atomicOr(s_mini_buffer[mini_offset + sample_pixel_id], pixel_bit);
#endif
		}
		subgroupMemoryBarrierShared();

#if WARP_SIZE == 32
		uint pixel_bitmask = s_buffer[buf_offset + gl_SubgroupInvocationID];
#else
		uvec2 pixel_bitmask = uvec2(s_mini_buffer[LIX], s_mini_buffer[LIX + LSIZE]);
#endif
		if(reduceSample(ctx, out_color, sample_s, pixel_bitmask))
			break;
		buf_offset += WARP_SIZE;
	}

	// TODO: check if encode+decode for out_color is really needed (to save 2 regs)
	ctx.out_color = SATURATE(out_color);
}

// ------------------------------------------------------------------------------------------------
// -------------------- Helper visualization functions --------------------------------------------

// Only used when debugging
shared uint s_vis_pixels[LSIZE];

void initVisualizeSamples() { s_vis_pixels[LIX] = 0; }

void visualizeSamples(uint sample_count) {
	uint buf_offset = (LIX >> WARP_SHIFT) * SEGMENT_SIZE;
	for(uint i = LIX & WARP_MASK; i < sample_count; i += WARP_SIZE) {
		uint pixel_id = s_buffer[buf_offset + i] & WARP_MASK;
		atomicAdd(s_vis_pixels[(LIX & ~WARP_MASK) + pixel_id], 1);
	}
}

void finishVisualizeSamples(ivec2 pixel_pos) {
	uint pixel_id = (pixel_pos.x & (RBLOCK_WIDTH - 1)) +
					((pixel_pos.y & (RBLOCK_HEIGHT - 1)) << RBLOCK_WIDTH_SHIFT);
	uint value = s_vis_pixels[(LIX & ~WARP_MASK) + pixel_id];
	vec3 color = gradientColor(value, uvec4(16, 64, 256, 1024));
	outputPixel(pixel_pos, vec4(SATURATE(color), 1.0));
}

#endif
