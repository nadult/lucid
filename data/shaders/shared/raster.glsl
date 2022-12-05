#ifndef _RASTER_GLSL_
#define _RASTER_GLSL_

// TODO: better documentation
// TODO: move all docs to one place at the top of the shader
// TODO: properly handling scenes with too many visible triangles / quads
// TODO: bug when pipeline takes 800ms to finish: when closing app, cannot destroy vk objects which are in use
// TODO: add synthetic test: 256 planes one after another
// TODO: cleanup in the beginning (group definitions)
// NOTE: converting integer multiplications to shifts does not increase perf

// Variable naming:
//   bid:      block (8x8) id; In 32x32 bin we have 4 x 4 = 16 blocks
//  hbid: half-block (8x4) id; In 32x32 bin we have 4 x 8 = 32 half-blocks
//  lbid: local block id (range: 0 up to num_subgroups - 1)

#include "compute_funcs.glsl"
#include "funcs.glsl"
#include "scanline.glsl"
#include "shading.glsl"
#include "structures.glsl"
#include "timers.glsl"

#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_shuffle_relative : require

#ifndef LSIZE
#error "LSIZE should be defined before including raster.glsl"
#endif

#define BIN_MASK (BIN_SIZE - 1)

#define BLOCK_SIZE 8
#define BLOCK_SHIFT 3

#define BLOCK_ROWS (BIN_SIZE / BLOCK_SIZE)
#define BLOCK_ROWS_SHIFT (BIN_SHIFT - BLOCK_SHIFT)
#define BLOCK_ROWS_MASK (BLOCK_ROWS - 1)

#define HBLOCK_WIDTH 8
#define HBLOCK_HEIGHT 4

#define HBLOCK_WIDTH_SHIFT 3
#define HBLOCK_HEIGHT_SHIFT 2

#if SUBGROUP_SIZE != 32 && SUBGROUP_SIZE != 64
#error "Currently only 32 & 64 subgroup size is supported"
#endif

#define HBLOCK_ROWS (BIN_SIZE / HBLOCK_HEIGHT)
#define HBLOCK_ROWS_SHIFT (BIN_SHIFT - HBLOCK_HEIGHT_SHIFT)
#define HBLOCK_ROWS_MASK (HBLOCK_ROWS - 1)

#define HBLOCK_COLS (BIN_SIZE / HBLOCK_WIDTH)
#define HBLOCK_COLS_SHIFT (BIN_SHIFT - HBLOCK_WIDTH_SHIFT)
#define HBLOCK_COLS_MASK (HBLOCK_COLS - 1)

#define NUM_BLOCKS (BLOCK_ROWS * BLOCK_ROWS)
#define NUM_HBLOCKS (HBLOCK_COLS * HBLOCK_ROWS)

// half-group is a 32-thread subgroup; It has the same number of threads
// needed to process all pixels within 8x4 half-block
#define HALFGROUP_MASK 31
#define HALFGROUP_SIZE 32
#define HALFGROUP_SHIFT 5
#define NUM_HALFGROUPS (LSIZE / HALFGROUP_SIZE)

#define SEGMENT_SIZE 256
#define SEGMENT_SHIFT 8

#define BASE_BUFFER_SIZE (LSIZE * 8)
#define FULL_BUFFER_SIZE (BASE_BUFFER_SIZE + LSIZE)

shared int s_num_bins, s_bin_id;
shared uint s_bin_quad_count, s_bin_quad_offset;
shared uint s_bin_tri_count, s_bin_tri_offset;
shared int s_raster_error;

void initBinLoader(uint level) {
	if(LIX == 0)
		s_num_bins = g_info.bin_level_counts[level];
}

bool loadNextBin(uint level) {
	if(LIX == 0) {
		uint level_idx;
		if(level == BIN_LEVEL_LOW)
			level_idx = atomicAdd(g_info.a_small_bins, 1);
		else
			level_idx = atomicAdd(g_info.a_high_bins, 1);

		if(level_idx < s_num_bins) {
			int bin_id =
				level == BIN_LEVEL_LOW ? LOW_LEVEL_BINS(level_idx) : HIGH_LEVEL_BINS(level_idx);
			s_bin_id = bin_id;
			int pos_y = bin_id / BIN_COUNT_X, pos_x = bin_id - pos_y * BIN_COUNT_X;
			s_bin_pos = ivec2(pos_x, pos_y) * BIN_SIZE;
			s_bin_quad_count = BIN_QUAD_COUNTS(bin_id);
			s_bin_quad_offset = BIN_QUAD_OFFSETS(bin_id);
			s_bin_tri_count = BIN_TRI_COUNTS(bin_id);
			s_bin_tri_offset = BIN_TRI_OFFSETS(bin_id);
			s_raster_error = 0;
		} else {
			s_bin_id = -1;
		}
	}
	barrier();
	return s_bin_id != -1;
}

shared uint s_buffer[FULL_BUFFER_SIZE + 1];

uvec3 rasterBinStep(inout ScanlineParams scan) {
#define SCAN_STEP(id)                                                                              \
	int min##id = int(max(max(scan.min[0], scan.min[1]), max(scan.min[2], 0.0)));                  \
	int max##id = int(min(min(scan.max[0], scan.max[1]), min(scan.max[2], BIN_SIZE))) - 1;         \
	if(min##id > max##id)                                                                          \
		min##id = BIN_MASK, max##id = 0;                                                           \
	scan.min += scan.step, scan.max += scan.step;

#define BX_MASK_ROW(id)                                                                            \
	((((1 << HBLOCK_COLS) - 1) << (min##id >> HBLOCK_WIDTH_SHIFT)) &                               \
	 (((1 << HBLOCK_COLS) - 1) >> (HBLOCK_COLS_MASK - (max##id >> HBLOCK_WIDTH_SHIFT))))
	SCAN_STEP(0);
	SCAN_STEP(1);
	SCAN_STEP(2);
	SCAN_STEP(3);

	uint bx_mask = BX_MASK_ROW(0) | BX_MASK_ROW(1) | BX_MASK_ROW(2) | BX_MASK_ROW(3);
	uint min_bits = (min0 << 0) | (min1 << 5) | (min2 << 10) | (min3 << 15);
	uint max_bits = (max0 << 0) | (max1 << 5) | (max2 << 10) | (max3 << 15);

#undef BX_MASK_ROW
#undef SCAN_STEP

	return uvec3(min_bits, max_bits, bx_mask);
}

vec2 rasterHalfBlockCentroid(uint tri_mins, uint tri_maxs, int startx, out uint num_frags) {
	ivec4 xmin = max(((ivec4(tri_mins) >> ivec4(0, 5, 10, 15)) & BIN_MASK) - startx, 0);
	ivec4 xmax = min(((ivec4(tri_maxs) >> ivec4(0, 5, 10, 15)) & BIN_MASK) - startx, 7);
	ivec4 count = max(xmax - xmin + 1, 0);
	vec4 cpx = vec4(xmin * 2 + count) * count;
	vec4 cpy = vec4(1.0, 3.0, 5.0, 7.0) * count;
	num_frags = count[0] + count[1] + count[2] + count[3];
	return vec2(cpx[0] + cpx[1] + cpx[2] + cpx[3], cpy[0] + cpy[1] + cpy[2] + cpy[3]);
}

uint rasterHalfBlockBits(uint tri_mins, uint tri_maxs, int startx, out uint num_frags) {
	ivec4 xmin = max(((ivec4(tri_mins) >> ivec4(0, 5, 10, 15)) & BIN_MASK) - startx, 0);
	ivec4 xmax = min(((ivec4(tri_maxs) >> ivec4(0, 5, 10, 15)) & BIN_MASK) - startx, 7);
	ivec4 count = max(xmax - xmin + 1, 0);
	num_frags = count[0] + count[1] + count[2] + count[3];
	uint min_bits = ((xmin[0] << 0) | (xmin[1] << 7) | (xmin[2] << 14) | (xmin[3] << 21)) &
					(7 | (7 << 7) | (7 << 14) | (7 << 21));
	uint count_bits = (count[0] << 3) | (count[1] << 10) | (count[2] << 17) | (count[3] << 24);
	return min_bits | count_bits;
}

uint rasterHalfBlockNumFrags(uint tri_mins, uint tri_maxs, int startx) {
	ivec4 xmin = max(((ivec4(tri_mins) >> ivec4(0, 5, 10, 15)) & BIN_MASK) - startx, 0);
	ivec4 xmax = min(((ivec4(tri_maxs) >> ivec4(0, 5, 10, 15)) & BIN_MASK) - startx, 7);
	ivec4 count = max(xmax - xmin + 1, 0);
	return count[0] + count[1] + count[2] + count[3];
}

uint rasterBlockDepth(vec2 cpos, uint tri_idx, float depth_range) {
	uint depth_offset = STORAGE_TRI_DEPTH_OFFSET + tri_idx;
	vec3 depth_eq = uintBitsToFloat(g_uvec4_storage[depth_offset].xyz);
	float ray_pos = depth_eq.x * cpos.x + (depth_eq.y * cpos.y + depth_eq.z);
	float depth = depth_range * SATURATE(inversesqrt(ray_pos + 1));
	return uint(depth);
}

ivec2 halfBlockPos(uint hbid) {
	uint rbx = hbid & HBLOCK_COLS_MASK;
	uint rby = hbid >> HBLOCK_COLS_SHIFT;
	return ivec2(rbx, rby);
}

uint fullBlockId(uint hbid) {
	ivec2 pos = halfBlockPos(hbid);
	return pos.x + ((pos.y >> 1) << HBLOCK_COLS_SHIFT);
}

uint halfBlockId(uint bid) {
	uint bx = bid & HBLOCK_COLS_MASK;
	uint by = bid >> HBLOCK_COLS_SHIFT;
	return bx + ((by * 2) << HBLOCK_COLS_SHIFT);
}

uvec2 halfBlockShift() { return uvec2(HBLOCK_WIDTH_SHIFT, HBLOCK_HEIGHT_SHIFT); }

ivec2 halfBlockPixelPos(uint hbid) {
	ivec2 pix_pos =
		ivec2(LIX & (HBLOCK_WIDTH - 1), (LIX >> HBLOCK_WIDTH_SHIFT) & (HBLOCK_HEIGHT - 1));
	return (halfBlockPos(hbid) << halfBlockShift()) + pix_pos;
}

uint swap(uint x, int mask, bool dir) {
	uint y = subgroupShuffleXor(x, mask);
	return (x < y) == dir ? y : x;
}

void sortBuffer(uint count, uint rcount, uint buf_offset, uint group_size, uint lid,
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
			value = swap(value, 0x10, bit != bit4);
			value = swap(value, 0x08, bit != bit3);
			value = swap(value, 0x04, bit != bit2);
			value = swap(value, 0x02, bit != bit1);
			value = swap(value, 0x01, bit != bit0);
			s_buffer[buf_offset + i] = value;
		}
		if(with_barriers)
			barrier();
	}
}

uint blockRowsToBits(uint rows) {
	uvec4 minx = (uvec4(rows) >> uvec4(0, 7, 14, 21)) & 7;
	uvec4 countx = (uvec4(rows) >> uvec4(3, 10, 17, 24)) & 15;
	uint bits0 = uint((1 << countx[0]) - 1) << (minx[0] + 0);
	uint bits1 = uint((1 << countx[1]) - 1) << (minx[1] + 8);
	uint bits2 = uint((1 << countx[2]) - 1) << (minx[2] + 16);
	uint bits3 = uint((1 << countx[3]) - 1) << (minx[3] + 24);
	return bits0 | bits1 | bits2 | bits3;
}

bool highTriDensity(uint tri_count, uint frag_count) {
	return frag_count < (tri_count << 3) && tri_count >= 16;
}

uint shadingBufferOffset() { return (LIX >> HALFGROUP_SHIFT) * (SEGMENT_SIZE + HALFGROUP_SIZE); }

// bits  0-11: current triangle idx
// bits 12-23: triangle count
// bit     24: high tri density
// bits 25-31: segment id, only 4 lower bits are needed
uint initUnpackSamples(uint tri_count, uint frag_count) {
	bool high_tri_density = highTriDensity(tri_count, frag_count);
	uint subgroup_id = LIX & HALFGROUP_MASK;
	return (high_tri_density ? subgroup_id | 0x08000000 : subgroup_id >> HBLOCK_HEIGHT_SHIFT) |
		   (tri_count << 12);
}

void unpackSamples(inout uint control_var, uint src_offset) {
	uint buf_offset = shadingBufferOffset();

	// Copying samples generated for current segment by last thread in previous round
	uint prev_offset = buf_offset + (LIX & HALFGROUP_MASK);
	s_buffer[prev_offset] = s_buffer[prev_offset + SEGMENT_SIZE];

	bool high_tri_density = (control_var & 0x08000000) != 0;
	uint segment_id = (control_var >> 28) & 0xf;
	uint i = control_var & 0xfff, tri_count = (control_var >> 12) & 0xfff;

	if(high_tri_density) {
		for(; i < tri_count; i += HALFGROUP_SIZE) {
			uvec2 tri_data = g_scratch_64[src_offset + i];
			uint tri_info = tri_data.x;
			uint seg_id = tri_data.y >> 28;
			if(seg_id != segment_id)
				break;

			uint tri_offset = tri_info & 0xff;
			uint tri_idx_shifted = tri_info & 0xffffff00;
			uint bits = blockRowsToBits(tri_data.y);
			while(bits != 0) {
				uint pixel_id = findLSB(bits);
				bits &= ~(1u << pixel_id);
				s_buffer[buf_offset + tri_offset] = pixel_id | tri_idx_shifted;
				tri_offset++;
			}
		}
	} else {
		uint y = LIX & (HBLOCK_HEIGHT - 1), row_shift = (y & 3) * 7;
		uint mask1 = y >= 1 ? ~0u : 0, mask2 = y >= 2 ? ~0u : 0;

		// TODO: najpierw wyznaczyæ offsety a póŸniej dopiero sample?
		for(; i < tri_count; i += HALFGROUP_SIZE / HBLOCK_HEIGHT) {
			uvec2 tri_block_data = g_scratch_64[src_offset + i];
			uint tri_data = tri_block_data.y, tri_info = tri_block_data.x;
			uint seg_id = tri_data >> 28;
			if(segment_id != seg_id)
				break;

			uint row_data = tri_data >> row_shift;
			uint countx = (row_data >> 3) & 15;
			uint prevx = countx + (subgroupShuffleUp(countx, 1) & mask1);
			prevx += (subgroupShuffleUp(prevx, 2) & mask2);

			uint tri_offset = (tri_info & 0xff) + prevx - countx;
			uint pixel_id = (y << 3) | (row_data & 7);
			uint value = pixel_id | (tri_info & 0xffffff00);
			uint max_offset = tri_offset + countx;
			while(tri_offset < max_offset)
				s_buffer[buf_offset + tri_offset++] = value++;
		}
	}

	// Increase segment_id (only lower 4 bits are needed) and updating triangle id
	control_var = ((control_var & 0xfffff000) + 0x10000000) | i;
}

void shadeAndReduceSamples(uint hbid, uint sample_count, in out ReductionContext ctx) {
	uint buf_offset = shadingBufferOffset();
	ivec2 hblock_pos = (halfBlockPos(hbid) << halfBlockShift()) + s_bin_pos;
	vec3 out_color = ctx.out_color;
	subgroupMemoryBarrierShared();

	for(uint i = 0; i < sample_count; i += HALFGROUP_SIZE) {
		uint invocation_id = LIX & HALFGROUP_MASK;
		uint sample_offset = buf_offset + invocation_id;
		uint value = s_buffer[sample_offset];

		s_buffer[sample_offset] = 0;
		subgroupMemoryBarrierShared();
		uvec2 sample_s;

		uint sample_id = i + invocation_id;
		if(sample_id < sample_count) {
			uint sample_pixel_id = value & HALFGROUP_MASK;
			uint tri_idx = value >> 8;
			ivec2 pix_pos = hblock_pos + ivec2(sample_pixel_id & 7, sample_pixel_id >> 3);
			float sample_depth;
			uint sample_color = shadeSample(pix_pos, tri_idx, sample_depth);
			sample_s = uvec2(sample_color, floatBitsToUint(sample_depth));

			uint pixel_bit = SUBGROUP_SIZE == 64 ? gl_SubgroupEqMask.x | gl_SubgroupEqMask.y :
												   gl_SubgroupEqMask.x;
			atomicOr(s_buffer[buf_offset + sample_pixel_id], pixel_bit);
		}
		subgroupMemoryBarrierShared();

		uint pixel_bitmask = s_buffer[sample_offset];
		if(reduceSample(ctx, out_color, sample_s, pixel_bitmask))
			break;
		buf_offset += HALFGROUP_SIZE;
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
	uint buf_offset = shadingBufferOffset();
	for(uint i = LIX & HALFGROUP_MASK; i < sample_count; i += HALFGROUP_SIZE) {
		uint pixel_id = s_buffer[buf_offset + i] & HALFGROUP_MASK;
		atomicAdd(s_vis_pixels[(LIX & ~HALFGROUP_MASK) + pixel_id], 1);
	}
	subgroupMemoryBarrierShared();
}

void finishVisualizeSamples(ivec2 pixel_pos) {
	uint pixel_id = (pixel_pos.x & (HBLOCK_WIDTH - 1)) +
					((pixel_pos.y & (HBLOCK_HEIGHT - 1)) << HBLOCK_WIDTH_SHIFT);
	uint value = s_vis_pixels[(LIX & ~HALFGROUP_MASK) + pixel_id];
	vec3 color = gradientColor(value, uvec4(8, 32, 128, 1024));
	outputPixel(pixel_pos, vec4(SATURATE(color), 1.0));
}

#endif
