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
//   bid: block (8x8) id; In 32x32 bin we have 4 x 4 = 16 blocks
//  rbid: raster block id: either 8x4 or 8x8
//  lbid: local block id (range: 0 up to NUM_WARPS - 1)
// lrbid: local raster block id
// Each block has 64 pixels, so we need 2 warps to process all pixels within a single block

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

#define NUM_BLOCKS (BLOCK_ROWS * BLOCK_ROWS)
#define NUM_RBLOCKS (RBLOCK_COLS * RBLOCK_ROWS)

#define SEGMENT_SIZE 256
#define SEGMENT_SHIFT 8

#define BASE_BUFFER_SIZE (LSIZE * 8)
#define FULL_BUFFER_SIZE (BASE_BUFFER_SIZE + LSIZE * (WARP_SIZE / 32))

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
	((((1 << RBLOCK_COLS) - 1) << (min##id >> RBLOCK_WIDTH_SHIFT)) &                               \
	 (((1 << RBLOCK_COLS) - 1) >> (RBLOCK_COLS_MASK - (max##id >> RBLOCK_WIDTH_SHIFT))))
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

uint rasterBlockDepth(vec2 cpos, uint tri_idx) {
	uint depth_offset = STORAGE_TRI_DEPTH_OFFSET + tri_idx;
	vec3 depth_eq = uintBitsToFloat(g_uvec4_storage[depth_offset].xyz);
	float ray_pos = depth_eq.x * cpos.x + (depth_eq.y * cpos.y + depth_eq.z);
	float depth = 0xffffe * SATURATE(inversesqrt(ray_pos + 1)); // 20 bits
	return uint(depth);
}

ivec2 renderBlockPos(uint rbid) {
	uint rbx = rbid & RBLOCK_COLS_MASK;
	uint rby = rbid >> RBLOCK_COLS_SHIFT;
	return ivec2(rbx, rby);
}

uint blockIdFromRender(uint rbid) {
#if RBLOCK_HEIGHT == BLOCK_HEIGHT
	return rbid;
#else
	ivec2 pos = renderBlockPos(rbid);
	return pos.x + ((pos.y >> 1) << RBLOCK_COLS_SHIFT);
#endif
}

uint blockIdToRender(uint bid) {
#if RBLOCK_HEIGHT == BLOCK_HEIGHT
	return bid;
#else
	uint bx = bid & RBLOCK_COLS_MASK;
	uint by = bid >> RBLOCK_COLS_SHIFT;
	return bx + ((by * 2) << RBLOCK_COLS_SHIFT);
#endif
}

uvec2 renderBlockShift() { return uvec2(RBLOCK_WIDTH_SHIFT, RBLOCK_HEIGHT_SHIFT); }

ivec2 renderBlockPixelPos(uint rbid) {
	ivec2 pix_pos =
		ivec2(LIX & (RBLOCK_WIDTH - 1), (LIX >> RBLOCK_WIDTH_SHIFT) & (RBLOCK_HEIGHT - 1));
	return (renderBlockPos(rbid) << renderBlockShift()) + pix_pos;
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

uint shadingBufferOffset() { return gl_SubgroupID * (SEGMENT_SIZE + WARP_SIZE); }

// bits  0-16: current_tri_idx + optional y
// bits 16-21: number of samples generated in previous round
// bit     31: high_tri_density
uint initUnpackSamples(uint tri_count, uint frag_count) {
	bool high_tri_density =
		BIN_LEVEL == BIN_LEVEL_HIGH && WARP_SIZE == 32 && highTriDensity(tri_count, frag_count);
	return (high_tri_density ? gl_SubgroupInvocationID | 0x80000000 :
							   gl_SubgroupInvocationID >> RBLOCK_HEIGHT_SHIFT);
}

void unpackSamples(inout uint control_var, uint tri_count, uint src_offset) {
	uint buf_offset = shadingBufferOffset();
	// Copying samples generated for current segment by last thread in previous round
	uint prev_offset = buf_offset + gl_SubgroupInvocationID;
	s_buffer[prev_offset] = s_buffer[prev_offset + SEGMENT_SIZE];

	uint base_offset = (control_var >> 16) & WARP_MASK;
	uint max_offset = 0;

	bool high_tri_density = BIN_LEVEL == BIN_LEVEL_HIGH && (control_var & 0x80000000) != 0;
	if(high_tri_density) {
		uint i = control_var & 0xffff;
		uint segment_id = (control_var >> 24) & 0xf;

		for(; i < tri_count; i += WARP_SIZE) {
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
			max_offset = tri_offset;
		}
		segment_id++;
		control_var = i | 0x80000000 | (segment_id << 24);
	} else {
		uint i = control_var & 0xffff;
		uint y = gl_SubgroupInvocationID & (RBLOCK_HEIGHT - 1), row_shift = (y & 3) * 7;
		uint mask1 = y >= 1 ? ~0u : 0, mask2 = y >= 2 ? ~0u : 0, mask3 = y >= 4 ? ~0u : 0;

		// TODO: najpierw wyznaczy� offsety a p�niej dopiero sample?
		for(; i < tri_count; i += WARP_SIZE / RBLOCK_HEIGHT) {
#if RBLOCK_HEIGHT == 8
			uint tri_data = g_scratch_64[src_offset + i][y >> 2];
			uint tri_info = g_scratch_32[src_offset + i];
#else
			uvec2 tri_block_data = g_scratch_64[src_offset + i];
			uint tri_data = tri_block_data.y, tri_info = tri_block_data.x;
#endif

			uint row_data = tri_data >> row_shift;
			uint countx = (row_data >> 3) & 15;
			uint prevx = countx + (subgroupShuffleUp(countx, 1) & mask1);
			prevx += (subgroupShuffleUp(prevx, 2) & mask2);
			if(RBLOCK_HEIGHT == 8)
				prevx += (subgroupShuffleUp(prevx, 4) & mask3);

			uint tri_offset = (tri_info & 0xff) + prevx - countx;
			if(tri_offset < max_offset)
				break;

			uint pixel_id = (y << 3) | (row_data & 7);
			uint value = pixel_id | (tri_info & 0xffffff00);
			max_offset = tri_offset + countx;
			while(tri_offset < max_offset)
				s_buffer[buf_offset + tri_offset++] = value++;
		}

		control_var = i;
	}

	// Last thread may have written some samples from next segment
	bool is_last_thread = max_offset >= SEGMENT_SIZE; // There can be only one
	uint last_thread = subgroupBallotFindLSB(subgroupBallot(is_last_thread));
	uint last_samples = subgroupShuffle(max_offset - SEGMENT_SIZE, last_thread);
	control_var |= last_samples << 16;
	subgroupMemoryBarrierShared();
}

void shadeAndReduceSamples(uint rbid, uint sample_count, in out ReductionContext ctx) {
	uint buf_offset = shadingBufferOffset();
	ivec2 rblock_pos = (renderBlockPos(rbid) << renderBlockShift()) + s_bin_pos;
	vec3 out_color = ctx.out_color;
	uint second_offset = (LIX & ~WARP_MASK) + BASE_BUFFER_SIZE + LSIZE;

	for(uint i = 0; i < sample_count; i += WARP_SIZE) {
		uint value = s_buffer[buf_offset + gl_SubgroupInvocationID];

		s_buffer[buf_offset + gl_SubgroupInvocationID] = 0;
#if WARP_SIZE == 64
		s_buffer[second_offset + gl_SubgroupInvocationID] = 0;
#endif
		subgroupMemoryBarrierShared();
		uvec2 sample_s;

		uint sample_id = i + gl_SubgroupInvocationID;
		if(sample_id < sample_count) {
			uint sample_pixel_id = value & WARP_MASK;
			uint tri_idx = value >> 8;
			ivec2 pix_pos = rblock_pos + ivec2(sample_pixel_id & 7, sample_pixel_id >> 3);
			float sample_depth;
			uint sample_color = shadeSample(pix_pos, tri_idx, sample_depth);
			sample_s = uvec2(sample_color, floatBitsToUint(sample_depth));

			uint bits_offset =
				WARP_SIZE == 64 && gl_SubgroupInvocationID >= 32 ? second_offset : buf_offset;
			uint pixel_bit =
				WARP_SIZE == 64 ? gl_SubgroupEqMask.x | gl_SubgroupEqMask.y : gl_SubgroupEqMask.x;
			atomicOr(s_buffer[bits_offset + sample_pixel_id], pixel_bit);
		}
		subgroupMemoryBarrierShared();

#if WARP_SIZE == 32
		uint pixel_bitmask = s_buffer[buf_offset + gl_SubgroupInvocationID];
#else
		uvec2 pixel_bitmask = uvec2(s_buffer[buf_offset + gl_SubgroupInvocationID],
									s_buffer[second_offset + gl_SubgroupInvocationID]);
		subgroupMemoryBarrierShared();
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
	uint buf_offset = shadingBufferOffset();
	for(uint i = gl_SubgroupInvocationID; i < sample_count; i += WARP_SIZE) {
		uint pixel_id = s_buffer[buf_offset + i] & WARP_MASK;
		atomicAdd(s_vis_pixels[(LIX & ~WARP_MASK) + pixel_id], 1);
	}
	subgroupMemoryBarrierShared();
}

void finishVisualizeSamples(ivec2 pixel_pos) {
	uint pixel_id = (pixel_pos.x & (RBLOCK_WIDTH - 1)) +
					((pixel_pos.y & (RBLOCK_HEIGHT - 1)) << RBLOCK_WIDTH_SHIFT);
	uint value = s_vis_pixels[(LIX & ~WARP_MASK) + pixel_id];
	vec3 color = gradientColor(value, uvec4(8, 32, 128, 1024));
	outputPixel(pixel_pos, vec4(SATURATE(color), 1.0));
}

#endif
