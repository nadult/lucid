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
#define INVALID_SEGMENT 0xffff

shared uint s_buffer[LSIZE * 8 + 1];
shared uint s_mini_buffer[LSIZE * (WARP_SIZE == 64 ? 2 : 1)];
shared uint s_segments[LSIZE * (WARP_SIZE == 64 ? 1 : 2)];

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
		(xmin[0] & 7) | ((xmin[1] & 7) << 4) | ((xmin[2] & 7) << 8) | ((xmin[3] & 7) << 12);
	uint count_bits = (count[0] << 16) | (count[1] << 20) | (count[2] << 24) | (count[3] << 28);
	return min_bits | count_bits;
}

uint rasterBlockDepth(vec2 cpos, uint tri_idx) {
	uint depth_offset = STORAGE_TRI_DEPTH_OFFSET + tri_idx;
	vec3 depth_eq = uintBitsToFloat(g_uvec4_storage[depth_offset].xyz);
	float ray_pos = depth_eq.x * cpos.x + (depth_eq.y * cpos.y + depth_eq.z);
	float depth = 0xffffe * SATURATE(inversesqrt(ray_pos + 1)); // 20 bits
	return uint(depth);
}

// TODO: this is only used in raster_low; Can we simplify it and use it n both versions?
ivec2 rasterBlockPos(uint rbid) {
#if WARP_SIZE == 64
	uint rbx = rbid & BLOCK_ROWS_MASK;
	uint rby = rbid >> BLOCK_ROWS_SHIFT;
#else
	uint rbx = (rbid >> 1) & BLOCK_ROWS_MASK;
	uint rby = (rbid & 1) + ((rbid >> (BLOCK_ROWS_SHIFT + 1)) << 1);
#endif
	return ivec2(rbx, rby);
}

uvec2 rasterBlockShift() { return uvec2(RBLOCK_WIDTH_SHIFT, RBLOCK_HEIGHT_SHIFT); }

ivec2 rasterBlockPixelPos(uint rbid) {
	ivec2 pix_pos =
		ivec2(LIX & (RBLOCK_WIDTH - 1), (LIX >> RBLOCK_WIDTH_SHIFT) & (RBLOCK_HEIGHT - 1));
	return (rasterBlockPos(rbid) << rasterBlockShift()) + pix_pos;
}

#endif
