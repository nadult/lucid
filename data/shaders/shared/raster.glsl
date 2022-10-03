#ifndef _RASTER_GLSL_
#define _RASTER_GLSL_

#include "compute_funcs.glsl"
#include "funcs.glsl"
#include "scanline.glsl"
#include "structures.glsl"

#define BIN_MASK (BIN_SIZE - 1)

// In this shader, we're using 8x8 blocks and 8x4 half blocks; TODO: better documentation
// TODO: move all docs to one place at the top of the shader
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

uvec2 rasterBlockPos(uint rbid) {
#if WARP_SIZE == 64
	uint rbx = rbid & BLOCK_ROWS_MASK;
	uint rby = rbid >> BLOCK_ROWS_SHIFT;
#else
	uint rbx = (rbid >> 1) & BLOCK_ROWS_MASK;
	uint rby = (rbid & 1) + ((rbid >> (BLOCK_ROWS_SHIFT + 1)) << 1);
#endif
	return uvec2(rbx, rby);
}

ivec2 rasterBlockPixelPos(uint rbid) {
	return ivec2(rasterBlockPos(rbid) << uvec2(RBLOCK_WIDTH_SHIFT, RBLOCK_HEIGHT_SHIFT));
}

#endif
