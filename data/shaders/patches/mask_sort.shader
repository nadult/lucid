// $$include funcs frustum viewport data

#define LIX gl_LocalInvocationIndex
#define LID gl_LocalInvocationID
#define LSIZE 256

// TODO: problem with uneven amounts of data in different tiles
// TODO: jak dobrze obsługiwać różnego rodzaju dystrybucje trójkątów ?

layout(local_size_x = LSIZE) in;

layout(std430, binding = 0) buffer buf0_ { TileCounters g_tiles; };
layout(std430, binding = 1) buffer buf1_ { uint g_block_counts[]; }; // TODO: 16-bits?
layout(std430, binding = 2) buffer buf2_ { uint g_block_tris[]; };
layout(std430, binding = 3) buffer buf3_ { uint g_block_keys[]; };

#define MAX_BLOCK_TRIS 2048

shared uint s_tile_inst_count, s_tile_inst_offset;
shared uvec2 s_sort_buffer[MAX_BLOCK_TRIS];

void sortTileBlocks() {
	uint N = s_tile_inst_count;
	// TODO: fix this
	uint TN = 32;
	while(TN < N)
		TN *= 2;

	uint nn = N;
	while(nn < TN) {
		uint count = min(TN - nn, LSIZE);
		if(LIX < count) // TODO: make sure that this key is bigger than all valid keys
			s_sort_buffer[nn + LIX] = uvec2(0xffffffff, 0xffffffff);
		nn += count;
	}
	barrier();

	// to zajmuje OK 36% czasu
	for(uint k = 2; k <= TN; k = 2 * k) {
		for(uint j = k >> 1; j > 0; j = j >> 1) {
			for(uint ti = 0; ti < TN; ti += LSIZE) {
				uint i = ti + LIX;
				if(i >= TN)
					continue;
				uint ixj = i ^ j;
				if((ixj) > i) {
					uvec2 ivalue = s_sort_buffer[i];
					uvec2 rvalue = s_sort_buffer[ixj];

					// TODO: merge branches?
					if((i & k) == 0 && ivalue.x > rvalue.x) {
						s_sort_buffer[i] = rvalue;
						s_sort_buffer[ixj] = ivalue;
					}
					if((i & k) != 0 && ivalue.x < rvalue.x) {
						s_sort_buffer[i] = rvalue;
						s_sort_buffer[ixj] = ivalue;
					}
				}
			}
			barrier();
		}
	}
}

void sortBinMasks(int bin_id) {
	for(int tile_id = 0; tile_id < TILES_PER_BIN; tile_id++) {
		barrier();
		if(LIX < BLOCKS_PER_TILE) {
			if(LIX == 0) {
				s_tile_inst_count = TILE_BLOCK_TRI_COUNTS(bin_id, tile_id);
				s_tile_inst_offset = TILE_BLOCK_TRI_OFFSETS(bin_id, tile_id);
			}
		}
		barrier();

		uint tile_count = s_tile_inst_count;
		if(tile_count == 0)
			continue;

		uint tile_offset = s_tile_inst_offset;
		for(uint i = LIX; i < tile_count; i += LSIZE)
			s_sort_buffer[i] = uvec2(g_block_keys[tile_offset + i], g_block_tris[tile_offset + i]);

		barrier();
		sortTileBlocks();

		barrier();
		for(uint i = LIX; i < tile_count; i += LSIZE) {
			uvec2 key_value = s_sort_buffer[i];
			if(i > 0) {
				uvec2 prev_key_value = s_sort_buffer[i - 1];
				if(prev_key_value.x > key_value.x)
					RECORD(prev_key_value.x, prev_key_value.y, key_value.x, key_value.y);
			}
			g_block_tris[tile_offset + i] = key_value.y;
		}
	}
}

shared int s_bin_id;

// TODO: some bins require a lot more computation than others
// 74us just to iterate over nothing...
int loadNextBin() {
	if(LIX == 0)
		s_bin_id = int(atomicAdd(g_tiles.sorted_bin_counter, 1));
	barrier();
	return s_bin_id;
}

void main() {
	int bin_id = loadNextBin();
	while(bin_id < BIN_COUNT) {
		barrier();
		sortBinMasks(bin_id);
		bin_id = loadNextBin();
	}
}
