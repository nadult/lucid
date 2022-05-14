// $$include data

#define LID gl_LocalInvocationID
#define LIX gl_LocalInvocationIndex
#define WGID gl_WorkGroupID

// TODO: this is too small for big number of bins!
#define LSIZE 128
#define TRIS_PER_THREAD 32
#define MAX_GROUP_QUADS (LSIZE * TRIS_PER_THREAD)

// Przydałaby się możliwość debugowania tego
// Jeśli moglibyśmy odpalić to np. na CPU
// Ale co zrobić z wątkami ?
// Zrobilibyśmy wirutalną maszynę operującą na 32-elementowych wektorach?
// moglibyśmy w takim kodzie dodać tyle assertów ile chcę
// problemem byłaby symulacja współbieżności?
// Zrobilibyśmy to normalnie na wątkach chyba
//
// Dużo łatwiej będzie zaimplementować lepsze asserty na gpu;

// TODO: be wary of bugs with barriers
// TODO: lepszy sposób na przechowywanie danych i share-owanie ich między C++ a glsl?
// TODO: rename tris to quads

layout(local_size_x = LSIZE) in;
layout(std430, binding = 0) buffer buf0_ { uint g_quad_aabbs[]; };
layout(std430, binding = 1) buffer buf1_ { BinCounters g_bins; };
layout(std430, binding = 2) buffer buf2_ { TileCounters g_tiles; };

shared int s_num_input_quads;

// TODO: zrobić na shortach? powinno sie zmiescic
shared int s_counts[BIN_COUNT];
shared int s_rows[BIN_COUNT_Y];

void countTri(uint quad_idx) {
	uint aabb = g_quad_aabbs[quad_idx];
	if(aabb == ~0u)
		return;
	int tsx = int(aabb & 0xff), tsy = int((aabb >> 8) & 0xff);
	int tex = int((aabb >> 16) & 0xff), tey = int((aabb >> 24));

	int bsx = tsx >> 2, bsy = tsy >> 2;
	int bex = tex >> 2, bey = tey >> 2;
	// ASSERT(bsx >= 0 && bsy >= 0);
	// ASSERT(bex <= BIN_COUNT_X && bey <= BIN_COUNT_Y);

	for(int by = bsy; by <= bey; by++)
		for(int bx = bsx; bx <= bex; bx++)
			atomicAdd(s_counts[bx + by * BIN_COUNT_X], 1);
}

shared bool finish;
shared uint s_quads_offset;

void estimateBins() {
	{
		s_num_input_quads = g_bins.num_input_quads;
		if(LIX == 0)
			finish = g_bins.num_binned_quads >= s_num_input_quads;
		barrier();
		if(finish)
			return;
	}

	for(int i = 0; i < BIN_COUNT; i += LSIZE)
		if(i + LIX < BIN_COUNT)
			s_counts[i + LIX] = 0;
	if(LIX == 0)
		s_quads_offset = atomicAdd(g_bins.num_binned_quads, MAX_GROUP_QUADS);
	barrier();

	uint tris_offset = s_quads_offset;
	int num_input_quads = s_num_input_quads;

	while(tris_offset < num_input_quads) {
		for(uint i = 0; i < TRIS_PER_THREAD; i++) {
			uint quad_idx = tris_offset + LSIZE * i + LIX;
			if(quad_idx < num_input_quads)
				countTri(quad_idx);
		}

		barrier();
		if(LIX == 0)
			s_quads_offset = atomicAdd(g_bins.num_binned_quads, MAX_GROUP_QUADS);
		barrier();
		tris_offset = s_quads_offset;
	}

	barrier();

	for(int i = 0; i < BIN_COUNT; i += LSIZE)
		if(i + LIX < BIN_COUNT && s_counts[i + LIX] > 0)
			atomicAdd(g_bins.bin_quad_counts[i + LIX], s_counts[i + LIX]);
}

// TODO: separate shader instead of phase
void computeOffsets() {
	// Loading tri counts
	for(int i = 0; i < BIN_COUNT; i += LSIZE)
		if(i + LIX < BIN_COUNT)
			s_counts[i + LIX] = g_bins.bin_quad_counts[i + LIX];
	barrier();

	// Computing per-bin tri offsets
	if(LIX < BIN_COUNT_Y) {
		for(int x = 1; x < BIN_COUNT_X; x++)
			s_counts[x + LIX * BIN_COUNT_X] += s_counts[x - 1 + LIX * BIN_COUNT_X];
		s_rows[LIX] = s_counts[BIN_COUNT_X - 1 + LIX * BIN_COUNT_X];
	}
	barrier();
	if(LIX < BIN_COUNT_X) {
		int prev_sum = 0;
		for(int y = 1; y < BIN_COUNT_Y; y++) {
			prev_sum += s_rows[y - 1];
			s_counts[LIX + y * BIN_COUNT_X] += prev_sum;
		}
	}
	barrier();
	for(int i = 0; i < BIN_COUNT; i += LSIZE)
		if(i + LIX < BIN_COUNT) {
			int cur_offset = s_counts[i + LIX] - g_bins.bin_quad_counts[i + LIX];
			g_bins.bin_quad_offsets     [i + LIX] = cur_offset;
			g_bins.bin_quad_offsets_temp[i + LIX] = cur_offset;
		}
	barrier();

	if(LIX == 0) {
		g_bins.num_estimated_quads = s_counts[BIN_COUNT - 1];
		g_bins.num_binned_quads = 0;
	}
}

uniform uint phase;

void main() {
	if(phase == 1)
		estimateBins();
	else if(phase == 2)
		computeOffsets();
}
