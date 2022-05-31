// clang-format off

// Thhese map directly to DrawCallOpts (lucid_base.h)
#define INST_HAS_VERTEX_COLORS		0x01
#define INST_HAS_VERTEX_TEX_COORDS	0x02
#define INST_HAS_VERTEX_NORMALS		0x04
#define INST_IS_OPAQUE				0x08
#define INST_TEX_OPAQUE				0x10
#define INST_HAS_UV_RECT			0x20
#define INST_HAS_TEXTURE			0x40
#define INST_HAS_COLOR				0x80

// Different reasons for rejection of triangles/quads during setup
#define REJECTION_TYPE_OTHER			0
#define REJECTION_TYPE_BACKFACE			1
#define REJECTION_TYPE_FRUSTUM			2
#define REJECTION_TYPE_BETWEEN_SAMPLES	3


// Per-bin number of quad counts, offsets, etc.
#define BIN_QUAD_COUNTS(idx)		g_counts[BIN_COUNT * 0 + (idx)]
#define BIN_QUAD_OFFSETS(idx)		g_counts[BIN_COUNT * 1 + (idx)]
#define BIN_QUAD_OFFSETS_TEMP(idx)	g_counts[BIN_COUNT * 2 + (idx)]

// Lists of bins of different quad density levels
#define MICRO_LEVEL_BINS(idx)		g_counts[BIN_COUNT * 3 + (idx)]
#define LOW_LEVEL_BINS(idx)			g_counts[BIN_COUNT * 4 + (idx)]
#define MEDIUM_LEVEL_BINS(idx)		g_counts[BIN_COUNT * 5 + (idx)]
#define HIGH_LEVEL_BINS(idx)		g_counts[BIN_COUNT * 6 + (idx)]

