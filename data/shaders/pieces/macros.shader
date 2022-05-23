// clang-format off

#define INST_HAS_VERTEX_COLORS		0x01
#define INST_HAS_VERTEX_TEX_COORDS	0x02
#define INST_HAS_VERTEX_NORMALS		0x04
#define INST_IS_OPAQUE				0x08
#define INST_TEX_OPAQUE				0x10
#define INST_HAS_UV_RECT			0x20
#define INST_HAS_TEXTURE			0x40

// Different reasons for rejection of triangles/quads during setup
#define REJECTED_OTHER				0
#define REJECTED_BACKFACE			1
#define REJECTED_FRUSTUM			2
#define REJECTED_BETWEEN_SAMPLES	3

#define REJECTED_TYPE_COUNT			4

