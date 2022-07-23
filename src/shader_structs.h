#include "lucid_base.h"

#include <fwk/math/matrix4.h>
#include <fwk/math_base.h>

namespace shader {

// Thread about using vec3 in glsl buffers:
// https://stackoverflow.com/questions/38172696

struct alignas(8) vec2 {
	vec2() : x(0.0f), y(0.0f) {}
	vec2(const fwk::float2 &rhs) : x(rhs.x), y(rhs.y) {}
	operator float2() const { return {x, y}; }

	float x, y;
};

struct alignas(16) vec4 {
	vec4() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}
	vec4(const fwk::float4 &rhs) : x(rhs.x), y(rhs.y), z(rhs.z), w(rhs.w) {}
	vec4(const fwk::float3 &rhs) : x(rhs.x), y(rhs.y), z(rhs.z), w(0.0f) {}
	operator float4() const { return {x, y, z, w}; }
	operator float3() const { return {x, y, z}; }

	float x, y, z, w;
};

struct alignas(16) mat4 {
	mat4() {}
	mat4(const Matrix4 &mat) : col{mat[0], mat[1], mat[2], mat[3]} {}

	vec4 col[4];
};

#include "../data/shaders/pieces/structures.shader"
#include "../data/shaders/structures.glsl"

static_assert(sizeof(LucidInfo) / sizeof(uint) == LUCID_INFO_SIZE);

}

#define LUCID_INFO_MEMBER_OFFSET(name) (offsetof(::shader::LucidInfo, name) / sizeof(u32))