#include "lucid_base.h"

#include <fwk/math/matrix4.h>
#include <fwk/math_base.h>

namespace shader {
using vec2 = fwk::float2;
using vec3 = fwk::float3;
using vec4 = fwk::float4;
using mat4 = fwk::Matrix4;

#include "../data/shaders/pieces/structures.shader"
#include "../data/shaders/structures.glsl"

static_assert(sizeof(LucidInfo) / sizeof(uint) == LUCID_INFO_SIZE);
static_assert(sizeof(Lighting) / sizeof(uint) == LIGHTING_STRUCT_SIZE);
static_assert(sizeof(SimpleDrawCall) / sizeof(uint) == SIMPLE_DRAW_CALL_STRUCT_SIZE);

}

#define LUCID_INFO_MEMBER_OFFSET(name) (offsetof(::shader::LucidInfo, name) / sizeof(u32))