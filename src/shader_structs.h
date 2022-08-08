#include "lucid_base.h"

namespace shader {
#include "../data/shaders/shared/structures.glsl"
static_assert(sizeof(LucidInfo) / sizeof(uint) == LUCID_INFO_SIZE);
}

#define LUCID_INFO_MEMBER_OFFSET(name) (offsetof(::shader::LucidInfo, name))