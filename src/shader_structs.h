#include "lucid_base.h"

namespace shader {
#include "../data/shaders/shared/structures.glsl"
}

#define LUCID_INFO_MEMBER_OFFSET(name) (offsetof(::shader::LucidInfo, name))