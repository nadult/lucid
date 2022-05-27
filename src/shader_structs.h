#include "lucid_base.h"

namespace shader {

#include "../data/shaders/pieces/structures.shader"

static_assert(sizeof(BinCounters) / sizeof(uint) == BIN_COUNTERS_SIZE);

}

#define BIN_COUNTERS_MEMBER_OFFSET(name) (offsetof(::shader::BinCounters, name) / sizeof(u32))