// Copyright (C) Krzysztof Jakubowski <nadult@fastmail.fm>
// This file is part of LucidRaster. See license.txt for details.

#include <fwk/sys/intellisense_fix.h>

#include <algorithm>
#include <fwk/dynamic.h>
#include <fwk/enum_flags.h>
#include <fwk/hash_map.h>
#include <fwk/index_range.h>
#include <fwk/io/xml.h>
#include <fwk/math/box_iter.h>
#include <fwk/math/constants.h>
#include <fwk/math/matrix4.h>
#include <fwk/math/random.h>
#include <fwk/math/rotation.h>
#include <fwk/math/triangle.h>
#include <fwk/maybe_ref.h>
#include <fwk/perf_base.h>
#include <fwk/small_vector.h>
#include <fwk/sys/assert.h>
#include <fwk/sys/expected.h>
#include <fwk/type_info_gen.h>
#include <fwk/variant.h>

#define DUMP FWK_DUMP
#define FATAL FWK_FATAL
