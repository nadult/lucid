// Copyright (C) Krzysztof Jakubowski <nadult@fastmail.fm>
// This file is part of LucidRaster. See license.txt for details.

#pragma once

#include "lucid_base.h"
#include <fwk/gfx/color.h>

struct TextureAtlas {
	struct Entry {
		int2 pos;
		int2 size;
		int2 border_tl, border_br;
	};

	struct Config {
		int round_elem_size = 1;
		int max_atlas_size = 16 * 1024;
	};

	static Maybe<TextureAtlas> make(vector<int2> sizes, Config);

	FRect uvRect(const Entry &, float inset_pixels = 0.0f) const;
	Image merge(CSpan<const Image *>, IColor background = ColorId::black) const;

	vector<Entry> entries;
	Config config;
	int2 size = {64, 64};
};
