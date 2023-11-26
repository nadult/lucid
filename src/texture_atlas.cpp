// Copyright (C) Krzysztof Jakubowski <nadult@fastmail.fm>
// This file is part of LucidRaster. See license.txt for details.

#include "texture_atlas.h"

#include <fwk/gfx/image.h>

#define STBRP_STATIC
#define STB_RECT_PACK_IMPLEMENTATION
#include "stb_rect_pack.h"

using Entry = TextureAtlas::Entry;
using Config = TextureAtlas::Config;

Maybe<TextureAtlas> TextureAtlas::make(vector<int2> sizes, Config config) {
	if(!sizes)
		return {};
	DASSERT(isPowerOfTwo(config.max_atlas_size));
	DASSERT(config.round_elem_size >= 1 && isPowerOfTwo(config.round_elem_size));

	int2 max_size;
	vector<int2> rsizes = sizes;
	for(int i : intRange(sizes)) {
		DASSERT(sizes[i].x > 0 && sizes[i].y > 0);
		max_size = vmax(max_size, sizes[i]);
		rsizes[i] = (rsizes[i] + int2(config.round_elem_size - 1)) / config.round_elem_size;
	}
	max_size = {nextPow2(max_size.x), nextPow2(max_size.y)};

	stbrp_context ctx;
	// TODO: how many nodes do we actually need?
	PodVector<stbrp_node> nodes(sizes.size() * 32);
	PodVector<stbrp_rect> rects(sizes.size());

	for(int i : intRange(sizes)) {
		rects[i].id = i;
		rects[i].x = rects[i].y = 0;
		rects[i].was_packed = 0;
		rects[i].w = rsizes[i].x;
		rects[i].h = rsizes[i].y;
	}

	while(max_size.x <= config.max_atlas_size && max_size.y <= config.max_atlas_size) {
		int2 max_rsize = max_size / config.round_elem_size;
		stbrp_init_target(&ctx, max_rsize.x, max_rsize.y, nodes.data(), nodes.size());
		if(stbrp_pack_rects(&ctx, rects.data(), rects.size())) {
			vector<Entry> entries(sizes.size());
			for(int i : intRange(rects)) {
				auto &rect = rects[i];
				auto &entry = entries[rect.id];
				entry.size = sizes[i];
				auto rsize = rsizes[i] * config.round_elem_size;
				entry.border_tl = (rsize - entry.size) / 2;
				entry.border_br = (rsize - entry.size) - entry.border_tl;
				entry.pos = int2(rect.x, rect.y) * config.round_elem_size + entry.border_tl;
			}
			return TextureAtlas{std::move(entries), config, max_size};
		}
		(max_size.x > max_size.y ? max_size.y : max_size.x) *= 2;
	}

	return none;
}

static void fillBorders(const Entry &entry, ImageView<IColor> tex) {
	int2 rsize = entry.size + entry.border_tl + entry.border_br;
	if(rsize == entry.size)
		return;

	int left = entry.border_tl.x, top = entry.border_tl.y;
	int right = entry.border_br.x, bottom = entry.border_br.y;
	int2 origin = entry.pos, size = entry.size;

	auto fill_corner = [&](int2 start, int w, int h, int2 src) {
		start += origin;
		src += origin;
		for(int y = 0; y < h; y++)
			for(int x = 0; x < w; x++)
				tex(start.x + x, start.y + y) = tex(src);
	};

	auto fill_rows = [&](int start_y, int h, int src_y) {
		for(int y = 0; y < h; y++) {
			int2 dst(origin.x, origin.y + start_y + y);
			for(int x = 0; x < size.x; x++)
				tex(dst.x + x, dst.y) = tex(origin.x + x, origin.y + src_y);
		}
	};

	auto fill_cols = [&](int start_x, int w, int src_x) {
		for(int y = 0; y < size.y; y++) {
			for(int x = 0; x < w; x++)
				tex(origin.x + start_x + x, origin.y + y) = tex(origin.x + src_x, origin.y + y);
		}
	};

	fill_corner({-left, -top}, left, top, {});
	fill_corner({size.x, -top}, right, top, {size.x - 1, 0});
	fill_corner({-left, size.y}, left, bottom, {0, size.y - 1});
	fill_corner({size.x, size.y}, right, bottom, {size.x - 1, size.y - 1});

	fill_rows(-top, top, 0);
	fill_rows(size.y, bottom, size.y - 1);

	fill_cols(-left, left, 0);
	fill_cols(size.x, right, size.x - 1);
}

Image TextureAtlas::merge(CSpan<const Image *> textures_, IColor background) const {
	DASSERT(textures_.size() == entries.size());
	Image out(size);
	out.fill(background);

	// TODO: use ImageViews instead
	vector<Image> textures;
	for(auto *texture : textures_) {
		DASSERT(baseFormat(texture->format()) == VBaseFormat::rgba8);
		PodVector<u8> data = texture->data();
		textures.emplace_back(data.reinterpret<IColor>(), texture->size(), VFormat::rgba8_unorm);
	}

	for(int i : intRange(textures)) {
		DASSERT(!textures[i].empty());
		out.blit(textures[i], entries[i].pos);
		fillBorders(entries[i], out.pixels<IColor>());
	}
	return out;
}

FRect TextureAtlas::uvRect(const Entry &entry, float inset_pixels) const {
	auto scale = vinv(float2(size));
	float2 p1 = (float2(entry.pos) + float2(inset_pixels)) * scale;
	float2 p2 = (float2(entry.pos + entry.size) - float2(inset_pixels)) * scale;
	return {p1, p2};
}
