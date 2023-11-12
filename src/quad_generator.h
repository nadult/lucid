// Copyright (C) Krzysztof Jakubowski <nadult@fastmail.fm>
// This file is part of LucidRaster. See license.txt for details.

#pragma once

#include "lucid_base.h"
#include <fwk/geom_base.h>

struct QuadNode {
	QuadNode(int t0, int t1) : tris{t0, t1} {};

	array<int, 2> tris;
	array<int, 4> verts;
	array<int, 4> conflicts = {-1, -1, -1, -1};
	float squareness;

	void addConflict(int idx) {
		if(isOneOf(idx, conflicts))
			return;
		for(auto &val : conflicts)
			if(val == -1) {
				val = idx;
				break;
			}
	}
	int otherTri(int idx) const { return idx == tris[0] ? tris[1] : tris[0]; }

	int degree() const {
		int deg = 0;
		for(int idx : conflicts)
			if(idx != -1)
				deg++;
		return deg;
	}
};

vector<array<int, 3>> triNeighbours(CSpan<array<int, 3>> tris);
Pair<vector<QuadNode>, vector<array<int, 3>>> quadNodes(CSpan<float3>, CSpan<array<int, 3>> tris,
														CSpan<array<int, 3>> tri_neighbours);
vector<array<int, 4>> genQuads(CSpan<array<int, 3>> tris, CSpan<array<int, 3>> tri_neighbours,
							   CSpan<QuadNode> quad_nodes, CSpan<array<int, 3>> tri_quads,
							   float square_weight);
