// Copyright (C) Krzysztof Jakubowski <nadult@fastmail.fm>
// This file is part of LucidRaster. See license.txt for details.

#include "quad_generator.h"

#include <fwk/heap.h>

float squareness(CSpan<float3, 4> corners) {
	float out = 0.0;
	array<float3, 4> edges;
	for(auto [i, j] : wrappedPairsRange(4))
		edges[i] = normalize(corners[j] - corners[i]);
	for(auto [i, j] : wrappedPairsRange(4))
		out += fabs(dot(edges[i], edges[j]));
	return (4.0f - out) * 0.25f;
}

vector<array<int, 3>> triNeighbours(CSpan<array<int, 3>> tris) {
	vector<array<int, 3>> out(tris.size(), array<int, 3>{{-1, -1, -1}});
	HashMap<pair<int, int>, int> edge_tri_map;
	edge_tri_map.reserve(tris.size() * 4);

	for(int i : intRange(tris)) {
		auto &tri = tris[i];
		for(int j = 0; j < 3; j++) {
			int v0 = tri[j], v1 = tri[j == 2 ? 0 : j + 1];
			edge_tri_map.emplace({v0, v1}, i);
		}
	}

	for(int i : intRange(tris)) {
		auto &tri = tris[i];
		for(int j = 0; j < 3; j++) {
			int v0 = tri[j], v1 = tri[j == 2 ? 0 : j + 1];
			auto it = edge_tri_map.find({v1, v0});
			if(it != edge_tri_map.end()) {
				if(it->value != i)
					out[i][j] = it->value;
			}
		}
	}

	return out;
}

static int findIndex(CSpan<int> values, int value) {
	for(int i : intRange(values))
		if(values[i] == value)
			return i;
	return -1;
}

//#define DEBUG_QUADGEN

Pair<vector<QuadNode>, vector<array<int, 3>>>
quadNodes(CSpan<float3> verts, CSpan<array<int, 3>> tris, CSpan<array<int, 3>> tri_neighbours) {
	vector<QuadNode> quads;
	quads.reserve(tri_neighbours.size() * 2 / 3);
	vector<array<int, 3>> tri_quads(tris.size(), array<int, 3>{{-1, -1, -1}});

	// Potrzebujemy tutaj mapę sąsiadów po krawędziach

	for(int idx0 : intRange(tris)) {
		auto &tri0 = tris[idx0];
		auto &tri0_neighbours = tri_neighbours[idx0];
		for(int i : intRange(3)) {
			int idx1 = tri0_neighbours[i];
			if(idx1 == -1 || tri_quads[idx0][i] != -1)
				continue;
			auto &tri1_neighbours = tri_neighbours[idx1];
			int j = findIndex(tri1_neighbours, idx0);
			if(j == -1)
				continue;

			int opposite_vert = -1;
			for(int ov : tris[idx1])
				if(!isOneOf(ov, tri0)) {
					opposite_vert = ov;
					break;
				}
			if(opposite_vert == -1)
				continue;

			int quad_idx = quads.size();
			auto &quad = quads.emplace_back(idx0, idx1);
			tri_quads[idx0][i] = quad_idx;
			tri_quads[idx1][j] = quad_idx;

			quad.verts = {{tri0[i], opposite_vert, tri0[(i + 1) % 3], tri0[(i + 2) % 3]}};
			float3 points[4];
			for(int k : intRange(4))
				points[k] = verts[quad.verts[k]];
			quad.squareness = squareness(points);
			//print("Quad %: [% %] sq: % idx:%\n", quad_idx, idx0, idx1, quad.squareness, i);
		}
	}

	for(auto &tri_quad : tri_quads)
		for(int i = 0; i < 3; i++) {
			int q0 = tri_quad[i];
			int q1 = tri_quad[i == 2 ? 0 : i + 1];
			if(q0 != -1 && q1 != -1) {
				quads[q0].addConflict(q1);
				quads[q1].addConflict(q0);
			}
		}

#ifdef DEBUG_QUADGEN
	for(int i : intRange(verts))
		print("Vertex %: %\n", i, verts[i]);
	for(int i : intRange(tris))
		print("Tri %: %\n", i, tris[i]);
	for(int i : intRange(quads))
		print("Quad %: v:% t:%  con:% sq:%\n", i, quads[i].verts, quads[i].tris, quads[i].conflicts,
			  quads[i].squareness);
#endif

	return {std::move(quads), std::move(tri_quads)};
}

vector<array<int, 4>> genQuads(CSpan<array<int, 3>> tris, CSpan<array<int, 3>> tri_neighbours,
							   CSpan<QuadNode> quads, CSpan<array<int, 3>> tri_quads,
							   float square_weight) {
	vector<array<int, 4>> out;
	out.reserve(tris.size() * 2 / 3);

	// 2:selected quad  1:removed quad
	vector<u8> visited_quads(quads.size(), 0);
	// At the end of the algorithm we will iterate in BFS manner over all tris
	// Tris will also be used to iterate over selected quads. First triangle
	// will be used for that (second will be disabled).
	vector<bool> visited_tris(tris.size(), false);

	vector<int> degree(quads.size(), 0);
	for(int qidx : intRange(quads))
		degree[qidx] = quads[qidx].degree();

	auto score = [&](int idx) {
		int deg = degree[idx];
		return deg - quads[idx].squareness * square_weight;
	};

	Heap<float> heap(quads.size());
	for(int qidx : intRange(quads)) {
		heap.insert(qidx, score(qidx));
#ifdef DEBUG_QUADGEN
		print("Quad % score: %\n", qidx, score(qidx));
#endif
	}

	// Computing maximum independent set with basic greedy algorithm
	// TODO: use better algorithm (with preprocessing):
	// http://www.ru.is/~mmh/papers/algo.pdf
	while(!heap.empty()) {
		auto [_, qidx] = heap.extractMin();
		if(visited_quads[qidx])
			continue;
		//print("Selected quad: %\n", qidx);

		auto &quad = quads[qidx];
		visited_quads[qidx] = 2;
		visited_tris[quad.tris[1]] = true;
		for(int nidx : quad.conflicts)
			if(nidx != -1) {
				auto &nquad = quads[nidx];
				visited_quads[nidx] = 1;
				//print("Removing quad: %\n", nidx);

				for(int nidx2 : nquad.conflicts)
					if(nidx2 != -1 && !visited_quads[nidx2]) {
						degree[nidx2]--;
						heap.update(nidx2, score(nidx2));
						//print("Lower degree for %: %\n", nidx2, degree[nidx2]);
					}
			}
	}

	auto get_selected_quad = [&](int tidx) {
		for(int qidx : tri_quads[tidx])
			if(qidx != -1 && visited_quads[qidx] == 2)
				return qidx;
		return -1;
	};

	// Trying to maintain original triangle order
	int num_degenerate = 0;
	for(int tidx : intRange(tris)) {
		if(visited_tris[tidx])
			continue;
		int qidx = get_selected_quad(tidx);
		if(qidx == -1) {
			out.emplace_back(tris[tidx][0], tris[tidx][1], tris[tidx][2], tris[tidx][2]);
			num_degenerate++;
		} else {
			out.emplace_back(quads[qidx].verts);
		}
	}

	print("Quadized: % tris -> % quads (% degenerate)\n", tris.size(), out.size(), num_degenerate);
	return out;
}
