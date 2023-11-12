// Copyright (C) Krzysztof Jakubowski <nadult@fastmail.fm>
// This file is part of LucidRaster. See license.txt for details.

// Original code source:
// https://github.com/GarageGames/Torque3D/blob/master/Engine/source/gfx/util/triListOpt.cpp
//
// Copyright (c) 2012 GarageGames, LLC
// Licensed under MIT

#include "lucid_base.h"

#include <fwk/list_node.h>
#include <map>

static constexpr int max_vertex_cache_size = 32;

struct VertData {
	int cache_pos = -1;
	float score = 0.0f;
	uint num_refs = 0;
	uint num_unadded_refs = 0;
	int *tri_index = nullptr;
};

struct TriData {
	ListNode node;
	float score = 0.0f;
	int vert_idx[3] = {0, 0, 0};
	bool is_in_list = false;
};

// Source: http://home.comcast.net/~tom_forsyth/papers/fast_vert_cache_opt.html
static float findVertexScore(const VertData &vertexData) {
	const float cache_decay_power = 1.5f;
	const float last_tri_score = 0.75f;
	const float valence_boost_scale = 2.0f;
	const float valence_boost_power = 0.5f;

	// If nobody needs this vertex, return -1.0
	if(vertexData.num_unadded_refs < 1)
		return -1.0f;
	float score = 0.0f;

	if(vertexData.cache_pos < 0) {
		// Vertex is not in FIFO cache - no score.
	} else {
		if(vertexData.cache_pos < 3) {
			// This vertex was used in the last triangle,
			// so it has a fixed score, whichever of the three
			// it's in. Otherwise, you can get very different
			// answers depending on whether you add
			// the triangle 1,2,3 or 3,1,2 - which is silly.
			score = last_tri_score;
		} else {
			DASSERT((vertexData.cache_pos < max_vertex_cache_size) &&
					"Out of range cache position for vertex");

			// Points for being high in the cache.
			const float Scaler = 1.0f / (max_vertex_cache_size - 3);
			score = 1.0f - (vertexData.cache_pos - 3) * Scaler;
			score = pow(score, cache_decay_power);
		}
	}

	// Bonus points for having a low number of tris still to
	// use the vert, so we get rid of lone verts quickly.
	float ValenceBoost = pow(vertexData.num_unadded_refs, -valence_boost_power);
	score += valence_boost_scale * ValenceBoost;

	return score;
}

class LRUCacheModel {
  public:
	LRUCacheModel(Span<VertData> vertices) : m_vertices(vertices) {
		m_entries.reserve(vertices.size());
	}

	void enforceSize(int max_size, Vector<uint> &outTrisToUpdate);
	void useVertex(int vidx);
	int getCachePosition(int vidx);

  private:
	static constexpr int null_entry = -1;

	struct Entry {
		int next = null_entry;
		int vidx = 0;
	};

	Span<VertData> m_vertices;
	vector<Entry> m_entries;
	vector<int> m_empty_entries;
	int m_head = null_entry;
};

void LRUCacheModel::useVertex(int vidx) {
	int search = m_head, last = null_entry;
	while(search != null_entry) {
		if(m_entries[search].vidx == vidx)
			break;
		last = search;
		search = m_entries[search].next;
	}

	// If this vertex wasn't found in the cache, create a new entry
	if(search == null_entry) {
		if(m_empty_entries) {
			search = m_empty_entries.back();
			m_empty_entries.pop_back();
			m_entries[search] = {null_entry, vidx};
		} else {
			search = m_entries.size();
			m_entries.emplace_back(null_entry, vidx);
		}
	}

	if(search != m_head) {
		// Unlink the entry from the linked list
		if(last != null_entry)
			m_entries[last].next = m_entries[search].next;
		// Vertex that got passed in is now at the head of the cache
		m_entries[search].next = m_head;
		m_head = search;
	}
}

void LRUCacheModel::enforceSize(int max_size, Vector<uint> &out_tris_to_update) {
	// Clear list of triangles to update scores for
	out_tris_to_update.clear();

	int length = 0;
	int next = m_head, last = null_entry;

	// Run through list, up to the max size
	while(next != null_entry && length < max_vertex_cache_size) {
		VertData &vert_data = m_vertices[m_entries[next].vidx];

		// Update cache position on verts still in cache
		vert_data.cache_pos = length++;

		for(int i = 0; i < vert_data.num_refs; i++) {
			const int &tri_idx = vert_data.tri_index[i];
			if(tri_idx > -1) {
				int j = 0;
				for(; j < out_tris_to_update.size(); j++)
					if(out_tris_to_update[j] == tri_idx)
						break;
				if(j == out_tris_to_update.size())
					out_tris_to_update.push_back(tri_idx);
			}
		}

		vert_data.score = findVertexScore(vert_data);
		last = next;
		next = m_entries[next].next;
	}

	// nullptr out the pointer to the next entry on the last valid entry
	m_entries[last].next = null_entry;
	// If next != nullptr, than we need to prune entries from the tail of the cache
	while(next != null_entry) {
		// Update cache position on verts which are going to get tossed from cache
		m_vertices[m_entries[next].vidx].cache_pos = -1;
		m_empty_entries.emplace_back(next);
		next = m_entries[next].next;
	}
}

int LRUCacheModel::getCachePosition(const int vidx) {
	int length = 0;
	int next = m_head;
	while(next != null_entry) {
		if(m_entries[next].vidx == vidx)
			return length;
		next = m_entries[next].next;
		length++;
	}

	return -1;
}

/// This method will look at the index buffer for a triangle list, and generate
/// a new index buffer which is optimized using Tom Forsyth's paper:
/// "Linear-Speed Vertex Cache Optimization"
/// http://home.comcast.net/~tom_forsyth/papers/fast_vert_cache_opt.html
/// @note Both 'indices' and 'outIndices' can point to the same memory.
/// TODO: this is still quite slow
void optimizeTriangleOrdering(const int num_verts, CSpan<int> indices, Span<int> out_indices) {
	if(num_verts == 0 || indices.size() == 0)
		return;

	DASSERT(indices.size() % 3 == 0);
	DASSERT(out_indices.size() == indices.size());
	int num_primitives = indices.size() / 3;

	// Step 1: initialization
	vector<VertData> vertex_data(num_verts);
	vector<TriData> tri_data(num_primitives);

	uint cur_idx = 0;
	uint num_refs = 0;
	for(int tri = 0; tri < num_primitives; tri++) {
		TriData &cur_tri = tri_data[tri];

		for(int c = 0; c < 3; c++) {
			int cur_vidx = indices[cur_idx];
			cur_tri.vert_idx[c] = cur_vidx;
			vertex_data[cur_vidx].num_unadded_refs++;
			num_refs++;
			cur_idx++;
		}
	}

	PodVector<int> refs(num_refs);
	num_refs = 0;
	for(int v = 0; v < num_verts; v++) {
		VertData &cur_vert = vertex_data[v];
		cur_vert.tri_index = &refs[num_refs];
		num_refs += cur_vert.num_unadded_refs;
		cur_vert.score = findVertexScore(cur_vert);
	}

	int next_next_best_tri_idx = -1, next_best_tri_idx = -1;
	float next_next_best_tri_score = -1.0f, next_best_tri_score = -1.0f;

	auto validate_tri_idx = [&](int idx) {
		if(idx > -1) {
			DASSERT(idx < num_primitives && "Out of range triangle index.");
			DASSERT(!tri_data[idx].is_in_list && "Triangle already in list, bad.");
		}
	};

	auto check_next_next_best = [&](float score, int idx) {
		if(score > next_next_best_tri_score) {
			next_next_best_tri_idx = idx;
			next_next_best_tri_score = score;
		}
	};

	auto check_next_best = [&](float score, int idx) {
		if(score > next_best_tri_score) {
			check_next_next_best(next_best_tri_score, next_best_tri_idx);
			next_best_tri_idx = idx;
			next_best_tri_score = score;
		}
		validate_tri_idx(next_best_tri_idx);
	};

	// TODO: use heap?
	std::map<float, List, std::greater<float>> ordered_tris;

	auto remove_ordered_tri = [&](int tri_idx) {
		auto &tri = tri_data[tri_idx];
		auto it = ordered_tris.find(tri.score);
		listRemove([&](int i) -> ListNode & { return tri_data[i].node; }, it->second, tri_idx);
		if(it->second.empty())
			ordered_tris.erase(it);
	};

	auto get_next_best_tris = [&]() {
		int count = 0;
		auto it = ordered_tris.begin();
		while(count < 2 && it != ordered_tris.end()) {
			auto idx = it->second.head;
			while(idx != -1 && count < 2) {
				check_next_best(it->first, idx);
				check_next_next_best(it->first, idx);
				idx = tri_data[idx].node.next;
				count++;
			}
			it++;
		}
	};

	auto add_ordered_tri = [&](int tri_idx) {
		auto &tri = tri_data[tri_idx];
		auto &list = ordered_tris[tri.score];
		listInsert([&](int i) -> ListNode & { return tri_data[i].node; }, list, tri_idx);
	};

	// Fill-in per-vertex triangle lists, and sum the scores of each vertex used
	// per-triangle, to get the starting triangle score
	cur_idx = 0;
	for(int tri = 0; tri < num_primitives; tri++) {
		TriData &cur_tri = tri_data[tri];
		for(int c = 0; c < 3; c++) {
			VertData &cur_vert = vertex_data[indices[cur_idx]];
			cur_vert.tri_index[cur_vert.num_refs++] = tri;
			cur_tri.score += cur_vert.score;
			cur_idx++;
		}
		add_ordered_tri(tri);
	}
	get_next_best_tris();

	// Step 2: Start emitting triangles...this is the emit loop
	LRUCacheModel lru_cache(vertex_data);

	for(int out_idx = 0; out_idx < indices.size();) {
		// If there is no next best triangle, than search for the next highest
		// scored triangle that isn't in the list already
		if(next_best_tri_idx < 0) {
			next_best_tri_score = next_next_best_tri_score = -1.0f;
			next_best_tri_idx = next_next_best_tri_idx = -1;
			get_next_best_tris();
		}
		DASSERT(next_best_tri_idx > -1);

		TriData &next_best_tri = tri_data[next_best_tri_idx];
		DASSERT(!next_best_tri.is_in_list);
		for(int i = 0; i < 3; i++) {
			out_indices[out_idx++] = int(next_best_tri.vert_idx[i]);
			VertData &cur_vert = vertex_data[next_best_tri.vert_idx[i]];
			cur_vert.num_unadded_refs--;
			for(int t = 0; t < cur_vert.num_refs; t++) {
				if(cur_vert.tri_index[t] == next_best_tri_idx) {
					cur_vert.tri_index[t] = -1;
					break;
				}
			}
			lru_cache.useVertex(next_best_tri.vert_idx[i]);
		}

		next_best_tri.is_in_list = true;
		remove_ordered_tri(next_best_tri_idx);

		// Enforce cache size, this will update the cache position of all verts
		// still in the cache. It will also update the score of the verts in the
		// cache, and give back a list of triangle indicies that need updating.
		vector<uint> tris_to_update;
		lru_cache.enforceSize(max_vertex_cache_size, tris_to_update);

		// Now update scores for triangles that need updates, and find the new best
		// triangle score/index
		next_best_tri_idx = -1;
		next_best_tri_score = -1.0f;

		// TODO: use idx directly
		for(auto itr = tris_to_update.begin(); itr != tris_to_update.end(); itr++) {
			TriData &tri = tri_data[*itr];

			// If this triangle isn't already emitted, re-score it
			if(!tri.is_in_list) {
				remove_ordered_tri(*itr);
				tri.score = 0.0f;
				for(int i = 0; i < 3; i++)
					tri.score += vertex_data[tri.vert_idx[i]].score;
				check_next_best(tri.score, *itr);
				check_next_next_best(tri.score, *itr);
				add_ordered_tri(*itr);
			}
		}

		// If there was no love finding a good triangle, than see if there is a
		// next-next-best triangle, and if there isn't one of those...well than
		// I guess we have to find one next time
		if(next_best_tri_idx < 0 && next_next_best_tri_idx > -1) {
			if(!tri_data[next_next_best_tri_idx].is_in_list) {
				next_best_tri_idx = next_next_best_tri_idx;
				next_best_tri_score = next_next_best_tri_score;
				validate_tri_idx(next_next_best_tri_idx);
			}

			// Nuke the next-next best
			next_next_best_tri_idx = -1;
			next_next_best_tri_score = -1.0f;
		}

		// Validate triangle we are marking as next-best
		validate_tri_idx(next_best_tri_idx);
	}
}
