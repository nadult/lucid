// Copyright (C) Krzysztof Jakubowski <nadult@fastmail.fm>
// This file is part of LucidRaster. See license.txt for details.

#include "bvh.h"
#include <fwk/format.h>
#include <fwk/math/box.h>
#include <fwk/math/triangle.h>

namespace {
inline int maxAxis(float3 vec) {
	return vec.y > vec.x ? vec.z > vec.y ? 2 : 1 : vec.z > vec.x ? 2 : 0;
}

inline float3 triangleCost(const Triangle3F &tri) {
	float3 a = tri[0], ba = tri[1] - tri[0], ca = tri[2] - tri[0];
	return a * 3.0f + ba + ca;
}

inline float triangleCost(const Triangle3F &tri, int axis) {
	float a = tri[0][axis], ba = tri[1][axis] - tri[0][axis], ca = tri[2][axis] - tri[0][axis];
	return a * 3.0f + ba + ca;
}
}

struct TriangleBVH::BuildContext {
	BuildContext(vector<Triangle3F> ttris, bool use_sah) : tris(std::move(ttris)) {
		DASSERT(!tris.empty());
		for(auto &tri : tris)
			DASSERT(!isNan(tri.points()));

		boxes.resize(tris.size());
		FBox sum = enclose(tris[0]);
		for(int n = 0; n < tris.size(); n++) {
			boxes[n] = enclose(tris[n]);
			sum = enclose(sum, boxes[n]);
		}

		nodes.reserve(tris.size() * 2);
		nodes.emplace_back(sum);
		costs.resize(tris.size());
		for(int n = 0; n < tris.size(); n++)
			costs[n] = triangleCost(tris[n]);

		indices.resize(tris.size());
		temp_buffer.resize(tris.size() * sizeof(Triangle3F));
		build(0, 0, tris.size(), 0, use_sah);
	}

	void makeLeaf(int nidx, int first, int count, int sdepth) {
		float3 bmin = boxes[first].min(), bmax = boxes[first].max();
		for(int n = 1; n < count; n++) {
			bmin = vmin(bmin, boxes[first + n].min());
			bmax = vmax(bmax, boxes[first + n].max());
		}
		nodes[nidx].bbox = {bmin, bmax};
		depth = max(depth, sdepth);
		nodes[nidx].first = first | 0x80000000;
		nodes[nidx].count = count;
	}

	void build(int nidx, int first, int count, int sdepth, bool use_sah) {
		auto bbox = nodes[nidx].bbox;

		if(count <= 4 || sdepth == max_depth - 1) {
			makeLeaf(nidx, first, count, sdepth);
			return;
		}

		int middle = count / 2, best_axis = maxAxis(bbox.size());

		Span<Triangle3F> ctris(tris.begin() + first, count);
		Span<float3> ccosts(costs.begin() + first, count);
		Span<FBox> cboxes(boxes.begin() + first, count);

		if(use_sah) {
			const float traverseCost = 0.0;
			const float isect_cost = 1.0;
			float min_cost = inf;
			float no_split_cost = isect_cost * count * bbox.surfaceArea();

			auto tindices = span(temp_buffer).reinterpret<pair<float, int>>();
			for(int axis = 0; axis <= 2; axis++) {
				for(int n = 0; n < count; n++)
					tindices[n] = {ccosts[n][axis], n};
				std::sort(&tindices[0], &tindices[count]);
				for(int n = 0; n < count; n++)
					indices[n] = tindices[n].second;

				Span<float> left_sa(span(temp_buffer).reinterpret<float>());
				Span<float> right_sa(left_sa.begin() + count, count);
				left_sa = span(left_sa.begin(), count);

				right_sa[count - 1] = cboxes[indices[count - 1]].surfaceArea();
				left_sa[0] = cboxes[indices[0]].surfaceArea();

				auto last_box = cboxes[indices[0]];
				for(int n = 1; n < count; n++) {
					last_box = enclose(last_box, cboxes[indices[n]]);
					left_sa[n] = last_box.surfaceArea();
				}

				last_box = cboxes[indices[count - 1]];
				for(int n = count - 2; n >= 0; n--) {
					last_box = enclose(last_box, cboxes[indices[n]]);
					right_sa[n] = last_box.surfaceArea();
				}

				for(int n = 1; n < count; n++) {
					float cost = left_sa[n - 1] * n + right_sa[n] * (count - n);
					if(cost < min_cost) {
						min_cost = cost;
						middle = n;
						best_axis = axis;
					}
				}
			}

			min_cost = traverseCost + isect_cost * min_cost;
			if(no_split_cost < min_cost)
				return makeLeaf(nidx, first, count, sdepth);
		}

		// TODO: why best_axis != 2 ???
		if(!use_sah || best_axis != 2) {
			for(int n = 0; n < count; n++)
				indices[n] = n;
			std::nth_element(&indices[0], &indices[middle], &indices[count], [&](int i1, int i2) {
				return ccosts[i1][best_axis] < ccosts[i2][best_axis];
			});
		}

		{ // Reordering data
			auto temp_tris = span(temp_buffer).reinterpret<Triangle3F>();
			auto temp_boxes = span(temp_buffer).reinterpret<FBox>();
			auto temp_costs = span(temp_buffer).reinterpret<float3>();

			for(int n = 0; n < count; n++)
				temp_tris[n] = ctris[indices[n]];
			for(int n = 0; n < count; n++)
				ctris[n] = temp_tris[n];
			for(int n = 0; n < count; n++)
				temp_boxes[n] = cboxes[indices[n]];
			for(int n = 0; n < count; n++)
				cboxes[n] = temp_boxes[n];
			for(int n = 0; n < count; n++)
				temp_costs[n] = ccosts[indices[n]];
			for(int n = 0; n < count; n++)
				ccosts[n] = temp_costs[n];
		}

		FBox left_box = cboxes[0];
		FBox right_box = cboxes[count - 1];

		for(int n = 1; n < middle; n++)
			left_box = enclose(left_box, cboxes[n]);
		for(int n = middle; n < count; n++)
			right_box = enclose(right_box, cboxes[n]);

		int sub_node = nodes.size();
		nodes[nidx].sub_node = sub_node;

		//	float maxDiff = -fconstant::inf;
		//	for(int ax = 0; ax < 3; ax++) {
		//		float diff = max(left_box.min[axis] - right_box.max[axis], right_box.min[axis] - left_box.max[axis]);
		//		if(diff > maxDiff) {
		//			maxDiff = diff;
		//			axis = ax;
		//		}
		//	}
		nodes[nidx].axis = best_axis;
		nodes[nidx].firstNode = left_box.min(best_axis) > right_box.min(best_axis) ? 1 : 0;
		nodes[nidx].firstNode = left_box.min(best_axis) == right_box.min(best_axis)
									? left_box.max(best_axis) < right_box.max(best_axis) ? 0 : 1
									: 0;

		nodes.push_back(Node(left_box));
		nodes.push_back(Node(right_box));

		build(sub_node + 0, first, middle, sdepth + 1, use_sah);
		build(sub_node + 1, first + middle, count - middle, sdepth + 1, use_sah);
	}

	int depth = 0;
	vector<Triangle3F> tris;
	vector<Node> nodes;
	PodVector<FBox> boxes;
	PodVector<float3> costs;
	PodVector<int> indices;
	PodVector<char> temp_buffer;
};

TriangleBVH::TriangleBVH(vector<Triangle3F> tris, int flags) {
	BuildContext context(std::move(tris), flags & use_sah);
	m_tris = std::move(context.tris);
	m_nodes = std::move(context.nodes);
	m_depth = context.depth;
	ASSERT(m_depth <= max_depth);
}

string TriangleBVH::info() const {
	double nodeBytes = m_nodes.size() * sizeof(Node);
	int objBytes = (sizeof(Triangle3F)) * m_tris.size();

	TextFormatter fmt;
	fmt.stdFormat("BVH Stats:\nTriangles: %d (%.2f KB)\n", (int)m_tris.size(), objBytes * 0.001);
	fmt.stdFormat("Nodes: %8d * %2d = %6.2fKB\n", (int)m_nodes.size(), (int)sizeof(Node),
				  nodeBytes * 0.001);
	fmt.stdFormat("~ %.0f bytes per triangle\n",
				  double(nodeBytes + objBytes) / double(m_tris.size()));
	fmt("Levels: %\n\n", m_depth);
	return fmt.text();
}
