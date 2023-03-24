#pragma once

#include "lucid_base.h"

class TriangleBVH {
  public:
	static constexpr int max_depth = 20;

	TriangleBVH(vector<Triangle3F>, int flags = use_sah);
	TriangleBVH() {}

	FBox bbox() const { return m_nodes[0].bbox; }
	string info() const;

	enum { use_sah = 4 };

	struct Node {
		Node() {}
		Node(const FBox &bbox) : bbox(bbox) {}
		bool IsLeaf() const { return sub_node & 0x80000000; }

		FBox bbox;
		union {
			int sub_node, first;
		};
		union {
			struct {
				short axis, firstNode;
			};
			int count;
		};
	};

	static_assert(sizeof(Node) == 32);

	struct BuildContext;

	vector<Triangle3F> m_tris;
	vector<Node> m_nodes;
	int m_depth;
};
