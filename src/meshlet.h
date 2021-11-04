#pragma once

#include "lucid_base.h"
#include <fwk/geom_base.h>

struct Meshlet {
	vector<float3> verts;
	vector<float3> normals;
	vector<float2> uvs;
	vector<IColor> colors;
	vector<array<u8, 3>> tris;
	uint material_id;
};

// Czy chcemy od razu robić quady?
// - najpierw byśmy zmergowali mesh w quady a później w meshlety?
//
// Czy meshlety to instancje?
// na razie z każdej instancji generujemy inny zestaw meshletów i tyle. Instancja może mapować się na N meshletów

struct SceneMesh;
struct Scene;

struct TriInfo {
	array<int, 3> verts;
	array<int, 3> neighbours;
	FBox bbox;
};

struct MeshPartition {
	vector<int> verts;
	vector<array<int, 3>> tris;
	FBox bbox;
};

vector<TriInfo> meshTriInfo(CSpan<float3> verts, CSpan<array<int, 3>> tris);

vector<MeshPartition> meshPartition(CSpan<float3>, const SceneMesh &, CSpan<TriInfo>,
									Interval<int> selected_tris, int max_tris, int max_verts);

void meshPartitionStats(CSpan<MeshPartition>, int max_tris, int max_verts);
void visualizeMeshPartitions(const Scene &, CSpan<MeshPartition>);
void meshletTest(const Scene &, float square_weight);
