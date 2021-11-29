#include "meshlet.h"

#include "quad_generator.h"
#include "scene.h"
#include <fwk/geom/graph.h>
#include <fwk/gfx/investigator3.h>
#include <fwk/gfx/visualizer3.h>
#include <fwk/heap.h>
#include <fwk/math/interval.h>
#include <fwk/static_vector.h>

vector<TriInfo> meshTriInfo(CSpan<float3> verts, CSpan<array<int, 3>> tris) {
	vector<TriInfo> out(tris.size());
	HashMap<pair<int, int>, int> edge_tri_map;
	edge_tri_map.reserve(tris.size() * 4);

	for(int i : intRange(tris)) {
		auto &tri = tris[i];
		array<float3, 3> positions{verts[tri[0]], verts[tri[1]], verts[tri[2]]};
		for(int j = 0; j < 3; j++) {
			int v0 = tri[j], v1 = tri[j == 2 ? 0 : j + 1];
			edge_tri_map.emplace({v0, v1}, i);
		}
		out[i].neighbours = {-1, -1, -1};
		out[i].verts = {tri[0], tri[1], tri[2]};
		out[i].bbox = enclose(positions);
	}

	for(int i : intRange(tris)) {
		auto &tri = tris[i];
		for(int j = 0; j < 3; j++) {
			int v0 = tri[j], v1 = tri[j == 2 ? 0 : j + 1];
			auto it = edge_tri_map.find({v1, v0});
			if(it != edge_tri_map.end()) {
				if(it->value != i)
					out[i].neighbours[j] = it->value;
			}
		}
	}

	return out;
}

vector<MeshPartition> toPartitions(CSpan<float3> verts, const SceneMesh &mesh,
								   CSpan<array<int, 4>> quads) {
	vector<MeshPartition> out;
	for(auto &quad : quads) {
		auto &part = out.emplace_back();
		if(quad[2] == quad[3]) {
			part.verts = cspan(quad).subSpan(0, 3);
			part.tris = {{quad[0], quad[1], quad[2]}};
			array<float3, 3> corners{{verts[quad[0]], verts[quad[1]], verts[quad[2]]}};
			part.bbox = enclose(corners);
		} else {
			part.verts = cspan(quad);
			part.tris = {{quad[0], quad[1], quad[2]}, {quad[0], quad[2], quad[3]}};
			array<float3, 4> corners{
				{verts[quad[0]], verts[quad[1]], verts[quad[2]], verts[quad[3]]}};
			part.bbox = enclose(corners);
		}
	}
	return out;
}

vector<MeshPartition> meshPartition(CSpan<float3> verts, const SceneMesh &mesh,
									CSpan<TriInfo> tri_infos, int max_tris, int max_verts) {
	vector<MeshPartition> out;
	static constexpr int unassigned_id = -1, front_id = -2;
	vector<int> tri_group_id(tri_infos.size(), unassigned_id);
	vector<char> selected_verts(verts.size(), 0);

	print("Input mesh: tris:% all_verts:% bbox:%\n", mesh.tris.size(), verts.size(),
		  mesh.bounding_box);
	print("Partition limits: max_tris:% max_verts:%\n", max_tris, max_verts);

	PodVector<int> selected_tris(mesh.tris.size());
	for(int i : intRange(mesh.tris))
		selected_tris[i] = i;
	auto bbox_size = mesh.bounding_box.size();
	int longest_axis = bbox_size[0] > bbox_size[1] ? 0 : 1;
	longest_axis = bbox_size[2] > bbox_size[longest_axis] ? 2 : longest_axis;

	std::sort(selected_tris.begin(), selected_tris.end(), [&](int a, int b) {
		auto &tri0 = tri_infos[a];
		auto &tri1 = tri_infos[b];
		return tri0.bbox.min(longest_axis) < tri1.bbox.min(longest_axis);
	});

	vector<int> front;
	for(int start_idx : selected_tris) {
		if(tri_group_id[start_idx] != unassigned_id)
			continue;

		int group_id = out.size();
		tri_group_id[start_idx] = group_id;
		out.emplace_back();
		auto &partition = out.back();
		partition.tris.reserve(max_tris);
		partition.verts.reserve(max_verts);
		front.clear();

		auto extend_front = [&](int src_idx) {
			for(auto idx : tri_infos[src_idx].neighbours)
				if(idx != -1 && tri_group_id[idx] == unassigned_id) {
					tri_group_id[idx] = front_id;
					front.emplace_back(idx);
				}
		};

		auto &start_tri = tri_infos[start_idx];
		FBox bbox = start_tri.bbox;
		partition.tris.emplace_back(start_tri.verts);
		insertBack(partition.verts, start_tri.verts);
		for(auto vidx : start_tri.verts)
			selected_verts[vidx] = 1;
		extend_front(start_idx);

		auto start_tri_nrm = Triangle3F(verts[start_tri.verts[0]], verts[start_tri.verts[1]],
										verts[start_tri.verts[2]])
								 .normal();

		// TODO: instead of merging allow jumping to neighbouring triangles in initial search

		while(!front.empty() && partition.tris.size() < max_tris &&
			  partition.verts.size() < max_verts) {
			Pair<Pair<int, float>, int> best = {{3, inf}, 0};
			for(int i : intRange(front)) {
				int idx = front[i];
				auto &ntri = tri_infos[idx];

				float3 ntri_nrm =
					Triangle3F(verts[ntri.verts[0]], verts[ntri.verts[1]], verts[ntri.verts[2]])
						.normal();
				float tri_dot = 1.5f - 0.5f * fabs(dot(start_tri_nrm, ntri_nrm));
				float max_sa = enclose(bbox, ntri.bbox).surfaceArea();

				int num_neighbours = 0;
				if(max_tris > 2)
					for(int nidx : tri_infos[idx].neighbours)
						if(nidx != -1 && tri_group_id[nidx] < 0)
							num_neighbours++;

				best = minFirst(best, {{num_neighbours, max_sa}, i});
			}

			int new_idx = front[best.second];
			front[best.second] = front.back();
			front.pop_back();

			auto &new_tri = tri_infos[new_idx];
			for(int vidx : new_tri.verts)
				if(!selected_verts[vidx]) {
					selected_verts[vidx] = 1;
					partition.verts.emplace_back(vidx);
				}
			partition.tris.emplace_back(new_tri.verts);
			bbox = enclose(bbox, new_tri.bbox);
			tri_group_id[new_idx] = group_id;
			extend_front(new_idx);
		}
		partition.bbox = bbox;

		for(auto vidx : partition.verts)
			selected_verts[vidx] = 0;
		for(auto nidx : front) {
			DASSERT_EQ(tri_group_id[nidx], front_id);
			tri_group_id[nidx] = unassigned_id;
		}
		makeSorted(partition.verts);

		print("- partition %: tris:% verts:% bbox_size:%\n", group_id, partition.tris.size(),
			  partition.verts.size(), partition.bbox.size());
		// TODO: re-index vertices
	}

	// Jak łączyć ze sobą partycje? po odległości? tak, żeby minimalizować bboxa
	// Zaczynać od najmniejszych?

	float merge_limit = 4.0f;
	std::sort(out.begin(), out.end(),
			  [](auto &a, auto &b) { return a.verts.size() < b.verts.size(); });

	// To jest trochę bez sensu, najlepiej od razu dobrze podzielić
	bool still_merging = true;
	while(still_merging) {
		still_merging = false;

		Pair<float, Pair<int>> best_pair = {inf, {-1, -1}};
		for(int i : intRange(out)) {
			for(int j = i + 1; j < out.size(); j++) {
				if(out[i].verts.size() + out[j].verts.size() > max_verts)
					break;
				if(out[i].tris.size() + out[j].tris.size() > max_tris)
					continue;

				FBox bbox = enclose(out[i].bbox, out[j].bbox);
				float score =
					bbox.surfaceArea() / (out[i].bbox.surfaceArea() + out[j].bbox.surfaceArea());
				best_pair = minFirst(best_pair, {score, {i, j}});
			}
		}

		auto [idx1, idx2] = best_pair.second;
		if(idx1 != -1 && best_pair.first <= merge_limit) {
			print("Merged % into %; score: %\n", idx1, idx2, best_pair.first);
			auto &par1 = out[idx1], &par2 = out[idx2];
			insertBack(par1.tris, par2.tris);
			insertBack(par1.verts, par2.verts);
			makeSortedUnique(par1.verts);
			out[idx2] = move(out.back());
			out.pop_back();
			std::sort(out.begin(), out.end(),
					  [](auto &a, auto &b) { return a.verts.size() < b.verts.size(); });
			still_merging = true;
		}
	}

	return out;
}

void meshPartitionStats(CSpan<MeshPartition> partitions, int max_tris, int max_verts) {
	// TODO: duplicated vertices count?
	double avg_tris = 0, avg_verts = 0, avg_sa = 0;
	int num_full_tris = 0, num_full_verts = 0;
	for(auto &partition : partitions) {
		avg_tris += partition.tris.size();
		avg_verts += partition.verts.size();
		avg_sa += partition.bbox.surfaceArea();
		if(partition.tris.size() == max_tris)
			num_full_tris++;
		if(partition.verts.size() == max_verts)
			num_full_verts++;
	}
	avg_tris = avg_tris / partitions.size();
	avg_verts = avg_verts / partitions.size();
	avg_sa /= partitions.size();
	printf("Partitions: %d\n  avg_tris:%.2f (%.2f %%)\n  avg_verts:%.2f (%.2f %%)\n",
		   partitions.size(), avg_tris, avg_tris / max_tris * 100.0, avg_verts,
		   avg_verts / max_verts * 100.0);
	printf("  avg_surface_area: %.2f\n", avg_sa);
	printf("  partitions with full tris: %.2f %%\n",
		   double(num_full_tris) / partitions.size() * 100);
	printf("  partitions with full verts: %.2f %%\n",
		   double(num_full_verts) / partitions.size() * 100);
}

void visualizeMeshPartitions(const Scene &scene, CSpan<MeshPartition> partitions) {
	vector<vector<Triangle3F>> partition_tris;
	CSpan<float3> verts = scene.positions;
	for(auto &partition : partitions) {
		vector<Triangle3F> tris;
		tris.reserve(partition.tris.size());
		for(auto tri : partition.tris)
			tris.emplace_back(verts[tri[0]], verts[tri[1]], verts[tri[2]]);
		partition_tris.emplace_back(move(tris));
	}

	vector<IColor> colors;
	for(auto color_id : all<ColorId>)
		if(!isOneOf(color_id, ColorId::black, ColorId::transparent)) {
			colors.emplace_back((IColor)lerp(FColor(color_id), FColor(ColorId::white), 0.0f));
			colors.emplace_back((IColor)lerp(FColor(color_id), FColor(ColorId::white), 0.1f));
			colors.emplace_back((IColor)lerp(FColor(color_id), FColor(ColorId::white), 0.2f));
		}

	auto vis_func = [&](Visualizer3 &vis, double2 mouse_pos) -> string {
		for(int i : intRange(partitions)) {
			IColor color = colors[i % colors.size()];
			vis(partition_tris[i], partition_tris[i].size() == 1 ? ColorId::black : color);
		}

		return "";
	};

	Investigator3 investigator(vis_func,  InvestigatorOpt::exit_with_space, {DBox(scene.bounding_box), none, 0.1f});
	investigator.run();
}

void meshletTest(const Scene &scene, float square_weight) {
	vector<MeshPartition> partitions;

	/*int max_tris = 64, max_verts = 64;
	for(int i : intRange(scene.meshes)) {
		auto tri_info = meshTriInfo(scene.positions, scene.meshes[i].tris);
		auto &mesh = scene.meshes[i];
		auto current = meshPartition(scene.positions, mesh, tri_info, max_tris, max_verts);
		insertBack(partitions, current);
	}*/

	int max_tris = 2, max_verts = 4;
	for(int i : intRange(scene.meshes)) {
		auto &mesh = scene.meshes[i];
		auto tri_neighbours = triNeighbours(scene.meshes[i].tris);
		auto [quad_nodes, tri_quads] = quadNodes(scene.positions, mesh.tris, tri_neighbours);
		auto qmesh = genQuads(mesh.tris, tri_neighbours, quad_nodes, tri_quads, square_weight);
		insertBack(partitions, toPartitions(scene.positions, mesh, qmesh));
	}

	meshPartitionStats(partitions, max_tris, max_verts);
	visualizeMeshPartitions(scene, partitions);
}
