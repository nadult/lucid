// Copyright (C) Krzysztof Jakubowski <nadult@fastmail.fm>
// This file is part of LucidRaster. See license.txt for details.

#include "wavefront_obj.h"

#include <fwk/hash_map.h>
#include <fwk/io/file_system.h>

// TODO: make parsing work without exceptions ?
Ex<void> WavefrontMaterial::load(ZStr file_path, vector<WavefrontMaterial> &out) {
	auto file_text = EX_PASS(loadFileString(file_path));

	int count = 0;
	WavefrontMaterial *new_mat = nullptr;

	for(string line : tokenize(file_text, '\n')) {
		if(line[0] == '#')
			continue;

		TextParser parser(line);
		Str element;
		parser >> element;
		if(!element)
			continue;

		if(element == "newmtl") {
			Str name;
			parser >> name;
			EXPECT(!name.empty());
			bool has_duplicate = anyOf(out, [=](auto &mat) { return mat.name == name; });
			if(has_duplicate) {
				print("Warning: ignoring duplicate material: '%'\n", name);
				new_mat = nullptr;
				continue;
			}

			out.emplace_back(name);
			new_mat = &out.back();
			count++;
		}

		if(!new_mat)
			continue;

		if(element.startsWith("map_")) {
			WavefrontMap map;
			vector<string> args;
			parser >> args;
			EXPECT(!args.empty());
			map.name = args.back();
			args.pop_back();
			auto locase_elem = toLower(element.substr(4));
			// kd, bump, ks, ao, roughness
			new_mat->maps.emplace_back(locase_elem, std::move(map));
		} else if(element == "d")
			parser >> new_mat->dissolve_factor;
		else if(element == "Kd")
			parser >> new_mat->diffuse;

		EX_CATCH();
	}

	print("Loaded % material% from: '%'\n", count, count > 0 ? "s" : "", file_path);
	return {};
}

Ex<WavefrontObject> WavefrontObject::load(ZStr path, i64 file_size_limit) {
	auto start_time = getTime();
	auto file_text = EX_PASS(loadFileString(path, file_size_limit));

	auto parse_time = getTime();
	vector<string> material_libs;
	vector<float3> src_positions, src_normals;
	vector<float2> src_tex_coords;
	int src_counts[3] = {0, 0, 0}, num_verts = 0;

	using MultiIndex = array<int, 3>;
	HashMap<MultiIndex, int> vertex_map;
	vector<array<int, 3>> tris;

	auto parseIndex = [&](Str text) {
		array<int, 3> idx = {0, 0, 0};
		idx[0] = atoi(text.data());

		auto slash_pos = text.find('/');
		if(slash_pos != -1) {
			text = text.advance(slash_pos + 1);
			slash_pos = text.find('/');
			if(slash_pos == -1) {
				idx[1] = atoi(text.data());
			} else if(slash_pos == 0) {
				idx[2] = atoi(text.data() + 1);
			} else {
				idx[1] = atoi(text.data());
				idx[2] = atoi(text.data() + slash_pos + 1);
			}
		}
		for(int j = 0; j < 3; j++)
			idx[j] = idx[j] < 0 ? idx[j] + src_counts[j] : idx[j] - 1;
		auto it = vertex_map.find(idx);
		if(it == vertex_map.end()) {
			vertex_map.emplace(idx, num_verts);
			return num_verts++;
		}
		return it->value;
	};

	struct UseMtl {
		int first_tri = 0;
		string mat_name;
	};
	vector<UseMtl> use_mtls;
	vector<WavefrontMaterial> materials;

	for(string line : tokenize(file_text, '\n')) {
		if(line[0] == '#')
			continue;
		TextParser parser(line);
		Str element;
		parser >> element;

		if(element == "o") {
			//	if(tris.size()>0) {
			//		objects.push_back(BaseScene::Object(verts,uvs,normals,tris));
			//		objects.back().name = strpbrk(line," \t") + 1;
			//		tris.clear();
			//	}
		} else if(element == "v") {
			float3 vert;
			float w = 1.0f;
			parser >> vert;
			if(!parser.empty())
				parser >> w;
			src_positions.emplace_back(vert);
			src_counts[0]++;
		} else if(element == "vt") {
			float2 uv(0.0f, 0.0f);
			float w = 0.0f;
			parser >> uv[0];
			if(!parser.empty())
				parser >> uv[1];
			if(!parser.empty())
				parser >> w;
			uv[1] = 1.0f - uv[1];
			src_tex_coords.push_back(uv);
			src_counts[1]++;
		} else if(element == "vn") {
			float3 vert;
			parser >> vert;
			src_normals.emplace_back(vert);
			src_counts[2]++;
		} else if(element == "f") {
			int indices[64];
			int count = 0;

			while(!parser.empty()) {
				if(count == arraySize(indices))
					return ERROR("Too many face indices (% is max): '%'", arraySize(indices), line);
				Str elem;
				parser >> elem;
				indices[count++] = parseIndex(elem);
			}

			for(int i = 1; i + 1 < count; i++)
				tris.emplace_back(indices[0], indices[i], indices[i + 1]);
		} else if(element == "usemtl") {
			Str mat_name;
			parser >> mat_name;
			EXPECT(!mat_name.empty());
			use_mtls.emplace_back(tris.size(), mat_name);
		} else if(element == "mtllib") {
			Str lib_name;
			parser >> lib_name;
			if(!anyOf(material_libs, lib_name))
				material_libs.emplace_back(lib_name);
		}
	}
	file_text.clear();

	auto dir_path = FilePath(path).parent();
	for(auto mtl_lib : material_libs)
		EXPECT(WavefrontMaterial::load(dir_path / mtl_lib, materials));

	WavefrontObject out;
	auto init_time = getTime();
	out.positions.resize(num_verts);
	out.tris = std::move(tris);
	for(auto &vindex : vertex_map) {
		int src_index = vindex.key[0];
		auto value =
			src_index >= 0 && src_index < src_counts[0] ? src_positions[src_index] : float3();
		out.positions[vindex.value] = value;
	}

	if(src_tex_coords) {
		out.tex_coords.resize(num_verts);
		for(auto &vindex : vertex_map) {
			int src_index = vindex.key[1];
			auto value =
				src_index >= 0 && src_index < src_counts[1] ? src_tex_coords[src_index] : float2();
			out.tex_coords[vindex.value] = value;
		}
	}

	if(src_normals) {
		out.normals.resize(num_verts);
		for(auto &vindex : vertex_map) {
			int src_index = vindex.key[2];
			auto value =
				src_index >= 0 && src_index < src_counts[2] ? src_normals[src_index] : float3();
			out.normals[vindex.value] = value;
		}
	}

	// TODO: handle groups

	if(use_mtls.empty()) {
		materials.emplace_back("default");
		out.material_groups.emplace_back(0, 0, out.tris.size());
	} else {
		int default_mat_idx = -1;

		for(int i = 0; i < use_mtls.size(); i++) {
			auto &use_mtl = use_mtls[i];
			int end_tri = i + 1 < use_mtls.size() ? use_mtls[i + 1].first_tri : out.tris.size();
			int mat_idx = -1;
			for(int j = 0; j < materials.size(); j++)
				if(materials[j].name == use_mtl.mat_name) {
					mat_idx = j;
					break;
				}
			if(mat_idx == -1) {
				print("Warning: material '%' not found, using default\n", use_mtl.mat_name);
				if(default_mat_idx == -1) {
					for(int j = 0; j < materials.size(); j++)
						if(materials[j].name == "default") {
							default_mat_idx = j;
							break;
						}
					if(default_mat_idx == -1) {
						default_mat_idx = materials.size();
						materials.emplace_back("default");
					}
				}
				mat_idx = default_mat_idx;
			}
			out.material_groups.emplace_back(mat_idx, use_mtl.first_tri,
											 end_tri - use_mtl.first_tri);
		}
	}
	out.materials = std::move(materials);
	out.resource_path = dir_path;
	auto finish_time = getTime();

	print("Loaded Wavefront OBJ in: % ms\n", int((finish_time - start_time) * 1000.0));
	print("  reading file: % ms\n  parsing: % ms\n  initializing scene: % ms\n",
		  int((parse_time - start_time) * 1000.0), int((init_time - parse_time) * 1000.0),
		  int((finish_time - init_time) * 1000.0));
	return out;
}
