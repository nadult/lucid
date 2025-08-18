// Copyright (C) Krzysztof Jakubowski <nadult@fastmail.fm>
// This file is part of LucidRaster. See license.txt for details.

#include "scene_convert.h"

#include "texture_atlas.h"
#include "wavefront_obj.h"
#include <fwk/gfx/image.h>
#include <fwk/io/file_stream.h>
#include <fwk/io/xml.h>
#include <fwk/sys/exception.h>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#pragma comment(lib, "assimp-vc143-mt.lib")
#pragma comment(lib, "draco.lib")

#include "../extern/tinyexr.h"

Ex<Image> loadExr(ZStr path) {
	float *exr_data = nullptr;
	const char *exr_error = nullptr;
	int width = 0, height = 0;
	int ret = LoadEXR(&exr_data, &width, &height, path.c_str(), &exr_error);
	if(ret != TINYEXR_SUCCESS) {
		auto error = FWK_ERROR("Error during loading EXR file '%': % (%)", path,
							   exr_error ? exr_error : "", ret);
		FreeEXRErrorMessage(exr_error);
		return error;
	}

	PodVector<float4> data({reinterpret_cast<float4 *>(exr_data), width * height});
	free(exr_data);
	return Image{std::move(data), {width, height}, VColorFormat::rgba32_sfloat};
}

vector<Image> panoramaToCubeMap(const Image &panorama) { return {}; }

InputScene::InputScene(string name, string path) : name(std::move(name)), path(std::move(path)) {}
InputScene::InputScene(const FilePath &root_path, CXmlNode node)
	: quad_squareness(node("quad_squareness", 1.0f)), merge_verts(node("merge_verts", false)),
	  flip_uv(node("flip_uv", false)), flip_yz(node("flip_yz", false)), pbr(node("pbr", false)) {
	name = node.tryAttrib("name", "");
	path = node.attrib("path");
	if(name.empty())
		name = FilePath(path).fileStem();
	env_map_path = node.tryAttrib("env_map", "");
	path = root_path / path;
}

Ex<vector<InputScene>> loadInputScenes(ZStr path) {
	auto doc = EX_PASS(XmlDocument::load(path));
	auto node = doc.child("scenes");
	ZStr root_path = node && node.hasAttrib("root_path") ? (ZStr)node("root_path") : "";

	vector<InputScene> out;
	if(node) {
		auto scene_node = node.child("scene");
		while(scene_node) {
			InputScene iscene(root_path, scene_node);
			EX_CATCH();
			out.emplace_back(std::move(iscene));
			scene_node = scene_node.sibling();
		}
	}

	return out;
}

SceneTexture convertTexture(ZStr path, SceneMapType map_type) {
	auto time = getTime();
	SceneTexture out;
	// TODO: what about duplicate names ?
	out.name = FilePath(path).fileStem();
	out.map_type = map_type;
	auto tex = Image::load(path);
	if(!tex) {
		tex.error().print();
		tex = Image({8, 8}, ColorId::purple);
		out.name += "_invalid";
	}
	out.is_opaque = allOf(tex->pixels<IColor>(), [](IColor col) { return col.a == 255; });
	out.mips.emplace_back(std::move(*tex));

	print("  loaded % (size:% opaque:%) in % msec\n", out.name, tex->size(), out.is_opaque,
		  int((getTime() - time) * 1000.0));

	return out;
}

static void makeTextureAtlas(Scene &scene, string name, bool is_opaque) {
	vector<int> selection;
	vector<int2> sizes;
	// TODO: use ImageView class
	vector<const Image *> pointers;
	vector<char> selected_map(scene.textures.size(), false);
	vector<int> inverse_map(scene.textures.size(), -1);

	for(int i : intRange(scene.textures)) {
		auto &tex = scene.textures[i];
		DASSERT(tex.map_type == SceneMapType::albedo &&
				"Atlas generation should be used when scene uses only albedo textures");

		if(tex.is_opaque == is_opaque) {
			selection.emplace_back(i);
			inverse_map[i] = sizes.size();
			sizes.emplace_back(tex.size());
			pointers.emplace_back(&tex.mips[0]);
			selected_map[i] = 1;
		}
	}
	if(selection.size() <= 1)
		return;

	auto atlas = TextureAtlas::make(sizes, {64, 16 * 1024});
	if(!atlas) {
		print("Failed to create texture atlas '%'\n", name);
		return;
	}

	print("Texture atlas '%' size: %\n", name, atlas->size);

	// TODO: clamped textures should have different borders than repeated
	// (but we would probably need some way to compare it with original version)
	auto tex = atlas->merge(pointers, is_opaque ? ColorId::black : ColorId::transparent);
	//tex.saveTGA(format("%_atlas.tga", name)).check();

	// Here we're assuming no instancing
	// TODO: we can also check for that: no more than 1 texture used for any given range of verts

	print("Remapping vertex UVS...\n");
	DASSERT(scene.hasTexCoords());
	vector<int> vertex_list;
	for(auto &mesh : scene.meshes) {
		auto &material = scene.materials[mesh.material_id];
		auto &map = material.maps[SceneMapType::albedo];
		if(!map || !selected_map[map.texture_id])
			continue;

		auto &tex = scene.textures[map.texture_id];
		int entry_index = inverse_map[map.texture_id];
		auto &entry = atlas->entries[entry_index];
		auto uv_rect = atlas->uvRect(entry);

		if(!tex.is_clamped) {
			// We will have to transform UV coords in the shader
			// We're decreasing uv_rect by half a pixel from each side to
			// make sure that it won't be interpolated with neighbouring texels
			// TODO: add repeated texture border? What about textures with pow of 2 size?
			map.uv_rect = atlas->uvRect(entry, 0.5f);
			continue;
		}

		// TODO: don't transform vertices? make uv-rects instead
		vertex_list.clear();
		for(auto &tri : mesh.tris)
			for(auto idx : tri)
				vertex_list.emplace_back(idx);
		makeSortedUnique(vertex_list); // TODO: slow

		// We're assuming here that there are no tricks in how UVs are used between different meshes
		float2 uv_scale = uv_rect.size();
		for(auto v : vertex_list)
			scene.tex_coords[v] = scene.tex_coords[v] * uv_scale + uv_rect.min();
	}

	// Some textures will be removed: let's reindex the rest
	vector<int> new_texture_index(scene.textures.size(), -1);
	int atlas_index = selection.front();
	SceneTexture atlas_tex;
	atlas_tex.name = name;
	atlas_tex.mips.emplace_back(std::move(tex));
	atlas_tex.is_clamped = true;
	atlas_tex.is_atlas = true;
	atlas_tex.is_opaque = is_opaque;
	scene.textures[atlas_index] = std::move(atlas_tex);
	int num_textures = atlas_index + 1;
	for(int i = 0; i < atlas_index; i++)
		new_texture_index[i] = i;
	for(int i = num_textures; i < scene.textures.size(); i++)
		if(!selected_map[i]) {
			new_texture_index[i] = num_textures;
			if(num_textures != i)
				scene.textures[num_textures] = std::move(scene.textures[i]);
			num_textures++;
		}
	scene.textures.resize(num_textures);

	auto update_tex_index = [&](SceneMaterial::Map &map) {
		auto &tex_id = map.texture_id;
		if(tex_id != -1)
			tex_id = selected_map[tex_id] ? atlas_index : new_texture_index[tex_id];
	};
	for(auto &material : scene.materials)
		for(auto &map : material.maps)
			update_tex_index(map);
}

static void detectClampedTextures(Scene &scene) {
	for(auto &texture : scene.textures)
		texture.is_clamped = true;
	for(auto &material : scene.materials)
		for(auto &map : material.maps)
			map.is_clamped = true;
	DASSERT(scene.hasTexCoords());

	for(auto &mesh : scene.meshes) {
		auto &material = scene.materials[mesh.material_id];
		if(!anyOf(material.maps, [](auto &map) { return bool(map); }))
			continue;

		float2 uv_min(inf), uv_max(-inf);
		for(auto &tri : mesh.tris)
			for(auto v : tri) {
				auto uv = scene.tex_coords[v];
				uv_min = vmin(uv_min, uv);
				uv_max = vmax(uv_max, uv);
			}

		bool clamped_uvs = uv_min.x >= 0.0 && uv_min.y >= 0.0 && uv_max.x <= 1.0 && uv_max.y <= 1.0;
		if(!clamped_uvs) {
			for(auto &map : material.maps)
				if(map) {
					scene.textures[map.texture_id].is_clamped = false;
					map.is_clamped = false;
				}
		}
	}
}

void rescale(Scene &scene, float target_scale = 100.0f) {
	auto box = scene.bounding_box;

	float3 offset = float3(); //-box.center();
	auto max_size = max(box.width(), box.height(), box.depth());
	float scale = target_scale / max_size;
	for(auto &pos : scene.positions)
		pos = (pos + offset) * scale;
	scene.bounding_box = (scene.bounding_box + offset) * scale;
	for(auto &mesh : scene.meshes)
		mesh.bounding_box = (mesh.bounding_box + offset) * scale;
}

void optimizeTriangleOrdering(const int num_verts, CSpan<int> indices, Span<int> out_indices);

Scene convertScene(WavefrontObject obj, const InputScene &iscene) {
	Scene out;

	auto total_time = getTime();
	out.positions = std::move(obj.positions);
	out.normals = std::move(obj.normals);
	out.tex_coords = std::move(obj.tex_coords);
	if(iscene.flip_yz) {
		for(auto &pos : out.positions)
			swap(pos.y, pos.z);
		for(auto &normal : out.normals) {
			swap(normal.y, normal.z);
			normal = -normal;
		}
	}
	if(iscene.flip_uv) {
		for(auto &uv : out.tex_coords)
			uv.y = 1.0f - uv.y;
	}

	vector<string> tex_paths;
	auto loadPbrTex = [&](WavefrontMap &map) {
		string tex_path = FilePath(obj.resource_path) / map.name;
		auto tex = convertTexture(tex_path, SceneMapType::pbr);
		return std::move(tex.mips[0]);
	};

	auto loadTex = [&](WavefrontMap &map, SceneMapType type) {
		SceneMaterial::Map out_map;
		if(!map.name.empty()) {
			string tex_path = FilePath(obj.resource_path) / map.name;
			int index = indexOf(tex_paths, tex_path);
			if(index == -1) {
				index = tex_paths.size();
				tex_paths.emplace_back(tex_path);
				out.textures.emplace_back(convertTexture(tex_path, type));
			}
			out_map.texture_id = index;
			out_map.is_opaque = out.textures[index].is_opaque;
		}
		return out_map;
	};

	out.materials.reserve(obj.materials.size());
	for(auto &mtl : obj.materials) {
		SceneMaterial mat;
		mat.name = mtl.name;
		mat.diffuse = mtl.diffuse;
		mat.opacity = mtl.dissolve_factor;
		WavefrontMap *pbr_ao = nullptr, *pbr_roughness = nullptr, *pbr_metallic = nullptr;

		for(auto &map : mtl.maps) {
			if(map.first == "kd")
				mat.maps[SceneMapType::albedo] = loadTex(map.second, SceneMapType::albedo);
			if(iscene.pbr) {
				if(map.first == "ao")
					pbr_ao = &map.second;
				if(map.first == "roughness")
					pbr_roughness = &map.second;
				if(map.first == "ks")
					pbr_metallic = &map.second;
				if(map.first == "bump") {
					// TODO: it can be normal or bump map
					mat.maps[SceneMapType::normal] = loadTex(map.second, SceneMapType::normal);
				}
			}
		}

		if(pbr_ao && pbr_roughness && pbr_metallic) {
			auto ao_tex = loadPbrTex(*pbr_ao);
			auto roughness_tex = loadPbrTex(*pbr_roughness);
			auto metallic_tex = loadPbrTex(*pbr_metallic);

			if(ao_tex.size() != roughness_tex.size() || ao_tex.size() != metallic_tex.size()) {
				print("Warning: invalid sizes of PBR textures for material: '%'\n"
					  "  AO: size:% '%'\n  roughness size:% '%'\n  metallic size:% '%'\n"
					  "  Skipping...\n",
					  mat.name, ao_tex.size(), pbr_ao->name, roughness_tex.size(),
					  pbr_roughness->name, metallic_tex.size(), pbr_metallic->name);
			} else {
				Image pbr_image(ao_tex.size());
				auto pbr_pixels = pbr_image.pixels<IColor>();
				auto ao_pixels = ao_tex.pixels<IColor>();
				auto roughness_pixels = roughness_tex.pixels<IColor>();
				auto metallic_pixels = metallic_tex.pixels<IColor>();
				for(int y = 0; y < pbr_image.height(); y++) {
					for(int x = 0; x < pbr_image.width(); x++) {
						pbr_pixels(x, y) = IColor(roughness_pixels(x, y).r, metallic_pixels(x, y).r,
												  ao_pixels(x, y).r);
					}
				}
				SceneTexture pbr_tex;
				pbr_tex.name = format("pbr_%", mat.name);
				pbr_tex.mips.emplace_back(std::move(pbr_image));
				pbr_tex.is_opaque = true;
				pbr_tex.map_type = SceneMapType::pbr;

				auto &map = mat.maps[SceneMapType::pbr];
				map.is_opaque = true;
				map.texture_id = out.textures.size();

				out.textures.emplace_back(std::move(pbr_tex));
			}
		}

		out.materials.emplace_back(std::move(mat));
	}

	out.meshes.reserve(obj.material_groups.size());
	for(auto &mtl_group : obj.material_groups) {
		SceneMesh mesh;
		mesh.material_id = mtl_group.material_id;
		mesh.tris =
			cspan(obj.tris).subSpan(mtl_group.first_tri, mtl_group.first_tri + mtl_group.num_tris);
		if(!mesh.tris)
			continue;
		out.meshes.emplace_back(mesh);
	}

	if(iscene.merge_verts && !out.tex_coords && !out.colors) {
		print("Merging duplicate vertices...\n");
		int old_count = out.numVerts();
		out.mergeVertices(5);
		int new_count = out.numVerts();
		if(old_count != new_count)
			print("Merged vertices: % -> %\n", old_count, new_count);
		// TODO: move bbox computaiton to separate function
	}

	print("Optimizing triangle order ");
	fflush(stdout);
	double time = getTime();
	for(auto &mesh : out.meshes) {
		// TODO: might be slow in case of large number of meshes
		auto old = mesh.tris;
		auto indices = span(mesh.tris).reinterpret<int>();
		optimizeTriangleOrdering(out.numVerts(), indices, indices);
		for(auto &idx : indices)
			DASSERT(idx >= 0 && idx < out.numVerts());
		print(".");
		fflush(stdout);
	}
	printf(" (%.2f sec)\n", getTime() - time);

	print("Computing bounding boxes...\n");
	out.bounding_box = enclose(out.positions);
	for(auto &mesh : out.meshes) {
		auto bmin = out.positions[mesh.tris[0][0]];
		auto bmax = bmin;
		for(auto &tri : mesh.tris)
			for(auto idx : tri) {
				auto point = out.positions[idx];
				bmin = vmin(bmin, point);
				bmax = vmax(bmax, point);
			}
		mesh.bounding_box = {bmin, bmax};
	}

	print("Compressing & generating mipmaps for textures...\n");
	if(out.hasTexCoords() && !iscene.pbr) {
		detectClampedTextures(out);
		makeTextureAtlas(out, format("%_opaque", iscene.name), true);
		makeTextureAtlas(out, format("%_transparent", iscene.name), false);
	}
	int max_atlas_mips = 6; // For atlas, other textures can have more ?

	for(auto &texture : out.textures) {
		if(texture.map_type != SceneMapType::albedo)
			continue; // TODO: compress normals & pbr

		auto size = texture.mips[0].size();
		print("- Processing texture: %\n", size);
		int num_mips = Image::maxMipmapLevels(size);
		if(texture.is_atlas)
			num_mips = min(num_mips, max_atlas_mips);
		texture.mips.reserve(num_mips);

		for(int i : intRange(num_mips - 1)) {
			auto &prev_mip = texture.mips[i];
			int2 mip_size(max(1, prev_mip.width() / 2), max(1, prev_mip.height() / 2));
			texture.mips.emplace_back(prev_mip.rescale(mip_size, ImageRescaleOpt::srgb));
		}

		auto format = texture.is_opaque ? VColorFormat::bc1_rgb_srgb : VColorFormat::bc3_rgba_srgb;
		for(auto &mip : texture.mips)
			mip = Image::compressBC(mip, format);
	}

	print("Generating quads...\n");
	out.generateQuads(iscene.quad_squareness);

	if(iscene.pbr)
		out.computeTangents();
	out.quantizeVectors();
	rescale(out);

	total_time = getTime() - total_time;
	print("Conversion finished in % seconds\n"
		  "- materials:% textures:% meshes:% tris:% quads:% verts:%\n\n",
		  total_time, out.materials.size(), out.textures.size(), out.meshes.size(), out.numTris(),
		  out.numQuads(), out.numVerts());

	return out;
}

void processNode(Scene &out, const aiScene *ai_scene, const aiNode *ai_node, aiMatrix4x4 trans) {
	trans *= ai_node->mTransformation;

	for(int m = 0; m < ai_node->mNumMeshes; m++) {
		auto *ai_mesh = ai_scene->mMeshes[ai_node->mMeshes[m]];
		if(!ai_mesh->HasPositions())
			continue;

		int vertex_offset = out.positions.size();
		out.positions.resize(vertex_offset + ai_mesh->mNumVertices);
		for(int i = 0; i < ai_mesh->mNumVertices; i++) {
			auto ai_vertex = ai_mesh->mVertices[i];
			ai_vertex *= trans;
			out.positions[vertex_offset + i] = {ai_vertex.x, ai_vertex.y, ai_vertex.z};
		}

		SceneMesh mesh;
		mesh.tris.resize(ai_mesh->mNumFaces);
		for(int i = 0; i < ai_mesh->mNumFaces; i++) {
			auto &ai_face = ai_mesh->mFaces[i];
			DASSERT(ai_face.mNumIndices == 3);
			mesh.tris[i] = {int(ai_face.mIndices[0]) + vertex_offset,
							int(ai_face.mIndices[1]) + vertex_offset,
							int(ai_face.mIndices[2]) + vertex_offset};
		}
		mesh.material_id = ai_mesh->mMaterialIndex;
		out.meshes.emplace_back(std::move(mesh));
	}

	for(int i = 0; i < ai_node->mNumChildren; i++)
		processNode(out, ai_scene, ai_node->mChildren[i], trans);
}

Ex<Scene> convertScene(const InputScene &iscene) {
	Scene out;

	auto start_time = getTime();
	Assimp::Importer importer;
	// TODO: import quads directly if possible
	auto import_flags = aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_GenNormals;
	const aiScene *ai_scene = importer.ReadFile(iscene.path, import_flags);
	EXPECT(ai_scene && ai_scene->mRootNode);
	EXPECT(!(ai_scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE));

	// TODO: handle instancing?
	processNode(out, ai_scene, ai_scene->mRootNode, aiMatrix4x4());

	if(iscene.flip_yz) {
		for(auto &pos : out.positions)
			swap(pos.y, pos.z);
		for(auto &normal : out.normals) {
			swap(normal.y, normal.z);
			normal = -normal;
		}
	}
	if(iscene.flip_uv) {
		for(auto &uv : out.tex_coords)
			uv.y = 1.0f - uv.y;
	}

	for(int i = 0; i < ai_scene->mNumMaterials; i++) {
		auto *ai_material = ai_scene->mMaterials[i];
		SceneMaterial material(ai_material->GetName().C_Str());

		float opacity = 1.0f;
		ai_material->Get(AI_MATKEY_OPACITY, opacity);

		aiColor4D color(1.0f, 1.0f, 1.0f, 1.0f);
		ai_material->Get(AI_MATKEY_COLOR_DIFFUSE, color);

		for(int j = 0; j < ai_material->mNumProperties; j++) {
			print("prop: %\n", ai_material->mProperties[j]->mKey.C_Str());
		}

		material.opacity = opacity;
		material.diffuse = {color.r, color.g, color.b};

		out.materials.emplace_back(material);
	}

	if(iscene.merge_verts && !out.tex_coords && !out.colors) {
		print("Merging duplicate vertices...\n");
		int old_count = out.numVerts();
		out.mergeVertices(5);
		int new_count = out.numVerts();
		if(old_count != new_count)
			print("Merged vertices: % -> %\n", old_count, new_count);
		// TODO: move bbox computaiton to separate function
	}

	print("Optimizing triangle order ");
	fflush(stdout);
	double time = getTime();
	for(auto &mesh : out.meshes) {
		// TODO: might be slow in case of large number of meshes
		auto old = mesh.tris;
		auto indices = span(mesh.tris).reinterpret<int>();
		optimizeTriangleOrdering(out.numVerts(), indices, indices);
		for(auto &idx : indices)
			DASSERT(idx >= 0 && idx < out.numVerts());
		print(".");
		fflush(stdout);
	}
	printf(" (%.2f sec)\n", getTime() - time);

	print("Computing bounding boxes...\n");
	out.bounding_box = enclose(out.positions);
	for(auto &mesh : out.meshes) {
		auto bmin = out.positions[mesh.tris[0][0]];
		auto bmax = bmin;
		for(auto &tri : mesh.tris)
			for(auto idx : tri) {
				auto point = out.positions[idx];
				bmin = vmin(bmin, point);
				bmax = vmax(bmax, point);
			}
		mesh.bounding_box = {bmin, bmax};
	}

	print("Compressing & generating mipmaps for textures...\n");
	if(out.hasTexCoords() && !iscene.pbr) {
		detectClampedTextures(out);
		makeTextureAtlas(out, format("%_opaque", iscene.name), true);
		makeTextureAtlas(out, format("%_transparent", iscene.name), false);
	}
	int max_atlas_mips = 6; // For atlas, other textures can have more ?

	for(auto &texture : out.textures) {
		if(texture.map_type != SceneMapType::albedo)
			continue; // TODO: compress normals & pbr

		auto size = texture.mips[0].size();
		print("- Processing texture: %\n", size);
		int num_mips = Image::maxMipmapLevels(size);
		if(texture.is_atlas)
			num_mips = min(num_mips, max_atlas_mips);
		texture.mips.reserve(num_mips);

		for(int i : intRange(num_mips - 1)) {
			auto &prev_mip = texture.mips[i];
			int2 mip_size(max(1, prev_mip.width() / 2), max(1, prev_mip.height() / 2));
			texture.mips.emplace_back(prev_mip.rescale(mip_size, ImageRescaleOpt::srgb));
		}

		auto format = texture.is_opaque ? VColorFormat::bc1_rgb_srgb : VColorFormat::bc3_rgba_srgb;
		for(auto &mip : texture.mips)
			mip = Image::compressBC(mip, format);
	}

	print("Generating quads...\n");
	out.generateQuads(iscene.quad_squareness);

	if(iscene.pbr)
		out.computeTangents();
	out.quantizeVectors();
	rescale(out);

	auto total_time = getTime() - start_time;
	print("Conversion finished in % seconds\n"
		  "- materials:% textures:% meshes:% tris:% quads:% verts:%\n\n",
		  total_time, out.materials.size(), out.textures.size(), out.meshes.size(), out.numTris(),
		  out.numQuads(), out.numVerts());

	return out;
}

void convertScenes(ZStr iscenes_path, vector<string> scenes_selection) {
	auto input_scenes = loadInputScenes(iscenes_path);
	if(!input_scenes) {
		input_scenes.error().print();
		return;
	}

	int num_converted = 0, num_failed = 0;
	auto scenes_path = mainPath() / "scenes";
	mkdirRecursive(scenes_path).check();

	for(auto iscene : *input_scenes) {
		if(scenes_selection && !isOneOf(iscene.name, scenes_selection))
			continue;

		auto dst_path = format("%/%.scene", scenes_path, iscene.name);
		print("*** Converting: % -> %\n", iscene.path, dst_path);
		auto time = getTime();

		/*
		auto wavefront_obj = WavefrontObject::load(iscene.path);
		if(!wavefront_obj) {
			print("  error while loading:\n");
			wavefront_obj.error().print();
			num_failed++;
			continue;
		}*/
		print("  loaded in % msec\n", int((getTime() - time) * 1000.0));

		auto scene = convertScene(iscene);
		if(!scene) {
			print("  error while loading:\n");
			scene.error().print();
			num_failed++;
			continue;
		}

		//auto scene = convertScene(std::move(*wavefront_obj), iscene);
		auto saver = fileSaver(dst_path);
		if(!saver) {
			print("  error while saving\n");
			saver.error().print();
			num_failed++;
			continue;
		}

		auto result = scene->save(*saver);
		if(!result) {
			print("  error while saving\n");
			result.error().print();
			num_failed++;
			continue;
		}
		num_converted++;
	}

	print("Converted succesfully % scenes; failures: %\n", num_converted, num_failed);
}
