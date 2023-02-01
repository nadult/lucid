#include "scene.h"

#include "texture_atlas.h"
#include "wavefront_obj.h"
#include <fwk/gfx/compressed_image.h>
#include <fwk/gfx/float_image.h>
#include <fwk/gfx/image.h>
#include <fwk/io/file_stream.h>

SceneTexture convertTexture(ZStr path) {
	auto time = getTime();
	SceneTexture out;
	// TODO: what about duplicate names ?
	out.name = FilePath(path).fileStem();
	auto tex = Image::load(path);
	if(!tex) {
		tex.error().print();
		tex = Image({32, 32}, ColorId::purple);
		out.name += "_invalid";
	}
	out.is_opaque = tex->isOpaque();
	out.plain_mips.emplace_back(move(*tex));

	print("  loaded % (size:% opaque:%) in % msec\n", out.name, tex->size(), out.is_opaque,
		  int((getTime() - time) * 1000.0));

	return out;
}

static void makeTextureAtlas(Scene &scene, string name, bool is_opaque) {
	vector<int> selection;
	vector<int2> sizes;
	vector<const Image *> pointers;
	vector<char> selected_map(scene.textures.size(), false);
	vector<int> inverse_map(scene.textures.size(), -1);

	for(int i : intRange(scene.textures)) {
		auto &tex = scene.textures[i];
		if(tex.is_opaque == is_opaque) {
			selection.emplace_back(i);
			inverse_map[i] = sizes.size();
			sizes.emplace_back(tex.plain_mips[0].size());
			pointers.emplace_back(&tex.plain_mips[0]);
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
		if(!material.diffuse_tex || !selected_map[material.diffuse_tex.id])
			continue;
		auto &tex = scene.textures[material.diffuse_tex.id];
		int entry_index = inverse_map[material.diffuse_tex.id];
		auto &entry = atlas->entries[entry_index];
		auto uv_rect = atlas->uvRect(entry);

		if(!tex.is_clamped) {
			// We will have to transform UV coords in the shader
			// We're decreasing uv_rect by half a pixel from each side to
			// make sure that it won't be interpolated with neighbouring texels
			// TODO: add repeated texture border? What about textures with pow of 2 size?
			material.diffuse_tex.uv_rect = atlas->uvRect(entry, 0.5f);
			continue;
		}

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
	atlas_tex.plain_mips.emplace_back(move(tex));
	atlas_tex.is_clamped = true;
	atlas_tex.is_atlas = true;
	atlas_tex.is_opaque = is_opaque;
	scene.textures[atlas_index] = move(atlas_tex);
	int num_textures = atlas_index + 1;
	for(int i = 0; i < atlas_index; i++)
		new_texture_index[i] = i;
	for(int i = num_textures; i < scene.textures.size(); i++)
		if(!selected_map[i]) {
			new_texture_index[i] = num_textures;
			if(num_textures != i)
				scene.textures[num_textures] = move(scene.textures[i]);
			num_textures++;
		}
	scene.textures.resize(num_textures);

	auto update_tex_index = [&](SceneMaterial::UsedTexture &tex) {
		if(tex.id != -1)
			tex.id = selected_map[tex.id] ? atlas_index : new_texture_index[tex.id];
	};
	for(auto &material : scene.materials) {
		update_tex_index(material.diffuse_tex);
		update_tex_index(material.normal_tex);
	}
}

static void detectClampedTextures(Scene &scene) {
	for(auto &texture : scene.textures)
		texture.is_clamped = true;
	for(auto &material : scene.materials) {
		material.diffuse_tex.is_clamped = true;
		material.normal_tex.is_clamped = true;
	}
	DASSERT(scene.hasTexCoords());

	for(auto &mesh : scene.meshes) {
		auto &material = scene.materials[mesh.material_id];
		if((!material.diffuse_tex && !material.normal_tex))
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
			if(material.diffuse_tex) {
				scene.textures[material.diffuse_tex.id].is_clamped = false;
				material.diffuse_tex.is_clamped = false;
			}
			if(material.normal_tex) {
				scene.textures[material.normal_tex.id].is_clamped = false;
				material.normal_tex.is_clamped = false;
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

Scene convertScene(WavefrontObject obj, bool flip_yz, Str scene_name, float squareness,
				   bool merge_verts) {
	Scene out;
	out.positions = move(obj.positions);
	out.normals = move(obj.normals);
	out.tex_coords = move(obj.tex_coords);
	if(flip_yz) {
		for(auto &pos : out.positions)
			swap(pos.y, pos.z);
		for(auto &normal : out.normals) {
			swap(normal.y, normal.z);
			normal = -normal;
		}
	}

	vector<string> tex_paths;
	auto loadTex = [&](WavefrontMap &map) {
		SceneMaterial::UsedTexture used_tex;
		if(!map.name.empty()) {
			string tex_path = FilePath(obj.resource_path) / map.name;
			int index = indexOf(tex_paths, tex_path);
			if(index == -1) {
				index = tex_paths.size();
				tex_paths.emplace_back(tex_path);
				out.textures.emplace_back(convertTexture(tex_path));
			}
			used_tex.id = index;
			used_tex.is_opaque = out.textures[index].is_opaque;
		}
		return used_tex;
	};

	out.materials.reserve(obj.materials.size());
	for(auto &mtl : obj.materials) {
		SceneMaterial mat;
		mat.name = mtl.name;
		mat.diffuse = mtl.diffuse;
		mat.opacity = mtl.dissolve_factor;
		mat.diffuse_tex = loadTex(mtl.maps[WavefrontMapType::diffuse]);
		// TODO: atlas support for other texture types
		//mat.normal_tex = loadTex(mtl.maps[WavefrontMapType::bump]);
		out.materials.emplace_back(move(mat));
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

	if(merge_verts && !out.tex_coords && !out.colors) {
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
	if(out.hasTexCoords()) {
		detectClampedTextures(out);

		makeTextureAtlas(out, format("%_opaque", scene_name), true);
		makeTextureAtlas(out, format("%_transparent", scene_name), false);
	}
	int max_atlas_mips = 6; // For atlas, other textures can have more ?

	for(auto &texture : out.textures) {
		print("- Processing texture: %\n", texture.plain_mips[0].size());
		auto &plain_tex = texture.plain_mips[0];
		auto format = texture.is_opaque ? VBlockFormat::srgb_bc1 : VBlockFormat::srgba_bc3;
		texture.block_mips.emplace_back(plain_tex, format);
		auto size = plain_tex.size();

		int max_mips = Image::maxMipmapLevels(size) - 1;
		if(texture.is_atlas)
			max_mips = min(max_mips, max_atlas_mips);

		for(int i : intRange(max_mips)) {
			int2 dsize(max(1, size.x / 2), max(1, size.y / 2));
			auto downsized_tex = plain_tex.rescale(dsize, ImageRescaleOpt::srgb);
			texture.block_mips.emplace_back(downsized_tex, format);
			swap(plain_tex, downsized_tex);
			size = dsize;
		}
		texture.plain_mips.clear();
	}

	print("Generating quads...\n");
	out.generateQuads(squareness);
	out.quantizeNormals();

	rescale(out);

	return out;
}

struct SceneInfo {
	ZStr path;
	float quad_squareness;
	bool merge_verts = false;
};

void convertScenes(ZStr source_path) {
	SceneInfo inputs[] = {
		{"dragon2.obj", 1.5, true},
		{"armadillo.obj", 1.5, true},
		{"thai.obj", 1.5, true},
		{"buddha.obj", 1.5, true},
		{"bunny.obj", 1.5},
		{"chestnut_tree/chestnut_tree01.obj", 2},
		{"chestnut_tree/chestnut_tree02.obj", 2},
		{"chestnut_tree/chestnut_tree03.obj", 2},
		{"conference/conference.obj", 3},
		{"dragon.obj", 1},
		{"gallery/gallery.obj", 0.5},
		{"hairball.obj", 1000},
		{"powerplant/powerplant.obj", 2},
		{"san_miguel/san-miguel.obj", 1},
		{"pine_tree/scrubPine.obj", 2},
		{"sponza/sponza.obj", 2},
		{"teapot/teapot.obj", 2},
		{"white_oak/white_oak.obj", 0.5},
	};

	int num_converted = 0, num_failed = 0;

	auto scenes_path = executablePath().parent() / "scenes";
	mkdirRecursive(scenes_path).check();

	for(auto [ipath, isquareness, imerge_verts] : inputs) {
		auto src_path = FilePath(source_path) / ipath;
		auto scene_name = src_path.fileStem();
		auto dst_path = format("%/%.scene", scenes_path, scene_name);
		print("*** Converting: % -> %\n", src_path, dst_path);
		auto time = getTime();
		auto wavefront_obj = WavefrontObject::load(src_path);
		if(!wavefront_obj) {
			print("  error while loading:\n");
			wavefront_obj.error().print();
			num_failed++;
			continue;
		}
		print("  loaded in % msec\n", int((getTime() - time) * 1000.0));

		bool flip_yz = ipath.find("chestnut_tree") != -1;
		auto scene =
			convertScene(move(*wavefront_obj), flip_yz, scene_name, isquareness, imerge_verts);

		print("  materials:% textures:% meshes:% tris:% quads:% verts:%\n\n",
			  scene.materials.size(), scene.textures.size(), scene.meshes.size(), scene.numTris(),
			  scene.numQuads(), scene.numVerts());
		time = getTime();

		auto saver = fileSaver(dst_path);
		if(!saver) {
			print("  error while saving\n");
			saver.error().print();
			num_failed++;
			continue;
		}

		auto result = scene.save(*saver);
		if(!result) {
			print("  error while saving\n");
			result.error().print();
			num_failed++;
			continue;
		}
		print("  saved in % seconds\n", getTime() - time);
		num_converted++;
	}

	print("Converted succesfully % scenes; failures: %\n", num_converted, num_failed);
}
