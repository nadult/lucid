// Copyright (C) Krzysztof Jakubowski <nadult@fastmail.fm>
// This file is part of LucidRaster. See license.txt for details.

#include "scene.h"

#include "quad_generator.h"
#include "shading.h"
#include <fwk/gfx/image.h>
#include <fwk/io/file_stream.h>
#include <fwk/math/ray.h>
#include <fwk/math/segment.h>
#include <fwk/vulkan/vulkan_buffer.h>
#include <fwk/vulkan/vulkan_device.h>
#include <fwk/vulkan/vulkan_image.h>
#include <fwk/vulkan/vulkan_pipeline.h>

Ex<void> SceneMesh::load(Stream &stream) {
	stream >> material_id >> colors_opaque >> tris >> quads >> num_degenerate_quads >> bounding_box;
	return stream.getValid();
}

Ex<void> SceneMesh::save(Stream &stream) const {
	stream << material_id << colors_opaque << tris << quads << num_degenerate_quads << bounding_box;
	return stream.getValid();
}

SceneMaterial::SceneMaterial(string name) : name(move(name)) {}
SceneMaterial::~SceneMaterial() = default;

Ex<void> SceneMaterial::load(Stream &sr) {
	sr >> name >> diffuse >> opacity;
	for(auto &map : maps)
		sr.unpack(map.texture_id, map.is_opaque, map.is_clamped, map.uv_rect);
	return sr.getValid();
}

Ex<void> SceneMaterial::save(Stream &sr) const {
	sr << name << diffuse << opacity;
	for(auto &map : maps)
		sr.pack(map.texture_id, map.is_opaque, map.is_clamped, map.uv_rect);
	return sr.getValid();
}

bool SceneMaterial::isOpaque() const {
	auto &albedo = maps[SceneMapType::albedo];
	return opacity == 1.0 && (albedo.texture_id == -1 || albedo.is_opaque);
}

void SceneMaterial::freeTextures() {
	for(auto &map : maps)
		map.vk_image = {};
}

SceneTexture::SceneTexture() = default;
FWK_COPYABLE_CLASS_IMPL(SceneTexture);

Ex<void> SceneTexture::load(Stream &sr) {
	u8 num_levels;
	VFormat format;
	sr >> name;
	sr.unpack(map_type, is_opaque, is_clamped, is_atlas, num_levels, format);
	mips.reserve(num_levels);
	// TODO: add serializing function for generic image?
	// TODO: verify mip sizes
	for(int i : intRange(num_levels)) {
		int2 size;
		int byte_size;
		sr.unpack(size, byte_size);
		EXPECT(size.x > 0 && size.x <= VulkanLimits::max_image_size);
		EXPECT(size.y > 0 && size.y <= VulkanLimits::max_image_size);
		EXPECT(byte_size > 0);
		EXPECT(byte_size == imageByteSize(format, size));
		if(!sr.addResources(byte_size))
			return sr.getValid();
		PodVector<u8> data(byte_size);
		sr.loadData(data);
		mips.emplace_back(std::move(data), size, format);
	}
	return sr.getValid();
}

Ex<void> SceneTexture::save(Stream &sr) const {
	DASSERT(!empty());
	sr << name;
	u8 num_levels = mips.size();
	sr.pack(map_type, is_opaque, is_clamped, is_atlas, num_levels, format());
	for(auto &mip : mips) {
		sr.pack(mip.size(), mip.data().size());
		sr.saveData(mip.data());
	}
	return sr.getValid();
}

Ex<void> SceneTexture::loadPlain(ZStr path) {
	*this = SceneTexture{};
	auto time = getTime();
	auto tex = EX_PASS(Image::load(path));
	is_opaque = allOf(tex.pixels<IColor>(), [](IColor col) { return col.a == 255; });
	mips.emplace_back(std::move(tex));
	print("Loaded texture '%' in % ms\n", path, (getTime() - time) * 1000.0);
	return {};
}

VFormat SceneTexture::format() const { return mips ? mips[0].format() : VFormat::rgba8_unorm; }
int2 SceneTexture::size() const { return mips ? mips[0].size() : int2(); }

Scene::Scene() = default;
FWK_MOVABLE_CLASS_IMPL(Scene);

Ex<Scene> Scene::load(ZStr path) {
	Scene scene;
	auto loader = EX_PASS(fileLoader(path));
	EXPECT(scene.load(loader));
	return scene;
}

Ex<void> Scene::load(Stream &stream) {
	EXPECT(stream.loadSignature("SCENE"));
	int num_meshes, num_materials, num_textures;
	stream.unpack(num_meshes, num_materials, num_textures);

	// Czemu nie zwracam po prostu expected za każdym razem ?
	EXPECT(num_meshes > 0 && num_materials > 0);
	// TODO: resource limits here

	stream >> positions >> colors >> tex_coords;
	stream >> normals >> tangents >> quantized_normals >> quantized_tangents;
	stream >> bounding_box;

	meshes.resize(num_meshes);
	for(auto &mesh : meshes) {
		EXPECT(mesh.load(stream));
		EXPECT(mesh.material_id >= 0 && mesh.material_id < num_materials);
	}

	materials.resize(num_materials);
	for(auto &material : materials) {
		EXPECT(material.load(stream));
		for(auto &map : material.maps)
			if(map.texture_id != -1)
				EXPECT(map.texture_id >= 0 && map.texture_id < num_textures);
	}

	textures.resize(num_textures);
	for(auto &texture : textures)
		EXPECT(texture.load(stream));

	return stream.getValid();
}

Ex<void> Scene::save(Stream &stream) const {
	stream.saveSignature("SCENE");
	stream.pack(meshes.size(), materials.size(), textures.size());
	stream << positions << colors << tex_coords;
	stream << normals << tangents << quantized_normals << quantized_tangents;
	stream << bounding_box;

	for(auto &mesh : meshes)
		EXPECT(mesh.save(stream));
	for(auto &material : materials)
		EXPECT(material.save(stream));
	for(auto &texture : textures)
		EXPECT(texture.save(stream));
	return stream.getValid();
}

void Scene::mergeVertices(int decimal_places) {
	// TODO: add support for colors & tex_coords
	DASSERT(tex_coords.empty());
	DASSERT(colors.empty());

	float scale = pow(10.0, decimal_places);
	float inv_scale = 1.0 / scale;

	struct NewVertex {
		float3 position;
		float3 normal;
		int count;
	};

	vector<NewVertex> new_vertices;
	PodVector<int> new_indices(positions.size());
	HashMap<float3, int> map;

	map.reserve(positions.size() * 3 / 2);
	new_vertices.reserve(positions.size());

	for(int i : intRange(positions)) {
		auto pos = positions[i];
		auto normal = normals.empty() ? float3(0, 1, 0) : normals[i];

		auto quantized_pos =
			float3(round(pos.x * scale), round(pos.y * scale), round(pos.z * scale));
		auto [it, added] = map.emplace(quantized_pos, -1);
		if(it->value == -1) {
			it->value = new_vertices.size();
			new_vertices.emplace_back(pos, normal, 1);
		} else {
			auto &ref = new_vertices[it->value];
			ref.position += pos;
			ref.normal += normal;
			ref.count++;
		}
		new_indices[i] = it->value;
	}

	if(new_vertices.size() == positions.size())
		return;

	{
		auto new_positions = transform(new_vertices, [](auto &new_vert) {
			return new_vert.position * (float(1.0) / float(new_vert.count));
		});
		positions.swap(new_positions);
	}

	if(normals) {
		vector<float3> new_normals;
		new_normals = transform(new_vertices, [](auto &new_vert) {
			return new_vert.normal * (float(1.0) / float(new_vert.count));
		});
		normals.swap(new_normals);
	}

	for(auto &mesh : meshes) {
		for(auto &tri : mesh.tris)
			for(auto &idx : tri)
				idx = new_indices[idx];
		for(auto &quad : mesh.quads)
			for(auto &idx : quad)
				idx = new_indices[idx];
	}
}

void Scene::generateQuads(float squareness) {
	for(auto &mesh : meshes) {
		auto tri_neighbours = triNeighbours(mesh.tris);
		auto [quad_nodes, tri_quads] = quadNodes(positions, mesh.tris, tri_neighbours);
		mesh.quads = genQuads(mesh.tris, tri_neighbours, quad_nodes, tri_quads, squareness);
		mesh.num_degenerate_quads = 0;
		for(auto &quad : mesh.quads)
			if(quad[2] == quad[3])
				mesh.num_degenerate_quads++;
	}
}

Maybe<Scene::Intersection> Scene::intersect(Segment3F segment) const {
	Intersection best = {-1, -1, {}};
	float best_dist = inf;
	Ray3F ray(segment.from, normalize(segment.to - segment.from));

	for(int m : intRange(meshes)) {
		auto &mesh = meshes[m];
		auto iparam = ray.isectParam(mesh.bounding_box);
		if(!iparam.valid() && iparam.closest() >= best_dist)
			continue;
		for(int t : intRange(mesh.tris)) {
			auto &itri = mesh.tris[t];
			Triangle3F tri(positions[itri[0]], positions[itri[1]], positions[itri[2]]);
			iparam = ray.isectParam(tri);
			if(iparam.valid() && iparam.closest() < best_dist) {
				best_dist = iparam.closest();
				best = {m, t, ray.at(best_dist)};
			}
		}
	}

	return best.mesh_id == -1 ? Maybe<Intersection>() : best;
}

void Scene::updatePrimitiveOffsets() {
	mesh_primitive_offsets.resize(meshes.size());
	int tri_off = 0, quad_off = 0;
	for(int i : intRange(meshes)) {
		auto &mesh = meshes[i];
		mesh_primitive_offsets[i] = {tri_off, quad_off};
		tri_off += mesh.tris.size();
		quad_off += mesh.quads.size();
	}
}

Ex<void> Scene::updateRenderingData(VulkanDevice &device) {
	updatePrimitiveOffsets();
	freeRenderingData();

	auto accel_struct_usage =
		mask(device.features() & VDeviceFeature::ray_tracing,
			 VBufferUsage::device_address | VBufferUsage::accel_struct_build_input_read_only);
	auto buf_usage = VBufferUsage::storage_buffer | VBufferUsage::transfer_dst;
	auto vb_usage = buf_usage | VBufferUsage::vertex_buffer | accel_struct_usage;
	auto ib_usage = buf_usage | VBufferUsage::index_buffer | accel_struct_usage;

	verts.positions = EX_PASS(VulkanBuffer::createAndUpload(device, positions, vb_usage));

	if(hasColors())
		verts.colors = EX_PASS(VulkanBuffer::createAndUpload(device, colors, vb_usage));
	if(hasTexCoords())
		verts.tex_coords = EX_PASS(VulkanBuffer::createAndUpload(device, tex_coords, vb_usage));
	if(hasQuantizedNormals())
		verts.normals = EX_PASS(VulkanBuffer::createAndUpload(device, quantized_normals, vb_usage));
	if(quantized_tangents)
		verts.tangents =
			EX_PASS(VulkanBuffer::createAndUpload(device, quantized_tangents, vb_usage));

	// TODO: merging shouldn't be needed; Add option to efficiently upload multiple spans?
	vector<u32> merged_tris, merged_quads;
	for(auto &mesh : meshes) {
		insertBack(merged_tris, cspan(mesh.tris).reinterpret<u32>());
		insertBack(merged_quads, cspan(mesh.quads).reinterpret<u32>());
	}
	tris_ib = EX_PASS(VulkanBuffer::createAndUpload(device, merged_tris, ib_usage));
	quads_ib = EX_PASS(VulkanBuffer::createAndUpload(device, merged_quads, ib_usage));

	for(auto &material : materials)
		for(auto map_type : all<SceneMapType>) {
			auto &map = material.maps[map_type];
			if(map.texture_id == -1)
				continue;

			bool is_srgb = map_type == SceneMapType::albedo;
			auto &tex = textures[map.texture_id];
			// TODO: makes no sense to keep both in main & gpu memory, especially because ogl is caching it too
			// TODO: mipmaps for plain textures
			if(!tex.vk_image) {
				DASSERT(!tex.empty());
				auto vk_image = EX_PASS(VulkanImage::createAndUpload(device, tex.mips));
				tex.vk_image = VulkanImageView::create(vk_image);
			}
			map.vk_image = tex.vk_image;
			map.is_opaque = tex.is_opaque;
		}

	return {};
}

u32 encodeNormalUint(const float3 &n) {
	uint x = uint(512.0 + n.x * 511.0) & 0x3ffu;
	uint y = uint(512.0 + n.y * 511.0) & 0x3ffu;
	uint z = uint(512.0 + n.z * 511.0) & 0x3ffu;
	return x | (y << 10) | (z << 20);
}

void Scene::computeTangents() {
	if(!hasTexCoords() || !hasNormals())
		return;

	tangents.clear();
	quantized_normals.clear();
	quantized_tangents.clear();
	tangents.resize(positions.size());

	for(auto &mesh : meshes) {
		for(auto &tri : mesh.tris) {
			auto v0 = positions[tri[0]];
			auto v1 = positions[tri[1]];
			auto v2 = positions[tri[2]];

			auto uv0 = tex_coords[tri[0]];
			auto uv1 = tex_coords[tri[1]];
			auto uv2 = tex_coords[tri[2]];

			auto edge0 = v1 - v0, edge1 = v2 - v0;
			auto delta_uv0 = uv1 - uv0, delta_uv1 = uv2 - uv0;
			float scale = 1.0f / (delta_uv0.x * delta_uv1.y - delta_uv1.x * delta_uv0.y);
			auto tangent = (delta_uv1.y * edge0 - delta_uv0.y * edge1) * scale;
			for(int i = 0; i < 3; i++)
				tangents[tri[i]] += tangent;
		}
	}

	for(auto &tangent : tangents)
		tangent = normalize(tangent);
}

void Scene::computeFlatVectors() {
	if(!hasTexCoords())
		return;

	normals.clear();
	tangents.clear();
	quantized_normals.clear();
	quantized_tangents.clear();
	normals.resize(positions.size());
	tangents.resize(positions.size());

	vector<bool> taken(positions.size(), false);

	int num_duplicated = 0;
	for(auto &mesh : meshes) {
		for(auto &tri : mesh.tris) {
			auto v0 = positions[tri[0]];
			auto v1 = positions[tri[1]];
			auto v2 = positions[tri[2]];

			auto uv0 = tex_coords[tri[0]];
			auto uv1 = tex_coords[tri[1]];
			auto uv2 = tex_coords[tri[2]];

			auto edge0 = v1 - v0, edge1 = v2 - v0;
			auto delta_uv0 = uv1 - uv0, delta_uv1 = uv2 - uv0;
			float scale = 1.0f / (delta_uv0.x * delta_uv1.y - delta_uv1.x * delta_uv0.y);
			auto normal = normalize(cross(edge0, edge1));
			auto tangent = (delta_uv1.y * edge0 - delta_uv0.y * edge1) * scale;

			int provoking_idx = -1;
			for(int i : intRange(3)) {
				auto idx = tri[i];
				if(!taken[idx]) {
					provoking_idx = i;
					taken[idx] = true;
					normals[idx] = normal;
					tangents[idx] = tangent;
					break;
				}
			}

			if(provoking_idx == -1) {
				positions.emplace_back(positions[tri[0]]);
				tex_coords.emplace_back(tex_coords[tri[0]]);
				if(colors)
					colors.emplace_back(colors[tri[0]]);
				normals.emplace_back(normal);
				tangents.emplace_back(tangent);
				tri[0] = taken.size();
				taken.emplace_back(true);
				num_duplicated++;
			} else {
				tri = {tri[provoking_idx], tri[(provoking_idx + 1) % 3],
					   tri[(provoking_idx + 2) % 3]};
			}
		}
	}

	if(num_duplicated > 0)
		print("Duplicated verts during flat vectors generation: %\n", num_duplicated);
}

void Scene::quantizeVectors() {
	if(normals) {
		quantized_normals = transform(normals, encodeNormalUint);
		normals.free();
	}
	if(tangents) {
		quantized_tangents = transform(tangents, encodeNormalUint);
		tangents.free();
	}
}

void Scene::freeRenderingData() {
	tris_ib = {};
	quads_ib = {};
	verts = {};
}

int Scene::numTris() const {
	int out = 0;
	for(auto &mesh : meshes)
		out += mesh.tris.size();
	return out;
}

int Scene::numQuads() const {
	int out = 0;
	for(auto &mesh : meshes)
		out += mesh.quads.size();
	return out;
}

bool Scene::hasSimpleTextures() const {
	int num_opaque = 0;
	for(auto &tex : textures) {
		if(tex.map_type != SceneMapType::albedo)
			return false;
		if(tex.is_opaque)
			num_opaque++;
	}
	int num_transparent = textures.size() - num_opaque;
	return num_opaque <= 1 && num_transparent <= 1;
}

vector<SceneDrawCall> Scene::draws(const Frustum &frustum) const {
	// TODO: frustum culling

	vector<SceneDrawCall> out;
	out.reserve(meshes.size());
	bool has_tex_coords = hasTexCoords();
	auto scene_opts = mask(hasColors(), DrawCallOpt::has_vertex_colors) |
					  mask(has_tex_coords, DrawCallOpt::has_vertex_tex_coords) |
					  mask(hasQuantizedNormals(), DrawCallOpt::has_vertex_normals);

	for(int i : intRange(meshes)) {
		auto &mesh = meshes[i];
		auto &material = materials[mesh.material_id];
		bool is_opaque = material.isOpaque() && mesh.colors_opaque;
		auto &albedo = material.maps[SceneMapType::albedo];
		auto opts = scene_opts | mask(is_opaque, DrawCallOpt::is_opaque) |
					mask(albedo.is_opaque, DrawCallOpt::tex_opaque) |
					mask(albedo.usesUvRect(), DrawCallOpt::has_uv_rect);
		if(has_tex_coords) {
			opts |= mask(!!material.maps[SceneMapType::albedo], DrawCallOpt::has_albedo_tex);
			opts |= mask(!!material.maps[SceneMapType::normal], DrawCallOpt::has_normal_tex);
			opts |= mask(!!material.maps[SceneMapType::pbr], DrawCallOpt::has_pbr_tex);
		}
		auto &offsets = mesh_primitive_offsets[i];
		out.emplace_back(mesh.bounding_box, mesh.material_id, mesh.tris.size(), offsets.first,
						 mesh.quads.size(), offsets.second, opts);
	}

	return out;
}

Pair<PVImageView> Scene::textureAtlasPair() const {
	Pair<PVImageView> out;
	for(auto &tex : textures) {
		if(!tex.vk_image)
			continue;
		if(tex.is_opaque && !out.first)
			out.first = tex.vk_image;
		if(!tex.is_opaque && !out.second)
			out.second = tex.vk_image;
	}
	return out;
}
