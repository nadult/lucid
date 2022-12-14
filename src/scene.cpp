#include "scene.h"

#include "quad_generator.h"
#include "shading.h"
#include <fwk/gfx/compressed_image.h>
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
	auto unpack_tex = [&](UsedTexture &tex) {
		sr.unpack(tex.id, tex.is_opaque, tex.is_clamped, tex.uv_rect);
	};
	unpack_tex(diffuse_tex);
	unpack_tex(normal_tex);
	return sr.getValid();
}

Ex<void> SceneMaterial::save(Stream &sr) const {
	sr << name << diffuse << opacity;
	for(auto &tex : {diffuse_tex, normal_tex})
		sr.pack(tex.id, tex.is_opaque, tex.is_clamped, tex.uv_rect);
	return sr.getValid();
}

string SceneMaterial::description() const {
	return format("Material: '%': diffuse:% opacity:% is_opaque:% diff_tex:% normal_tex:%\n", name,
				  diffuse, opacity, isOpaque(), diffuse_tex.id, normal_tex.id);
}

bool SceneMaterial::isOpaque() const {
	return opacity == 1.0 && (diffuse_tex.id == -1 || diffuse_tex.is_opaque);
}

void SceneMaterial::freeTextures() {
	diffuse_tex.vk_image = {};
	normal_tex.vk_image = {};
}

SceneTexture::SceneTexture() = default;
FWK_COPYABLE_CLASS_IMPL(SceneTexture);

// TODO: this should be handled by Image & CompressedImage
Ex<void> SceneTexture::load(Stream &sr) {
	bool is_compressed;
	u8 num_levels;
	sr >> name;
	sr.unpack(is_compressed, is_opaque, is_clamped, is_atlas, num_levels);
	for(int i = 0; i < num_levels; i++)
		if(is_compressed) {
			int2 size;
			VBlockFormat format;
			sr.unpack(size, format);
			// TODO: turn this to function
			int data_size = imageSize(format, size.x, size.y);
			if(!sr.addResources(data_size))
				return sr.getValid();
			PodVector<u8> data(data_size);
			sr.loadData(data);
			block_mips.emplace_back(CompressedImage{move(data), size, format});
		} else {
			int2 size;
			sr >> size;
			if(!sr.addResources(size.x * size.y))
				return sr.getValid();
			PodVector<IColor> data(size.x * size.y);
			sr.loadData(data);
			plain_mips.emplace_back(Image{move(data), size});
		}
	return sr.getValid();
}

Ex<void> SceneTexture::save(Stream &sr) const {
	DASSERT(plain_mips.empty() != block_mips.empty());
	sr << name;
	u8 num_levels = block_mips ? block_mips.size() : plain_mips.size();
	sr.pack(!!block_mips, is_opaque, is_clamped, is_atlas, num_levels);
	if(block_mips) {
		for(auto &tex : block_mips) {
			sr.pack(tex.size(), tex.format());
			sr.saveData(tex.data());
		}
	} else {
		for(auto &tex : plain_mips) {
			sr << tex.size();
			sr.saveData(tex.data().reinterpret<char>());
		}
	}
	return sr.getValid();
}

Ex<void> SceneTexture::loadPlain(ZStr path) {
	*this = SceneTexture{};
	auto time = getTime();
	auto tex = EX_PASS(Image::load(path));
	is_opaque = tex.isOpaque();
	plain_mips.emplace_back(move(tex));
	print("Loaded texture '%' in % ms\n", path, (getTime() - time) * 1000.0);
	return {};
}

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

	// Czemu nie zwracam po prostu expected za ka??dym razem ?
	EXPECT(num_meshes > 0 && num_materials > 0);
	// TODO: resource limits here

	stream >> positions >> colors >> tex_coords >> normals >> quantized_normals >> bounding_box;

	meshes.resize(num_meshes);
	for(auto &mesh : meshes) {
		EXPECT(mesh.load(stream));
		EXPECT(mesh.material_id >= 0 && mesh.material_id < num_materials);
	}

	materials.resize(num_materials);
	for(auto &material : materials) {
		EXPECT(material.load(stream));
		if(material.diffuse_tex.id != -1)
			EXPECT(material.diffuse_tex.id >= 0 && material.diffuse_tex.id < num_textures);
	}

	textures.resize(num_textures);
	for(auto &texture : textures)
		EXPECT(texture.load(stream));

	return stream.getValid();
}

Ex<void> Scene::save(Stream &stream) const {
	stream.saveSignature("SCENE");
	stream.pack(meshes.size(), materials.size(), textures.size());
	stream << positions << colors << tex_coords << normals << quantized_normals << bounding_box;
	for(auto &mesh : meshes)
		EXPECT(mesh.save(stream));
	for(auto &material : materials)
		EXPECT(material.save(stream));
	for(auto &texture : textures)
		EXPECT(texture.save(stream));
	return stream.getValid();
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

	auto &cmds = device.cmdQueue();
	auto buf_usage = VBufferUsage::storage_buffer | VBufferUsage::transfer_dst;
	auto vb_usage = buf_usage | VBufferUsage::vertex_buffer;
	auto ib_usage = buf_usage | VBufferUsage::index_buffer;

	verts.pos = EX_PASS(VulkanBuffer::create<float3>(device, numVerts(), vb_usage));
	EXPECT(cmds.upload(verts.pos, positions));

	if(hasColors()) {
		verts.col = EX_PASS(VulkanBuffer::create<IColor>(device, numVerts(), vb_usage));
		EXPECT(cmds.upload(verts.col, colors));
	}
	if(hasTexCoords()) {
		verts.tex = EX_PASS(VulkanBuffer::create<float2>(device, numVerts(), vb_usage));
		EXPECT(cmds.upload(verts.tex, tex_coords));
	}
	if(hasQuantizedNormals()) {
		verts.nrm = EX_PASS(VulkanBuffer::create<u32>(device, numVerts(), vb_usage));
		EXPECT(cmds.upload(verts.nrm, quantized_normals));
	}

	tris_ib = EX_PASS(VulkanBuffer::create<u32>(device, numTris() * 3, ib_usage));
	quads_ib = EX_PASS(VulkanBuffer::create<u32>(device, numQuads() * 4, ib_usage));
	// TODO: merging shouldn't be needed; Add upload version which returns Span<> ?
	vector<u32> merged_tris, merged_quads;
	for(auto &mesh : meshes) {
		insertBack(merged_tris, cspan(mesh.tris).reinterpret<u32>());
		insertBack(merged_quads, cspan(mesh.quads).reinterpret<u32>());
	}
	EXPECT(cmds.upload(tris_ib, merged_tris));
	EXPECT(cmds.upload(quads_ib, merged_quads));

	auto loadTex = [&](SceneMaterial::UsedTexture &used_tex) -> Ex<> {
		if(used_tex.id == -1)
			return {};
		auto &tex = textures[used_tex.id];
		// TODO: makes no sense to keep both in main & gpu memory, especially because ogl is caching it too
		// TODO: mipmaps for plain textures
		if(!tex.vk_image) {
			DASSERT(tex.block_mips || tex.plain_mips);
			if(tex.block_mips) {
				auto vk_image = EX_PASS(VulkanImage::createAndUpload(device.ref(), tex.block_mips));
				tex.vk_image = VulkanImageView::create(device.ref(), vk_image);
			} else {
				// TODO: different format for opaque tex? R8G8B8?
				auto vk_image = EX_PASS(VulkanImage::createAndUpload(device.ref(), tex.plain_mips));
				tex.vk_image = VulkanImageView::create(device.ref(), vk_image);
			}
		}
		used_tex.vk_image = tex.vk_image;
		used_tex.is_opaque = tex.is_opaque;
		return {};
	};

	for(auto &material : materials) {
		EXPECT(loadTex(material.diffuse_tex));
		EXPECT(loadTex(material.normal_tex));
	}

	return {};
}

u32 encodeNormalUint(const float3 &n) {
	uint x = uint(512.0 + n.x * 511.0) & 0x3ffu;
	uint y = uint(512.0 + n.y * 511.0) & 0x3ffu;
	uint z = uint(512.0 + n.z * 511.0) & 0x3ffu;
	return x | (y << 10) | (z << 20);
}

void Scene::quantizeNormals() {
	if(!hasNormals())
		return;
	quantized_normals = transform(normals, encodeNormalUint);
	normals.free();
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
		bool tex_opaque = material.diffuse_tex.is_opaque;
		auto opts = scene_opts | mask(is_opaque, DrawCallOpt::is_opaque) |
					mask(tex_opaque, DrawCallOpt::tex_opaque) |
					mask(!material.diffuse_tex.is_clamped, DrawCallOpt::has_uv_rect) |
					mask(material.diffuse_tex && has_tex_coords, DrawCallOpt::has_texture);
		auto &offsets = mesh_primitive_offsets[i];
		out.emplace_back(mesh.bounding_box, material.diffuse_tex.uv_rect, mesh.material_id,
						 mesh.tris.size(), offsets.first, mesh.quads.size(), offsets.second, opts);
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
