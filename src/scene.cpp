#include "scene.h"

#include "quad_generator.h"
#include "shading.h"
#include <fwk/gfx/compressed_image.h>
#include <fwk/gfx/gl_buffer.h>
#include <fwk/gfx/gl_format.h>
#include <fwk/gfx/gl_program.h>
#include <fwk/gfx/gl_texture.h>
#include <fwk/gfx/gl_vertex_array.h>
#include <fwk/gfx/image.h>
#include <fwk/gfx/opengl.h>
#include <fwk/io/file_stream.h>
#include <fwk/math/ray.h>
#include <fwk/math/segment.h>

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
	diffuse_tex.gl_handle = {};
	normal_tex.gl_handle = {};
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
			GlFormat format;
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

	// Czemu nie zwracam po prostu expected za każdym razem ?
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

void Scene::updateRenderingData() {
	return; // TODO: fixme

	updatePrimitiveOffsets();
	freeRenderingData();

	auto usage = BufferUsage::static_draw;

	PBuffer pos_vb = GlBuffer::make(BufferType::array, numVerts() * sizeof(float3), usage);
	copy(pos_vb->map<float3>(AccessMode::write_only), positions);
	pos_vb->unmap();

	PBuffer tex_vb, col_vb, nrm_vb;
	if(hasColors()) {
		col_vb = GlBuffer::make(BufferType::array, numVerts() * sizeof(IColor), usage);
		copy(col_vb->map<IColor>(AccessMode::write_only), colors);
		col_vb->unmap();
	}
	if(hasTexCoords()) {
		tex_vb = GlBuffer::make(BufferType::array, numVerts() * sizeof(float2), usage);
		copy(tex_vb->map<float2>(AccessMode::write_only), tex_coords);
		tex_vb->unmap();
	}
	if(hasQuantizedNormals()) {
		nrm_vb = GlBuffer::make(BufferType::array, numVerts() * sizeof(u32), usage);
		copy(nrm_vb->map<u32>(AccessMode::write_only), quantized_normals);
		nrm_vb->unmap();
	}

	tris_ib = GlBuffer::make(BufferType::element_array, numTris() * 3 * sizeof(u32), usage);
	quads_ib = GlBuffer::make(BufferType::element_array, numQuads() * 4 * sizeof(u32), usage);
	auto tris_data = tris_ib->map<u32>(AccessMode::write_only);
	auto quads_data = quads_ib->map<u32>(AccessMode::write_only);
	int tri_offset = 0, quad_offset = 0;
	for(auto &mesh : meshes) {
		copy(tris_data.subSpan(tri_offset), cspan(mesh.tris).reinterpret<u32>());
		copy(quads_data.subSpan(quad_offset), cspan(mesh.quads).reinterpret<u32>());
		tri_offset += mesh.tris.size() * 3;
		quad_offset += mesh.quads.size() * 4;
	}
	tris_ib->unmap();
	quads_ib->unmap();

	auto vert_attribs = defaultVertexAttribs<float3, IColor, float2, u32>();
	mesh_vao = GlVertexArray::make();
	mesh_vao->set({pos_vb, col_vb, tex_vb, nrm_vb}, vert_attribs, tris_ib, IndexType::uint32);

	auto loadTex = [&](SceneMaterial::UsedTexture &used_tex) {
		if(used_tex.id == -1)
			return;
		auto &tex = textures[used_tex.id];
		// TODO: makes no sense to keep both in main & gpu memory, especially because ogl is caching it too
		// TODO: mipmaps for plain textures
		if(!tex.gl_texture) {
			DASSERT(tex.block_mips || tex.plain_mips);
			if(tex.block_mips) {
				tex.gl_texture.emplace(tex.block_mips);
			} else {
				print("Loading plain texture (%)\n", tex.plain_mips[0].size());
				auto format = tex.is_opaque ? GlFormat::srgb8 : GlFormat::srgba8;
				tex.gl_texture.emplace(tex.plain_mips, format);
			}

			tex.gl_texture->setFiltering(TextureFilterOpt::linear);
			auto wrap_opt = tex.is_clamped ? TextureWrapOpt::clamp_to_edge : TextureWrapOpt::repeat;
			tex.gl_texture->setWrapping(wrap_opt);
		}
		used_tex.gl_handle = tex.gl_texture;
		used_tex.is_opaque = tex.is_opaque;
	};

	for(auto &material : materials) {
		loadTex(material.diffuse_tex);
		loadTex(material.normal_tex);
	}
	testGlError("updateRenderingData");
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
	mesh_vao = {};
	tris_ib = {};
	quads_ib = {};
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

Pair<PTexture> Scene::textureAtlasPair() const {
	Pair<PTexture> out;
	for(auto &tex : textures) {
		if(!tex.gl_texture)
			continue;
		if(tex.is_opaque && !out.first)
			out.first = tex.gl_texture;
		if(!tex.is_opaque && !out.second)
			out.second = tex.gl_texture;
	}
	return out;
}
