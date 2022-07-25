#pragma once

#include "lucid_base.h"
#include <fwk/gfx/color.h>
#include <fwk/vulkan_base.h>

struct WavefrontObject;

struct SceneTexture {
	SceneTexture();
	FWK_COPYABLE_CLASS(SceneTexture);

	Ex<void> loadPlain(ZStr path);
	Ex<void> load(Stream &);
	Ex<void> save(Stream &) const;

	string name;
	PVImage vk_image;
	vector<Image> plain_mips;
	vector<CompressedImage> block_mips;
	bool is_opaque = true;
	bool is_clamped = false;
	bool is_atlas = false;
};

struct SceneMaterial {
	struct UsedTexture {
		explicit operator bool() const { return id != -1; }

		PVImageView vk_image;
		FRect uv_rect = FRect(0, 0, 1, 1);
		bool is_opaque = false;
		bool is_clamped = true;
		int id = -1;
	};

	SceneMaterial(string name = "");
	~SceneMaterial();

	Ex<void> load(Stream &);
	Ex<void> save(Stream &) const;

	string description() const;
	bool isOpaque() const;
	void freeTextures();

	string name;
	float3 diffuse = float3(1);
	float opacity = 1.0f;
	UsedTexture diffuse_tex, normal_tex;
};

struct SceneMesh {
	Ex<void> load(Stream &);
	Ex<void> save(Stream &) const;

	using Tri = array<int, 3>;
	using Quad = array<int, 4>;
	vector<Tri> tris;
	vector<Quad> quads;
	FBox bounding_box;
	bool colors_opaque = true;
	int material_id = 0;
	int num_degenerate_quads = 0;
};

struct Scene {
	static Ex<Scene> loadObj(ZStr path);
	static Ex<Scene> loadAssimp(ZStr path);
	static Ex<Scene> load(ZStr Path);

	Ex<void> load(Stream &);
	Ex<void> save(Stream &) const;
	void generateQuads(float squareness_weight);

	int numTris() const;
	int numQuads() const;
	int numVerts() const { return positions.size(); }
	bool hasColors() const { return colors && colors.size() == positions.size(); }
	bool hasNormals() const { return normals && normals.size() == positions.size(); }
	bool hasQuantizedNormals() const {
		return quantized_normals && quantized_normals.size() == positions.size();
	}
	bool hasTexCoords() const { return tex_coords && tex_coords.size() == positions.size(); }

	struct Intersection {
		int mesh_id, tri_id;
		float3 pos;
	};

	Maybe<Intersection> intersect(Segment3F) const;

	vector<float3> positions;
	vector<IColor> colors;
	vector<float2> tex_coords;
	vector<float3> normals;
	vector<uint> quantized_normals;

	string resource_path;
	vector<SceneTexture> textures;
	vector<SceneMaterial> materials;
	vector<SceneMesh> meshes;
	FBox bounding_box;

	// ------ Rendering data --------------------------------------------------

	void updatePrimitiveOffsets();
	Ex<void> updateRenderingData(VDeviceRef);
	void freeRenderingData();
	void quantizeNormals();

	vector<SceneDrawCall> draws(const Frustum &) const;
	Pair<PVImage> textureAtlasPair() const;

	VertexArray verts;
	PVBuffer tris_ib, quads_ib;

	vector<Pair<int>> mesh_primitive_offsets;
};
