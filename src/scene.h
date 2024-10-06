// Copyright (C) Krzysztof Jakubowski <nadult@fastmail.fm>
// This file is part of LucidRaster. See license.txt for details.

#pragma once

#include "lucid_base.h"
#include <fwk/enum_map.h>
#include <fwk/gfx/color.h>
#include <fwk/vulkan_base.h>

struct WavefrontObject;

DEFINE_ENUM(SceneMapType, albedo, normal, pbr);
using SceneMapTypes = EnumFlags<SceneMapType>;

struct SceneTexture {
	SceneTexture();
	FWK_COPYABLE_CLASS(SceneTexture);

	Ex<void> loadPlain(ZStr path);
	Ex<void> load(Stream &);
	Ex<void> save(Stream &) const;

	bool empty() const { return size() == int2(0, 0); }
	VFormat format() const;
	int2 size() const;

	string name;
	PVImageView vk_image;
	vector<Image> mips;
	SceneMapType map_type = SceneMapType::albedo;
	bool is_opaque = true;
	bool is_clamped = false;
	bool is_atlas = false;
};

struct SceneMaterial {
	struct Map {
		explicit operator bool() const { return texture_id != -1; }

		// clamped maps don't have to use uv_rect, because their uv_coordinates are transformed
		bool usesUvRect() const { return !is_clamped && uv_rect != FRect(0, 0, 1, 1); }

		PVImageView vk_image;
		FRect uv_rect = FRect(0, 0, 1, 1);
		bool is_opaque = false;
		bool is_clamped = true;
		int texture_id = -1;
	};

	SceneMaterial(string name = "");
	~SceneMaterial();

	Ex<void> load(Stream &);
	Ex<void> save(Stream &) const;

	bool isOpaque() const;
	void freeTextures();

	string name;
	float3 diffuse = float3(1);
	float opacity = 1.0f;
	EnumMap<SceneMapType, Map> maps;
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
	Scene();
	FWK_MOVABLE_CLASS(Scene);

	static Ex<Scene> load(ZStr Path);
	Ex<void> load(Stream &);
	Ex<void> save(Stream &) const;

	void mergeVertices(int decimal_places = 3);
	void generateQuads(float squareness_weight);

	int numTris() const;
	int numQuads() const;
	int numVerts() const { return positions.size(); }
	bool hasColors() const { return colors && colors.size() == positions.size(); }

	// TODO: remove these?
	bool hasNormals() const { return normals && normals.size() == positions.size(); }
	bool hasQuantizedNormals() const {
		return quantized_normals && quantized_normals.size() == positions.size();
	}

	bool hasTexCoords() const { return tex_coords && tex_coords.size() == positions.size(); }

	// Only albedo and at most 2 textures (one opaque, one alpha)
	bool hasSimpleTextures() const;

	struct Intersection {
		int mesh_id, tri_id;
		float3 pos;
	};

	Maybe<Intersection> intersect(Segment3F) const;

	vector<float3> positions;
	vector<IColor> colors;
	vector<float2> tex_coords;
	vector<float3> normals;
	vector<float3> tangents;
	vector<uint> quantized_normals;
	vector<uint> quantized_tangents;

	string id, resource_path;
	vector<SceneTexture> textures;
	vector<SceneMaterial> materials;
	vector<SceneMesh> meshes;
	FBox bounding_box;

	// ------ Rendering data --------------------------------------------------

	void updatePrimitiveOffsets();
	Ex<> updateRenderingData(VulkanDevice &);
	void freeRenderingData();
	void computeTangents();
	void quantizeVectors();

	void computeFlatVectors();
	void quantizeFlatVectors();

	vector<SceneDrawCall> draws(const Frustum &) const;
	Pair<PVImageView> textureAtlasPair() const;

	VertexArray verts;
	VBufferSpan<u32> tris_ib, quads_ib;

	vector<Pair<int>> mesh_primitive_offsets;
};
