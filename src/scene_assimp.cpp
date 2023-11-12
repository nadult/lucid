// Copyright (C) Krzysztof Jakubowski <nadult@fastmail.fm>
// This file is part of LucidRaster. See license.txt for details.

#include "scene.h"

#include "shading.h"

#include <assimp/Importer.hpp>
#include <assimp/material.h>
#include <assimp/matrix4x4.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/types.h>

static float3 fromAssimp(const aiVector3D &vec) { return {vec.x, vec.y, vec.z}; }
static IColor fromAssimp(const aiColor4D &col) {
	return (IColor)FColor(col.r, col.g, col.b, col.a);
}

static string fromAssimp(const aiString &str) { return str.C_Str(); }

static Matrix4 fromAssimp(const aiMatrix4x4 &mat) {
	static_assert(sizeof(aiMatrix4x4) == sizeof(Matrix4));
	Matrix4 out;
	memcpy(&out, &mat, sizeof(mat));
	return out;
}

Ex<Scene> Scene::loadAssimp(ZStr input_path) {
	Assimp::Importer importer;
	auto ai_flags = aiProcess_Triangulate | aiProcess_SortByPType | aiProcess_JoinIdenticalVertices;
	const aiScene *ai_scene = importer.ReadFile(input_path.c_str(), ai_flags);

	if(!ai_scene) {
		return ERROR("Error while loading '%' with assimp:%\n", input_path,
					 importer.GetErrorString());
	}

	Scene scene;
	scene.meshes.resize(ai_scene->mNumMeshes);

	for(uint i = 0; i < ai_scene->mNumMeshes; i++) {
		auto &src_mesh = *ai_scene->mMeshes[i];
		auto &dst_mesh = scene.meshes[i];

		dst_mesh.positions.resize(src_mesh.mNumVertices);
		for(uint j = 0; j < src_mesh.mNumVertices; j++)
			dst_mesh.positions[j] = fromAssimp(src_mesh.mVertices[j]);

		dst_mesh.tex_coords.resize(src_mesh.mNumUVComponents[0]);
		for(uint j = 0; j < src_mesh.mNumUVComponents[0]; j++)
			dst_mesh.tex_coords[j] = fromAssimp(src_mesh.mTextureCoords[0][j]).xy();

		if(src_mesh.HasNormals()) {
			dst_mesh.normals.resize(src_mesh.mNumVertices);
			for(uint j = 0; j < src_mesh.mNumVertices; j++)
				dst_mesh.normals[j] = fromAssimp(src_mesh.mNormals[j]);
		}

		if(src_mesh.HasVertexColors(0)) {
			dst_mesh.colors.resize(src_mesh.mNumVertices);
			for(uint j = 0; j < src_mesh.mNumVertices; j++)
				dst_mesh.colors[j] = fromAssimp(src_mesh.mColors[0][j]);
		}

		dst_mesh.tris.resize(src_mesh.mNumFaces);
		dst_mesh.bounding_box = enclose(dst_mesh.positions);
		for(uint j = 0; j < src_mesh.mNumFaces; j++) {
			auto &face = src_mesh.mFaces[j];
			dst_mesh.tris[j] = {int(face.mIndices[0]), int(face.mIndices[1]),
								int(face.mIndices[2])};
		}
	}

	for(uint i = 0; i < ai_scene->mNumMaterials; i++) {
		auto &src = *ai_scene->mMaterials[i];
		auto &dst = scene.materials.emplace_back(fromAssimp(src.GetName()));

		aiColor3D color(0.f, 0.f, 0.f);
		src.Get(AI_MATKEY_COLOR_DIFFUSE, color);
		float opacity = 1.0f;
		src.Get(AI_MATKEY_OPACITY, opacity);

		dst.diffuse = {color.b, color.g, color.r};
		dst.opacity = opacity;

		aiString texName;
		if(src.Get(AI_MATKEY_TEXTURE(aiTextureType_DIFFUSE, 0), texName) == aiReturn_SUCCESS)
			dst.maps[MaterialMapType::diffuse].texture_name = fromAssimp(texName);
		if(src.Get(AI_MATKEY_TEXTURE(aiTextureType_NORMALS, 0), texName) == aiReturn_SUCCESS)
			dst.maps[MaterialMapType::bump].texture_name = fromAssimp(texName);
	}

	struct Node {
		aiNode *node;
		Matrix4 trans;
	};
	vector<Node> nodes;
	nodes.emplace_back(ai_scene->mRootNode, fromAssimp(ai_scene->mRootNode->mTransformation));

	for(int i = 0; i < nodes.size(); i++) {
		auto node = nodes[i];
		for(uint j = 0; j < node.node->mNumMeshes; j++) {
			auto mesh_id = node.node->mMeshes[j];
			auto &mesh = *ai_scene->mMeshes[mesh_id];
			Instance new_inst;
			new_inst.first_tri = 0;
			new_inst.mesh_id = mesh_id;
			new_inst.material_id = mesh.mMaterialIndex;
			new_inst.num_tris = mesh.mNumFaces;
			// TODO: rough, compute exactly ?
			new_inst.bounding_box = scene.meshes[mesh_id].bounding_box;
			new_inst.trans = node.trans;
			scene.instances.emplace_back(new_inst);
		}
		for(uint j = 0; j < node.node->mNumChildren; j++) {
			auto *child = node.node->mChildren[j];
			nodes.emplace_back(child, node.trans * fromAssimp(child->mTransformation));
		}
	}

	return scene;
}
