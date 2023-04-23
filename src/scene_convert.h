#pragma once

#include "scene.h"

struct InputScene {
	InputScene(string name, string path);
	InputScene(const FilePath &root_path, CXmlNode);

	string name, path;
	float quad_squareness = 1.0f;
	bool merge_verts = false;
	bool flip_uv = false;
	bool flip_yz = false;
	bool pbr = false;
};

Ex<vector<InputScene>> loadInputScenes(ZStr path);

// TODO: warning / logging system needed
SceneTexture convertTexture(ZStr path);
Scene convertScene(WavefrontObject);
void convertScenes(ZStr scenes_path);
