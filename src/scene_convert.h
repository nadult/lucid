// Copyright (C) Krzysztof Jakubowski <nadult@fastmail.fm>
// This file is part of LucidRaster. See license.txt for details.

#pragma once

#include "scene.h"

struct InputScene {
	InputScene(string name, string path);
	InputScene(const FilePath &root_path, CXmlNode);

	string name, path;
	string env_map_path;
	float quad_squareness = 1.0f;
	bool merge_verts = false;
	bool flip_uv = false;
	bool flip_yz = false;
	bool pbr = false;
};

Ex<vector<InputScene>> loadInputScenes(ZStr path);

// TODO: warning / logging system needed

vector<Image> panoramaToCubeMap(const Image &);
Ex<Image> loadExr(ZStr path);
SceneTexture convertTexture(ZStr path);
Scene convertScene(WavefrontObject);
void convertScenes(ZStr scenes_path);
