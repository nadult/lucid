#pragma once

#include "scene.h"

// TODO: warning / logging system needed
SceneTexture convertTexture(ZStr path);
Scene convertScene(WavefrontObject);
void convertScenes(ZStr source_path);
