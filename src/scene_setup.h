#pragma once

#include "scene.h"
#include <fwk/gfx/camera_variant.h>

DEFINE_ENUM(SceneSetupId, boxes, teapot, bunny, hairball, white_oak, power_plant);

// TODO: rename to SceneSetup
class SceneSetup {
  public:
	SceneSetup(string name);
	virtual ~SceneSetup();
	virtual void doMenu(){};
	virtual Ex<> updateScene() = 0;

	string name;
	Maybe<Scene> scene;
	RenderConfig render_config;

	vector<CameraVariant> views;
	int view_id = 0;
	Maybe<CameraVariant> camera;
};

class BoxesSetup final : public SceneSetup {
  public:
	BoxesSetup();
	void doMenu() override;
	Ex<> updateScene() override;

  private:
	int3 m_current_dims;
	int3 m_dims = {10, 10, 10};
	float m_box_size = 0.5f;
	float m_box_dist = 0.1f;
};

class LoadedSetup final : public SceneSetup {
  public:
	LoadedSetup(string name);
	Ex<> updateScene() override;

	static vector<string> findAll();
};
