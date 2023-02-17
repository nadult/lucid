#pragma once

#include "scene.h"
#include <fwk/gfx/camera_variant.h>

class SceneSetup {
  public:
	SceneSetup(string name);
	virtual ~SceneSetup();
	virtual void doMenu(VDeviceRef){};
	virtual Ex<> updateScene(VDeviceRef) = 0;

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
	void doMenu(VDeviceRef) override;
	Ex<> updateScene(VDeviceRef) override;

  private:
	int3 m_current_dims;
	int3 m_dims = {10, 10, 10};
	float m_box_size = 0.5f;
	float m_box_dist = 0.1f;
};

class PlanesSetup final : public SceneSetup {
  public:
	PlanesSetup();
	void doMenu(VDeviceRef) override;
	Ex<> updateScene(VDeviceRef) override;

  private:
	int m_current_planes = 0;
	int m_num_planes = 32;
	float m_plane_size = 4.0f;
	float m_plane_dist = 0.1f;
};

class LoadedSetup final : public SceneSetup {
  public:
	LoadedSetup(string name);
	Ex<> updateScene(VDeviceRef) override;

	static vector<string> findAll();
};
