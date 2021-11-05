#pragma once

#include "lucid_renderer.h"
#include "shading.h"
#include <fwk/gfx/camera_control.h>
#include <fwk/gfx/gl_texture.h>
#include <fwk/menu/imgui_wrapper.h>
#include <fwk/perf_base.h>

class LucidRenderer;
class SimpleRenderer;
class SceneSetup;
struct Scene;

DEFINE_ENUM(RenderingMode, simple, trans, mixed);

class LucidApp {
  public:
	LucidApp();
	~LucidApp();

	void setConfig(const AnyConfig &);
	static Maybe<AnyConfig> loadConfig();
	void saveConfig() const;

	void selectSetup(ZStr name);
	void selectSetup(int idx);

	void switchView();

	bool updateViewport();
	void updateRenderer();

	void doMenu(Renderer2D &renderer_2d);
	bool handleInput(vector<InputEvent> events, float time_diff);
	bool tick(float time_diff);

	void drawScene();

	bool mainLoop(GlDevice &device);
	static bool mainLoop(GlDevice &device, void *this_ptr);

	void printPerfStats();

  private:
	void printSceneStats(const Scene &);

	Dynamic<Font> m_font;
	Maybe<float2> m_mouse_pos;
	ImGuiWrapper m_imgui;
	Dynamic<perf::Analyzer> m_perf_analyzer;
	PTexture m_depth_buffer;
	PFramebuffer m_clear_fbo;
	SceneLighting m_lighting;
	Maybe<float3> m_picked_pos;

	Dynamic<LucidRenderer> m_lucid_renderer;
	Dynamic<SimpleRenderer> m_simple_renderer;
	HashMap<FilePath, double> m_shader_times;
	LucidRenderOpts m_lucid_opts = none;
	bool m_wireframe_mode = false;
	bool m_test_meshlets = false;
	bool m_show_stats = false;
	RenderingMode m_rendering_mode = RenderingMode::simple;

	IRect m_viewport;
	CameraControl m_cam_control;
	vector<Dynamic<SceneSetup>> m_setups;
	TextureFilteringParams m_filtering_params;
	float m_square_weight = 0.5f;
	int m_setup_idx = -1;

	struct StatSample {
		double min = inf, max = -inf, sum = 0.0;
		u64 count = 0;
	};

	struct StatPoint {
		perf::ExecId exec_id;
		ZStr short_name;
	};

	vector<StatPoint> selectPerfPoints() const;
	void updatePerfStats();

	vector<vector<perf::Frame>> m_stats;
	double m_last_time = -1.0;
	bool m_gather_perf_stats = true;
	int m_prev_setup_idx = -1;
	int m_skip_frame_id = 0;

	//bool m_imgui_demo = false;
};
