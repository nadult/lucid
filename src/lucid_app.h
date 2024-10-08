// Copyright (C) Krzysztof Jakubowski <nadult@fastmail.fm>
// This file is part of LucidRaster. See license.txt for details.

#pragma once

#include "lucid_renderer.h"
#include "path_tracer.h"
#include "shading.h"
#include <fwk/gfx/camera_control.h>
#include <fwk/gui/gui.h>
#include <fwk/perf_base.h>

#include <fwk/vulkan/vulkan_storage.h> // TODO: shouldn't be needed
#include <fwk/vulkan_base.h>

class LucidRenderer;
class SimpleRenderer;
class PbrRenderer;
class PathTracer;
class SceneSetup;
struct Scene;

DEFINE_ENUM(RenderingMode, simple, lucid, mixed, pbr, path_trace);

class LucidApp {
  public:
	LucidApp(VWindowRef, VDeviceRef);
	~LucidApp();

	void setConfig(const AnyConfig &);
	static Maybe<AnyConfig> loadConfig();
	void saveConfig() const;

	void selectSetup(ZStr name);
	void selectSetup(int idx);

	void switchView();

	bool updateViewport();
	Ex<void> updateRenderer();
	Ex<> updateEnvMap();

	void doMenu();
	bool handleInput(vector<InputEvent> events, float time_diff);
	bool tick(float time_diff);

	void clearScreen(const RenderContext &);
	void drawFrame();
	void drawScene();

	bool mainLoop();
	static bool mainLoop(VulkanWindow &, void *this_ptr);

	void printPerfStats();

  private:
	void showStatsMenu(const Scene &);
	void showSceneStats(const Scene &);
	void showRasterStats(const Scene &);

	VWindowRef m_window;
	VDeviceRef m_device;
	PVRenderPass m_gui_render_pass;
	Gui m_gui;

	Maybe<float2> m_mouse_pos;
	Dynamic<perf::Analyzer> m_perf_analyzer;
	SceneLighting m_lighting;
	Maybe<float3> m_picked_pos;

	Dynamic<ShaderCompiler> m_shader_compiler;
	Dynamic<PathTracer> m_path_tracer;
	Dynamic<LucidRenderer> m_lucid_renderer;
	Dynamic<SimpleRenderer> m_simple_renderer;
	Dynamic<PbrRenderer> m_pbr_renderer;
	LucidRenderOpts m_lucid_opts = none;
	PathTracerOpts m_path_tracer_opts = none;
	bool m_wireframe_mode = false;
	bool m_test_meshlets = false;
	bool m_show_stats = false;
	bool m_verify_lucid_info = true;
	int m_select_stats_tab = -1, m_selected_stats_tab = 0;
	RenderingMode m_rendering_mode = RenderingMode::simple;

	IRect m_viewport;
	CameraControl m_cam_control;
	vector<Dynamic<SceneSetup>> m_setups;
	float m_square_weight = 0.5f;
	int m_setup_idx = -1;

	bool m_is_picking_block = false;
	bool m_is_final_pick = false;

	struct StatPoint {
		perf::ExecId exec_id;
		ZStr short_name;
	};

	vector<StatPoint> selectPerfPoints() const;
	void updatePerfStats();

	vector<vector<perf::Frame>> m_stats;
	double m_last_time = -1.0, m_last_shader_update_time = -1.0;
	bool m_gather_perf_stats = true;
	int m_prev_setup_idx = -1;
	int m_skip_frame_id = 0;
	int m_scene_frame_id = 0;
};
