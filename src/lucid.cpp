#include "lucid_app.h"
#include "scene_convert.h"

#include <fwk/any_config.h>
#include <fwk/gfx/gl_device.h>
#include <fwk/gfx/opengl.h>
#include <fwk/perf/manager.h>
#include <fwk/perf/thread_context.h>

void initShaderCombiner();
Ex<void> loadShaderPieces();

// Select more powerful GPU if more than 1 available
extern "C" {
_declspec(dllexport) uint NvOptimusEnablement = 1;
_declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}

int main(int argc, char **argv) {
	double time = getTime();
	IRect window_rect(int2(1200, 700));

	// TODO: xml loading is still messy
	Maybe<AnyConfig> config = LucidApp::loadConfig();
	GlDeviceConfig gl_config;
	gl_config.flags = GlDeviceOpt::resizable | GlDeviceOpt::allow_hidpi | GlDeviceOpt::centered;
	gl_config.version = 4.4;

	for(int n = 1; n < argc; n++) {
		string argument = argv[n];
		if(argument == "--convert-scenes") {
			ASSERT(n + 1 < argc && "Invalid nr of arguments");
			convertScenes(argv[n + 1]);
			return 0;
		} else if(argument == "--vsync") {
			gl_config.flags |= GlDeviceOpt::vsync;
		} else if(argument == "--msaa") {
			ASSERT(n + 1 < argc && "Invalid nr of arguments");
			gl_config.multisampling = clamp(atoi(argv[n + 1]), 1, 16);
			n++;
		} else {
			FATAL("Unsupported argument: %s", argument.c_str());
		}
	}

	if(config) {
		if(auto *rect = config->get<IRect>("window_rect")) {
			window_rect = *rect;
			gl_config.flags &= ~GlDeviceOpt::centered;
		}
		if(config->get<bool>("window_maximized", false))
			gl_config.flags |= GlDeviceOpt::maximized;
	}

	GlDevice gl_device;
	// TODO: make sure that window rect overlaps some display rect and sanitize if needed
	gl_device.createWindow("Lucid rasterizer", window_rect, gl_config);

	print("OpenGL info:\n%\n", gl_info->toString());
	if(gl_info->vendor == GlVendor::nvidia) {
		// TODO: add it to libfwk, find similar for intel and amd ?
		GLint warp_size, warps_per_sm, sm_count;
		glGetIntegerv(GL_WARP_SIZE_NV, &warp_size);
		glGetIntegerv(GL_WARPS_PER_SM_NV, &warps_per_sm);
		glGetIntegerv(GL_SM_COUNT_NV, &sm_count);
		print("NVIDIA OpenGL info:\n warp size:    %\n warps per SM: %\n SM count:     %\n",
			  warp_size, warps_per_sm, sm_count);
	}

	initShaderCombiner();
	loadShaderPieces().get();

	Dynamic<perf::Manager> perf_manager;
	Dynamic<perf::ThreadContext> perf_ctx;
	if(true) {
		perf_manager.emplace();
		perf_ctx.emplace(1024);
	}

	LucidApp app;
	if(config)
		app.setConfig(*config);
	gl_device.runMainLoop(LucidApp::mainLoop, &app);
	app.printPerfStats();

	return 0;
}
