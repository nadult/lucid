#include "lucid_app.h"
#include "scene_convert.h"

#include <fwk/any_config.h>
#include <fwk/gfx/gl_device.h>
#include <fwk/gfx/opengl.h>
#include <fwk/perf/manager.h>
#include <fwk/perf/thread_context.h>

void initShaderCombiner();
Ex<void> loadShaderPieces();

int main(int argc, char **argv) {
	double time = getTime();
	int2 resolution(1200, 700);

	// TODO: xml loading is still messy
	Maybe<AnyConfig> config = LucidApp::loadConfig();
	GlDeviceFlags flags = GlDeviceOpt::resizable;

	for(int n = 1; n < argc; n++) {
		string argument = argv[n];
		if(argument == "--convert-scenes") {
			ASSERT(n + 1 < argc && "Invalid nr of arguments");
			convertScenes(argv[n + 1]);
			return 0;
		} else if(argument == "--vsync") {
			flags |= GlDeviceOpt::vsync;
		} else {
			FATAL("Unsupported argument: %s", argument.c_str());
		}
	}

	GlDevice gl_device;
	gl_device.createWindow("Lucid rasterizer", resolution, flags, GlProfile::core, 4.3);

	print("OpenGL info:\n%\n", gl_info->toString());

	if(config) {
		if(auto *rect = config->get<IRect>("window_rect")) {
			// TODO: sanitize ?
			gl_device.setWindowRect(*rect);
		}
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
