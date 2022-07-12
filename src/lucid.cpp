#include "lucid_app.h"
#include "scene_convert.h"

#include <fwk/any_config.h>
#include <fwk/gfx/gl_device.h>
#include <fwk/gfx/opengl.h>
#include <fwk/perf/manager.h>
#include <fwk/perf/thread_context.h>

#include <fwk/vulkan/vulkan_buffer.h>
#include <fwk/vulkan/vulkan_device.h>
#include <fwk/vulkan/vulkan_framebuffer.h>
#include <fwk/vulkan/vulkan_image.h>
#include <fwk/vulkan/vulkan_instance.h>
#include <fwk/vulkan/vulkan_memory_manager.h>
#include <fwk/vulkan/vulkan_pipeline.h>
#include <fwk/vulkan/vulkan_render_graph.h>
#include <fwk/vulkan/vulkan_shader.h>
#include <fwk/vulkan/vulkan_swap_chain.h>
#include <fwk/vulkan/vulkan_window.h>

void initShaderCombiner();
Ex<void> loadShaderPieces();

Ex<int> exMain(int argc, char **argv) {
	IRect window_rect(int2(1200, 700));

	// TODO: xml loading is still messy
	Maybe<AnyConfig> config = LucidApp::loadConfig();
	VulkanInstanceSetup setup;
	setup.debug_levels = VDebugLevel::warning | VDebugLevel::error;
	setup.debug_types = all<VDebugType>;
	// TODO: cleanup in flags
	auto window_flags = VWindowFlag::resizable | VWindowFlag::centered | VWindowFlag::allow_hidpi;
	uint multisampling = 1;

	for(int n = 1; n < argc; n++) {
		string argument = argv[n];
		if(argument == "--convert-scenes") {
			ASSERT(n + 1 < argc && "Invalid nr of arguments");
			convertScenes(argv[n + 1]);
			return 0;
		} else if(argument == "--vsync") {
			window_flags |= VWindowFlag::vsync;
		} else if(argument == "--msaa") {
			ASSERT(n + 1 < argc && "Invalid nr of arguments");
			multisampling = clamp(atoi(argv[n + 1]), 1, 16);
			n++;
		} else {
			FATAL("Unsupported argument: %s", argument.c_str());
		}
	}

	// TODO: create instance on a thread, in the meantime load resources?
	auto instance = EX_PASS(VulkanInstance::create(setup));

	// TODO: handle config
	/*if(config) {
		auto display_rects = VulkanWindow::displayRects();
		if(auto *rect = config->get<IRect>("window_rect")) {
			window_rect = GlDevice::sanitizeWindowRect(display_rects, *rect);
			gl_config.flags &= ~GlDeviceOpt::centered;
		}
		if(config->get<bool>("window_maximized", false))
			gl_config.flags |= GlDeviceOpt::maximized;
	}*/

	auto window = EX_PASS(VulkanWindow::create(instance, "Lucid rasterizer", IRect(0, 0, 1280, 720),
											   VulkanWindowConfig{window_flags}));

	VulkanDeviceSetup dev_setup;
	auto pref_device = instance->preferredDevice(window->surfaceHandle(), &dev_setup.queues);
	if(!pref_device)
		return ERROR("Couldn't find a suitable Vulkan device");
	auto device = EX_PASS(instance->createDevice(*pref_device, dev_setup));
	auto phys_info = instance->info(device->physId());
	print("Selected Vulkan physical device: %\nDriver version: %\n",
		  phys_info.properties.deviceName, phys_info.properties.driverVersion);

	auto swap_chain = EX_PASS(VulkanSwapChain::create(
		device, window, {.preferred_present_mode = VPresentMode::immediate}));
	EXPECT(device->createRenderGraph(swap_chain));

	//initShaderCombiner();
	//loadShaderPieces().get();

	Dynamic<perf::Manager> perf_manager;
	Dynamic<perf::ThreadContext> perf_ctx;
	if(false) { // TODO: fixme
		perf_manager.emplace();
		perf_ctx.emplace(1024);
	}

	LucidApp app(window, device);
	if(config)
		app.setConfig(*config);
	app.updateViewport();
	app.updateRenderer();
	window->runMainLoop(LucidApp::mainLoop, &app);
	app.printPerfStats();

	return 0;
}

int main(int argc, char **argv) {
	auto result = exMain(argc, argv);

	if(!result) {
		result.error().print();
		return 1;
	}
	return *result;
}