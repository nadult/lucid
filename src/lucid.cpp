#include "lucid_app.h"
#include "scene_convert.h"

#include <fwk/any_config.h>
#include <fwk/perf/manager.h>
#include <fwk/perf/thread_context.h>

#include <fwk/vulkan/vulkan_buffer.h>
#include <fwk/vulkan/vulkan_command_queue.h>
#include <fwk/vulkan/vulkan_device.h>
#include <fwk/vulkan/vulkan_framebuffer.h>
#include <fwk/vulkan/vulkan_image.h>
#include <fwk/vulkan/vulkan_instance.h>
#include <fwk/vulkan/vulkan_memory_manager.h>
#include <fwk/vulkan/vulkan_pipeline.h>
#include <fwk/vulkan/vulkan_shader.h>
#include <fwk/vulkan/vulkan_swap_chain.h>
#include <fwk/vulkan/vulkan_window.h>

#include <fwk/libs_msvc.h>

Ex<int> exMain(int argc, char **argv) {
	IRect window_rect = IRect({1280, 720}) + int2(32, 32);

	// TODO: xml loading is still messy
	Maybe<AnyConfig> config = LucidApp::loadConfig();
	VInstanceSetup setup;
	auto window_flags = VWindowFlag::resizable | VWindowFlag::centered | VWindowFlag::allow_hidpi |
						VWindowFlag::sleep_when_minimized;
	uint multisampling = 1;
#ifdef NDEBUG
	bool debug_mode = false;
#else
	bool debug_mode = true;
#endif

	VSwapChainSetup swap_chain_setup;
	// TODO: UI is configure for Unorm, shouldn't we use SRGB by default?
	swap_chain_setup.preferred_formats = {{VK_FORMAT_B8G8R8A8_UNORM}};
	swap_chain_setup.preferred_present_mode = VPresentMode::immediate;
	swap_chain_setup.usage =
		VImageUsage::color_att | VImageUsage::storage | VImageUsage::transfer_dst;
	swap_chain_setup.initial_layout = VImageLayout::general;

	for(int n = 1; n < argc; n++) {
		string argument = argv[n];
		if(argument == "--convert-scenes") {
			convertScenes(mainPath() / "input_scenes.xml");
			return 0;
		} else if(argument == "--vsync") {
			swap_chain_setup.preferred_present_mode = VPresentMode::fifo;
		} else if(argument == "--vulkan-debug") {
			debug_mode = true;
		} else if(argument == "--no-vulkan-debug") {
			debug_mode = false;
		} else if(argument == "--msaa") {
			ASSERT(n + 1 < argc && "Invalid nr of arguments");
			multisampling = clamp(atoi(argv[n + 1]), 1, 16);
			n++;
		} else {
			FATAL("Unsupported argument: %s", argument.c_str());
		}
	}

	if(debug_mode) {
		setup.debug_levels = VDebugLevel::warning | VDebugLevel::error;
		setup.debug_types = all<VDebugType>;
	}

	// TODO: create instance on a thread, in the meantime load resources?
	auto instance = EX_PASS(VulkanInstance::create(setup));

	if(config) {
		// TODO: first initialize SDL?
		auto display_rects = VulkanWindow::displayRects();
		if(auto *rect = config->get<IRect>("window_rect")) {
			window_rect = VulkanWindow::sanitizeWindowRect(display_rects, *rect);
			window_flags &= ~VWindowFlag::centered;
		}
		if(config->get<bool>("window_maximized", false))
			window_flags |= VWindowFlag::maximized;
	}

	auto window = EX_PASS(VulkanWindow::create(instance, "LucidRaster", window_rect, window_flags));

	VDeviceSetup dev_setup;
	dev_setup.features.emplace();
	dev_setup.features->shaderInt64 = VK_TRUE;
	dev_setup.features->samplerAnisotropy = VK_TRUE;
	dev_setup.features->fillModeNonSolid = VK_TRUE;
	auto pref_device = instance->preferredDevice(window->surfaceHandle(), &dev_setup.queues);
	if(!pref_device)
		return ERROR("Couldn't find a suitable Vulkan device");
	dev_setup.extensions = {"VK_EXT_shader_subgroup_vote", "VK_EXT_shader_subgroup_ballot"};
	auto device = EX_PASS(instance->createDevice(*pref_device, dev_setup));
	auto phys_info = instance->info(device->physId());
	print("Selected Vulkan physical device: %\nDriver version: %\n",
		  phys_info.properties.deviceName, phys_info.properties.driverVersion);
	device->addSwapChain(EX_PASS(VulkanSwapChain::create(device, window, swap_chain_setup)));

	Dynamic<perf::Manager> perf_manager;
	Dynamic<perf::ThreadContext> perf_ctx;
	if(true) {
		perf_manager.emplace();
		perf_ctx.emplace(1024);
	}

	LucidApp app(window, device);
	if(config)
		app.setConfig(*config);
	app.updateViewport();
	EXPECT(app.updateRenderer());
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