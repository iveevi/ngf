#include <imgui/imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include <implot.h>

#include "context.hpp"
#include "glio.hpp"

constexpr vk::DescriptorSetLayoutBinding texture_at(uint32_t binding,
	vk::ShaderStageFlagBits extra = vk::ShaderStageFlagBits::eMeshEXT)
{
	return vk::DescriptorSetLayoutBinding {
		binding, vk::DescriptorType::eCombinedImageSampler,
		1, vk::ShaderStageFlagBits::eMeshEXT | extra
	};
}

const std::array <vk::DescriptorSetLayoutBinding, 8> meshlet_dslbs {
	// Complexes
	texture_at(0, vk::ShaderStageFlagBits::eTaskEXT),

	// Vertices
	texture_at(1, vk::ShaderStageFlagBits::eTaskEXT),

	// Features
	texture_at(2),

	// Bias vectors
	texture_at(3),

	// Layer weights
	texture_at(4),
	texture_at(5),
	texture_at(6),
	texture_at(7),
};

const std::array <vk::DescriptorSetLayoutBinding, 1> environment_dslbs {
	vk::DescriptorSetLayoutBinding {
		0, vk::DescriptorType::eCombinedImageSampler,
		1, vk::ShaderStageFlagBits::eFragment
	},
};

const std::unordered_map <std::string, FragmentShaderInfo> fragment_shaders {
	{ "Shaded", { SHADERS_DIRECTORY "/shaded.frag" } },
	{ "Patches", { SHADERS_DIRECTORY "/patches.frag" } },
	{ "Normals", { SHADERS_DIRECTORY "/normals.frag" } },
	{ "Depth", { SHADERS_DIRECTORY "/depth.frag" } },
	{ "Wireframe", {
		SHADERS_DIRECTORY "/fill.frag",
		vk::CullModeFlagBits::eNone,
		vk::PolygonMode::eLine
	} },
};

// Resizing framebuffers
void DeviceRenderContext::resize()
{
	// TODO: return a tuple from the skeleton...
	// e.g. littlevk::std::...
	device.waitIdle();

	// Resize the swapchain
	littlevk::Skeleton::resize();

	// Resize the depth buffers
	for (auto db : depth)
		littlevk::destroy_image(device, db);

	// Allocate new ones
	depth.clear();
	for (size_t i = 0; i < swapchain.images.size(); i++) {
		littlevk::Image depth_buffer = bind(device, memory_properties, dal)
			.image(window->extent,
				vk::Format::eD32Sfloat,
				vk::ImageUsageFlagBits::eDepthStencilAttachment,
				vk::ImageAspectFlagBits::eDepth);

		depth.push_back(depth_buffer);
	}

	// Create the framebuffers
	littlevk::FramebufferGenerator generator(device, render_pass, window->extent, dal);
	for (size_t i = 0; i < swapchain.images.size(); i++)
		generator.add(swapchain.image_views[i], depth[i].view);

	framebuffers = generator.unpack();
}

// Allocating images
littlevk::Image DeviceRenderContext::upload_texture(const Texture &tex) const
{
	littlevk::Image image;
	littlevk::Buffer staging;

	std::tie(image, staging) = bind(device, memory_properties, dal)
		.image((uint32_t) tex.width, (uint32_t) tex.height,
			vk::Format::eR8G8B8A8Unorm,
			vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
			vk::ImageAspectFlagBits::eColor)
		.buffer(tex.pixels, vk::BufferUsageFlagBits::eTransferSrc);

	littlevk::submit_now(device, command_pool, graphics_queue,
		[&](const vk::CommandBuffer &cmd) {
			littlevk::transition(cmd, image,
					vk::ImageLayout::eUndefined,
					vk::ImageLayout::eTransferDstOptimal);

			littlevk::copy_buffer_to_image(cmd, image, staging,
					vk::ImageLayout::eTransferDstOptimal);

			littlevk::transition(cmd, image,
					vk::ImageLayout::eTransferDstOptimal,
					vk::ImageLayout::eShaderReadOnlyOptimal);
		}
	);

	// Free interim data
	littlevk::destroy_buffer(device, staging);

	return image;
}

// Construction
void DeviceRenderContext::configure_imgui(DeviceRenderContext &engine)
{
	// Allocate descriptor pool
	vk::DescriptorPoolSize pool_sizes[] = {
		{ vk::DescriptorType::eSampler, 1 << 10 },
	};

	vk::DescriptorPoolCreateInfo pool_info = {};
	pool_info.poolSizeCount = sizeof(pool_sizes) / sizeof(pool_sizes[0]);
	pool_info.pPoolSizes = pool_sizes;
	pool_info.maxSets = 1 << 10;

	engine.imgui_descriptor_pool = littlevk::descriptor_pool
	(
		engine.device,
		pool_info
	).unwrap(engine.dal);

	// Configure ImGui
	ImGui::CreateContext();
	ImGui::StyleColorsDark();

	ImGui_ImplGlfw_InitForVulkan(engine.window->handle, true);

	ImGui_ImplVulkan_InitInfo init_info = {};
	init_info.Instance = littlevk::detail::get_vulkan_instance();
	init_info.PhysicalDevice = engine.phdev;
	init_info.Device = engine.device;
	init_info.QueueFamily = littlevk::find_graphics_queue_family(engine.phdev);
	init_info.Queue = engine.graphics_queue;
	init_info.PipelineCache = nullptr;
	init_info.DescriptorPool = engine.imgui_descriptor_pool;
	init_info.Allocator = nullptr;
	init_info.MinImageCount = 2;
	init_info.ImageCount = 2;
	init_info.CheckVkResultFn = nullptr;
	init_info.RenderPass = engine.render_pass;
	init_info.Subpass = 0;

	ImGui_ImplVulkan_Init(&init_info);

	// Upload fonts
	ImGui_ImplVulkan_CreateFontsTexture();

	// Configure ImPlot as well
	ImPlot::CreateContext();
}

DeviceRenderContext DeviceRenderContext::from(const vk::PhysicalDevice &phdev, const std::vector <const char *> &extensions, size_t fsize)
{
	DeviceRenderContext engine;

	engine.phdev = phdev;
	engine.memory_properties = phdev.getMemoryProperties();
	engine.dal = new littlevk::Deallocator(engine.device);

	// Analyze the properties
	vk::PhysicalDeviceMeshShaderPropertiesEXT ms_properties = {};
	vk::PhysicalDeviceProperties2 properties = {};
	properties.pNext = &ms_properties;

	phdev.getProperties2(&properties);
	ulog_info("testbed", "physical device properties:\n");
	ulog_info("testbed", "  max (task) payload memory: %d KB\n", ms_properties.maxTaskPayloadSize / 1024);
	ulog_info("testbed", "  max (task) shared memory: %d KB\n", ms_properties.maxTaskSharedMemorySize / 1024);
	ulog_info("testbed", "  max (mesh) shared memory: %d KB\n", ms_properties.maxMeshSharedMemorySize / 1024);
	ulog_info("testbed", "  max output vertices: %d\n", ms_properties.maxMeshOutputVertices);
	ulog_info("testbed", "  max output primitives: %d\n", ms_properties.maxMeshOutputPrimitives);
	ulog_info("testbed", "  max work group invocations: %d\n", ms_properties.maxMeshWorkGroupInvocations);

	// Configure the features
	vk::PhysicalDeviceMeshShaderFeaturesEXT ms_ft = {};
	vk::PhysicalDeviceMaintenance4FeaturesKHR m4_ft = {};
	vk::PhysicalDeviceSeparateDepthStencilLayoutsFeaturesKHR separation = {};
	vk::PhysicalDeviceRobustness2FeaturesEXT robustness = {};
	vk::PhysicalDeviceFeatures2KHR ft = {};

	ft.features.independentBlend = true;
	ft.features.fillModeNonSolid = true;
	ft.features.geometryShader = true;

	// TODO: littlevk next_chain(...)
	ft.pNext = &ms_ft;
	ms_ft.pNext = &m4_ft;
	m4_ft.pNext = &separation;
	separation.pNext = &robustness;

	phdev.getFeatures2(&ft);

	ulog_info("testbed", "features:\n");
	ulog_info("testbed", "  task shaders: %s\n", ms_ft.taskShader ? "true" : "false");
	ulog_info("testbed", "  mesh shaders: %s\n", ms_ft.meshShader ? "true" : "false");
	ulog_info("testbed", "  multiview: %s\n", ms_ft.multiviewMeshShader ? "true" : "false");
	ulog_info("testbed", "  m4: %s\n", m4_ft.maintenance4 ? "true" : "false");

	ms_ft.multiviewMeshShader = vk::False;
	ms_ft.primitiveFragmentShadingRateMeshShader = vk::False;

	separation.separateDepthStencilLayouts = vk::True;

	robustness.nullDescriptor = vk::True;

	// Initialize the device and surface
	engine.skeletonize(phdev, { 1920, 1080 }, "Neural Geometry Fields Testbed",
			extensions, ft, vk::PresentModeKHR::eImmediate);

	// Create the render pass
	engine.render_pass = littlevk::RenderPassAssembler(engine.device, engine.dal)
		.add_attachment(littlevk::default_color_attachment(engine.swapchain.format))
		.add_attachment(littlevk::default_depth_attachment())
		.add_subpass(vk::PipelineBindPoint::eGraphics)
			.color_attachment(0, vk::ImageLayout::eColorAttachmentOptimal)
			.depth_attachment(1, vk::ImageLayout::eDepthStencilAttachmentOptimal)
			.done();

	// Get everything to the correct size
	engine.resize();

	// Allocate command buffers
	engine.command_pool = littlevk::command_pool(engine.device,
		vk::CommandPoolCreateInfo {
			vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
			littlevk::find_graphics_queue_family(phdev)
		}
	).unwrap(engine.dal);

	engine.command_buffers = engine.device.allocateCommandBuffers({
		engine.command_pool,
		vk::CommandBufferLevel::ePrimary, 2
	});

	// Allocate descriptor pool
	vk::DescriptorPoolSize pool_sizes[] = {
		{ vk::DescriptorType::eStorageBuffer, 1 << 10 },
	};

	vk::DescriptorPoolCreateInfo pool_info = {};
	pool_info.poolSizeCount = sizeof(pool_sizes) / sizeof(pool_sizes[0]);
	pool_info.pPoolSizes = pool_sizes;
	pool_info.maxSets = 1 << 10;

	engine.descriptor_pool = littlevk::descriptor_pool
	(
		engine.device,
		pool_info
	).unwrap(engine.dal);

	// Present syncronization
	engine.sync = littlevk::present_syncronization(engine.device, 2).unwrap(engine.dal);

	// Configure pipelines
	configure_imgui(engine);

	using standalone::readfile;

	const std::string entry = "main";
	const std::filesystem::path mesh_shader = SHADERS_DIRECTORY "/ngf.mesh";
	const std::filesystem::path task_shader = SHADERS_DIRECTORY "/ngf.task";

	const littlevk::shader::Defines defines {
		{ "FEATURE_SIZE", std::to_string(fsize) }
	};

	for (const auto &[key, info] : fragment_shaders) {
		auto bundle = littlevk::ShaderStageBundle(engine.device, engine.dal)
			.file(mesh_shader, vk::ShaderStageFlagBits::eMeshEXT, entry, {}, defines)
			.file(task_shader, vk::ShaderStageFlagBits::eTaskEXT, entry, {}, defines)
			.file(info.path, vk::ShaderStageFlagBits::eFragment, entry, {}, defines);

		engine.primaries[key] = littlevk::PipelineAssembler <littlevk::eGraphics> (engine.device, engine.window, engine.dal)
			.with_render_pass(engine.render_pass, 0)
			.with_shader_bundle(bundle)
			.with_dsl_bindings(meshlet_dslbs)
			.polygon_mode(info.fill)
			.cull_mode(info.culling)
			.with_push_constant <TaskData> (vk::ShaderStageFlagBits::eMeshEXT | vk::ShaderStageFlagBits::eTaskEXT)
			.with_push_constant <ShadingData> (vk::ShaderStageFlagBits::eFragment, sizeof(TaskData));
	}

	auto environment_bundle = littlevk::ShaderStageBundle(engine.device, engine.dal)
		.file(SHADERS_DIRECTORY "/quad.vert", vk::ShaderStageFlagBits::eVertex)
		.file(SHADERS_DIRECTORY "/env.frag", vk::ShaderStageFlagBits::eFragment);

	engine.environment = littlevk::PipelineAssembler <littlevk::eGraphics> (engine.device, engine.window, engine.dal)
		.with_render_pass(engine.render_pass, 0)
		.with_shader_bundle(environment_bundle)
		.cull_mode(vk::CullModeFlagBits::eNone)
		.depth_stencil(false, false)
		.with_dsl_bindings(environment_dslbs)
		.with_push_constant <RayFrame> (vk::ShaderStageFlagBits::eFragment);

	// Other configurations
	engine.camera.from(engine.aspect_ratio());

	// Configure callbacks
	GLFWwindow *win = engine.window->handle;

	glfwSetWindowUserPointer(win, &engine.camera_transform);
	glfwSetMouseButtonCallback(win, button_callback);
	glfwSetCursorPosCallback(win, cursor_callback);

	return engine;
}
