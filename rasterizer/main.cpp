#include <cstdlib>
#include <iostream>
#include <iterator>
#include <optional>
#include <fstream>

#include <glm/glm.hpp>

#include <littlevk/littlevk.hpp>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include <implot.h>
#include <vulkan/vulkan_enums.hpp>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#include "common.hpp"

struct Texture {
	int width;
	int height;
	int channels;
	std::vector <uint8_t> pixels;
};

Texture load_texture(const std::filesystem::path &path)
{
	std::string tr = path.string();
	if (!std::filesystem::exists(path)) {
		ulog_error("load_texture", "load_texture: could not find path : %s\n", tr.c_str());
		return {};
	}

	int width;
	int height;
	int channels;

	stbi_set_flip_vertically_on_load(true);

	uint8_t *pixels = stbi_load(tr.c_str(), &width, &height, &channels, 4);

	std::vector <uint8_t> vector;
	vector.resize(width * height * 4);
	memcpy(vector.data(), pixels, vector.size() * sizeof(uint8_t));

	return Texture {
		.width = width,
		.height = height,
		.channels = channels,
		.pixels = vector
	};
}

struct Engine;

struct MVP {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
};

struct alignas(16) TaskData {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;

	// TODO: procedural texture synthesis?
	int flags;
	int resolution;
	alignas(16) glm::vec3 viewing;
	float time;
};

struct alignas(16) ShadingData {
	alignas(16) glm::vec3 viewing;
	alignas(16) glm::vec3 color;
};

struct MouseInfo {
	bool left_drag = false;
	bool voided = true;
	float last_x = 0.0f;
	float last_y = 0.0f;
} static mouse;

void button_callback(GLFWwindow *window, int button, int action, int mods)
{
	// Ignore if on ImGui window
	ImGuiIO &io = ImGui::GetIO();
	io.AddMouseButtonEvent(button, action);

	if (ImGui::GetIO().WantCaptureMouse)
		return;

	if (button == GLFW_MOUSE_BUTTON_LEFT) {
		mouse.left_drag = (action == GLFW_PRESS);
		if (action == GLFW_RELEASE)
			mouse.voided = true;
	}
}

void cursor_callback(GLFWwindow *window, double xpos, double ypos)
{
	Transform *camera_transform = (Transform *) glfwGetWindowUserPointer(window);

	// Ignore if on ImGui window
	ImGuiIO &io = ImGui::GetIO();
	io.MousePos = ImVec2(xpos, ypos);

	if (io.WantCaptureMouse)
		return;

	if (mouse.voided) {
		mouse.last_x = xpos;
		mouse.last_y = ypos;
		mouse.voided = false;
	}

	float xoffset = xpos - mouse.last_x;
	float yoffset = ypos - mouse.last_y;

	mouse.last_x = xpos;
	mouse.last_y = ypos;

	constexpr float sensitivity = 0.001f;
	xoffset *= sensitivity;
	yoffset *= sensitivity;

	if (mouse.left_drag) {
		camera_transform->rotation.x += yoffset;
		camera_transform->rotation.y -= xoffset;

		if (camera_transform->rotation.x > 89.0f)
			camera_transform->rotation.x = 89.0f;
		if (camera_transform->rotation.x < -89.0f)
			camera_transform->rotation.x = -89.0f;
	}
}

constexpr vk::DescriptorSetLayoutBinding texture_at(uint32_t binding, vk::ShaderStageFlagBits extra = vk::ShaderStageFlagBits::eMeshEXT)
{
	return vk::DescriptorSetLayoutBinding {
		binding, vk::DescriptorType::eCombinedImageSampler,
		1, vk::ShaderStageFlagBits::eMeshEXT | extra
	};
}

constexpr std::array <vk::DescriptorSetLayoutBinding, 8> meshlet_dslbs {
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

constexpr std::array <vk::DescriptorSetLayoutBinding, 1> environment_dslbs {
	vk::DescriptorSetLayoutBinding {
		0, vk::DescriptorType::eCombinedImageSampler,
		1, vk::ShaderStageFlagBits::eFragment
	},
};

struct FragmentShaderInfo {
	std::filesystem::path path;
	vk::CullModeFlags culling = vk::CullModeFlagBits::eBack;
	vk::PolygonMode fill = vk::PolygonMode::eFill;
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

struct Engine : littlevk::Skeleton {
	vk::PhysicalDevice phdev;
	vk::PhysicalDeviceMemoryProperties memory_properties;

	littlevk::Deallocator *dal = nullptr;

	vk::RenderPass render_pass;
	vk::CommandPool command_pool;
	vk::DescriptorPool descriptor_pool;

	std::vector <littlevk::Image> depth;
	std::vector <vk::Framebuffer> framebuffers;

	std::vector <vk::CommandBuffer> command_buffers;

	littlevk::PresentSyncronization sync;

	// Pipelines
	// littlevk::Pipeline meshlet;
	std::unordered_map <std::string, littlevk::Pipeline> primaries;

	littlevk::Pipeline environment;

	// ImGui resources
	vk::DescriptorPool imgui_descriptor_pool;

	// View parameters
	Camera camera;
	Transform camera_transform;

	MVP mvp;

	// Other frame information
	float last_time;

	// Resizing framebuffers
	void resize() {
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
	littlevk::Image upload_texture(const Texture &tex) {
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
				littlevk::transition(cmd, image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
				littlevk::copy_buffer_to_image(cmd, image, staging, vk::ImageLayout::eTransferDstOptimal);
				littlevk::transition(cmd, image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
			}
		);

		// Free interim data
		littlevk::destroy_buffer(device, staging);

		return image;
	}

	// Construction
	static void configure_imgui(Engine &engine) {
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

	static Engine from(const vk::PhysicalDevice &phdev, const std::vector <const char *> &extensions, size_t fsize) {
		Engine engine;

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

	// TODO: deallocate
};

static bool valid_window(const Engine &engine)
{
	return glfwWindowShouldClose(engine.window->handle) == 0;
}

void handle_key_input(Engine &engine, Transform &camera_transform)
{
	constexpr float speed = 2.5f;

	float delta = speed * float(glfwGetTime() - engine.last_time);
	engine.last_time = glfwGetTime();

	GLFWwindow *win = engine.window->handle;

	glm::vec3 velocity(0.0f);
	if (glfwGetKey(win, GLFW_KEY_S) == GLFW_PRESS)
		velocity.z -= delta;
	else if (glfwGetKey(win, GLFW_KEY_W) == GLFW_PRESS)
		velocity.z += delta;

	if (glfwGetKey(win, GLFW_KEY_D) == GLFW_PRESS)
		velocity.x -= delta;
	else if (glfwGetKey(win, GLFW_KEY_A) == GLFW_PRESS)
		velocity.x += delta;

	if (glfwGetKey(win, GLFW_KEY_E) == GLFW_PRESS)
		velocity.y += delta;
	else if (glfwGetKey(win, GLFW_KEY_Q) == GLFW_PRESS)
		velocity.y -= delta;

	glm::quat q = glm::quat(camera_transform.rotation);
	velocity = q * glm::vec4(velocity, 0.0f);
	camera_transform.position += velocity;
}

std::optional <std::pair <vk::CommandBuffer, littlevk::SurfaceOperation>> new_frame(Engine &engine, size_t frame)
{
	// Handle input
	handle_key_input(engine, engine.camera_transform);

	// Update camera state before passing to render hooks
	engine.camera.aspect = engine.aspect_ratio();
	engine.mvp.view = engine.camera.view_matrix(engine.camera_transform);
	engine.mvp.proj = engine.camera.perspective_matrix();

	// Get next image
	littlevk::SurfaceOperation op;
	op = littlevk::acquire_image(engine.device, engine.swapchain.swapchain, engine.sync[frame]);
	if (op.status == littlevk::SurfaceOperation::eResize) {
		engine.resize();
		return std::nullopt;
	}

	vk::CommandBuffer cmd = engine.command_buffers[frame];
	cmd.begin(vk::CommandBufferBeginInfo {});

	littlevk::viewport_and_scissor(cmd, littlevk::RenderArea(engine.window));

	// Record command buffer
	return std::make_pair(cmd, op);
}

void end_frame(const Engine &engine, const vk::CommandBuffer &cmd, size_t frame)
{
	cmd.end();

	// Submit command buffer while signaling the semaphore
	vk::PipelineStageFlags wait_stage = vk::PipelineStageFlagBits::eColorAttachmentOutput;

	vk::SubmitInfo submit_info {
		1, &engine.sync.image_available[frame],
		&wait_stage,
		1, &cmd,
		1, &engine.sync.render_finished[frame]
	};

	engine.graphics_queue.submit(submit_info, engine.sync.in_flight[frame]);
}

void present_frame(Engine &engine, const littlevk::SurfaceOperation &op, size_t frame)
{
	// Send image to the screen
	littlevk::SurfaceOperation pop = littlevk::present_image(engine.present_queue, engine.swapchain.swapchain, engine.sync[frame], op.index);
	if (pop.status == littlevk::SurfaceOperation::eResize)
		engine.resize();
}

void render_pass_begin(const Engine &engine, const vk::CommandBuffer &cmd, const littlevk::SurfaceOperation &op, const glm::vec4 &color)
{
	const auto &rpbi = littlevk::default_rp_begin_info <2>
		(engine.render_pass, engine.framebuffers[op.index], engine.window)
		.clear_value(0, vk::ClearColorValue {
			std::array <float, 4> { color.x, color.y, color.z, color.w }
		});

	return cmd.beginRenderPass(rpbi, vk::SubpassContents::eInline);
}

void render_pass_end(const Engine &engine, const vk::CommandBuffer &cmd)
{
	return cmd.endRenderPass();
}

int main(int argc, char *argv[])
{
	littlevk::config().enable_validation_layers = false;
	littlevk::config().enable_logging = false;

	if (argc < 2) {
		ulog_error("testbed", "Usage: testbed <ngf>\n");
		return EXIT_FAILURE;
	}

	std::string path_ngf = argv[1];

	// Load the neural geometry field
	constexpr int32_t LAYERS = 4;

	struct Tensor {
		std::vector <float> vec;
		int32_t width;
		int32_t height;
	};

	struct NGF {
		std::vector <glm::ivec4> patches;
		std::vector <glm::vec4> vertices;
		std::vector <float> features;

		uint32_t patch_count;
		uint32_t feature_size;

		std::array <Tensor, LAYERS> weights;
		std::array <Tensor, LAYERS> biases;
	} ngf;

	{
		std::ifstream fin(path_ngf);
		ulog_assert(fin.good(), "Bad ngf file %s\n", path_ngf.c_str());

		int32_t sizes[3];
		fin.read(reinterpret_cast <char *> (sizes), sizeof(sizes));
		ulog_info("ngf io", "%d patches, %d vertices, %d feature size\n", sizes[0], sizes[1], sizes[2]);

		std::vector <glm::ivec4> patches;
		std::vector <glm::vec3> vertices;
		std::vector <float> features;

		patches.resize(sizes[0]);
		vertices.resize(sizes[1]);
		features.resize(sizes[1] * sizes[2]);

		ngf.patch_count = sizes[0];
		ngf.feature_size = sizes[2];

		fin.read(reinterpret_cast <char *> (vertices.data()), vertices.size() * sizeof(glm::vec3));
		fin.read(reinterpret_cast <char *> (features.data()), features.size() * sizeof(float));
		fin.read(reinterpret_cast <char *> (patches.data()), patches.size() * sizeof(glm::ivec4));

		ulog_info("ngf io", "read patches data\n");

		std::array <Tensor, LAYERS> weights;
		for (int32_t i = 0; i < LAYERS; i++) {
			int32_t sizes[2];
			fin.read(reinterpret_cast <char *> (sizes), sizeof(sizes));
			ulog_info("ngf io", "weight matrix with size %d x %d\n", sizes[0], sizes[1]);

			Tensor w;
			w.width = sizes[0];
			w.height = sizes[1];
			w.vec.resize(sizes[0] * sizes[1]);
			fin.read(reinterpret_cast <char *> (w.vec.data()), w.vec.size() * sizeof(float));

			weights[i] = w;
		}

		std::array <Tensor, LAYERS> biases;
		for (int32_t i = 0; i < LAYERS; i++) {
			int32_t size;
			fin.read(reinterpret_cast <char *> (&size), sizeof(size));
			ulog_info("ngf io", "bias vector with size %d\n", size);

			Tensor w;
			w.width = size;
			w.height = 1;
			w.vec.resize(size);
			fin.read(reinterpret_cast <char *> (w.vec.data()), w.vec.size() * sizeof(float));

			biases[i] = w;
		}

		ngf.patches = patches;
		ngf.features = features;
		ngf.weights = weights;
		ngf.biases = biases;

		// Need special care for vertices to align them properly
		ngf.vertices.resize(vertices.size());
		for (int32_t i = 0; i < vertices.size(); i++)
			ngf.vertices[i] = glm::vec4(vertices[i], 0.0f);
	}

	// Configure renderer
	static const std::vector <const char *> extensions {
		VK_EXT_MESH_SHADER_EXTENSION_NAME,
		VK_EXT_ROBUSTNESS_2_EXTENSION_NAME,
		VK_KHR_MAINTENANCE_4_EXTENSION_NAME,
		VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME,
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
	};

	auto predicate = [](const vk::PhysicalDevice &phdev) {
		return littlevk::physical_device_able(phdev, extensions);
	};

	vk::PhysicalDevice phdev = littlevk::pick_physical_device(predicate);

	// Initialization
	Engine engine = Engine::from(phdev, extensions, ngf.feature_size);

	engine.camera_transform.position = glm::vec3 { 0, 0, -2.3 };

	// Device buffers for the neural geometry field
	struct {
		littlevk::Buffer vertices;
		littlevk::Buffer features;
		littlevk::Buffer patches;
		littlevk::Buffer network;
		vk::DescriptorSet dset;
	} vk_ngf;

	// Concatenate the neural network weights
	std::vector <float> network;
	for (int32_t i = 0; i < LAYERS; i++) {
		network.insert(network.begin(), ngf.weights[i].vec.begin(), ngf.weights[i].vec.end());
		network.insert(network.begin(), ngf.biases[i].vec.begin(), ngf.biases[i].vec.end());
	}

	// Concatenate the biases into a single buffer
	std::vector <float> biases;
	for (int32_t i = 0; i < LAYERS; i++)
		biases.insert(biases.end(), ngf.biases[i].vec.begin(), ngf.biases[i].vec.end());

	// Align to vec4 size
	size_t fixed = (biases.size() + 3)/4;
	biases.resize(4 * fixed);

	littlevk::Image bias_texture;
	littlevk::Buffer bias_buffer;

	// Weights (transposed)
	size_t w0c = ngf.weights[0].height;
	std::vector <float> W0(64 * w0c);
	std::vector <float> W1(64 * 64);
	std::vector <float> W2(64 * 64);
	std::vector <float> W3 = ngf.weights[3].vec;

	for (int i = 0; i < 64; i++) {
		for (int j = 0; j < 64; j++) {
			W1[j * 64 + i] = ngf.weights[1].vec[i * 64 + j];
			W2[j * 64 + i] = ngf.weights[2].vec[i * 64 + j];
		}

		for (int j = 0; j < w0c; j++)
			W0[j * 64 + i] = ngf.weights[0].vec[i * w0c + j];
	}

	std::vector <float> all;

	// Feature vector
	std::vector <glm::vec4> features(ngf.patch_count * ngf.feature_size);

	for (size_t i = 0; i < ngf.patch_count; i++) {
		glm::ivec4 complex = ngf.patches[i];
		for (size_t j = 0; j < ngf.feature_size; j++) {
			float f0 = ngf.features[complex.x * ngf.feature_size + j];
			float f1 = ngf.features[complex.y * ngf.feature_size + j];
			float f2 = ngf.features[complex.z * ngf.feature_size + j];
			float f3 = ngf.features[complex.w * ngf.feature_size + j];
			features[i * ngf.feature_size + j] = glm::vec4(f0, f1, f2, f3);
		}
	}

	littlevk::Image complex_texture;
	littlevk::Buffer complex_buffer;

	littlevk::Image vertex_texture;
	littlevk::Buffer vertex_buffer;

	littlevk::Image feature_texture;
	littlevk::Buffer feature_buffer;

	littlevk::Image W0_texture;
	littlevk::Buffer W0_buffer;

	littlevk::Image W1_texture;
	littlevk::Buffer W1_buffer;

	littlevk::Image W2_texture;
	littlevk::Buffer W2_buffer;

	littlevk::Image W3_texture;
	littlevk::Buffer W3_buffer;

	std::tie(bias_texture, bias_buffer,
			complex_texture, complex_buffer,
			vertex_texture, vertex_buffer,
			feature_texture, feature_buffer,
			W0_texture, W0_buffer,
			W1_texture, W1_buffer,
			W2_texture, W2_buffer,
			W3_texture, W3_buffer) = littlevk::linked_device_allocator(engine.device, engine.memory_properties, engine.dal)
		.image(fixed, 1,
			vk::Format::eR32G32B32A32Sfloat,
			vk::ImageUsageFlagBits::eSampled
				| vk::ImageUsageFlagBits::eTransferDst,
			vk::ImageAspectFlagBits::eColor,
			vk::ImageType::e1D,
			vk::ImageViewType::e1D)
		.buffer(biases, vk::BufferUsageFlagBits::eTransferSrc)
		.image(ngf.patch_count, 1,
			vk::Format::eR32G32B32A32Sfloat,
			vk::ImageUsageFlagBits::eSampled
				| vk::ImageUsageFlagBits::eTransferDst,
			vk::ImageAspectFlagBits::eColor,
			vk::ImageType::e1D,
			vk::ImageViewType::e1D)
		.buffer(ngf.patches, vk::BufferUsageFlagBits::eTransferSrc)
		.image(ngf.vertices.size(), 1,
			vk::Format::eR32G32B32A32Sfloat,
			vk::ImageUsageFlagBits::eSampled
				| vk::ImageUsageFlagBits::eTransferDst,
			vk::ImageAspectFlagBits::eColor,
			vk::ImageType::e1D,
			vk::ImageViewType::e1D)
		.buffer(ngf.vertices, vk::BufferUsageFlagBits::eTransferSrc)
		.image(ngf.feature_size, ngf.patch_count,
			vk::Format::eR32G32B32A32Sfloat,
			vk::ImageUsageFlagBits::eSampled
				| vk::ImageUsageFlagBits::eTransferDst,
			vk::ImageAspectFlagBits::eColor,
			vk::ImageType::e2D,
			vk::ImageViewType::e2D)
		.buffer(features, vk::BufferUsageFlagBits::eTransferSrc)
		.image(16, w0c,
			vk::Format::eR32G32B32A32Sfloat,
			vk::ImageUsageFlagBits::eSampled
				| vk::ImageUsageFlagBits::eTransferDst,
			vk::ImageAspectFlagBits::eColor,
			vk::ImageType::e2D,
			vk::ImageViewType::e2D)
		.buffer(W0, vk::BufferUsageFlagBits::eTransferSrc)
		.image(16, 64,
			vk::Format::eR32G32B32A32Sfloat,
			vk::ImageUsageFlagBits::eSampled
				| vk::ImageUsageFlagBits::eTransferDst,
			vk::ImageAspectFlagBits::eColor,
			vk::ImageType::e2D,
			vk::ImageViewType::e2D)
		.buffer(W1, vk::BufferUsageFlagBits::eTransferSrc)
		.image(16, 64,
			vk::Format::eR32G32B32A32Sfloat,
			vk::ImageUsageFlagBits::eSampled
				| vk::ImageUsageFlagBits::eTransferDst,
			vk::ImageAspectFlagBits::eColor,
			vk::ImageType::e2D,
			vk::ImageViewType::e2D)
		.buffer(W2, vk::BufferUsageFlagBits::eTransferSrc)
		.image(16, 3,
			vk::Format::eR32G32B32A32Sfloat,
			vk::ImageUsageFlagBits::eSampled
				| vk::ImageUsageFlagBits::eTransferDst,
			vk::ImageAspectFlagBits::eColor,
			vk::ImageType::e2D,
			vk::ImageViewType::e2D)
		.buffer(W3, vk::BufferUsageFlagBits::eTransferSrc);

	// Peform all the transfers
	auto copy_to_texure = [](const vk::CommandBuffer &cmd, const littlevk::Image &texture, const littlevk::Buffer &buffer) {
		littlevk::transition(cmd, texture,
				vk::ImageLayout::eUndefined,
				vk::ImageLayout::eTransferDstOptimal);

		littlevk::copy_buffer_to_image(cmd, texture, buffer,
				vk::ImageLayout::eTransferDstOptimal);

		littlevk::transition(cmd, texture,
				vk::ImageLayout::eTransferDstOptimal,
				vk::ImageLayout::eShaderReadOnlyOptimal);
	};

	littlevk::submit_now(engine.device, engine.command_pool, engine.graphics_queue,
		[&](const vk::CommandBuffer &cmd) {
			copy_to_texure(cmd, bias_texture, bias_buffer);
			copy_to_texure(cmd, complex_texture, complex_buffer);
			copy_to_texure(cmd, vertex_texture, vertex_buffer);
			copy_to_texure(cmd, feature_texture, feature_buffer);

			copy_to_texure(cmd, W0_texture, W0_buffer);
			copy_to_texure(cmd, W1_texture, W1_buffer);
			copy_to_texure(cmd, W2_texture, W2_buffer);
			copy_to_texure(cmd, W3_texture, W3_buffer);
		}
	);

	std::tie(vk_ngf.vertices, vk_ngf.features, vk_ngf.patches, vk_ngf.network) = littlevk::linked_device_allocator(engine.device, engine.memory_properties, engine.dal)
		.buffer(ngf.vertices, vk::BufferUsageFlagBits::eStorageBuffer)
		.buffer(ngf.features, vk::BufferUsageFlagBits::eStorageBuffer)
		.buffer(ngf.patches, vk::BufferUsageFlagBits::eStorageBuffer)
		.buffer(network, vk::BufferUsageFlagBits::eStorageBuffer);

	vk_ngf.dset = littlevk::bind(engine.device, engine.descriptor_pool)
		.allocate_descriptor_sets(*engine.primaries["Shaded"].dsl).front();

	vk::Sampler floating_sampler = littlevk::SamplerAssembler(engine.device, engine.dal);

	auto SROO = vk::ImageLayout::eShaderReadOnlyOptimal;

	littlevk::bind(engine.device, vk_ngf.dset, meshlet_dslbs)
		.update(0, 0, floating_sampler, complex_texture.view, SROO)
		.update(1, 0, floating_sampler, vertex_texture.view, SROO)
		.update(2, 0, floating_sampler, feature_texture.view, SROO)
		.update(3, 0, floating_sampler, bias_texture.view, SROO)
		.update(4, 0, floating_sampler, W0_texture.view, SROO)
		.update(5, 0, floating_sampler, W1_texture.view, SROO)
		.update(6, 0, floating_sampler, W2_texture.view, SROO)
		.update(7, 0, floating_sampler, W3_texture.view, SROO)
		.finalize();

	// Environment map
	vk::Sampler sampler = littlevk::SamplerAssembler(engine.device, engine.dal);

	Texture tex = load_texture("resources/environment.hdr");
	littlevk::Image dtex = engine.upload_texture(tex);

	vk::DescriptorSet env_dset = littlevk::bind(engine.device, engine.descriptor_pool)
		.allocate_descriptor_sets(*engine.environment.dsl).front();

	littlevk::bind(engine.device, env_dset, environment_dslbs)
		.update(0, 0, sampler, dtex.view, vk::ImageLayout::eShaderReadOnlyOptimal)
		.finalize();

	// Plotting data
	std::string key = "Normals";

	int resolution = 15;
	bool backface_culling = false;

	size_t frame = 0;
	while (valid_window(engine)) {
		// Get events
		glfwPollEvents();

		// Frame
		auto frame_info = new_frame(engine, frame);
		if (!frame_info)
			continue;

		auto [cmd, op] = *frame_info;

		glm::vec4 color(1.0f);
		if (key == "Depth")
			color = glm::vec4(0.0f);

		render_pass_begin(engine, cmd, op, color);

		if (key == "Shaded") {
			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, engine.environment.handle);

			RayFrame rayframe = engine.camera.rayframe(engine.camera_transform);

			cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
					engine.environment.layout,
					0, { env_dset }, nullptr);

			cmd.pushConstants <RayFrame> (engine.environment.layout,
				vk::ShaderStageFlagBits::eFragment,
				0, rayframe);

			cmd.bindVertexBuffers(0, { VK_NULL_HANDLE }, { 0 });
			cmd.draw(6, 1, 0, 0);
		}

		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, engine.primaries[key].handle);

		cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
				engine.primaries[key].layout,
				0, { vk_ngf.dset }, nullptr);

		float time = glfwGetTime();

		// Task/Mesh shader push constants
		int flags = 0;
		flags |= int(backface_culling);

		TaskData ngf_pc {
			Transform().matrix(),
			engine.mvp.view,
			engine.mvp.proj,
			flags, resolution,
			glm::vec3(glm::inverse(ngf_pc.view) * glm::vec4(0, 0, 1, 0)),
			time,
		};

		cmd.pushConstants <TaskData> (engine.primaries[key].layout,
			vk::ShaderStageFlagBits::eMeshEXT | vk::ShaderStageFlagBits::eTaskEXT,
			0, ngf_pc);

		// Fragment shader push constants
		ShadingData shading_pc {
			.viewing = glm::vec3(glm::inverse(ngf_pc.view) * glm::vec4(0, 0, 1, 0)),
			.color = glm::vec3(0.6f, 0.5f, 1.0f),
		};

		cmd.pushConstants <ShadingData> (engine.primaries[key].layout,
			vk::ShaderStageFlagBits::eFragment,
			sizeof(TaskData), shading_pc);

		cmd.drawMeshTasksEXT(ngf.patch_count, 1, 1);

		// ImGui pass
		{
			static std::vector <float> frametimes;
			static const size_t WINDOW = 60;

			ImGui_ImplVulkan_NewFrame();
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();

			ImGui::Begin("Panel");

			float fr = ImGui::GetIO().Framerate;
			float ft = 1.0f/fr;

			frametimes.push_back(ft);
			if (frametimes.size() > WINDOW)
				frametimes.erase(frametimes.begin());

			auto tflags = ImGuiTreeNodeFlags_DefaultOpen;
			if (ImGui::CollapsingHeader("Performance", tflags)) {
				float max = -1e10f;
				for (float ft : frametimes)
					max = std::max(max, ft);

				ImGui::Text("%05.2f ms per frame (average)", ft * 1000.0f);
				ImGui::Text("%05.2f ms per frame (max)", max * 1000.0f);
				ImGui::Text("%04d frames per second", (int) fr);
			}

			if (ImGui::CollapsingHeader("Statistics", tflags)) {
				ImGui::Text("%d patches", ngf.patch_count);
			}

			if (ImGui::CollapsingHeader("Render mode", tflags)) {
				for (const auto &[k, _] : fragment_shaders) {
					if (ImGui::RadioButton(k.c_str(), key == k))
						key = k;
				}
			}

			if (ImGui::CollapsingHeader("Options", tflags)) {
				ImGui::Checkbox("Backface culling (aprox.)", &backface_culling);
				ImGui::DragInt("Tessellation", (int *) &resolution, 0.05f, 2, 15);
			}

			ImGui::End();

			ImGui::Render();
			ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
		 }

		render_pass_end(engine, cmd);

		// Complete the frame
		end_frame(engine, cmd, frame);

		// Present the frame and submit
		present_frame(engine, op, frame);

		// Post frame
		frame = 1 - frame;
	}
}
