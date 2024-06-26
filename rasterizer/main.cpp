#include <cstdlib>
#include <iostream>
#include <optional>
#include <fstream>

#include <glm/glm.hpp>

#include <littlevk/littlevk.hpp>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include <implot.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

#include "common.hpp"

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

	glm::vec2 extent;
	int resolution;
	float time;
};

struct alignas(16) ShadingData {
	alignas(16) glm::vec3 viewing;
	alignas(16) glm::vec3 color;
	int mode;
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

constexpr std::array <vk::DescriptorSetLayoutBinding, 4> meshlet_dslbs {
	vk::DescriptorSetLayoutBinding { 0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eMeshEXT | vk::ShaderStageFlagBits::eTaskEXT },
	vk::DescriptorSetLayoutBinding { 1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eMeshEXT },
	vk::DescriptorSetLayoutBinding { 2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eMeshEXT | vk::ShaderStageFlagBits::eTaskEXT },
	vk::DescriptorSetLayoutBinding { 3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eMeshEXT },
};

struct Engine : littlevk::Skeleton {
	vk::PhysicalDevice phdev;
	vk::PhysicalDeviceMemoryProperties memory_properties;

	littlevk::Deallocator *dal = nullptr;

	vk::RenderPass render_pass;
	vk::CommandPool command_pool;
	vk::DescriptorPool descriptor_pool;

	std::vector <vk::Framebuffer> framebuffers;
	std::vector <vk::CommandBuffer> command_buffers;

	littlevk::PresentSyncronization sync;

	// Pipelines
	littlevk::Pipeline meshlet;

	// ImGui resources
	vk::DescriptorPool imgui_descriptor_pool;

	// View parameters
	Camera camera;
	Transform camera_transform;

	MVP mvp;

	// Other frame information
	float last_time;

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

		ImGui_ImplVulkan_Init(&init_info);

		// Upload fonts
		ImGui_ImplVulkan_CreateFontsTexture();

		// Configure ImPlot as well
		ImPlot::CreateContext();
	}

	static Engine from(const vk::PhysicalDevice &phdev, const std::vector <const char *> &extensions) {
		Engine engine;

		engine.phdev = phdev;
		engine.memory_properties = phdev.getMemoryProperties();
		engine.dal = new littlevk::Deallocator(engine.device);

		// Analyze the properties
		vk::PhysicalDeviceMeshShaderPropertiesEXT ms_properties = {};
		vk::PhysicalDeviceProperties2 properties = {};
		properties.pNext = &ms_properties;

		phdev.getProperties2(&properties);
		printf("properties:\n");
		printf("  max (task) payload memory: %d KB\n", ms_properties.maxTaskPayloadSize / 1024);
		printf("  max (task) shared memory: %d KB\n", ms_properties.maxTaskSharedMemorySize / 1024);
		printf("  max (mesh) shared memory: %d KB\n", ms_properties.maxMeshSharedMemorySize / 1024);
		printf("  max output vertices: %d\n", ms_properties.maxMeshOutputVertices);
		printf("  max output primitives: %d\n", ms_properties.maxMeshOutputPrimitives);
		printf("  max work group invocations: %d\n", ms_properties.maxMeshWorkGroupInvocations);

		// Configure the features
		vk::PhysicalDeviceMeshShaderFeaturesEXT ms_ft = {};
		vk::PhysicalDeviceMaintenance4FeaturesKHR m4_ft = {};
		vk::PhysicalDeviceFeatures2KHR ft = {};

		ft.features.independentBlend = true;
		ft.features.fillModeNonSolid = true;
		ft.features.geometryShader = true;

		ft.pNext = &ms_ft;
		ms_ft.pNext = &m4_ft;

		phdev.getFeatures2(&ft);

		printf("features:\n");
		printf("  task shaders: %s\n", ms_ft.taskShader ? "true" : "false");
		printf("  mesh shaders: %s\n", ms_ft.meshShader ? "true" : "false");
		printf("  multiview: %s\n", ms_ft.multiviewMeshShader ? "true" : "false");
		printf("  m4: %s\n", m4_ft.maintenance4 ? "true" : "false");

		ms_ft.multiviewMeshShader = vk::False;
		ms_ft.primitiveFragmentShadingRateMeshShader = vk::False;

		// Initialize the device and surface
		engine.skeletonize(phdev, { 1920, 1080 }, "Neural Geometry Fields Testbed", extensions, ft, vk::PresentModeKHR::eImmediate);

		// Create the render pass
		engine.render_pass = littlevk::RenderPassAssembler(engine.device, engine.dal)
			.add_attachment(littlevk::default_color_attachment(engine.swapchain.format))
			.add_attachment(littlevk::default_depth_attachment())
			.add_subpass(vk::PipelineBindPoint::eGraphics)
				.color_attachment(0, vk::ImageLayout::eColorAttachmentOptimal)
				.depth_attachment(1, vk::ImageLayout::eDepthStencilAttachmentOptimal)
				.done();

		// Create the depth buffer
		littlevk::Image depth_buffer = bind(engine.device, engine.memory_properties, engine.dal)
			.image(engine.window->extent,
				vk::Format::eD32Sfloat,
				vk::ImageUsageFlagBits::eDepthStencilAttachment,
				vk::ImageAspectFlagBits::eDepth);

		// Create framebuffers from the swapchain
		littlevk::FramebufferGenerator generator(engine.device, engine.render_pass, engine.window->extent, engine.dal);
		for (const auto &view : engine.swapchain.image_views)
			generator.add(view, depth_buffer.view);

		engine.framebuffers = generator.unpack();

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

		auto bundle = littlevk::ShaderStageBundle(engine.device, engine.dal)
			.file(SHADERS_DIRECTORY "/ngf.mesh", vk::ShaderStageFlagBits::eMeshEXT)
			.file(SHADERS_DIRECTORY "/ngf.task", vk::ShaderStageFlagBits::eTaskEXT)
			.file(SHADERS_DIRECTORY "/ngf.frag", vk::ShaderStageFlagBits::eFragment);

		engine.meshlet = littlevk::PipelineAssembler <littlevk::eGraphics> (engine.device, engine.window, engine.dal)
			.with_render_pass(engine.render_pass, 0)
			.with_shader_bundle(bundle)
			.with_dsl_bindings(meshlet_dslbs)
			.with_push_constant <TaskData> (vk::ShaderStageFlagBits::eMeshEXT | vk::ShaderStageFlagBits::eTaskEXT)
			.with_push_constant <ShadingData> (vk::ShaderStageFlagBits::eFragment, sizeof(TaskData));

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

void render_pass_begin(const Engine &engine, const vk::CommandBuffer &cmd, const littlevk::SurfaceOperation &op)
{
	const auto &rpbi = littlevk::default_rp_begin_info <2>
		(engine.render_pass, engine.framebuffers[op.index], engine.window)
		.clear_value(0, vk::ClearColorValue {
			std::array <float, 4> { 1.0f, 1.0f, 1.0f, 1.0f }
		});

	return cmd.beginRenderPass(rpbi, vk::SubpassContents::eInline);
}

void render_pass_end(const Engine &engine, const vk::CommandBuffer &cmd)
{
	return cmd.endRenderPass();
}

int main(int argc, char *argv[])
{
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
		ulog_assert(ngf.feature_size == 20, "testbed", "Expected an NGF with feature size of 20.\n");

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
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
		VK_EXT_MESH_SHADER_EXTENSION_NAME,
		VK_KHR_MAINTENANCE_4_EXTENSION_NAME,
		VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME,
	};

	auto predicate = [](const vk::PhysicalDevice &phdev) {
		return littlevk::physical_device_able(phdev, extensions);
	};

	vk::PhysicalDevice phdev = littlevk::pick_physical_device(predicate);

	// Initialization
	Engine engine = Engine::from(phdev, extensions);

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

	std::tie(vk_ngf.vertices, vk_ngf.features, vk_ngf.patches, vk_ngf.network) = littlevk::linked_device_allocator(engine.device, engine.memory_properties, engine.dal)
		.buffer(ngf.vertices, vk::BufferUsageFlagBits::eStorageBuffer)
		.buffer(ngf.features, vk::BufferUsageFlagBits::eStorageBuffer)
		.buffer(ngf.patches, vk::BufferUsageFlagBits::eStorageBuffer)
		.buffer(network, vk::BufferUsageFlagBits::eStorageBuffer);

	vk_ngf.dset = littlevk::bind(engine.device, engine.descriptor_pool)
		.allocate_descriptor_sets(*engine.meshlet.dsl).front();

	littlevk::bind(engine.device, vk_ngf.dset, meshlet_dslbs)
		.update(0, 0, *vk_ngf.vertices, 0, sizeof(glm::vec4) * ngf.vertices.size())
		.update(1, 0, *vk_ngf.features, 0, sizeof(float) * ngf.features.size())
		.update(2, 0, *vk_ngf.patches, 0, sizeof(glm::ivec4) * ngf.patches.size())
		.update(3, 0, *vk_ngf.network, 0, sizeof(float) * network.size())
		.finalize();

	// Plotting data
	int mode = 0;
	int resolution = 15;

	size_t frame = 0;
	while (valid_window(engine)) {
		// Get events
		glfwPollEvents();

		// Frame
		auto frame_info = new_frame(engine, frame);
		if (!frame_info)
			continue;

		auto [cmd, op] = *frame_info;

		render_pass_begin(engine, cmd, op);

		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, engine.meshlet.handle);
		cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, engine.meshlet.layout, 0, { vk_ngf.dset }, nullptr);

		float time = glfwGetTime();

		// Task/Mesh shader push constants
		TaskData ngf_pc {
			Transform().matrix(),
			engine.mvp.view,
			engine.mvp.proj,
			{ engine.window->extent.width, engine.window->extent.height },
			resolution,
			time
		};

		cmd.pushConstants <TaskData> (engine.meshlet.layout,
			vk::ShaderStageFlagBits::eMeshEXT | vk::ShaderStageFlagBits::eTaskEXT,
			0, ngf_pc);

		// Fragment shader push constants
		ShadingData shading_pc {
			.viewing = glm::vec3(glm::inverse(ngf_pc.view) * glm::vec4(0, 0, 1, 0)),
			.color = glm::vec3(0.6f, 0.5f, 1.0f),
			.mode = mode
		};

		cmd.pushConstants <ShadingData> (engine.meshlet.layout,
			vk::ShaderStageFlagBits::eFragment,
			sizeof(TaskData), shading_pc);

		cmd.drawMeshTasksEXT(ngf.patch_count, 1, 1);

		// ImGui pass
		{
			ImGui_ImplVulkan_NewFrame();
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();

			ImGui::Begin("Info");

			float ft = ImGui::GetIO().DeltaTime;

			ImGui::Text("Performance");

			ImGui::Text("%05.2f ms per frame", ft * 1000.0f);
			ImGui::Text("%04d frames per second", int(1.0/ft));

			ImGui::Separator();

			ImGui::Text("Statistics");
			ImGui::Spacing();
			ImGui::Text("%d active patches", ngf.patch_count);

			ImGui::Separator();

			static const std::unordered_map <uint32_t, std::string> mode_descriptions {
				{ 0, "Patches" },
				{ 1, "Normal" },
				{ 2, "Shaded" }
			};

			ImGui::Text("Render mode");
			for (const auto &[m, desc] : mode_descriptions) {
				if (ImGui::RadioButton(desc.c_str(), mode == m))
					mode = m;
			}

			ImGui::Separator();

			ImGui::Text("Tessellation");
			ImGui::DragInt("Resolution", (int *) &resolution, 0.05f, 2, 15);

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
