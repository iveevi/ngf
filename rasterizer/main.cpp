#include <cstdlib>
#include <iostream>
#include <optional>

#include <glm/glm.hpp>

#include <littlevk/littlevk.hpp>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include "mesh.hpp"
#include "microlog.h"
#include "util.hpp"

struct Vertex {
	glm::vec3 position;
	glm::vec3 normal;
};

static constexpr vk::VertexInputBindingDescription vertex_binding {
	0, sizeof(Vertex), vk::VertexInputRate::eVertex,
};

static constexpr std::array <vk::VertexInputAttributeDescription, 2> vertex_attributes {
	vk::VertexInputAttributeDescription {
		0, 0, vk::Format::eR32G32B32Sfloat, 0
	},

	vk::VertexInputAttributeDescription {
		1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, normal)
	}
};

struct Engine;

struct VulkanMesh {
	littlevk::Buffer vertices;
	littlevk::Buffer triangles;
	size_t indices;

	static VulkanMesh from(const Engine &, const Mesh &);
};

struct BasePushConstants {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
};

struct Pipeline {
	vk::Pipeline pipeline;
	vk::PipelineLayout layout;
};

Pipeline ppl_normals
(
		const vk::Device      &device,
		const vk::RenderPass  &rp,
		const vk::Extent2D    &extent,
		littlevk::Deallocator *dal
)
{
	Pipeline ppl;

	// Read shader source
	std::string vertex_source = readfile("../mesh.vert.glsl");
	std::string fragment_source = readfile("../normals.frag.glsl");

	// Compile shader modules
	vk::ShaderModule vertex_module = littlevk::shader::compile(
		device, vertex_source,
		vk::ShaderStageFlagBits::eVertex
	).unwrap(dal);

	vk::ShaderModule fragment_module = littlevk::shader::compile(
		device, fragment_source,
		vk::ShaderStageFlagBits::eFragment
	).unwrap(dal);

	// Create the pipeline
	vk::PushConstantRange push_constant_range {
		vk::ShaderStageFlagBits::eVertex,
		0, sizeof(BasePushConstants)
	};

	ppl.layout = littlevk::pipeline_layout(
		device,
		vk::PipelineLayoutCreateInfo {
			{}, {}, push_constant_range
		}
	).unwrap(dal);

	littlevk::pipeline::GraphicsCreateInfo pipeline_info;
	pipeline_info.vertex_binding = vertex_binding;
	pipeline_info.vertex_attributes = vertex_attributes;
	pipeline_info.vertex_shader = vertex_module;
	pipeline_info.fragment_shader = fragment_module;
	pipeline_info.extent = extent;
	pipeline_info.pipeline_layout = ppl.layout;
	pipeline_info.render_pass = rp;
	pipeline_info.fill_mode = vk::PolygonMode::eFill;
	pipeline_info.cull_mode = vk::CullModeFlagBits::eNone;
	pipeline_info.dynamic_viewport = true;

	ppl.pipeline = littlevk::pipeline::compile(device, pipeline_info).unwrap(dal);

	return ppl;
}

struct MouseInfo {
	bool drag = false;
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
		mouse.drag = (action == GLFW_PRESS);
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

	if (mouse.drag) {
		camera_transform->rotation.x += yoffset;
		camera_transform->rotation.y -= xoffset;

		if (camera_transform->rotation.x > 89.0f)
			camera_transform->rotation.x = 89.0f;
		if (camera_transform->rotation.x < -89.0f)
			camera_transform->rotation.x = -89.0f;
	}
}

struct Engine : littlevk::Skeleton {
	vk::PhysicalDevice phdev;
	vk::PhysicalDeviceMemoryProperties memory_properties;

	littlevk::Deallocator *dal = nullptr;

	vk::RenderPass render_pass;
	vk::CommandPool command_pool;

	std::vector <vk::Framebuffer> framebuffers;
	std::vector <vk::CommandBuffer> command_buffers;

	littlevk::PresentSyncronization sync;

	// Pipelines
	Pipeline normals;

	// ImGui resources
	vk::DescriptorPool imgui_descriptor_pool;

	// View parameters
	Camera    camera;
	Transform camera_transform;

	BasePushConstants push_constants;

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

		ImGui_ImplVulkan_Init(&init_info, engine.render_pass);

		// Upload fonts
		littlevk::submit_now
		(
		 	engine.device,
			engine.command_pool,
			engine.graphics_queue,
			[&](const vk::CommandBuffer &cmd) {
				ImGui_ImplVulkan_CreateFontsTexture(cmd);
			}
		);
	}

	static Engine from(const vk::PhysicalDevice &phdev) {
		Engine engine;
		engine.skeletonize(phdev, { 1000, 1000 }, "Neural Geometry Fields Testbed");

		engine.phdev = phdev;
		engine.memory_properties = phdev.getMemoryProperties();
		engine.dal = new littlevk::Deallocator(engine.device);

		// Create the render pass
		engine.render_pass = littlevk::default_color_depth_render_pass
			(engine.device, engine.swapchain.format).unwrap(engine.dal);

		// Create the depth buffer
		littlevk::ImageCreateInfo depth_info {
			engine.window->extent.width,
			engine.window->extent.height,
			vk::Format::eD32Sfloat,
			vk::ImageUsageFlagBits::eDepthStencilAttachment,
			vk::ImageAspectFlagBits::eDepth,
		};

		littlevk::Image depth_buffer = littlevk::image(
			engine.device,
			depth_info, engine.memory_properties
		).unwrap(engine.dal);

		// Create framebuffers from the swapchain
		littlevk::FramebufferSetInfo fb_info;
		fb_info.swapchain = &engine.swapchain;
		fb_info.render_pass = engine.render_pass;
		fb_info.extent = engine.window->extent;
		fb_info.depth_buffer = &depth_buffer.view;

		engine.framebuffers = littlevk::framebuffers
			(engine.device, fb_info).unwrap(engine.dal);

		// Allocate command buffers
		engine.command_pool = littlevk::command_pool
		(
			engine.device,
			vk::CommandPoolCreateInfo {
				vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
				littlevk::find_graphics_queue_family(phdev)
			}
		).unwrap(engine.dal);

		engine.command_buffers = engine.device.allocateCommandBuffers({
			engine.command_pool,
			vk::CommandBufferLevel::ePrimary, 2
		});

		// Present syncronization
		engine.sync = littlevk::present_syncronization(engine.device, 2).unwrap(engine.dal);

		// Configure pipelines
		configure_imgui(engine);

		engine.normals = ppl_normals
		(
			engine.device,
			engine.render_pass,
			engine.window->extent,
			engine.dal
		);

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

VulkanMesh VulkanMesh::from(const Engine &engine, const Mesh &m)
{
	VulkanMesh vm;

	vm.indices = 3 * m.triangles.size();

	vm.vertices = littlevk::buffer
	(
		engine.device,
		interleave_attributes(m), // TODO: in an engine, put flags for which to include
		vk::BufferUsageFlagBits::eVertexBuffer,
		engine.memory_properties
	).unwrap(engine.dal);

	vm.triangles = littlevk::buffer
	(
	 	engine.device,
		m.triangles,
		vk::BufferUsageFlagBits::eIndexBuffer,
		engine.memory_properties
	).unwrap(engine.dal);

	return vm;
}

void handle_key_input(Engine &engine, Transform &camera_transform)
{
	constexpr float speed = 2.5f;

	float delta = speed * float(glfwGetTime() - engine.last_time);
	engine.last_time = glfwGetTime();

	// TODO: littlevk io system
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
	engine.push_constants.view = engine.camera.view_matrix(engine.camera_transform);
	engine.push_constants.proj = engine.camera.perspective_matrix();

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
	// TODO: littlevk shortcut for this...
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

const Pipeline &activate_pipeline(const Engine &engine, const vk::CommandBuffer &cmd)
{
	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, engine.normals.pipeline);
	return engine.normals;
}

int main(int argc, char *argv[])
{
	if (argc < 2) {
		ulog_error("testbed", "Usage: testbed <reference mesh>\n");
		return EXIT_FAILURE;
	}

	std::string path = argv[1];

	Mesh reference = *Mesh::load(path).begin();
	reference = Mesh::normalize(reference);

	ulog_info("testbed", "Loaded mesh with %d vertices and %d faces\n", reference.vertices.size(), reference.triangles.size());

	// Configure renderer
	auto predicate = [](vk::PhysicalDevice phdev) {
		return littlevk::physical_device_able(phdev, {
			VK_KHR_SWAPCHAIN_EXTENSION_NAME
		});
	};

	vk::PhysicalDevice phdev = littlevk::pick_physical_device(predicate);

	// Initialization
	Engine engine = Engine::from(phdev);

	VulkanMesh vk_ref = VulkanMesh::from(engine, reference);
	for (int i = 0; i < 1000000; i++)
		ulog_progress("bar", i/1000000.0f);

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

		const auto &ppl = activate_pipeline(engine, cmd);

		BasePushConstants push_constants = engine.push_constants;
		push_constants.model = Transform().matrix();

		cmd.pushConstants <BasePushConstants>
		(
			ppl.layout,
			vk::ShaderStageFlagBits::eVertex,
			0, push_constants
		);

		cmd.bindVertexBuffers(0, { vk_ref.vertices.buffer }, { 0 });
		cmd.bindIndexBuffer(vk_ref.triangles.buffer, 0, vk::IndexType::eUint32);
		cmd.drawIndexed(vk_ref.indices, 1, 0, 0, 0);

		render_pass_end(engine, cmd);

		end_frame(engine, cmd, frame);
		present_frame(engine, op, frame);

		// Post frame
		frame = 1 - frame;
	}
}
