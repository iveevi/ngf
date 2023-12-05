#include "common.hpp"

void Transform::from(const glm::vec3 &position_, const glm::vec3 &rotation_, const glm::vec3 &scale_)
{
	position = position_;
	rotation = rotation_;
	scale = scale_;
}

glm::mat4 Transform::matrix() const
{
	glm::mat4 pmat = glm::translate(glm::mat4(1.0f), position);
	glm::mat4 rmat = glm::mat4_cast(glm::quat(glm::radians(rotation)));
	glm::mat4 smat = glm::scale(glm::mat4(1.0f), scale);
	return pmat * rmat * smat;
}

glm::vec3 Transform::right() const
{
	glm::quat q = glm::quat(rotation);
	return glm::normalize(glm::vec3(q * glm::vec4(1.0f, 0.0f, 0.0f, 0.0f)));
}

glm::vec3 Transform::up() const
{
	glm::quat q = glm::quat(rotation);
	return glm::normalize(glm::vec3(q * glm::vec4(0.0f, 1.0f, 0.0f, 0.0f)));
}

glm::vec3 Transform::forward() const
{
	glm::quat q = glm::quat(rotation);
	return glm::normalize(glm::vec3(q * glm::vec4(0.0f, 0.0f, 1.0f, 0.0f)));
}

std::tuple <glm::vec3, glm::vec3, glm::vec3> Transform::axes() const
{
	glm::quat q = glm::quat(rotation);
	return std::make_tuple(
		glm::normalize(glm::vec3(q * glm::vec4(1.0f, 0.0f, 0.0f, 0.0f))),
		glm::normalize(glm::vec3(q * glm::vec4(0.0f, 1.0f, 0.0f, 0.0f))),
		glm::normalize(glm::vec3(q * glm::vec4(0.0f, 0.0f, 1.0f, 0.0f)))
	);
}

const std::string wireframe_vertex_shader_source = R"(
#version 450

layout (location = 0) in vec3 position;

layout (push_constant) uniform PushConstants {
	mat4 model;
	mat4 view;
	mat4 proj;
} push_constants;

void main()
{
	gl_Position = push_constants.proj * push_constants.view * push_constants.model * vec4(position, 1.0);
	gl_Position.y = -gl_Position.y;
	gl_Position.z = (gl_Position.z + gl_Position.w) / 2.0;
}
)";

const std::string wireframe_fragment_shader_source = R"(
#version 450

layout (location = 0) out vec4 fragment;

void main()
{
	fragment = vec4(0.0);
}
)";

const std::string solid_vertex_shader_source = R"(
#version 450

layout (location = 0) in vec3 position;

layout (push_constant) uniform PushConstants {
	mat4 model;
	mat4 view;
	mat4 proj;
} push_constants;

void main()
{
	gl_Position = push_constants.proj * push_constants.view * push_constants.model * vec4(position, 1.0);
	gl_Position.y = -gl_Position.y;
	gl_Position.z = (gl_Position.z + gl_Position.w) / 2.0;
}
)";

const std::string solid_fragment_shader_source = R"(
#version 450

layout (location = 0) out vec4 fragment;

vec3 color_wheel[] = vec3[16] (
	vec3(0.880, 0.520, 0.520),
	vec3(0.880, 0.655, 0.520),
	vec3(0.880, 0.790, 0.520),
	vec3(0.835, 0.880, 0.520),
	vec3(0.700, 0.880, 0.520),
	vec3(0.565, 0.880, 0.520),
	vec3(0.520, 0.880, 0.610),
	vec3(0.520, 0.880, 0.745),
	vec3(0.520, 0.880, 0.880),
	vec3(0.520, 0.745, 0.880),
	vec3(0.520, 0.610, 0.880),
	vec3(0.565, 0.520, 0.880),
	vec3(0.700, 0.520, 0.880),
	vec3(0.835, 0.520, 0.880),
	vec3(0.880, 0.520, 0.790),
	vec3(0.880, 0.520, 0.655)
);

// Murmur hash
uint hash(uint h)
{
	h ^= h >> 16;
	h *= 0x85ebca6b;
	h ^= h >> 13;
	h *= 0xc2b2ae35;
	h ^= h >> 16;
	return h;
}

void main()
{
	uint index = hash(gl_PrimitiveID) % 16;
	fragment = vec4(color_wheel[index], 1.0);
}
)";

const std::string normal_vertex_shader_source = R"(
#version 450

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;

layout (push_constant) uniform PushConstants {
	mat4 model;
	mat4 view;
	mat4 proj;
} push_constants;

layout (location = 0) out vec3 out_normal;

void main()
{
	gl_Position = push_constants.proj * push_constants.view * push_constants.model * vec4(position, 1.0);
	gl_Position.y = -gl_Position.y;
	gl_Position.z = (gl_Position.z + gl_Position.w) / 2.0;
	out_normal = normalize(mat3(push_constants.model) * normal);
}
)";

const std::string normal_fragment_shader_source = R"(
#version 450

layout (location = 0) in vec3 in_normal;
layout (location = 0) out vec4 fragment;

void main()
{
	fragment = vec4(0.5 * in_normal + 0.5, 1.0);
}
)";

const std::string shaded_vertex_shader_source = R"(
#version 450

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;

layout (push_constant) uniform PushConstants {
	mat4 model;
	mat4 view;
	mat4 proj;
} push_constants;

layout (location = 0) out vec3 out_normal;

void main()
{
	gl_Position = push_constants.proj * push_constants.view * push_constants.model * vec4(position, 1.0);
	gl_Position.y = -gl_Position.y;
	gl_Position.z = (gl_Position.z + gl_Position.w) / 2.0;
	out_normal = normalize(mat3(push_constants.model) * normal);
}
)";

const std::string shaded_fragment_shader_source = R"(
#version 450

layout (location = 0) in vec3 in_normal;
layout (location = 0) out vec4 fragment;

void main()
{
	vec3 light = normalize(vec3(1.0, 1.0, 1.0));
	float intensity = max(0.0, dot(in_normal, light));
	fragment = vec4(intensity, intensity, intensity, 1.0);
}
)";

void Renderer::configure_imgui()
{
	// Allocate descriptor pool
	vk::DescriptorPoolSize pool_sizes[] = {
		{ vk::DescriptorType::eSampler, 1 << 10 },
	};

	vk::DescriptorPoolCreateInfo pool_info = {};
	pool_info.poolSizeCount = sizeof(pool_sizes) / sizeof(pool_sizes[0]);
	pool_info.pPoolSizes = pool_sizes;
	pool_info.maxSets = 1 << 10;

	imgui_descriptor_pool = littlevk::descriptor_pool(device, pool_info).unwrap(dal);

	// Configure ImGui
	ImGui::CreateContext();
	ImGui::StyleColorsDark();

	ImGui_ImplGlfw_InitForVulkan(window->handle, true);

	ImGui_ImplVulkan_InitInfo init_info = {};
	init_info.Instance = littlevk::detail::get_vulkan_instance();
	init_info.PhysicalDevice = phdev;
	init_info.Device = device;
	init_info.QueueFamily = littlevk::find_graphics_queue_family(phdev);
	init_info.Queue = graphics_queue;
	init_info.PipelineCache = nullptr;
	init_info.DescriptorPool = imgui_descriptor_pool;
	init_info.Allocator = nullptr;
	init_info.MinImageCount = 2;
	init_info.ImageCount = 2;
	init_info.CheckVkResultFn = nullptr;

	ImGui_ImplVulkan_Init(&init_info, render_pass);

	// Upload fonts
	littlevk::submit_now(device, command_pool, graphics_queue,
		[&](const vk::CommandBuffer &cmd) {
			ImGui_ImplVulkan_CreateFontsTexture(cmd);
		}
	);
}

void Renderer::configure_point()
{
	static constexpr vk::VertexInputBindingDescription vertex_binding {
		0, sizeof(glm::vec3), vk::VertexInputRate::eVertex
	};

	static constexpr std::array <vk::VertexInputAttributeDescription, 1> vertex_attributes {
		vk::VertexInputAttributeDescription {
			0, 0, vk::Format::eR32G32B32Sfloat, 0
		},
	};

	// Compile shader modules
	vk::ShaderModule vertex_module = littlevk::shader::compile(
		device, wireframe_vertex_shader_source,
		vk::ShaderStageFlagBits::eVertex
	).unwrap(dal);

	vk::ShaderModule fragment_module = littlevk::shader::compile(
		device, wireframe_fragment_shader_source,
		vk::ShaderStageFlagBits::eFragment
	).unwrap(dal);

	// Create the pipeline
	vk::PushConstantRange push_constant_range {
		vk::ShaderStageFlagBits::eVertex,
		0, sizeof(push_constants)
	};

	point.pipeline_layout = littlevk::pipeline_layout(
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
	pipeline_info.extent = window->extent;
	pipeline_info.pipeline_layout = point.pipeline_layout;
	pipeline_info.render_pass = render_pass;
	pipeline_info.fill_mode = vk::PolygonMode::ePoint;
	pipeline_info.cull_mode = vk::CullModeFlagBits::eNone;
	pipeline_info.dynamic_viewport = true;

	point.pipeline = littlevk::pipeline::compile(device, pipeline_info).unwrap(dal);
}

void Renderer::configure_wireframe()
{
	static constexpr vk::VertexInputBindingDescription vertex_binding {
		0, sizeof(glm::vec3), vk::VertexInputRate::eVertex
	};

	static constexpr std::array <vk::VertexInputAttributeDescription, 1> vertex_attributes {
		vk::VertexInputAttributeDescription {
			0, 0, vk::Format::eR32G32B32Sfloat, 0
		},
	};

	// Compile shader modules
	vk::ShaderModule vertex_module = littlevk::shader::compile(
		device, wireframe_vertex_shader_source,
		vk::ShaderStageFlagBits::eVertex
	).unwrap(dal);

	vk::ShaderModule fragment_module = littlevk::shader::compile(
		device, wireframe_fragment_shader_source,
		vk::ShaderStageFlagBits::eFragment
	).unwrap(dal);

	// Create the pipeline
	vk::PushConstantRange push_constant_range {
		vk::ShaderStageFlagBits::eVertex,
		0, sizeof(push_constants)
	};

	wireframe.pipeline_layout = littlevk::pipeline_layout(
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
	pipeline_info.extent = window->extent;
	pipeline_info.pipeline_layout = wireframe.pipeline_layout;
	pipeline_info.render_pass = render_pass;
	pipeline_info.fill_mode = vk::PolygonMode::eLine;
	pipeline_info.cull_mode = vk::CullModeFlagBits::eNone;
	pipeline_info.dynamic_viewport = true;

	wireframe.pipeline = littlevk::pipeline::compile(device, pipeline_info).unwrap(dal);
}

void Renderer::configure_solid()
{
	static constexpr vk::VertexInputBindingDescription vertex_binding {
		0, sizeof(glm::vec3), vk::VertexInputRate::eVertex
	};

	static constexpr std::array <vk::VertexInputAttributeDescription, 1> vertex_attributes {
		vk::VertexInputAttributeDescription {
			0, 0, vk::Format::eR32G32B32Sfloat, 0
		},
	};

	// Compile shader modules
	vk::ShaderModule vertex_module = littlevk::shader::compile(
		device, solid_vertex_shader_source,
		vk::ShaderStageFlagBits::eVertex
	).unwrap(dal);

	vk::ShaderModule fragment_module = littlevk::shader::compile(
		device, solid_fragment_shader_source,
		vk::ShaderStageFlagBits::eFragment
	).unwrap(dal);

	// Create the pipeline
	vk::PushConstantRange push_constant_range {
		vk::ShaderStageFlagBits::eVertex,
		0, sizeof(push_constants)
	};

	solid.pipeline_layout = littlevk::pipeline_layout(
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
	pipeline_info.extent = window->extent;
	pipeline_info.pipeline_layout = solid.pipeline_layout;
	pipeline_info.render_pass = render_pass;
	pipeline_info.cull_mode = vk::CullModeFlagBits::eNone;
	pipeline_info.dynamic_viewport = true;

	solid.pipeline = littlevk::pipeline::compile(device, pipeline_info).unwrap(dal);
}

void Renderer::configure_normal()
{
	static constexpr vk::VertexInputBindingDescription vertex_binding {
		0, 2 * sizeof(glm::vec3), vk::VertexInputRate::eVertex
	};

	static constexpr std::array <vk::VertexInputAttributeDescription, 2> vertex_attributes {
		vk::VertexInputAttributeDescription {
			0, 0, vk::Format::eR32G32B32Sfloat, 0
		},
		vk::VertexInputAttributeDescription {
			1, 0, vk::Format::eR32G32B32Sfloat, sizeof(glm::vec3)
		},
	};

	// Compile shader modules
	vk::ShaderModule vertex_module = littlevk::shader::compile(
		device, normal_vertex_shader_source,
		vk::ShaderStageFlagBits::eVertex
	).unwrap(dal);

	vk::ShaderModule fragment_module = littlevk::shader::compile(
		device, normal_fragment_shader_source,
		vk::ShaderStageFlagBits::eFragment
	).unwrap(dal);

	// Create the pipeline
	vk::PushConstantRange push_constant_range {
		vk::ShaderStageFlagBits::eVertex,
		0, sizeof(push_constants)
	};

	normal.pipeline_layout = littlevk::pipeline_layout(
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
	pipeline_info.extent = window->extent;
	pipeline_info.pipeline_layout = normal.pipeline_layout;
	pipeline_info.render_pass = render_pass;
	pipeline_info.cull_mode = vk::CullModeFlagBits::eNone;
	pipeline_info.dynamic_viewport = true;

	normal.pipeline = littlevk::pipeline::compile(device, pipeline_info).unwrap(dal);
}

void Renderer::configure_shaded()
{
	static constexpr vk::VertexInputBindingDescription vertex_binding {
		0, 2 * sizeof(glm::vec3), vk::VertexInputRate::eVertex
	};

	static constexpr std::array <vk::VertexInputAttributeDescription, 2> vertex_attributes {
		vk::VertexInputAttributeDescription {
			0, 0, vk::Format::eR32G32B32Sfloat, 0
		},
		vk::VertexInputAttributeDescription {
			1, 0, vk::Format::eR32G32B32Sfloat, sizeof(glm::vec3)
		},
	};

	// Compile shader modules
	vk::ShaderModule vertex_module = littlevk::shader::compile(
		device, shaded_vertex_shader_source,
		vk::ShaderStageFlagBits::eVertex
	).unwrap(dal);

	vk::ShaderModule fragment_module = littlevk::shader::compile(
		device, shaded_fragment_shader_source,
		vk::ShaderStageFlagBits::eFragment
	).unwrap(dal);

	// Create the pipeline
	vk::PushConstantRange push_constant_range {
		vk::ShaderStageFlagBits::eVertex,
		0, sizeof(push_constants)
	};

	shaded.pipeline_layout = littlevk::pipeline_layout(
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
	pipeline_info.extent = window->extent;
	pipeline_info.pipeline_layout = shaded.pipeline_layout;
	pipeline_info.render_pass = render_pass;
	pipeline_info.cull_mode = vk::CullModeFlagBits::eNone;
	pipeline_info.dynamic_viewport = true;

	shaded.pipeline = littlevk::pipeline::compile(device, pipeline_info).unwrap(dal);
}

Renderer::Renderer(const vk::PhysicalDevice& phdev_) : phdev(phdev_)
{
	mem_props = phdev.getMemoryProperties();
	skeletonize(phdev, { 1000, 1000 }, "Neural Subdivision Complexes", vk::PresentModeKHR::eImmediate);

	dal = new littlevk::Deallocator(device);

	// Create the render pass
	render_pass = littlevk::default_color_depth_render_pass(device, swapchain.format).unwrap(dal);

	// Create the depth buffer
	littlevk::ImageCreateInfo depth_info {
		window->extent.width,
		window->extent.height,
		vk::Format::eD32Sfloat,
		vk::ImageUsageFlagBits::eDepthStencilAttachment,
		vk::ImageAspectFlagBits::eDepth,
	};

	littlevk::Image depth_buffer = littlevk::image(
		device,
		depth_info, mem_props
	).unwrap(dal);

	// Create framebuffers from the swapchain
	littlevk::FramebufferSetInfo fb_info;
	fb_info.swapchain = &swapchain;
	fb_info.render_pass = render_pass;
	fb_info.extent = window->extent;
	fb_info.depth_buffer = &depth_buffer.view;

	framebuffers = littlevk::framebuffers(device, fb_info).unwrap(dal);

	// Allocate command buffers
	command_pool = littlevk::command_pool(device,
		vk::CommandPoolCreateInfo {
			vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
			littlevk::find_graphics_queue_family(phdev)
		}
	).unwrap(dal);

	command_buffers = device.allocateCommandBuffers({
		command_pool, vk::CommandBufferLevel::ePrimary, 2
	});

	// Create the pipelines
	configure_point();
	configure_wireframe();
	configure_solid();
	configure_normal();
	configure_shaded();

	// Setup for ImGui
	configure_imgui();

	// Present syncronization
	sync = littlevk::present_syncronization(device, 2).unwrap(dal);

	// Configure camera from aspect ratio
	camera.from(float(window->extent.width) / float(window->extent.height));

	// Configure callbacks
	glfwSetWindowUserPointer(window->handle, this);
	glfwSetMouseButtonCallback(window->handle, button_callback);
	glfwSetCursorPosCallback(window->handle, cursor_callback);
}

Renderer::~Renderer()
{
	// TODO: clean up imgui
	device.waitIdle();
	delete dal;
}

void Renderer::render()
{
	// Handle input
	handle_key_input();

	// Update camera state before passing to render hooks
	camera.aspect = aspect_ratio();
	push_constants.view = camera.view_matrix(camera_transform);
	push_constants.proj = camera.perspective_matrix();

	// Get next image
	littlevk::SurfaceOperation op;
	op = littlevk::acquire_image(device, swapchain.swapchain, sync[frame]);
	if (op.status == littlevk::SurfaceOperation::eResize) {
		resize();
		return;
	}

	// Record command buffer
	vk::CommandBuffer &cmd = command_buffers[frame];

	const auto &rpbi = littlevk::default_rp_begin_info <2>
		(render_pass, framebuffers[op.index], window)
		.clear_value(0, vk::ClearColorValue {
			std::array <float, 4> { 1.0f, 1.0f, 1.0f, 1.0f }
		});

	cmd.begin(vk::CommandBufferBeginInfo {});

	// Hooks before render pass
	for (auto &hook : prerender_hooks) {
		if (std::holds_alternative <Cmd_Image_Hook> (hook))
			std::get <Cmd_Image_Hook> (hook)(cmd, swapchain.images[op.index]);
		else
			std::get <Cmd_Hook> (hook)(cmd);
	}

	cmd.beginRenderPass(rpbi, vk::SubpassContents::eInline);

	littlevk::viewport_and_scissor(cmd, littlevk::RenderArea(window));

	ImGui_ImplVulkan_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	// Hooks inside render pass
	for (auto &hook : hooks) {
		if (std::holds_alternative <Cmd_Image_Hook> (hook))
			std::get <Cmd_Image_Hook> (hook)(cmd, swapchain.images[op.index]);
		else
			std::get <Cmd_Hook> (hook)(cmd);
	}

	ImGui::Render();
	ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

	cmd.endRenderPass();

	// Hooks after render pass
	for (auto &hook : postrender_hooks) {
		if (std::holds_alternative <Cmd_Image_Hook> (hook))
			std::get <Cmd_Image_Hook> (hook)(cmd, swapchain.images[op.index]);
		else
			std::get <Cmd_Hook> (hook)(cmd);
	}

	cmd.end();

	// Submit command buffer while signaling the semaphore
	// TODO: littlevk shortcut for this...
	vk::PipelineStageFlags wait_stage = vk::PipelineStageFlagBits::eColorAttachmentOutput;

	vk::SubmitInfo submit_info {
		1, &sync.image_available[frame],
		&wait_stage,
		1, &cmd,
		1, &sync.render_finished[frame]
	};

	graphics_queue.submit(submit_info, sync.in_flight[frame]);

	// Post submit hooks
	for (auto &hook : postsubmit_hooks)
		hook();

	// Send image to the screen
	op = littlevk::present_image(present_queue, swapchain.swapchain, sync[frame], op.index);
	if (op.status == littlevk::SurfaceOperation::eResize)
		resize();

	frame = 1 - frame;
}

void Renderer::resize()
{
	littlevk::Skeleton::resize();

	// Recreate the depth buffer
	littlevk::ImageCreateInfo depth_info {
		window->extent.width,
		window->extent.height,
		vk::Format::eD32Sfloat,
		vk::ImageUsageFlagBits::eDepthStencilAttachment,
		vk::ImageAspectFlagBits::eDepth,
	};

	littlevk::Image depth_buffer = littlevk::image(
		device,
		depth_info, mem_props
	).unwrap(dal);

	// Recreate framebuffers from the swapchain
	littlevk::FramebufferSetInfo fb_info;
	fb_info.swapchain = &swapchain;
	fb_info.render_pass = render_pass;
	fb_info.extent = window->extent;
	fb_info.depth_buffer = &depth_buffer.view;

	framebuffers = littlevk::framebuffers(device, fb_info).unwrap(dal);
}

void Renderer::handle_key_input()
{
	constexpr float speed = 2.5f;

	float delta = speed * float(glfwGetTime() - last_time);
	last_time = glfwGetTime();

	glm::vec3 velocity(0.0f);
	if (glfwGetKey(window->handle, GLFW_KEY_S) == GLFW_PRESS)
		velocity.z -= delta;
	else if (glfwGetKey(window->handle, GLFW_KEY_W) == GLFW_PRESS)
		velocity.z += delta;

	if (glfwGetKey(window->handle, GLFW_KEY_D) == GLFW_PRESS)
		velocity.x -= delta;
	else if (glfwGetKey(window->handle, GLFW_KEY_A) == GLFW_PRESS)
		velocity.x += delta;

	if (glfwGetKey(window->handle, GLFW_KEY_E) == GLFW_PRESS)
		velocity.y += delta;
	else if (glfwGetKey(window->handle, GLFW_KEY_Q) == GLFW_PRESS)
		velocity.y -= delta;

	glm::quat q = glm::quat(camera_transform.rotation);
	velocity = q * glm::vec4(velocity, 0.0f);
	camera_transform.position += velocity;
}

void Renderer::button_callback(GLFWwindow *window, int button, int action, int mods)
{
	Renderer *renderer = (Renderer *) glfwGetWindowUserPointer(window);

	// Ignore if on ImGui window
	ImGuiIO &io = ImGui::GetIO();
	io.AddMouseButtonEvent(button, action);

	if (ImGui::GetIO().WantCaptureMouse)
		return;

	auto &m = renderer->mouse;
	if (button == GLFW_MOUSE_BUTTON_LEFT) {
		m.drag = (action == GLFW_PRESS);
		if (action == GLFW_RELEASE)
			m.voided = true;
	}
}

void Renderer::cursor_callback(GLFWwindow *window, double xpos, double ypos)
{
	Renderer *renderer = (Renderer *) glfwGetWindowUserPointer(window);

	// Ignore if on ImGui window
	ImGuiIO &io = ImGui::GetIO();
	io.MousePos = ImVec2(xpos, ypos);

	if (io.WantCaptureMouse)
		return;

	auto &m = renderer->mouse;
	if (m.voided) {
		m.last_x = xpos;
		m.last_y = ypos;
		m.voided = false;
	}

	float xoffset = xpos - m.last_x;
	float yoffset = ypos - m.last_y;

	m.last_x = xpos;
	m.last_y = ypos;

	constexpr float sensitivity = 0.001f;
	xoffset *= sensitivity;
	yoffset *= sensitivity;

	if (m.drag) {
		renderer->camera_transform.rotation.x += yoffset;
		renderer->camera_transform.rotation.y -= xoffset;

		if (renderer->camera_transform.rotation.x > 89.0f)
			renderer->camera_transform.rotation.x = 89.0f;
		if (renderer->camera_transform.rotation.x < -89.0f)
			renderer->camera_transform.rotation.x = -89.0f;
	}
}

// Query utilities
bool Renderer::should_close()
{
	return glfwWindowShouldClose(window->handle);
}

void Renderer::poll()
{
	glfwPollEvents();
}
