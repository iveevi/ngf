#include <iostream>
#include <random>

#include <omp.h>

#include <glm/glm.hpp>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include <littlevk/littlevk.hpp>

#define MESH_LOAD_SAVE
#include "argparser.hpp"
#include "mesh.hpp"

struct push_constants {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
};

const char *vertex_shader = R"(
#version 450

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;

layout (push_constant) uniform VertexPushConstants {
	mat4 model;
	mat4 view;
	mat4 proj;
};

layout (location = 0) out vec3 out_normal;

void main()
{
	gl_Position = proj * view * model * vec4(position, 1.0);
	gl_Position.y = -gl_Position.y;
	gl_Position.z = (gl_Position.z + gl_Position.w) / 2.0;
	out_normal = mat3(transpose(inverse(model))) * normal;
}
)";

const char *shaded_fragment_shader = R"(
#version 450

layout (location = 0) in vec3 in_normal;

layout (location = 0) out vec4 fragment;

void main()
{
	vec3 light_direction = normalize(vec3(1.0, 1.0, 1.0));
	float light_intensity = max(0.0, dot(in_normal, light_direction));
	vec3 color = vec3(light_intensity + 0.1);
	fragment = vec4(color, 1.0);
}
)";

const char *transparent_fragment_shader = R"(
#version 450

layout (location = 0) out vec4 fragment;

void main()
{
	fragment = vec4(1.0, 0.5, 0.5, 0.5);
}
)";

struct wireframe_push_constants {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
	glm::vec3 color;
};

const char *wireframe_fragment_shader = R"(
#version 450

layout (push_constant) uniform FragmentPushConstants {
	layout (offset = 192) vec3 color;
};

layout (location = 0) out vec4 fragment;

void main()
{
	fragment = vec4(color, 1.0);
}
)";

struct Viewer : littlevk::Skeleton {
	// Different viewing modes
	enum class Mode : uint32_t {
		Shaded,
		Transparent,
		Wireframe,
		Count
	};

	// General Vulkan resources
	vk::PhysicalDevice phdev;
	vk::PhysicalDeviceMemoryProperties mem_props;

	littlevk::Deallocator *dal = nullptr;

	vk::RenderPass render_pass;

	using Pipeline = std::pair <vk::PipelineLayout, vk::Pipeline>;
	std::array <Pipeline, (uint32_t) Mode::Count> pipelines;

	std::vector <vk::Framebuffer> framebuffers;

	vk::CommandPool command_pool;
	std::vector <vk::CommandBuffer> command_buffers;

	vk::DescriptorPool imgui_pool;

	littlevk::PresentSyncronization sync;

	// Constructor loads a device and starts the initialization process
	Viewer() {
		// Load Vulkan physical device
		auto predicate = [](const vk::PhysicalDevice &dev) {
			return littlevk::physical_device_able(dev, {
				VK_KHR_SWAPCHAIN_EXTENSION_NAME,
			});
		};

		vk::PhysicalDevice dev = littlevk::pick_physical_device(predicate);
		skeletonize(dev, { 2560, 1440 }, "Viewer");
		from(dev);
	}

	// Vertex properties
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

	// Initialize the viewer
	void from(const vk::PhysicalDevice &phdev_) {
		// Copy the physical device and properties
		phdev = phdev_;
		mem_props = phdev.getMemoryProperties();

		// Configure basic resources
		dal = new littlevk::Deallocator(device);

		// Create the render pass
		std::array <vk::AttachmentDescription, 2> attachments {
			littlevk::default_color_attachment(swapchain.format),
			littlevk::default_depth_attachment(),
		};

		std::array <vk::AttachmentReference, 1> color_attachments {
			vk::AttachmentReference {
				0, vk::ImageLayout::eColorAttachmentOptimal,
			}
		};

		vk::AttachmentReference depth_attachment {
			1, vk::ImageLayout::eDepthStencilAttachmentOptimal,
		};

		vk::SubpassDescription subpass {
			{}, vk::PipelineBindPoint::eGraphics,
			{}, color_attachments,
			{}, &depth_attachment,
		};

		render_pass = littlevk::render_pass(
			device,
			vk::RenderPassCreateInfo {
				{}, attachments, subpass
			}
		).unwrap(dal);

		// Create a depth buffer
		littlevk::ImageCreateInfo depth_info {
			window->extent.width,
			window->extent.height,
			vk::Format::eD32Sfloat,
			vk::ImageUsageFlagBits::eDepthStencilAttachment,
			vk::ImageAspectFlagBits::eDepth,
		};

		littlevk::Image depth_buffer = littlevk::image(
			device,
			depth_info,
			mem_props
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

		// Compile shader modules
		vk::ShaderModule vertex_module = littlevk::shader::compile(
			device, std::string(vertex_shader),
			vk::ShaderStageFlagBits::eVertex
		).unwrap(dal);

		vk::ShaderModule shaded_fragment_module = littlevk::shader::compile(
			device, std::string(shaded_fragment_shader),
			vk::ShaderStageFlagBits::eFragment
		).unwrap(dal);

		vk::ShaderModule transparent_fragment_module = littlevk::shader::compile(
			device, std::string(transparent_fragment_shader),
			vk::ShaderStageFlagBits::eFragment
		).unwrap(dal);

		vk::ShaderModule wireframe_fragment_module = littlevk::shader::compile(
			device, std::string(wireframe_fragment_shader),
			vk::ShaderStageFlagBits::eFragment
		).unwrap(dal);

		// Create the pipeline
		vk::PushConstantRange push_constant_range {
			vk::ShaderStageFlagBits::eVertex,
			0, sizeof(push_constants)
		};

		littlevk::pipeline::GraphicsCreateInfo pipeline_info;
		pipeline_info.vertex_binding = vertex_binding;
		pipeline_info.vertex_attributes = vertex_attributes;
		pipeline_info.vertex_shader = vertex_module;
		pipeline_info.extent = window->extent;
		pipeline_info.render_pass = render_pass;
		pipeline_info.cull_mode = vk::CullModeFlagBits::eNone;


		{
			vk::PipelineLayout pipeline_layout = littlevk::pipeline_layout(
				device,
				vk::PipelineLayoutCreateInfo {
					{}, nullptr, push_constant_range
				}
			).unwrap(dal);

			pipeline_info.fragment_shader = shaded_fragment_module;
			pipeline_info.pipeline_layout = pipeline_layout;

			pipelines[0].first = pipeline_layout;
			pipelines[0].second = littlevk::pipeline::compile(device, pipeline_info).unwrap(dal);
		}

		{
			vk::PipelineLayout pipeline_layout = littlevk::pipeline_layout(
				device,
				vk::PipelineLayoutCreateInfo {
					{}, nullptr, push_constant_range
				}
			).unwrap(dal);

			pipeline_info.fragment_shader = transparent_fragment_module;
			pipeline_info.pipeline_layout = pipeline_layout;

			pipelines[1].first = pipeline_layout;
			pipelines[1].second = littlevk::pipeline::compile(device, pipeline_info).unwrap(dal);
		}

		{
			std::array <vk::PushConstantRange, 2> push_constant_ranges {
				vk::PushConstantRange {
					vk::ShaderStageFlagBits::eVertex,
					0, 3 * sizeof(glm::mat4)
				},
				vk::PushConstantRange {
					vk::ShaderStageFlagBits::eFragment,
					offsetof(wireframe_push_constants, color), sizeof(glm::vec3)
				},
			};

			vk::PipelineLayout pipeline_layout = littlevk::pipeline_layout(
				device,
				vk::PipelineLayoutCreateInfo {
					{}, nullptr, push_constant_ranges
				}
			).unwrap(dal);

			pipeline_info.fragment_shader = wireframe_fragment_module;
			pipeline_info.pipeline_layout = pipeline_layout;
			pipeline_info.fill_mode = vk::PolygonMode::eLine;

			pipelines[2].first = pipeline_layout;
			pipelines[2].second = littlevk::pipeline::compile(device, pipeline_info).unwrap(dal);
		}

		// Create the syncronization objects
		sync = littlevk::present_syncronization(device, 2).unwrap(dal);

		// Configure ImGui
		ImGui::CreateContext();
		ImGui_ImplGlfw_InitForVulkan(window->handle, true);

		// Allow popups
		ImGui::GetIO().ConfigFlags |= ImGuiWindowFlags_Popup;

		std::array <vk::DescriptorPoolSize, 4> imgui_pool_sizes {
			vk::DescriptorPoolSize {
				vk::DescriptorType::eSampler,
				1000
			},
			vk::DescriptorPoolSize {
				vk::DescriptorType::eCombinedImageSampler,
				1000
			},
			vk::DescriptorPoolSize {
				vk::DescriptorType::eSampledImage,
				1000
			},
			vk::DescriptorPoolSize {
				vk::DescriptorType::eUniformBuffer,
				1000
			},
		};

		imgui_pool = littlevk::descriptor_pool(
			device,
			vk::DescriptorPoolCreateInfo {
				{
					vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
				},
				1000, imgui_pool_sizes
			}
		).unwrap(dal);

		ImGui_ImplVulkan_InitInfo init_info {};
		init_info.Instance = littlevk::detail::get_vulkan_instance();
		init_info.PhysicalDevice = phdev;
		init_info.Device = device;
		init_info.QueueFamily = littlevk::find_graphics_queue_family(phdev);
		init_info.Queue = graphics_queue;
		init_info.DescriptorPool = imgui_pool;
		init_info.MinImageCount = 2;
		init_info.ImageCount = 2;

		ImGui_ImplVulkan_Init(&init_info, render_pass);

		// Create font atlas
		littlevk::submit_now(device, command_pool, graphics_queue,
			[&](const vk::CommandBuffer &cmd) {
				ImGui_ImplVulkan_CreateFontsTexture(cmd);
			}
		);

		ImGui_ImplVulkan_DestroyFontUploadObjects();
	}

	// Local resources
	struct MeshResource {
		littlevk::Buffer vertex_buffer;
		littlevk::Buffer index_buffer;
		size_t index_count;
		Mode mode;
		bool enabled = true;
		glm::vec3 color = { 0.3, 0.7, 0.3 };
	};

	std::map <std::string, MeshResource> meshes;

	void add(const std::string &name, const Mesh &mesh, Mode mode) {
		MeshResource res;

		Mesh local = mesh;
		recompute_normals(local);

		// Interleave the vertex data
		std::vector <glm::vec3> vertices;
		for (size_t i = 0; i < local.vertices.size(); i++) {
			vertices.push_back(local.vertices[i]);
			vertices.push_back(local.normals[i]);
		}

		res.vertex_buffer = littlevk::buffer(device,
			vertices,
			vk::BufferUsageFlagBits::eVertexBuffer,
			mem_props
		).unwrap(dal);

		res.index_buffer = littlevk::buffer(device,
			mesh.triangles,
			vk::BufferUsageFlagBits::eIndexBuffer,
			mem_props
		).unwrap(dal);

		res.index_count = mesh.triangles.size() * 3;
		res.mode = mode;

		meshes[name] = res;
	}

	void refresh(const std::string &name, const Mesh &mesh) const {
		auto it = meshes.find(name);
		if (it == meshes.end())
			return;

		Mesh local = mesh;
		recompute_normals(local);

		// Interleave the vertex data
		std::vector <glm::vec3> vertices;
		for (size_t i = 0; i < local.vertices.size(); i++) {
			vertices.push_back(local.vertices[i]);
			vertices.push_back(local.normals[i]);
		}

		littlevk::upload(device, it->second.vertex_buffer, vertices);
	}

	MeshResource *ref(const std::string &name) const {
		auto it = meshes.find(name);
		if (it == meshes.end())
			return nullptr;

		return (MeshResource *) &it->second;
	}

	// Camera state
	struct {
		float radius = 10.0f;
		float theta = 0.0f;
		float phi = 0.0f;
		float fov = 45.0f;

		glm::mat4 proj(const vk::Extent2D &ext) const {
			return glm::perspective(
				glm::radians(fov),
				(float) ext.width / (float) ext.height,
				0.1f, 1e5f
			);
		}

		glm::mat4 view() const {
			glm::vec3 eye = {
				radius * std::sin(theta),
				radius * std::sin(phi),
				radius * std::cos(theta)
			};

			return glm::lookAt(eye, { 0, 0, 0 }, { 0, 1, 0 });
		}
	} camera;

	// Rendering a frame
	size_t frame = 0;

	void render() {
		littlevk::SurfaceOperation op;
                op = littlevk::acquire_image(device, swapchain.swapchain, sync, frame);

		// Start empty render pass
		std::array <vk::ClearValue, 2> clear_values {
			vk::ClearColorValue { std::array <float, 4> { 0.0f, 0.0f, 0.0f, 1.0f } },
			vk::ClearDepthStencilValue { 1.0f, 0 }
		};

		vk::RenderPassBeginInfo render_pass_info {
			render_pass, framebuffers[op.index],
			vk::Rect2D { {}, window->extent },
			clear_values
		};

		// Record command buffer
		vk::CommandBuffer &cmd = command_buffers[frame];

		cmd.begin(vk::CommandBufferBeginInfo {});
		cmd.beginRenderPass(render_pass_info, vk::SubpassContents::eInline);

		// Configure the push constants
		push_constants constants;

		constants.proj = camera.proj(window->extent);
		constants.view = camera.view();
		constants.model = glm::mat4(1.0f);

		// Draw main interface
		for (const auto &[name, res] : meshes) {
			if (!res.enabled)
				continue;

			uint32_t mode = (uint32_t) res.mode;
			if (res.mode == Mode::Wireframe) {
				wireframe_push_constants wf_constants;
				wf_constants.proj = constants.proj;
				wf_constants.view = constants.view;
				wf_constants.model = constants.model;
				wf_constants.color = res.color;

				cmd.pushConstants(
					pipelines[mode].first,
					vk::ShaderStageFlagBits::eVertex,
					0, 3 * sizeof(glm::mat4), &wf_constants);
				
				cmd.pushConstants(
					pipelines[mode].first,
					vk::ShaderStageFlagBits::eFragment,
					offsetof(wireframe_push_constants, color), sizeof(glm::vec3), &wf_constants.color);
			} else {
				cmd.pushConstants <push_constants> (
					pipelines[mode].first,
					vk::ShaderStageFlagBits::eVertex,
					0, constants);
			}

			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines[mode].second);
			cmd.bindVertexBuffers(0, *res.vertex_buffer, { 0 });
			cmd.bindIndexBuffer(*res.index_buffer, 0, vk::IndexType::eUint32);
			cmd.drawIndexed(res.index_count, 1, 0, 0, 0);
		}

		// Draw ImGui
		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		static constexpr const char *mode_names[] = {
			// TODO: triangles with false positive coloring
			"Shaded", "Transparent", "Wireframe",
		};

		ImGui::Begin("Meshes");
		for (auto &[name, res] : meshes) {
			ImGui::Checkbox(name.c_str(), &res.enabled);

			// Select mode button for now
			for (size_t i = 0; i < 3; i++) {
				ImGui::SameLine();

				std::string bstr = mode_names[i] + ("##" + name);
				if (ImGui::Button(bstr.c_str()))
					res.mode = (Mode) i;
			}
		}

		ImGui::End();

		ImGui::Render();
		ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

		cmd.endRenderPass();
		cmd.end();

		// Submit command buffer while signaling the semaphore
		vk::PipelineStageFlags wait_stage = vk::PipelineStageFlagBits::eColorAttachmentOutput;

		vk::SubmitInfo submit_info {
			1, &sync.image_available[frame],
			&wait_stage,
			1, &cmd,
			1, &sync.render_finished[frame]
		};

		graphics_queue.submit(submit_info, sync.in_flight[frame]);

                op = littlevk::present_image(present_queue, swapchain.swapchain, sync, op.index);
		frame = 1 - frame;
	}

	bool destroy() override {
		device.waitIdle();

		ImGui_ImplVulkan_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();

		delete dal;
		return littlevk::Skeleton::destroy();
	}
};

// Mouse callback
Viewer *viewer_ptr = nullptr;
bool mouse_pressed = false;

void mouse_button_callback(GLFWwindow *win, int button, int action, int mods)
{
	if (button == GLFW_MOUSE_BUTTON_LEFT) {
		if (action == GLFW_RELEASE)
			mouse_pressed = false;
		else if (action == GLFW_PRESS)
			mouse_pressed = true;

		// glfwSetInputMode(win, GLFW_CURSOR,
		// 	mouse_pressed ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
	}
}

void cursor_position_callback(GLFWwindow *win, double x, double y)
{
	static double last_x;
	static double last_y;
	static bool first = true;

	if (viewer_ptr && mouse_pressed) {
		if (first) {
			last_x = x;
			last_y = y;
			first = false;
		}

		double delta_x = x - last_x;
		double delta_y = y - last_y;

		constexpr float sensitivity = 0.0001f;
		// viewer_ptr->camera.rot.x += delta_y * sensitivity;
		// viewer_ptr->camera.rot.y += delta_x * sensitivity;
		//
		// if (viewer_ptr->camera.rot.x < -M_PI / 2.0f)
		// 	viewer_ptr->camera.rot.x = -M_PI / 2.0f;
		// if (viewer_ptr->camera.rot.x > M_PI / 2.0f)
		// 	viewer_ptr->camera.rot.x = M_PI / 2.0f;
	}

	last_x = x;
	last_y = y;
}

// Bounding box of mesh
std::pair <glm::vec3, glm::vec3> bound(const Mesh &mesh)
{
	glm::vec3 max = mesh.vertices[0];
	glm::vec3 min = mesh.vertices[0];
	for (const glm::vec3 &v : mesh.vertices) {
		max = glm::max(max, v);
		min = glm::min(min, v);
	}

	return { max, min };
}

// Closest point on triangle
__forceinline__ __host__ __device__
glm::vec3 triangle_closest_point(const glm::vec3 v0, const glm::vec3 &v1, const glm::vec3 &v2, const glm::vec3 &p)
{
	glm::vec3 B = v0;
	glm::vec3 E1 = v1 - v0;
	glm::vec3 E2 = v2 - v0;
	glm::vec3 D = B - p;

	float a = glm::dot(E1, E1);
	float b = glm::dot(E1, E2);
	float c = glm::dot(E2, E2);
	float d = glm::dot(E1, D);
	float e = glm::dot(E2, D);
	float f = glm::dot(D, D);

	float det = a * c - b * b;
	float s = b * e - c * d;
	float t = b * d - a * e;

	if (s + t <= det) {
		if (s < 0.0f) {
			if (t < 0.0f) {
				if (d < 0.0f) {
					s = glm::clamp(-d / a, 0.0f, 1.0f);
					t = 0.0f;
				} else {
					s = 0.0f;
					t = glm::clamp(-e / c, 0.0f, 1.0f);
				}
			} else {
				s = 0.0f;
				t = glm::clamp(-e / c, 0.0f, 1.0f);
			}
		} else if (t < 0.0f) {
			s = glm::clamp(-d / a, 0.0f, 1.0f);
			t = 0.0f;
		} else {
			float invDet = 1.0f / det;
			s *= invDet;
			t *= invDet;
		}
	} else {
		if (s < 0.0f) {
			float tmp0 = b + d;
			float tmp1 = c + e;
			if (tmp1 > tmp0) {
				float numer = tmp1 - tmp0;
				float denom = a - 2 * b + c;
				s = glm::clamp(numer / denom, 0.0f, 1.0f);
				t = 1 - s;
			} else {
				t = glm::clamp(-e / c, 0.0f, 1.0f);
				s = 0.0f;
			}
		} else if (t < 0.0f) {
			if (a + d > b + e) {
				float numer = c + e - b - d;
				float denom = a - 2 * b + c;
				s = glm::clamp(numer / denom, 0.0f, 1.0f);
				t = 1 - s;
			} else {
				s = glm::clamp(-e / c, 0.0f, 1.0f);
				t = 0.0f;
			}
		} else {
			float numer = c + e - b - d;
			float denom = a - 2 * b + c;
			s = glm::clamp(numer / denom, 0.0f, 1.0f);
			t = 1.0f - s;
		}
	}

	return B + s * E1 + t * E2;
}

// Closest point caching acceleration structure
struct dev_cas_grid {
	glm::vec3 min;
	glm::vec3 max;
	glm::vec3 bin_size;

	glm::vec3 *vertices;
	glm::uvec3 *triangles;

	uint32_t *query_triangles;
	uint32_t *index0;
	uint32_t *index1;

	uint32_t vertex_count;
	uint32_t triangle_count;
	uint32_t resolution;
};

struct closest_point_kinfo {
	glm::vec3 *points;
	glm::vec3 *closest;

	uint32_t point_count;
};

__global__
void closest_point_kernel(dev_cas_grid cas, closest_point_kinfo kinfo)
{
	uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t stride = blockDim.x * gridDim.x;

	for (uint32_t i = tid; i < kinfo.point_count; i += stride) {
		glm::vec3 point = kinfo.points[i];
		glm::vec3 closest;
		
		glm::vec3 bin_flt = glm::clamp((point - cas.min) / cas.bin_size,
				glm::vec3(0), glm::vec3(cas.resolution - 1));

		glm::ivec3 bin = glm::ivec3(bin_flt);
		uint32_t bin_index = bin.x + bin.y * cas.resolution + bin.z * cas.resolution * cas.resolution;

		uint32_t index0 = cas.index0[bin_index];
		uint32_t index1 = cas.index1[bin_index];

		float min_dist = FLT_MAX;
		for (uint32_t j = index0; j < index1; j++) {
			uint32_t triangle_index = cas.query_triangles[j];
			glm::uvec3 triangle = cas.triangles[triangle_index];

			glm::vec3 v0 = cas.vertices[triangle.x];
			glm::vec3 v1 = cas.vertices[triangle.y];
			glm::vec3 v2 = cas.vertices[triangle.z];

			// TODO: prune triangles that are too far away (based on bbox)?
			glm::vec3 candidate = triangle_closest_point(v0, v1, v2, point);
			float dist = glm::distance(point, candidate);

			if (dist < min_dist) {
				min_dist = dist;
				closest = candidate;
			}
		}

		kinfo.closest[i] = closest;
	}
}

struct cas_grid {
	Mesh ref;

	glm::vec3 min;
	glm::vec3 max;
	
	uint32_t resolution;
	glm::vec3 bin_size;

	using query_bin = std::vector <uint32_t>;
	std::vector <query_bin> overlapping_triangles;
	std::vector <query_bin> query_triangles;
	
	dev_cas_grid dev_cas;

	// Construct from mesh
	cas_grid(const Mesh &ref_, uint32_t resolution_)
			: ref(ref_), resolution(resolution_) {
		uint32_t size = resolution * resolution * resolution;
		overlapping_triangles.resize(size);
		query_triangles.resize(size);

		// Put triangles into bins
		std::tie(max, min) = bound(ref);
		glm::vec3 extent = max - min;
		bin_size = extent / (float) resolution;

		for (size_t i = 0; i < ref.triangles.size(); i++) {
			const Triangle &triangle = ref.triangles[i];

			// Triangle belongs to all bins it intersects
			glm::vec3 v0 = ref.vertices[triangle[0]];
			glm::vec3 v1 = ref.vertices[triangle[1]];
			glm::vec3 v2 = ref.vertices[triangle[2]];

			glm::vec3 tri_min = glm::min(glm::min(v0, v1), v2);
			glm::vec3 tri_max = glm::max(glm::max(v0, v1), v2);

			glm::vec3 min_bin = glm::clamp((tri_min - min) / bin_size, glm::vec3(0), glm::vec3(resolution - 1));
			glm::vec3 max_bin = glm::clamp((tri_max - min) / bin_size, glm::vec3(0), glm::vec3(resolution - 1));

			for (int x = min_bin.x; x <= max_bin.x; x++) {
				for (int y = min_bin.y; y <= max_bin.y; y++) {
					for (int z = min_bin.z; z <= max_bin.z; z++) {
						int index = x + y * resolution + z * resolution * resolution;
						overlapping_triangles[index].push_back(i);
					}
				}
			}
		}

		dev_cas.vertices = 0;
		dev_cas.triangles = 0;

		dev_cas.query_triangles = 0;
		dev_cas.index0 = 0;
		dev_cas.index1 = 0;
	}

	uint32_t to_index(const glm::ivec3 &bin) const {
		return bin.x + bin.y * resolution + bin.z * resolution * resolution;
	}

	uint32_t to_index(const glm::vec3 &p) const {
		glm::vec3 bin_flt = glm::clamp((p - min) / bin_size, glm::vec3(0), glm::vec3(resolution - 1));
		glm::ivec3 bin = glm::ivec3(bin_flt);
		return to_index(bin);
	}

	// Find the complete set of query triangles for a point
	std::unordered_set <uint32_t> closest_triangles(const glm::vec3 &p) const {
		// Get the current bin
		glm::vec3 bin_flt = glm::clamp((p - min) / bin_size, glm::vec3(0), glm::vec3(resolution - 1));
		glm::ivec3 bin = glm::ivec3(bin_flt);
		uint32_t bin_index = to_index(p);

		// Find the closest non-empty bins
		std::vector <glm::ivec3> closest_bins;

		if (!overlapping_triangles[bin_index].empty()) {
			closest_bins.push_back(bin);
		} else {
			std::vector <glm::ivec3> plausible_bins;
			std::queue <glm::ivec3> queue;

			std::unordered_set <glm::ivec3> visited;
			bool stop = false;

			queue.push(bin);
			while (!queue.empty()) {
				glm::ivec3 current = queue.front();
				queue.pop();

				// If visited, continue
				if (visited.find(current) != visited.end())
					continue;

				visited.insert(current);

				// If non-empty, add to plausible bins and continue
				uint32_t current_index = current.x + current.y * resolution + current.z * resolution * resolution;
				if (!overlapping_triangles[current_index].empty()) {
					plausible_bins.push_back(current);

					// Also set the stop flag to stop adding neighbors
					stop = true;
					continue;
				}

				if (stop)
					continue;

				int dx[] = { -1, 0, 0, 1, 0, 0 };
				int dy[] = { 0, -1, 0, 0, 1, 0 };
				int dz[] = { 0, 0, -1, 0, 0, 1 };

				// Add all neighbors to queue...
				for (int i = 0; i < 6; i++) {
					glm::ivec3 next = current + glm::ivec3(dx[i], dy[i], dz[i]);
					if (next.x < 0 || next.x >= resolution ||
						next.y < 0 || next.y >= resolution ||
						next.z < 0 || next.z >= resolution)
						continue;

					// ...if not visited
					if (visited.find(next) == visited.end())
						queue.push(next);
				}
			}

			// Sort plausible bins by distance
			std::sort(plausible_bins.begin(), plausible_bins.end(),
				[&](const glm::ivec3 &a, const glm::ivec3 &b) {
					return glm::distance(bin_flt, glm::vec3(a)) < glm::distance(bin_flt, glm::vec3(b));
				}
			);

			assert(!plausible_bins.empty());

			// Add first one always; stop adding when difference is larger than voxel size
			closest_bins.push_back(plausible_bins[0]);
			for (uint32_t i = 1; i < plausible_bins.size(); i++) {
				glm::vec3 a = glm::vec3(plausible_bins[i - 1]);
				glm::vec3 b = glm::vec3(plausible_bins[i]);

				if (glm::distance(a, b) > 1.1f)
					break;

				closest_bins.push_back(plausible_bins[i]);
			}
		}

		assert(!closest_bins.empty());

		// Within the final collection, make sure to search immediate neighbors
		std::unordered_set <uint32_t> final_bins;

		for (const glm::ivec3 &bin : closest_bins) {
			int dx[] = { 0, -1, 0, 0, 1, 0, 0 };
			int dy[] = { 0, 0, -1, 0, 0, 1, 0 };
			int dz[] = { 0, 0, 0, -1, 0, 0, 1 };

			for (int i = 0; i < 7; i++) {
				glm::ivec3 next = bin + glm::ivec3(dx[i], dy[i], dz[i]);
				if (next.x < 0 || next.x >= resolution ||
					next.y < 0 || next.y >= resolution ||
					next.z < 0 || next.z >= resolution)
					continue;

				uint32_t next_index = to_index(next);
				if (!overlapping_triangles[next_index].empty())
					final_bins.insert(next_index);
			}
		}

		std::unordered_set <uint32_t> final_triangles;
		for (uint32_t bin_index : final_bins) {
			for (uint32_t index : overlapping_triangles[bin_index])
				final_triangles.insert(index);
		}

		return final_triangles;
	}

	// Load the cached query triangles if not already loaded
	bool precache_query(const glm::vec3 &p) {
		// Check if the bin is already cached
		uint32_t bin_index = to_index(p);
		// printf("  Precaching bin %d\n", bin_index);
		// printf("  p = (%f, %f, %f)\n", p.x, p.y, p.z);
		// printf("  max = (%f, %f, %f)\n", max.x, max.y, max.z);
		// printf("  min = (%f, %f, %f)\n", min.x, min.y, min.z);
		assert(bin_index < query_triangles.size());

		if (!query_triangles[bin_index].empty())
			return false;

		// Otherwise, load the bin
		auto set = closest_triangles(p);
		query_triangles[bin_index] = query_bin(set.begin(), set.end());
		return true;
	}

	// Precache a collection of query points
	bool precache_query(const std::vector <glm::vec3> &points) {
		uint32_t any_count = 0;
		for (const glm::vec3 &p : points)
			any_count += precache_query(p);

		printf("Cache hit rate: %f\n", 1 - (float) any_count / points.size());
		return any_count > 0;
	}

	// Single point query
	glm::vec3 query(const glm::vec3 &p) const {
		// Assuming the point is precached already
		uint32_t bin_index = to_index(p);
		assert(bin_index < overlapping_triangles.size());

		const std::vector <uint32_t> &bin = query_triangles[bin_index];
		assert(bin.size() > 0);

		float min_dist = FLT_MAX;
		
		glm::vec3 closest = p;
		for (uint32_t index : bin) {
			const Triangle &tri = ref.triangles[index];
			glm::vec3 a = ref.vertices[tri[0]];
			glm::vec3 b = ref.vertices[tri[1]];
			glm::vec3 c = ref.vertices[tri[2]];

			glm::vec3 point = triangle_closest_point(a, b, c, p);

			float dist = glm::distance(p, point);
			if (dist < min_dist) {
				min_dist = dist;
				closest = point;
			}
		}

		return closest;
	}

	// Host-side query
	void query(const std::vector <glm::vec3> &sources, std::vector <glm::vec3> &dst) const {
		// Assuming all elements are precached already
		// and that the dst vector is already allocated
		assert(sources.size() == dst.size());

		#pragma omp parallel for
		for (uint32_t i = 0; i < sources.size(); i++) {
			uint32_t bin_index = to_index(sources[i]);
			dst[i] = query(sources[i]);
		}
	}

	void precache_device() {
		dev_cas.min = min;
		dev_cas.max = max;
		dev_cas.bin_size = bin_size;

		dev_cas.resolution = resolution;
		dev_cas.vertex_count = ref.vertices.size();
		dev_cas.triangle_count = ref.triangles.size();

		std::vector <uint32_t> linear_query_triangles;
		std::vector <uint32_t> index0;
		std::vector <uint32_t> index1;

		uint32_t size = resolution * resolution * resolution;
		uint32_t offset = 0;

		for (uint32_t i = 0; i < size; i++) {
			uint32_t query_size = query_triangles[i].size();
			linear_query_triangles.insert(linear_query_triangles.end(),
					query_triangles[i].begin(),
					query_triangles[i].end());

			index0.push_back(offset);
			index1.push_back(offset + query_size);
			offset += query_size;
		}

		// Free old memory
		if (dev_cas.vertices != nullptr)
			cudaFree(dev_cas.vertices);

		if (dev_cas.triangles != nullptr)
			cudaFree(dev_cas.triangles);

		if (dev_cas.query_triangles != nullptr)
			cudaFree(dev_cas.query_triangles);

		if (dev_cas.index0 != nullptr)
			cudaFree(dev_cas.index0);

		if (dev_cas.index1 != nullptr)
			cudaFree(dev_cas.index1);

		// Allocate new memory
		cudaMalloc(&dev_cas.vertices, sizeof(glm::vec3) * ref.vertices.size());
		cudaMalloc(&dev_cas.triangles, sizeof(glm::uvec3) * ref.triangles.size());

		cudaMalloc(&dev_cas.query_triangles, sizeof(uint32_t) * linear_query_triangles.size());
		cudaMalloc(&dev_cas.index0, sizeof(uint32_t) * index0.size());
		cudaMalloc(&dev_cas.index1, sizeof(uint32_t) * index1.size());

		cudaMemcpy(dev_cas.vertices, ref.vertices.data(), sizeof(glm::vec3) * ref.vertices.size(), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_cas.triangles, ref.triangles.data(), sizeof(glm::uvec3) * ref.triangles.size(), cudaMemcpyHostToDevice);

		cudaMemcpy(dev_cas.query_triangles, linear_query_triangles.data(), sizeof(uint32_t) * linear_query_triangles.size(), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_cas.index0, index0.data(), sizeof(uint32_t) * index0.size(), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_cas.index1, index1.data(), sizeof(uint32_t) * index1.size(), cudaMemcpyHostToDevice);
	}

	void query_device(closest_point_kinfo kinfo) {
		dim3 block(256);
		dim3 grid((kinfo.point_count + block.x - 1) / block.x);

		closest_point_kernel <<< grid, block >>> (dev_cas, kinfo);
	}
};

// Closest point queries
// struct closest_points_kinfo {
// 	glm::vec3 *points;
// 	glm::vec3 *closest;
// 	
// 	glm::vec3 *targets;
// 	glm::uvec3 *triangles;
//
// 	uint32_t point_count;
// 	uint32_t target_count;
// 	uint32_t triangle_count;
// };
//
// __global__
// void closest_points(closest_points_kinfo kinfo)
// {
// 	uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
// 	uint32_t stride = blockDim.x * gridDim.x;
//
// 	for (uint32_t i = tid; i < kinfo.point_count; i += stride) {
// 		glm::vec3 point = kinfo.points[i];
// 		glm::vec3 closest;
//
// 		float min_dist = FLT_MAX;
// 		for (uint32_t j = 0; j < kinfo.triangle_count; j++) {
// 			glm::uvec3 triangle = kinfo.triangles[j];
// 			glm::vec3 v0 = kinfo.targets[triangle.x];
// 			glm::vec3 v1 = kinfo.targets[triangle.y];
// 			glm::vec3 v2 = kinfo.targets[triangle.z];
//
// 			glm::vec3 p = triangle_closest_point(v0, v1, v2, point);
//
// 			float dist = glm::distance(point, p);
// 			if (dist < min_dist) {
// 				min_dist = dist;
// 				closest = p;
// 			}
// 		}
//
// 		kinfo.closest[i] = closest;
// 	}
// }

int main(int argc, char *argv[])
{
	// Load arguments
	if (argc != 3) {
		printf("Usage: %s <filename> <resolution>\n", argv[0]);
		return 1;
	}

	std::filesystem::path path = std::filesystem::weakly_canonical(argv[1]);
	size_t resolution = std::atoi(argv[2]);

	// Load mesh
	Mesh mesh = load_mesh(path);
	printf("Loaded mesh with %lu vertices and %lu triangles\n", mesh.vertices.size(), mesh.triangles.size());
	printf("Resolution: %lu\n", resolution);

	// Place triangles into the bins on all the planes
	std::vector <std::vector <size_t>> bins_xy(resolution * resolution);
	std::vector <std::vector <size_t>> bins_xz(resolution * resolution);
	std::vector <std::vector <size_t>> bins_yz(resolution * resolution);

	glm::vec3 ext_min = glm::vec3(std::numeric_limits <float> ::max());
	glm::vec3 ext_max = glm::vec3(std::numeric_limits <float> ::min());

	for (const glm::vec3 &vertex : mesh.vertices) {
		ext_min = glm::min(ext_min, vertex);
		ext_max = glm::max(ext_max, vertex);
	}

	for (size_t i = 0; i < mesh.triangles.size(); i++) {
		const Triangle &tri = mesh.triangles[i];

		// Find the bounding box of the triangle
		glm::vec3 min = glm::vec3(std::numeric_limits <float> ::max());
		glm::vec3 max = glm::vec3(std::numeric_limits <float> ::lowest());

		for (size_t j = 0; j < 3; ++j) {
			glm::vec3 v = mesh.vertices[tri[j]];
			v = (v - ext_min) / (ext_max - ext_min);
			min = glm::min(min, v);
			max = glm::max(max, v);
		}

		// Find the bins that the triangle overlaps
		glm::vec3 min_bin = glm::floor(min * glm::vec3(resolution));
		glm::vec3 max_bin = glm::ceil(max * glm::vec3(resolution));

		for (size_t x = min_bin.x; x < max_bin.x; x++) {
			for (size_t y = min_bin.y; y < max_bin.y; y++)
				bins_xy[x + y * resolution].push_back(i);
		}

		for (size_t x = min_bin.x; x < max_bin.x; x++) {
			for (size_t z = min_bin.z; z < max_bin.z; z++)
				bins_xz[x + z * resolution].push_back(i);
		}

		for (size_t y = min_bin.y; y < max_bin.y; y++) {
			for (size_t z = min_bin.z; z < max_bin.z; z++)
				bins_yz[y + z * resolution].push_back(i);
		}
	}

	float load_xy = 0.0f;
	float load_xz = 0.0f;
	float load_yz = 0.0f;

	for (const auto &bin : bins_xy)
		load_xy += bin.size();

	for (const auto &bin : bins_xz)
		load_xz += bin.size();

	for (const auto &bin : bins_yz)
		load_yz += bin.size();

	printf("Average bin load: %f (xy), %f (xz), %f (yz)\n",
		load_xy / bins_xy.size(),
		load_xz / bins_xz.size(),
		load_yz / bins_yz.size());

	// Find all voxels that are occupied in the dense grid using an interior check
	std::vector <uint32_t> voxels(resolution * resolution * resolution, 0);

	auto ray_x_triangle = [&](size_t tindex, const glm::vec3 &x, const glm::vec3 &d) -> bool {
		const Triangle &tri = mesh.triangles[tindex];

		glm::vec3 v0 = mesh.vertices[tri[0]];
		glm::vec3 v1 = mesh.vertices[tri[1]];
		glm::vec3 v2 = mesh.vertices[tri[2]];

		glm::vec3 e1 = v1 - v0;
		glm::vec3 e2 = v2 - v0;
		glm::vec3 p = cross(d, e2);

		float a = dot(e1, p);
		if (std::abs(a) < 1e-6)
			return false;

		float f = 1.0 / a;
		glm::vec3 s = x - v0;
		float u = f * dot(s, p);

		if (u < 0.0 || u > 1.0)
			return false;

		glm::vec3 q = cross(s, e1);
		float v = f * dot(d, q);

		if (v < 0.0 || u + v > 1.0)
			return false;

		float t = f * dot(e2, q);
		return t > 1e-6;
	};

	uint32_t voxel_count = 0;

	#pragma omp parallel for reduction(+:voxel_count)
	for (uint32_t i = 0; i < voxels.size(); i++) {
		constexpr glm::vec3 dx { 1, 0, 0 };
		constexpr glm::vec3 dy { 0, 1, 0 };
		constexpr glm::vec3 dz { 0, 0, 1 };

		uint32_t x = i % resolution;
		uint32_t y = (i / resolution) % resolution;
		uint32_t z = i / (resolution * resolution);

		glm::vec3 center = glm::vec3(x + 0.5f, y + 0.5f, z + 0.5f) / (float) resolution;
		center = center * (ext_max - ext_min) + ext_min;

		// Check if the voxel is inside the mesh
		uint32_t xy_count = 0;
		uint32_t xy_count_neg = 0;

		for (size_t tindex : bins_xy[x + y * resolution]) {
			if (ray_x_triangle(tindex, center, dz))
				xy_count++;

			if (ray_x_triangle(tindex, center, -dz))
				xy_count_neg++;
		}

		uint32_t xz_count = 0;
		uint32_t xz_count_neg = 0;

		for (size_t tindex : bins_xz[x + z * resolution]) {
			if (ray_x_triangle(tindex, center, dy))
				xz_count++;

			if (ray_x_triangle(tindex, center, -dy))
				xz_count_neg++;
		}

		uint32_t yz_count = 0;
		uint32_t yz_count_neg = 0;

		for (size_t tindex : bins_yz[y + z * resolution]) {
			if (ray_x_triangle(tindex, center, dx))
				yz_count++;

			if (ray_x_triangle(tindex, center, -dx))
				yz_count_neg++;
		}

		bool xy_in = (xy_count % 2 == 1) && (xy_count_neg % 2 == 1);
		bool xz_in = (xz_count % 2 == 1) && (xz_count_neg % 2 == 1);
		bool yz_in = (yz_count % 2 == 1) && (yz_count_neg % 2 == 1);

		if (xy_in || xz_in || yz_in) {
			voxels[i] = 1;
			voxel_count++;
		}
	}

	printf("Voxel count: %u/%u\n", voxel_count, voxels.size());

	// Adding voxel elements to a mesh
	auto add_voxel = [&](Mesh &mesh, const size_t i) {
		uint32_t x = i % resolution;
		uint32_t y = (i / resolution) % resolution;
		uint32_t z = i / (resolution * resolution);

		// Load all eight corners of the voxel
		glm::vec3 corners[8];

		corners[0] = glm::vec3(x, y, z) / (float) resolution;
		corners[1] = glm::vec3(x + 1, y, z) / (float) resolution;
		corners[2] = glm::vec3(x, y + 1, z) / (float) resolution;
		corners[3] = glm::vec3(x + 1, y + 1, z) / (float) resolution;
		corners[4] = glm::vec3(x, y, z + 1) / (float) resolution;
		corners[5] = glm::vec3(x + 1, y, z + 1) / (float) resolution;
		corners[6] = glm::vec3(x, y + 1, z + 1) / (float) resolution;
		corners[7] = glm::vec3(x + 1, y + 1, z + 1) / (float) resolution;

		for (glm::vec3 &v : corners)
			v = v * (ext_max - ext_min) + ext_min;

		// Fill the data in the mesh
		uint32_t base = mesh.vertices.size();

		for (const glm::vec3 &v : corners)
			mesh.vertices.push_back(v);

		mesh.triangles.push_back({ base + 0, base + 1, base + 2 });
		mesh.triangles.push_back({ base + 1, base + 3, base + 2 });
		mesh.triangles.push_back({ base + 0, base + 2, base + 4 });
		mesh.triangles.push_back({ base + 2, base + 6, base + 4 });
		mesh.triangles.push_back({ base + 0, base + 4, base + 1 });
		mesh.triangles.push_back({ base + 1, base + 4, base + 5 });
		mesh.triangles.push_back({ base + 1, base + 5, base + 3 });
		mesh.triangles.push_back({ base + 3, base + 5, base + 7 });
		mesh.triangles.push_back({ base + 2, base + 3, base + 6 });
		mesh.triangles.push_back({ base + 3, base + 7, base + 6 });
		mesh.triangles.push_back({ base + 4, base + 6, base + 5 });
		mesh.triangles.push_back({ base + 5, base + 6, base + 7 });
	};

	// Create a mesh out of the voxels
	Mesh voxel_mesh;

	for (size_t i = 0; i < voxels.size(); i++) {
		if (voxels[i] == 0)
			continue;

		add_voxel(voxel_mesh, i);
	}

	voxel_mesh.normals.resize(voxel_mesh.vertices.size());
	voxel_mesh = deduplicate(voxel_mesh);

	printf("Voxel mesh: %u vertices, %u triangles\n", voxel_mesh.vertices.size(), voxel_mesh.triangles.size());

	// Adjaceny for the voxel elements
	std::unordered_map <uint32_t, std::vector<uint32_t>> voxel_adjacency;

	for (size_t i = 0; i < voxels.size(); i++) {
		if (voxels[i] == 0)
			continue;

		// Find all neighbors that are valid
		int dx[6] = { -1, 1, 0, 0, 0, 0 };
		int dy[6] = { 0, 0, -1, 1, 0, 0 };
		int dz[6] = { 0, 0, 0, 0, -1, 1 };

		int32_t x = i % resolution;
		int32_t y = (i / resolution) % resolution;
		int32_t z = i / (resolution * resolution);

		for (int j = 0; j < 6; j++) {
			int32_t nx = x + dx[j];
			int32_t ny = y + dy[j];
			int32_t nz = z + dz[j];

			if (nx < 0 || nx >= resolution || ny < 0 || ny >= resolution || nz < 0 || nz >= resolution)
				continue;

			int32_t nindex = nx + ny * resolution + nz * resolution * resolution;

			if (voxels[nindex] == 0)
				continue;

			voxel_adjacency[i].push_back(nindex);
		}
	}

	float adj_avg = 0.0f;
	for (const auto &p : voxel_adjacency)
		adj_avg += p.second.size();

	printf("Average adjacency: %f\n", adj_avg / (float) voxel_adjacency.size());

	// Remove voxels with only one neighbor
	for (auto it = voxel_adjacency.begin(); it != voxel_adjacency.end(); ) {
		if (it->second.size() <= 1) {
			printf("Removing voxel %u\n", it->first);
			it = voxel_adjacency.erase(it);
		} else
			it++;
	}

	// Find all the connected components and select the largest one
	std::vector <std::unordered_set <uint32_t>> components;

	std::unordered_set <uint32_t> remaining;
	for (auto [v, _] : voxel_adjacency)
		remaining.insert(v);

	while (remaining.size() > 0) {
		uint32_t seed = *remaining.begin();

		std::unordered_set <uint32_t> component;
		std::queue <uint32_t> queue;

		queue.push(seed);
		component.insert(seed);
		remaining.erase(seed);

		while (!queue.empty()) {
			uint32_t v = queue.front();
			queue.pop();

			for (uint32_t n : voxel_adjacency[v]) {
				if (remaining.count(n) == 0)
					continue;

				queue.push(n);
				component.insert(n);
				remaining.erase(n);
			}
		}

		components.push_back(component);
	}

	printf("Found %u components\n", components.size());
	for (size_t i = 0; i < components.size(); i++)
		printf("Component %u: %u voxels\n", i, components[i].size());

	std::sort(components.begin(), components.end(),
		[](const auto &a, const auto &b) {
			return a.size() > b.size();
		}
	);
	
	std::unordered_set <uint32_t> working_set = components[0];

	// Extract the voxel mesh
	Mesh reduced_mesh;
	for (size_t i = 0; i < voxels.size(); i++) {
		if (working_set.count(i) == 0)
			continue;

		add_voxel(reduced_mesh, i);
	}

	reduced_mesh.normals.resize(reduced_mesh.vertices.size());
	reduced_mesh = deduplicate(reduced_mesh);

	// Remove all the duplicate triangles
	auto triangle_eq = [](const Triangle &t1, const Triangle &t2) {
		// Any permutation of the vertices is considered equal
		bool perm1 = (t1[0] == t2[0] && t1[1] == t2[1] && t1[2] == t2[2]);
		bool perm2 = (t1[0] == t2[0] && t1[1] == t2[2] && t1[2] == t2[1]);
		bool perm3 = (t1[0] == t2[1] && t1[1] == t2[0] && t1[2] == t2[2]);
		bool perm4 = (t1[0] == t2[1] && t1[1] == t2[2] && t1[2] == t2[0]);
		bool perm5 = (t1[0] == t2[2] && t1[1] == t2[0] && t1[2] == t2[1]);
		bool perm6 = (t1[0] == t2[2] && t1[1] == t2[1] && t1[2] == t2[0]);
		return perm1 || perm2 || perm3 || perm4 || perm5 || perm6;
	};

	auto triangle_hash = [](const Triangle &t) {
		std::hash <uint32_t> hasher;
		return hasher(t[0]) ^ hasher(t[1]) ^ hasher(t[2]);
	};

	std::unordered_set <Triangle, decltype(triangle_hash), decltype(triangle_eq)> triangles(0, triangle_hash, triangle_eq);
	std::unordered_set <Triangle, decltype(triangle_hash), decltype(triangle_eq)> triangles_to_remove(0, triangle_hash, triangle_eq);

	for (const auto &t : reduced_mesh.triangles) {
		if (triangles.count(t) > 0)
			triangles_to_remove.insert(t);
		else
			triangles.insert(t);
	}

	printf("Number of triangles to remove: %u\n", triangles_to_remove.size());

	Mesh skin_mesh = reduced_mesh;

	// for (auto it = skin_mesh.triangles.begin(); it != skin_mesh.triangles.end(); ) {
	// 	if (triangles_to_remove.count(*it))
	// 		it = skin_mesh.triangles.erase(it);
	// 	else
	// 		it++;
	// }

	std::vector <Triangle> new_triangles;
	for (const auto &t : skin_mesh.triangles) {
		if (triangles_to_remove.count(t) == 0)
			new_triangles.push_back(t);
	}

	skin_mesh.triangles = new_triangles;
	skin_mesh = deduplicate(skin_mesh);

	printf("Skin mesh has %u vertices and %u triangles\n", skin_mesh.vertices.size(), skin_mesh.triangles.size());

	Viewer viewer;
	viewer.add("mesh", mesh, Viewer::Mode::Shaded);
	viewer.add("skin", skin_mesh, Viewer::Mode::Wireframe);

	viewer_ptr = &viewer;
	viewer.ref("skin")->color = { 1.0f, 0.0f, 1.0f };

	glm::vec3 color_wheel[] = {
		{ 0.910, 0.490, 0.490 },
		{ 0.910, 0.700, 0.490 },
		{ 0.910, 0.910, 0.490 },
		{ 0.700, 0.910, 0.490 },
		{ 0.490, 0.910, 0.490 },
		{ 0.490, 0.910, 0.700 },
		{ 0.490, 0.910, 0.910 },
		{ 0.490, 0.700, 0.910 },
		{ 0.490, 0.490, 0.910 },
		{ 0.700, 0.490, 0.910 },
		{ 0.910, 0.490, 0.910 },
		{ 0.910, 0.490, 0.700 }
	};

	// auto octree = simple_octree(mesh);
	// for (size_t i = 0; i < octree.size(); i++) {
	// 	Mesh tmp = mesh;
	// 	tmp.triangles.clear();
	// 	
	// 	for (size_t i : octree[i].triangles)
	// 		tmp.triangles.push_back(mesh.triangles[i]);
	//
	// 	viewer.add("octree_" + std::to_string(i), tmp, Viewer::Mode::Wireframe);
	// 	viewer.ref("octree_" + std::to_string(i))->color = color_wheel[i % 12];
	// }

	// Create the acceleration structure for point queries on the target mesh
	cas_grid cas = cas_grid(mesh, 128);
	cas.precache_query(skin_mesh.vertices);
	cas.precache_device();

	printf("CAS grid has been preloaded\n");

	// Optimization (while rendering)
	closest_point_kinfo kinfo;
	kinfo.point_count = skin_mesh.vertices.size();

	cudaMalloc(&kinfo.points, sizeof(glm::vec3) * skin_mesh.vertices.size());
	cudaMalloc(&kinfo.closest, sizeof(glm::vec3) * skin_mesh.vertices.size());

	// auto [vgraph, egraph, dual] = build_graphs(skin_mesh);

	while (true) {
		GLFWwindow *window = viewer.window->handle;

		// Check window close state
		glfwPollEvents();
		if (glfwWindowShouldClose(window))
			break;

		// Other inputs
		if (glfwGetKey(window, GLFW_KEY_EQUAL) == GLFW_PRESS)
			viewer.camera.radius *= 1.1f;
		else if (glfwGetKey(window, GLFW_KEY_MINUS) == GLFW_PRESS)
			viewer.camera.radius /= 1.1f;

		if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
			viewer.camera.theta -= 0.1f;
		else if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
			viewer.camera.theta += 0.1f;

		if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
			viewer.camera.phi += 0.1f;
		else if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
			viewer.camera.phi -= 0.1f;

		viewer.camera.phi = glm::clamp(viewer.camera.phi, float(-M_PI / 2.0f), float(M_PI / 2.0f));

		// Optimize the skin mesh around the target (original) mesh
		std::vector <glm::vec3> host_closest(skin_mesh.vertices.size());

		// TODO: memcpy async while caching
		cudaMemcpy(kinfo.points, skin_mesh.vertices.data(),
				sizeof(glm::vec3) * skin_mesh.vertices.size(),
				cudaMemcpyHostToDevice);

		bool updated = cas.precache_query(skin_mesh.vertices);
		if (updated)
			cas.precache_device();

		// cas.query(skin_mesh.vertices, host_closest);
		cas.query_device(kinfo);

		cudaMemcpy(host_closest.data(), kinfo.closest,
				sizeof(glm::vec3) * skin_mesh.vertices.size(),
				cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();

		std::vector <glm::vec3> gradients;
		gradients.reserve(skin_mesh.vertices.size());

		for (uint32_t i = 0; i < skin_mesh.vertices.size(); i++) {
			const glm::vec3 &v = skin_mesh.vertices[i];
			const glm::vec3 &w = host_closest[i];
			gradients[i] = (w - v);
		}

		// TODO: adam optimizer...
		for (uint32_t i = 0; i < skin_mesh.vertices.size(); i++)
			skin_mesh.vertices[i] += 0.01f * gradients[i];

		viewer.refresh("skin", skin_mesh);
		viewer.render();
	}

	viewer.destroy();
}
