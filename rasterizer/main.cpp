#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

#include <glm/glm.hpp>

#include <littlevk/littlevk.hpp>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include <assimp/scene.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>

#include "microlog.h"

static std::string readfile(const std::string &path)
{
	std::ifstream file(path);
	ulog_assert(file.is_open(), "Could not open file: %s", path.c_str());

	std::stringstream buffer;
	buffer << file.rdbuf();

	return buffer.str();
}

struct mesh {
	std::vector <glm::vec3> vertices;
	std::vector <glm::vec3> normals;
	std::vector <glm::ivec3> triangles;

	static std::vector <mesh> load(const std::filesystem::path &);
};

mesh assimp_process_mesh(aiMesh *m, const aiScene *scene, const std::string &dir)
{

	std::vector <glm::vec3> vertices;
	std::vector <glm::vec3> normals;
        std::vector <glm::ivec3> indices;

	// Process all the mesh's vertices
	for (uint32_t i = 0; i < m->mNumVertices; i++) {
		vertices.push_back({
			m->mVertices[i].x,
			m->mVertices[i].y,
			m->mVertices[i].z
		});

		if (m->HasNormals()) {
			normals.push_back({
				m->mNormals[i].x,
				m->mNormals[i].y,
				m->mNormals[i].z
			});
		} else {
			normals.push_back({ 0.0f, 0.0f, 0.0f });
		}
	}

	// Process all the mesh's triangles
	for (uint32_t i = 0; i < m->mNumFaces; i++) {
		aiFace face = m->mFaces[i];
		ulog_assert(face.mNumIndices == 3, "process_mesh", "Only triangles are supported, got %d-sided polygon instead\n", face.mNumIndices);
		indices.push_back({
			face.mIndices[0],
			face.mIndices[1],
			face.mIndices[2]
		});
	}

	return mesh { vertices, normals, indices };
}

std::vector <mesh> assimp_process_node(aiNode *node, const aiScene *scene, const std::string &directory)
{
	std::vector <mesh> meshes;

	// Process all the node's meshes (if any)
	for (uint32_t i = 0; i < node->mNumMeshes; i++) {
		aiMesh *m = scene->mMeshes[node->mMeshes[i]];
		mesh pm = assimp_process_mesh(m, scene, directory);
		meshes.push_back(pm);
	}

	// Recusively process all the node's children
	for (uint32_t i = 0; i < node->mNumChildren; i++) {
		auto pn = assimp_process_node(node->mChildren[i], scene, directory);
		meshes.insert(meshes.begin(), pn.begin(), pn.end());
	}

	return meshes;
}

std::vector <mesh> mesh::load(const std::filesystem::path &path)
{
	Assimp::Importer importer;
	ulog_assert(std::filesystem::exists(path), "loader", "File \"%s\" does not exist\n", path.c_str());

	// Read scene
	const aiScene *scene;
	scene = importer.ReadFile(path, aiProcess_GenNormals | aiProcess_Triangulate);

	// Check if the scene was loaded
	if ((!scene | scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) || !scene->mRootNode) {
		ulog_error("loader", "Assimp error: \"%s\"\n", importer.GetErrorString());
		return {};
	}

	return assimp_process_node(scene->mRootNode, scene, path.parent_path());
}

struct vertex {
	glm::vec3 position;
	glm::vec3 normal;
};

static constexpr vk::VertexInputBindingDescription vertex_binding {
	0, sizeof(vertex), vk::VertexInputRate::eVertex,
};

static constexpr std::array <vk::VertexInputAttributeDescription, 2> vertex_attributes {
	vk::VertexInputAttributeDescription {
		0, 0, vk::Format::eR32G32B32Sfloat, 0
	},

	vk::VertexInputAttributeDescription {
		1, 0, vk::Format::eR32G32B32Sfloat, offsetof(vertex, normal)
	}
};

struct Engine;

struct vulkan_mesh {
	littlevk::Buffer vertices;
	littlevk::Buffer triangles;
	size_t indices;

	static vulkan_mesh from(const Engine &, const mesh &);
};

struct mvp_push_constants {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
};

struct Pipeline {
	vk::Pipeline pipeline;
	vk::PipelineLayout layout;
};

Pipeline ppl_normals(const vk::Device &device, const vk::RenderPass &rp, const vk::Extent2D &extent, littlevk::Deallocator *dal)
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
		0, sizeof(mvp_push_constants)
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
	pipeline_info.fill_mode = vk::PolygonMode::ePoint;
	pipeline_info.cull_mode = vk::CullModeFlagBits::eNone;
	pipeline_info.dynamic_viewport = true;

	ppl.pipeline = littlevk::pipeline::compile(device, pipeline_info).unwrap(dal);

	return ppl;
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
			(engine.device, pool_info).unwrap(engine.dal);

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
		littlevk::submit_now(engine.device, engine.command_pool, engine.graphics_queue,
			[&](const vk::CommandBuffer &cmd) {
				ImGui_ImplVulkan_CreateFontsTexture(cmd);
			}
		);
	}

	static Engine from(const vk::PhysicalDevice &phdev) {
		Engine engine;
		engine.skeletonize(phdev, { 1000, 1000 }, "NGF Testbed");

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
			(engine.device,
			 vk::CommandPoolCreateInfo {
				vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
				littlevk::find_graphics_queue_family(phdev)
			}).unwrap(engine.dal);

		engine.command_buffers = engine.device.allocateCommandBuffers
			({ engine.command_pool, vk::CommandBufferLevel::ePrimary, 2 });

		// Configure pipelines
		configure_imgui(engine);

		engine.normals = ppl_normals(engine.device, engine.render_pass, engine.window->extent, engine.dal);

		return engine;
	}

	// TODO: deallocate
};

static bool valid_window(const Engine &engine)
{
	return glfwWindowShouldClose(engine.window->handle) == 0;
}

vulkan_mesh vulkan_mesh::from(const Engine &engine, const mesh &m)
{
	vulkan_mesh vm;

	vm.indices = 3 * m.triangles.size();
	vm.vertices = littlevk::buffer(engine.device,
		m.vertices,
		vk::BufferUsageFlagBits::eVertexBuffer,
		engine.memory_properties).unwrap(engine.dal);
	vm.triangles = littlevk::buffer(engine.device,
		m.triangles,
		vk::BufferUsageFlagBits::eVertexBuffer,
		engine.memory_properties).unwrap(engine.dal);

	return vm;
}

int main(int argc, char *argv[])
{
	if (argc < 2) {
		ulog_error("testbed", "Usage: testbed <reference mesh>\n");
		return EXIT_FAILURE;
	}

	std::string path = argv[1];

	mesh reference = *mesh::load(path).begin();

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

	vulkan_mesh vk_ref = vulkan_mesh::from(engine, reference);

	while (valid_window(engine)) {
		// Get events
		glfwPollEvents();
	}
}
