#pragma once

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include <littlevk/littlevk.hpp>

#include "mesh.hpp"
	
struct Camera {
	glm::vec3 position = { 0, 0, -10 };
	glm::vec3 rotation = { 0, 0, 0 };

	float pitch = 0.0f;
	float yaw = 0.0f;
	float fov = 45.0f;

	glm::mat4 proj(const vk::Extent2D &) const;
	glm::mat4 view() const;
	void move(const glm::vec3 &);
	void rotate(const glm::vec2 &);
};

struct Allocator {
	vk::Device device;
	littlevk::Deallocator *dal;
	vk::PhysicalDeviceMemoryProperties mem_props;
};

struct VectorQuantity {
	// starting point, endpoint
	littlevk::Buffer buffer;
};

// TODO: color maps...

struct Viewer : littlevk::Skeleton {
	// Different viewing modes
	enum class Mode : uint32_t {
		Shaded,
		Normal,
		Transparent,
		Wireframe,
		FaceColor,
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
	Viewer();

	// Initialize the viewer
	void from(const vk::PhysicalDevice &phdev_);

	// Local resources
	struct MeshResource {
		vk::Device device;

		littlevk::Buffer vertex_buffer;
		littlevk::Buffer index_buffer;
		littlevk::Buffer unindexed_vertex_buffer;
		size_t index_count;

		Mode mode;
		bool enabled = true;
		glm::vec3 color = { 0.3, 0.7, 0.3 };

		// Additional visualization quantities
		VectorQuantity vecs;

		void set_face_colors(const std::vector <glm::vec3> &colors) {
			assert(colors.size() == index_count / 3);

			std::vector <glm::vec3> unindexed_vertices;
			unindexed_vertices.resize(index_count);

			littlevk::download(device, unindexed_vertex_buffer, unindexed_vertices);
			for (size_t i = 0; i < index_count/3; i++)
				unindexed_vertices[i * 3 + 2] = colors[i];

			littlevk::upload(device, unindexed_vertex_buffer, unindexed_vertices);
		}
	};

	std::map <std::string, MeshResource> meshes;

	void add(const std::string &name, const Mesh &mesh, Mode mode);
	void refresh(const std::string &name, const Mesh &mesh);
	void replace(const std::string &name, const Mesh &mesh);
	MeshResource *ref(const std::string &name);
	void clear();

	// Camera state
	Camera camera;

	// Rendering a frame
	size_t frame = 0;

	void render();
	bool destroy() override;
};
