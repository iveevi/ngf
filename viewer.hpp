#pragma once

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include <littlevk/littlevk.hpp>

#include "mesh.hpp"

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
	Viewer();

	// Initialize the viewer
	void from(const vk::PhysicalDevice &phdev_);

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

	void add(const std::string &name, const Mesh &mesh, Mode mode);
	void refresh(const std::string &name, const Mesh &mesh);
	void replace(const std::string &name, const Mesh &mesh);
	MeshResource *ref(const std::string &name);
	void clear();

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

	void render();
	bool destroy() override;
};
