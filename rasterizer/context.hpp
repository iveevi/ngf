#pragma once

#include <glm/glm.hpp>

#include <littlevk/littlevk.hpp>

#include "io.hpp"
#include "common.hpp"

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

struct FragmentShaderInfo {
	std::filesystem::path path;
	vk::CullModeFlags culling = vk::CullModeFlagBits::eBack;
	vk::PolygonMode fill = vk::PolygonMode::eFill;
};

extern const std::array <vk::DescriptorSetLayoutBinding, 8> meshlet_dslbs;
extern const std::array <vk::DescriptorSetLayoutBinding, 1> environment_dslbs;
extern const std::unordered_map <std::string, FragmentShaderInfo> fragment_shaders;

struct DeviceRenderContext : littlevk::Skeleton {
	vk::PhysicalDevice phdev;
	vk::PhysicalDeviceMemoryProperties memory_properties;

	littlevk::Deallocator dal;

	vk::RenderPass render_pass;
	vk::CommandPool command_pool;
	vk::DescriptorPool descriptor_pool;

	std::vector <littlevk::Image> depth;
	std::vector <vk::Framebuffer> framebuffers;

	std::vector <vk::CommandBuffer> command_buffers;

	littlevk::PresentSyncronization sync;

	// Pipelines
	std::unordered_map <std::string, littlevk::Pipeline> primaries;

	littlevk::Pipeline environment;

	// ImGui resources
	vk::DescriptorPool imgui_descriptor_pool;

	// View parameters
	Camera camera;
	Transform camera_transform;

	// TODO: deallocate
	void resize();
	littlevk::Image upload_texture(const Texture &);
	static void configure_imgui(DeviceRenderContext &);
	static DeviceRenderContext from(const vk::PhysicalDevice &, const std::vector <const char *> &, size_t);
};

