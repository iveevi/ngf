#pragma once

#include <glm/glm.hpp>

#include <littlevk/littlevk.hpp>

// Vertex type
struct Vertex {
	glm::vec3 position;
	glm::vec3 normal;
};

// Shader push constants
struct BasePushConstants {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
};

// TODO: use the same...
struct alignas(16) NGFPushConstants {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;

	glm::vec2 extent;
	float time;
};

struct ShadingPushConstants {
	alignas(16) glm::vec3 viewing;
	alignas(16) glm::vec3 color;
	uint32_t mode;
};

// General pipeline structure
struct Pipeline {
	vk::Pipeline pipeline;
	vk::PipelineLayout layout;
	vk::DescriptorSetLayout dsl;
};

Pipeline ppl_normals(const vk::Device &, const vk::RenderPass &, const vk::Extent2D &, littlevk::Deallocator *);
Pipeline ppl_ngf(const vk::Device &, const vk::RenderPass &, const vk::Extent2D &, littlevk::Deallocator *);
