#pragma once

#include <array>
#include <fstream>
#include <functional>
#include <iostream>
#include <vector>

#include <omp.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>

#include <littlevk/littlevk.hpp>

#include <imgui/imgui.h>
#include <imgui/backends/imgui_impl_glfw.h>
#include <imgui/backends/imgui_impl_vulkan.h>

#include "microlog.h"

struct Transform {
	glm::vec3 position = glm::vec3(0.0f);
	glm::vec3 rotation = glm::vec3(0.0f);
	glm::vec3 scale = glm::vec3(1.0f);

	void from(const glm::vec3 &, const glm::vec3 &, const glm::vec3 &);
	
	glm::mat4 matrix() const;

	glm::vec3 right() const;
	glm::vec3 up() const;
	glm::vec3 forward() const;

	std::tuple <glm::vec3, glm::vec3, glm::vec3> axes() const;
};

struct Camera {
	float aspect = 1.0f;
	float fov = 45.0f;
	float near = 0.1f;
	float far = 1000.0f;

	void from(float aspect_, float fov_ = 45.0f, float near_ = 0.1f, float far_ = 1000.0f) {
		aspect = aspect_;
		fov = fov_;
		near = near_;
		far = far_;
	}

	glm::mat4 perspective_matrix() const {
		return glm::perspective(
			glm::radians(fov),
			aspect, near, far
		);
	}

	static glm::mat4 view_matrix(const Transform &transform) {
		auto [right, up, forward] = transform.axes();
		return glm::lookAt(
			transform.position,
			transform.position + forward,
			up
		);
	}
};

struct Renderer : littlevk::Skeleton {
	// Essential resources
	vk::PhysicalDevice phdev;
	vk::PhysicalDeviceMemoryProperties mem_props;

	littlevk::Deallocator *dal = nullptr;

	vk::RenderPass render_pass;
	vk::CommandPool command_pool;

	std::vector <vk::Framebuffer> framebuffers;
	std::vector <vk::CommandBuffer> command_buffers;

	littlevk::PresentSyncronization sync;

	// ImGui resources
	vk::DescriptorPool imgui_descriptor_pool;

	void configure_imgui();

	// Pipeline resources
	struct Pipeline {
		vk::Pipeline pipeline;
		vk::PipelineLayout pipeline_layout;
	};

	Pipeline wireframe;
	Pipeline solid;
	Pipeline shaded;

	void configure_wireframe();
	void configure_solid();
	void configure_shaded();

	struct push_constants_struct {
		glm::mat4 model;
		glm::mat4 view;
		glm::mat4 proj;
	} push_constants;

	Camera camera;
	Transform camera_transform = {};

	struct {
		bool drag = false;
		bool voided = true;
		float last_x = 0.0f;
		float last_y = 0.0f;
	} mouse;

	void from(const vk::PhysicalDevice &);

	// Render hooks for rendering
	using RenderHook = std::function <void (const vk::CommandBuffer &)>;

	std::vector <RenderHook> hooks = {};

	// Frame rendering
	// TODO: immediate mode presentation
	size_t frame = 0;

	void render();

	// Keyboard input
	float last_time = 0.0f;

	void handle_key_input();
	static void button_callback(GLFWwindow *, int, int, int);
	static void cursor_callback(GLFWwindow *, double, double);

	// Query utilities
	bool should_close();
	void poll();
};
