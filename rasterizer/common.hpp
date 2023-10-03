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

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "microlog.h"

// Global constants
constexpr size_t MAXIMUM_SAMPLE_RATE = 16;
constexpr size_t COMPLEX_BATCH_SIZE = MAXIMUM_SAMPLE_RATE * MAXIMUM_SAMPLE_RATE;
constexpr size_t DNN_INTERIM_SIZE = 128;

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
	Pipeline normal;
	Pipeline shaded;

	void configure_wireframe();
	void configure_solid();
	void configure_normal();
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

// Loading meshes
struct geometry {
	std::vector <glm::vec3> vertices;
	std::vector <glm::vec3> normals;
	std::vector <glm::uvec3> indices;
};

std::vector <float> interleave_attributes(const geometry &);

struct loader {
	std::vector <geometry> meshes;

	loader(const std::filesystem::path &path);

	void process_node(aiNode *, const aiScene *, const std::string &);
	void process_mesh(aiMesh *, const aiScene *, const std::string &);

	const geometry &get(uint32_t) const;
};

// Global state structures
struct neural_network {
	// Host buffers
	std::vector <float> Wm0c;
	std::vector <float> Wm1c;
	std::vector <float> Wm2c;

	// Device (CUDA) buffers
	float *d_Wm0c;
	float *d_Wm1c;
	float *d_Wm2c;

	float *d_embedded;
	float *d_interim_one;
	float *d_interim_two;

	uint32_t W0;
	uint32_t H0;

	uint32_t W1;
	uint32_t H1;

	uint32_t W2;
	uint32_t H2;
} extern g_dnn;

struct subdivision_complexes {
	// Host buffers
	std::vector <glm::uvec4> complexes;
	std::vector <glm::vec3>  vertices;
	std::vector <float>      features;

	// Device (CUDA) buffers
	glm::uvec4	         *d_complexes;
	glm::vec3	         *d_vertices;
	float		         *d_features;

	uint32_t	         complex_count;
	uint32_t	         vertex_count;
	uint32_t                 feature_count;
} extern g_sdc;

void dnn_read(FILE *);
void sdc_read(FILE *);

std::vector <glm::vec3> eval(uint32_t);
std::vector <glm::vec3> eval_normals(uint32_t);

std::vector <glm::vec3> eval_cuda(uint32_t);
