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
#include "util.hpp"

// Global constants
constexpr size_t MAXIMUM_SAMPLE_RATE = 32;
constexpr size_t COMPLEX_BATCH_SIZE  = MAXIMUM_SAMPLE_RATE * MAXIMUM_SAMPLE_RATE;
constexpr size_t DNN_INTERIM_SIZE    = 128;
constexpr size_t FREQUENCIES         = 10;

// Error handling
#define CUDA_CHECK(call) { \
	cudaError_t status = call; \
	ulog_assert(status == cudaSuccess, "CUDA", "%s\n", cudaGetErrorString(status)); \
}

#define CUDA_CHECK_SYNCED() { \
	cudaDeviceSynchronize(); \
	cudaError_t status = cudaGetLastError(); \
	ulog_assert(status == cudaSuccess, "CUDA", "%s\n", cudaGetErrorString(status)); \
}

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

	Pipeline point;
	Pipeline wireframe;
	Pipeline solid;
	Pipeline normal;
	Pipeline shaded;

	void configure_point();
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

	Renderer(const vk::PhysicalDevice &);
	~Renderer();

	// Render hooks for rendering
	using Cmd_Hook = std::function <void (const vk::CommandBuffer &)>;
	using Cmd_Image_Hook = std::function <void (const vk::CommandBuffer &, const vk::Image &)>;
	using Cmd_Active_Hook = std::variant <Cmd_Hook, Cmd_Image_Hook>;

	using Void_Hook = std::function <void ()>;

	std::vector <Cmd_Active_Hook> hooks = {};
	std::vector <Cmd_Active_Hook> prerender_hooks = {};
	std::vector <Cmd_Active_Hook> postrender_hooks = {};

	std::vector <Void_Hook> postsubmit_hooks = {};

	// Frame rendering
	// TODO: immediate mode presentation
	size_t frame = 0;

	void render();
	void resize();

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
	std::array <std::vector <float>, 4> Wm_c;
	std::array <std::vector <float>, 4> Wm;
	std::array <std::vector <float>, 4> Bs;

	// Device (CUDA) buffers
	std::array <float *, 4> d_Wm_c;

	// Dimensions
	std::array <uint32_t, 4> Ws;
	std::array <uint32_t, 4> Hs;

	// Activation functions
	std::array <std::function <float (float)>, 3> activations;
} extern g_dnn;

struct subdivision_complexes {
	// Host buffers
	std::vector <glm::ivec4> complexes;
	std::vector <glm::vec3>  vertices;
	std::vector <float>      features;

	// Device (CUDA) buffers
	glm::ivec4	         *d_complexes;
	glm::vec3	         *d_vertices;
	float		         *d_features;

	// float                    *d_flat;
	float4                   *d_flat;

	uint32_t	         complex_count;
	uint32_t	         vertex_count;
	uint32_t                 feature_size;

	uint32_t ffwd_size() const {
		return 3 * (2 * FREQUENCIES + 1) + feature_size;
	}

	constexpr uint32_t vertex_encoding_size() const {
		return 3 * (2 * FREQUENCIES + 1);
	}
} extern g_sdc;

// Reading data
void read(FILE *);

// Evaluation
std::vector <glm::vec3> eval(uint32_t);
// std::vector <glm::vec3> eval_normals(uint32_t);

std::vector <glm::vec3> eval_cuda(uint32_t);

// Timing utilities
struct timeframes {
	static timeframe *current;
	static std::deque <timeframe> frames;

	static void push() {
		frames.emplace_back(timeframe {});
		current = &frames.back();
	}

	static void pop() {
		frames.pop_back();
		current = &frames.back();
	}
};
