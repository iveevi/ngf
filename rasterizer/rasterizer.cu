#include <array>
#include <fstream>
#include <iostream>
#include <vector>

#include <glm/glm.hpp>

#include <omp.h>

#include <littlevk/littlevk.hpp>

#include "microlog.h"

const std::string vertex_shader_source = R"(
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

const std::string fragment_shader_source = R"(
#version 450

layout (location = 0) out vec4 fragment;

void main()
{
	fragment = vec4(1.0);
}
)";

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

	// Pipeline resources
	vk::Pipeline pipeline;
	vk::PipelineLayout pipeline_layout;

	// Vertex properties
	static constexpr vk::VertexInputBindingDescription vertex_binding {
		0, sizeof(glm::vec3), vk::VertexInputRate::eVertex
	};

	static constexpr std::array <vk::VertexInputAttributeDescription, 1> vertex_attributes {
		vk::VertexInputAttributeDescription {
			0, 0, vk::Format::eR32G32B32Sfloat, 0
		},
	};

	struct push_constants_struct {
		glm::mat4 model;
		glm::mat4 view;
		glm::mat4 proj;
	} push_constants;

	struct {
		bool drag = false;
		bool voided = true;
		float last_x = 0.0f;
		float last_y = 0.0f;
	} mouse;

	void from(const vk::PhysicalDevice& phdev_) {
		phdev = phdev_;
		mem_props = phdev.getMemoryProperties();
		skeletonize(phdev, { 1000, 1000 }, "zzz");

		dal = new littlevk::Deallocator(device);

		// Create the render pass
		std::array <vk::AttachmentDescription, 2> attachments {
			littlevk::default_color_attachment(swapchain.format),
			littlevk::default_depth_attachment()
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
			{}, &depth_attachment
		};

		render_pass = littlevk::render_pass(
			device,
			vk::RenderPassCreateInfo {
				{}, attachments, subpass
			}
		).unwrap(dal);

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

		// Compile shader modules
		vk::ShaderModule vertex_module = littlevk::shader::compile(
			device, vertex_shader_source,
			vk::ShaderStageFlagBits::eVertex
		).unwrap(dal);

		vk::ShaderModule fragment_module = littlevk::shader::compile(
			device, fragment_shader_source,
			vk::ShaderStageFlagBits::eFragment
		).unwrap(dal);

		// Create the pipeline
		vk::PushConstantRange push_constant_range {
			vk::ShaderStageFlagBits::eVertex,
			0, sizeof(push_constants)
		};

		pipeline_layout = littlevk::pipeline_layout(
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
		pipeline_info.pipeline_layout = pipeline_layout;
		pipeline_info.render_pass = render_pass;
		pipeline_info.fill_mode = vk::PolygonMode::eLine;
		pipeline_info.cull_mode = vk::CullModeFlagBits::eNone;

		pipeline = littlevk::pipeline::compile(device, pipeline_info).unwrap(dal);

		// Present syncronization
		sync = littlevk::present_syncronization(device, 2).unwrap(dal);

		// Configure callbacks
		// glfwSetWindowUserPointer(window->handle, this);
		// glfwSetMouseButtonCallback(window->handle, button_callback);
		// glfwSetCursorPosCallback(window->handle, cursor_callback);
	}

	// Frame rendering
	// TODO: immediate mode presentation
	size_t frame = 0;

	void render() {
		// Handle input
		handle_key_input();

		// Get next image
		littlevk::SurfaceOperation op;
                op = littlevk::acquire_image(device, swapchain.swapchain, sync, frame);

		// Record command buffer
		vk::CommandBuffer &cmd = command_buffers[frame];

		vk::RenderPassBeginInfo render_pass_info = littlevk::default_rp_begin_info <2>
				(render_pass, framebuffers[op.index], window);

		cmd.begin(vk::CommandBufferBeginInfo {});
		cmd.beginRenderPass(render_pass_info, vk::SubpassContents::eInline);

		cmd.endRenderPass();
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

		// Send image to the screen
		op = littlevk::present_image(present_queue, swapchain.swapchain, sync, op.index);
		frame = 1 - frame;
	}

	// Keyboard input
	float last_time = 0.0f;

	void handle_key_input() {
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

		// glm::quat q = glm::quat(camera_transform.rotation);
		// velocity = q * glm::vec4(velocity, 0.0f);
		// camera_transform.position += velocity;
	}

	// Query utilities
	bool should_close() {
		return glfwWindowShouldClose(window->handle);
	}

	void poll() {
		glfwPollEvents();
	}
};

struct neural_network {
	// Host buffers
	std::vector <float> Wm0c;
	std::vector <float> Wm1c;
	std::vector <float> Wm2c;

	// Device (CUDA) buffers
	float *d_Wm0c;
	float *d_Wm1c;
	float *d_Wm2c;

	uint32_t W0;
	uint32_t H0;

	uint32_t W1;
	uint32_t H1;

	uint32_t W2;
	uint32_t H2;
} g_dnn;

// Batched version (true matrix multiplication)
// TODO: concat into a single matrix multiplication (e.g. remove the bias and add as the last column of the weight matrix)
std::vector <float> matmul(const std::vector <float> &W, const std::vector <float> &Xbatched,
		const std::vector <float> &B,
		float (*act)(float),
		size_t in, size_t out)
{
	// Check sizes
	ulog_assert(W.size() == in * out,      "matmul", "W size mismatch\n");
	ulog_assert(Xbatched.size() % in == 0, "matmul", "X size mismatch\n");
	ulog_assert(B.size() == out,           "matmul", "B size mismatch\n");

	// Prepare result
	size_t batch = Xbatched.size() / in;

	std::vector <float> Ybatched(out * batch);

	// Perform matrix multiplication and add bias
	#pragma omp parallel
	for (size_t b = 0; b < batch; b++) {
		for (size_t i = 0; i < out; i++) {
			float sum = 0.0f;

			for (size_t j = 0; j < in; j++)
				sum += W[i * in + j] * Xbatched[b * in + j];

			Ybatched[b * out + i] = act ? act(sum + B[i]) : sum + B[i];
		}
	}

	return Ybatched;
}

// Matrix multiplication with bias
template <float (*act)(float) = nullptr>
std::vector <float> matmul_biased(const std::vector <float> &W,
		const std::vector <float> &Xbatched,
		size_t in, size_t out)
{
	// Check sizes
	ulog_assert(W.size() == (in + 1) * out, "matmul_biased", "W size mismatch\n");
	ulog_assert(Xbatched.size() % in == 0,  "matmul_biased", "X size mismatch\n");

	// Prepare result
	size_t batch = Xbatched.size() / in;

	std::vector <float> Ybatched(out * batch);

	// Perform matrix multiplication and add bias
	#pragma omp parallel
	for (size_t b = 0; b < batch; b++) {
		float *Yrow = &Ybatched[b * out];
		for (size_t i = 0; i < out; i++) {
			const float *Wrow = &W[i * (in + 1)];
			float sum = Wrow[in];
			for (size_t j = 0; j < in; j++)
				sum += Wrow[j] * Xbatched[b * in + j];

			if constexpr (act)
				Yrow[i] = act(sum);
			else
				Yrow[i] = sum;
		}
	}

	return Ybatched;
}

// TODO: batch by each (or several complexes; set some maximum cache/interim memory amt)

// Combine weights and biases into a single matrix
std::vector <float> combine(const std::vector <float> &W, const std::vector <float> &B)
{
	size_t rows = B.size();
	size_t cols = W.size() / B.size();

	ulog_assert(W.size() == rows * cols, "combine", "W size mismatch\n");

	std::vector <float> Wc(rows * (cols + 1));
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++)
			Wc[i * (cols + 1) + j] = W[i * cols + j];
		Wc[i * (cols + 1) + cols] = B[i];
	}

	return Wc;
}

void dnn_read(FILE *file)
{
	// Read neural network data
	ulog_info("dnn_read", "Neural Network:\n");

	uint32_t W0 = 0;
	uint32_t H0 = 0;

	fread((char *) &W0, sizeof(W0), 1, file);
	fread((char *) &H0, sizeof(H0), 1, file);
	ulog_info("dnn_read", "  > Layer 1: %d x %d + %d\n", W0, H0, W0);

	std::vector <float> Wm0(W0 * H0);
	std::vector <float> Bs0(W0);

	fread((char *) Wm0.data(), sizeof(float), W0 * H0, file);
	fread((char *) Bs0.data(), sizeof(float), W0, file);

	// Combined layer (as one matrix)
	std::vector <float> Wm0c = combine(Wm0, Bs0);

	uint32_t W1 = 0;
	uint32_t H1 = 0;

	fread((char *) &W1, sizeof(W1), 1, file);
	fread((char *) &H1, sizeof(H1), 1, file);
	ulog_info("dnn_read", "  > Layer 2: %d x %d + %d\n", W1, H1, W1);

	std::vector <float> Wm1(W1 * H1);
	std::vector <float> Bs1(W1);

	fread((char *) Wm1.data(), sizeof(float), W1 * H1, file);
	fread((char *) Bs1.data(), sizeof(float), W1, file);

	std::vector <float> Wm1c = combine(Wm1, Bs1);

	uint32_t W2 = 0;
	uint32_t H2 = 0;

	fread((char *) &W2, sizeof(W2), 1, file);
	fread((char *) &H2, sizeof(H2), 1, file);
	ulog_info("dnn_read", "  > Layer 3: %d x %d + %d\n", W2, H2, W2);

	std::vector <float> Wm2(W2 * H2);
	std::vector <float> Bs2(W2);

	fread((char *) Wm2.data(), sizeof(float), W2 * H2, file);
	fread((char *) Bs2.data(), sizeof(float), W2, file);

	std::vector <float> Wm2c = combine(Wm2, Bs2);

	g_dnn.W0 = W0;
	g_dnn.H0 = H0;

	g_dnn.W1 = W1;
	g_dnn.H1 = H1;

	g_dnn.W2 = W2;
	g_dnn.H2 = H2;

	g_dnn.Wm0c = Wm0c;
	g_dnn.Wm1c = Wm1c;
	g_dnn.Wm2c = Wm2c;

	// Transfer to device
	cudaMalloc((void **) &g_dnn.d_Wm0c, Wm0c.size() * sizeof(float));
	cudaMalloc((void **) &g_dnn.d_Wm1c, Wm1c.size() * sizeof(float));
	cudaMalloc((void **) &g_dnn.d_Wm2c, Wm2c.size() * sizeof(float));

	cudaMemcpy(g_dnn.d_Wm0c, Wm0c.data(), Wm0c.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(g_dnn.d_Wm1c, Wm1c.data(), Wm1c.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(g_dnn.d_Wm2c, Wm2c.data(), Wm2c.size() * sizeof(float), cudaMemcpyHostToDevice);
}

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
} g_sdc;

void sdc_read(FILE *file)
{
	// Read size data
	uint32_t sizes[3];
	fread((char *) sizes, sizeof(uint32_t), 3, file);

	g_sdc.complex_count = sizes[0];
	g_sdc.vertex_count = sizes[1];
	g_sdc.feature_count = sizes[2];

	ulog_info("sdc_read", "Neural Subdivision Complexes:\n");
	ulog_info("sdc_read", "  > %4d complexes\n", g_sdc.complex_count);
	ulog_info("sdc_read", "  > %4d vertices\n", g_sdc.vertex_count);
	ulog_info("sdc_read", "  > %4d encoding features\n", g_sdc.feature_count);

	// Read complexes data
	std::vector <glm::uvec4> complexes(g_sdc.complex_count);
	fread((char *) complexes.data(), sizeof(glm::uvec4), g_sdc.complex_count, file);

	// Read corner vertices, normals, and their features
	std::vector <glm::vec3> vertices(g_sdc.vertex_count);
	fread((char *) vertices.data(), sizeof(glm::vec3), g_sdc.vertex_count, file);

	// Corner feature vectors
	std::vector <float> features(g_sdc.vertex_count * g_sdc.feature_count);
	fread((char *) features.data(), sizeof(float), g_sdc.vertex_count * g_sdc.feature_count, file);

	g_sdc.complexes = complexes;
	g_sdc.vertices = vertices;
	g_sdc.features = features;

	// Transfer to device
	cudaMalloc(&g_sdc.d_complexes, g_sdc.complex_count * sizeof(glm::uvec4));
	cudaMalloc(&g_sdc.d_vertices, g_sdc.vertex_count * sizeof(glm::vec3));
	cudaMalloc(&g_sdc.d_features, g_sdc.vertex_count * g_sdc.feature_count * sizeof(float));

	cudaMemcpy(g_sdc.d_complexes, complexes.data(), g_sdc.complex_count * sizeof(glm::uvec4), cudaMemcpyHostToDevice);
	cudaMemcpy(g_sdc.d_vertices, vertices.data(), g_sdc.vertex_count * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	cudaMemcpy(g_sdc.d_features, features.data(), g_sdc.vertex_count * g_sdc.feature_count * sizeof(float), cudaMemcpyHostToDevice);
}

// TODO: microlog...
std::vector <glm::vec3> eval(size_t sample_rate)
{
	uint32_t vertex_count = sample_rate * sample_rate * g_sdc.complex_count;

	std::vector <glm::vec3> lerped_P;
	std::vector <float>     lerped_E;

	lerped_P.resize(vertex_count);
	lerped_E.resize(vertex_count * g_sdc.feature_count);

	uint32_t i = 0;
	for (uint32_t c = 0; c < g_sdc.complex_count; c++) {
		const glm::uvec4 &complex = g_sdc.complexes[c];

		glm::vec3 v00 = g_sdc.vertices[complex.x];
		glm::vec3 v10 = g_sdc.vertices[complex.y];
		glm::vec3 v01 = g_sdc.vertices[complex.w];
		glm::vec3 v11 = g_sdc.vertices[complex.z];

		auto get_feature = [&](uint32_t v) {
			std::vector <float> f(g_sdc.feature_count);
			for (uint32_t i = 0; i < g_sdc.feature_count; i++)
				f[i] = g_sdc.features[v * g_sdc.feature_count + i];
			return f;
		};

		auto set_feature = [&](uint32_t v, std::vector <float> &f) {
			for (uint32_t i = 0; i < g_sdc.feature_count; i++)
				lerped_E[v * g_sdc.feature_count + i] = f[i];
		};

		std::vector <float> f00 = get_feature(complex.x);
		std::vector <float> f10 = get_feature(complex.y);
		std::vector <float> f01 = get_feature(complex.w);
		std::vector <float> f11 = get_feature(complex.z);

		for (uint32_t ix = 0; ix < sample_rate; ix++) {
			for (uint32_t iy = 0; iy < sample_rate; iy++) {
				float u = (float) ix / (sample_rate - 1);
				float v = (float) iy / (sample_rate - 1);

				{
					glm::vec3 lp00 = v00 * u * v;
					glm::vec3 lp10 = v10 * (1.0f - u) * v;
					glm::vec3 lp01 = v01 * u * (1.0f - v);
					glm::vec3 lp11 = v11 * (1.0f - u) * (1.0f - v);
					lerped_P[i] = lp00 + lp10 + lp01 + lp11;
				}

				{
					std::vector <float> f(g_sdc.feature_count);
					for (uint32_t k = 0; k < g_sdc.feature_count; k++) {
						float f00k = f00[k] * u * v;
						float f10k = f10[k] * (1.0f - u) * v;
						float f01k = f01[k] * u * (1.0f - v);
						float f11k = f11[k] * (1.0f - u) * (1.0f - v);
						f[k] = f00k + f10k + f01k + f11k;
					}

					set_feature(i, f);
				}

				i++;
			}
		}
	}

	constexpr uint32_t lines = 3;

	// Construct the network input with embeddings
	// TODO: get these from the file as well...
	constexpr uint32_t L = 8;
	constexpr uint32_t K = 16;

	uint32_t embedded_size = g_sdc.feature_count + 3 * (2 * L + 1);
	ulog_assert(embedded_size == g_dnn.H0, "eval", "embedded_size != g_dnn.H0 [%u != %u]\n", embedded_size, g_dnn.H0);

	std::vector <float> embedded;
	embedded.resize(vertex_count * embedded_size);

	for (uint32_t i = 0; i < vertex_count; i++) {
		float *vembedded = &embedded[i * embedded_size];

		// First copy the feature
		float *lfeature = &lerped_E[i * g_sdc.feature_count];
		for (uint32_t k = 0; k < g_sdc.feature_count; k++)
			vembedded[k] = lfeature[k];

		// Positional encoding
		const glm::vec3 &p = lerped_P[i];

		std::vector <float> pos_enc;
		pos_enc.push_back(p.x);
		pos_enc.push_back(p.y);
		pos_enc.push_back(p.z);

		for (uint32_t j = 0; j < L; j++) {
			glm::vec3 sp = glm::sin(powf(2.0f, j) * p);
			glm::vec3 cp = glm::cos(powf(2.0f, j) * p);
			pos_enc.push_back(sp.x);
			pos_enc.push_back(sp.y);
			pos_enc.push_back(sp.z);
			pos_enc.push_back(cp.x);
			pos_enc.push_back(cp.y);
			pos_enc.push_back(cp.z);
		}

		ulog_assert(pos_enc.size() == 3 * (2 * L + 1), "eval", "pos_enc.size() != 3 * (2 * L + 1) [%lu != %u]\n", pos_enc.size(), 3 * (2 * L + 1));

		for (uint32_t k = 0; k < pos_enc.size(); k++)
			vembedded[g_sdc.feature_count + k] = pos_enc[k];
	}

	// Evaluate the first network layer
	std::vector <float> hidden;

	hidden = matmul_biased <sinf> (g_dnn.Wm0c, embedded, embedded_size, g_dnn.W0);
	ulog_assert(hidden.size() == vertex_count * g_dnn.W0, "eval", "hidden.size() != vertex_count * g_dnn.W0 [%lu != %u]\n", hidden.size(), vertex_count * g_dnn.W0);

	hidden = matmul_biased <sinf> (g_dnn.Wm1c, hidden, g_dnn.W0, g_dnn.W1);
	ulog_assert(hidden.size() == vertex_count * g_dnn.W1, "eval", "hidden.size() != vertex_count * g_dnn.W1 [%lu != %u]\n", hidden.size(), vertex_count * g_dnn.W1);
	ulog_assert(g_dnn.W2 == 3, "eval", "W2 != 3");

	hidden = matmul_biased <nullptr> (g_dnn.Wm2c, hidden, g_dnn.W1, g_dnn.W2);
	ulog_assert(hidden.size() == vertex_count * g_dnn.W2, "eval", "hidden.size() != vertex_count * g_dnn.W2 [%lu != %u]\n", hidden.size(), vertex_count * g_dnn.W2);

	// Apply displacements
	glm::vec3 *displacements = (glm::vec3 *) hidden.data();

	std::vector <glm::vec3> final_P(vertex_count);
	for (uint32_t i = 0; i < vertex_count; i++)
		final_P[i] = lerped_P[i] + displacements[i];

	return final_P;
}

// TODO: group the complexes by proximity and batch the culling and drawing processes...

std::vector <std::array <uint32_t, 3>> nsc_indices(const std::vector <glm::vec3> &vertices, size_t complexe_count, size_t sample_rate)
{
	std::vector <std::array <uint32_t, 3>> tris;

	uint32_t t = 0;
	for (uint32_t c = 0; c < g_sdc.complex_count; c++) {
		uint32_t base = c * sample_rate * sample_rate;
		for (uint32_t ix = 0; ix < sample_rate - 1; ix++) {
			for (uint32_t iy = 0; iy < sample_rate - 1; iy++) {
				uint32_t i0 = ix + sample_rate * iy;
				uint32_t i1 = i0 + 1;
				uint32_t i2 = i0 + sample_rate;
				uint32_t i3 = i2 + 1;

				glm::vec3 p0 = vertices[base + i0];
				glm::vec3 p1 = vertices[base + i1];
				glm::vec3 p2 = vertices[base + i2];
				glm::vec3 p3 = vertices[base + i3];

				float d03 = glm::length(p0 - p3);
				float d12 = glm::length(p1 - p2);

				if (d03 < d12) {
					tris.push_back({ base + i0, base + i1, base + i3 });
					tris.push_back({ base + i0, base + i3, base + i2 });
				} else {
					tris.push_back({ base + i0, base + i1, base + i2 });
					tris.push_back({ base + i1, base + i3, base + i2 });
				}
			}
		}
	}

	return tris;
}

int main(int argc, char *argv[])
{
	// Expect a filename
	if (argc < 2) {
		printf("./rasterizer <nsc binary>\n");
		return 1;
	}

	// Open the file
	FILE *file = fopen(argv[1], "rb");
	if (!file) {
		fprintf(stderr, "Could not open file %s\n", argv[1]);
		return 1;
	}

	sdc_read(file);
	dnn_read(file);

	constexpr uint32_t rate = 16;

	// Render
	auto predicate = [](vk::PhysicalDevice phdev) {
		return littlevk::physical_device_able(phdev, {
			VK_KHR_SWAPCHAIN_EXTENSION_NAME
		});
	};

	vk::PhysicalDevice phdev;
	phdev = littlevk::pick_physical_device(predicate);

	Renderer renderer;
	renderer.from(phdev);

	while (!renderer.should_close()) {
		renderer.render();
		renderer.poll();
	}
}
