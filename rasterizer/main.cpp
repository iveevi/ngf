#include <cstdlib>
#include <optional>
#include <fstream>

#include <glm/glm.hpp>

#include <littlevk/littlevk.hpp>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>

#include "common.hpp"
#include "glio.hpp"
#include "vkutil.hpp"
#include "context.hpp"

struct MVP {
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
};

bool valid_window(const DeviceRenderContext &engine)
{
	return glfwWindowShouldClose(engine.window.handle) == 0;
}

// Interface rendering
struct Options {
	bool backface_culling = false;
	int resolution = 15;
	std::string key = "Shaded";
};

struct Statistics {
	size_t patch_count;
};

std::string imgui_pass(const vk::CommandBuffer &cmd, Options &options, const Statistics &stats, size_t kbs)
{
	static std::vector <float> frametimes;
	static const size_t WINDOW = 60;
	static std::string current_path = "";

	std::string path;

	ImGui_ImplVulkan_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	ImGui::Begin("Panel");

	float fr = ImGui::GetIO().Framerate;
	float ft = 1.0f/fr;

	frametimes.push_back(ft);
	if (frametimes.size() > WINDOW)
		frametimes.erase(frametimes.begin());

	auto tflags = ImGuiTreeNodeFlags_DefaultOpen;

	if (ImGui::CollapsingHeader("Info", tflags)) {
		ImGui::Text("WASD to move horizontally");
		ImGui::Text("QE to vertically");
		ImGui::Text("Mouse to change view");
	}

	if (ImGui::CollapsingHeader("Performance", tflags)) {
		float max = -1e10f;
		for (float ft : frametimes)
			max = std::max(max, ft);

		ImGui::Text("%05.2f ms per frame (average)", ft * 1000.0f);
		ImGui::Text("%05.2f ms per frame (max)", max * 1000.0f);
		ImGui::Text("%04d frames per second", (int) fr);
	}

	if (ImGui::CollapsingHeader("Statistics", tflags)) {
		ImGui::Text("%4ld patches", stats.patch_count);
		ImGui::Text("%4ld KB", kbs);
	}

	if (ImGui::CollapsingHeader("Render mode", tflags)) {
		for (const auto &[k, _] : fragment_shaders) {
			if (ImGui::RadioButton(k.c_str(), options.key == k))
				options.key = k;
		}
	}

	if (ImGui::CollapsingHeader("Models", tflags)) {
		static const std::map <std::string, std::filesystem::path> models {
			{ "armadillo", "resources/models/armadillo.bin" },
			{ "buddha",    "resources/models/buddha.bin"    },
			{ "dragon",    "resources/models/dragon.bin"    },
			{ "ganesha",   "resources/models/ganesha.bin"   },
			{ "nefertiti", "resources/models/nefertiti.bin" },
		};

		for (const auto &[k, p] : models) {
			if (ImGui::RadioButton(k.c_str(), current_path == p)) {
				ulog_info(__FUNCTION__, "loading %s\n", k.c_str());
				current_path = path = p;
			}
		}
	}

	if (ImGui::CollapsingHeader("Options", tflags)) {
		ImGui::Checkbox("Backface culling (aprox.)", &options.backface_culling);
		ImGui::DragInt("Tessellation", (int *) &options.resolution, 0.05f, 2, 15);
	}

	ImGui::End();

	ImGui::Render();
	ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

	return path;
}

struct HomogenizedNGF {
	std::vector <glm::ivec4> patches;
	std::vector <glm::vec4> vertices;
	std::vector <glm::vec4> features;
	std::vector <float> biases;
	std::vector <float> W0;
	std::vector <float> W1;
	std::vector <float> W2;
	std::vector <float> W3;

	static HomogenizedNGF from(const NGF &);
};

HomogenizedNGF HomogenizedNGF::from(const NGF &ngf)
{
	// Concatenate the biases into a single buffer
	std::vector <float> biases;
	for (int32_t i = 0; i < LAYERS; i++)
		biases.insert(biases.end(), ngf.biases[i].begin(), ngf.biases[i].end());

	// Align to vec4 size
	size_t fixed = (biases.size() + 3)/4;
	biases.resize(4 * fixed);

	// Weights (transposed)
	size_t w0c = ngf.weights[0].height;
	std::vector <float> W0(64 * w0c);
	std::vector <float> W1(64 * 64);
	std::vector <float> W2(64 * 64);
	std::vector <float> W3 = ngf.weights[3];

	for (size_t i = 0; i < 64; i++) {
		for (size_t j = 0; j < 64; j++) {
			W1[j * 64 + i] = ngf.weights[1][i * 64 + j];
			W2[j * 64 + i] = ngf.weights[2][i * 64 + j];
		}

		for (size_t j = 0; j < w0c; j++)
			W0[j * 64 + i] = ngf.weights[0][i * w0c + j];
	}

	// Feature vector
	std::vector <glm::vec4> features(ngf.patch_count * ngf.feature_size);

	for (size_t i = 0; i < ngf.patch_count; i++) {
		glm::ivec4 complex = ngf.patches[i];
		for (size_t j = 0; j < ngf.feature_size; j++) {
			float f0 = ngf.features[complex.x * ngf.feature_size + j];
			float f1 = ngf.features[complex.y * ngf.feature_size + j];
			float f2 = ngf.features[complex.z * ngf.feature_size + j];
			float f3 = ngf.features[complex.w * ngf.feature_size + j];
			features[i * ngf.feature_size + j] = glm::vec4(f0, f1, f2, f3);
		}
	}

	return HomogenizedNGF {
		.patches = ngf.patches,
		.vertices = ngf.vertices,
		.features = features,
		.biases = biases,
		.W0 = W0, .W1 = W1,
		.W2 = W2, .W3 = W3,
	};
}

// Configuring descriptor sets
vk::DescriptorSet neural_geometry_image(DeviceRenderContext &engine, const HomogenizedNGF &hngf)
{
	static const auto allocator = [&] <typename T> (const std::vector <T> &buffer, vk::Extent2D extent,
			vk::ImageType type = vk::ImageType::e2D,
			vk::Format format = vk::Format::eR32G32B32A32Sfloat) {
		return general_allocator(engine.device, engine.memory_properties,
				engine.command_pool, engine.graphics_queue,
				engine.dal, buffer, extent, type, format);
	};

	// Allocate textures for the NGF
	uint32_t patches = hngf.patches.size();
	uint32_t features = hngf.features.size() / patches;
	uint32_t vertices = hngf.vertices.size();
	uint32_t biases = hngf.biases.size();
	uint32_t wsize = hngf.W0.size() >> 6;

	auto complex_texture = allocator(hngf.patches, { patches, 1 },
			vk::ImageType::e1D, vk::Format::eR32G32B32A32Sint);

	auto vertex_texture = allocator(hngf.vertices, { vertices, 1 }, vk::ImageType::e1D);
	auto feature_texture = allocator(hngf.features, { features, patches });

	auto bias_texture = allocator(hngf.biases, { biases, 1 }, vk::ImageType::e1D);
	auto W0_texture = allocator(hngf.W0, { 16, wsize });
	auto W1_texture = allocator(hngf.W1, { 16, 64 });
	auto W2_texture = allocator(hngf.W2, { 16, 64 });
	auto W3_texture = allocator(hngf.W3, { 16, 3 });

	// Bind the resources
	vk::DescriptorSet dset = littlevk::bind(engine.device, engine.descriptor_pool)
		.allocate_descriptor_sets(*engine.primaries.at("Shaded").dsl).front();

	vk::Sampler floating_sampler = littlevk::SamplerAssembler(engine.device, engine.dal);

	auto SROO = vk::ImageLayout::eShaderReadOnlyOptimal;

	littlevk::bind(engine.device, dset, meshlet_dslbs)
		.update(0, 0, floating_sampler, complex_texture.view, SROO)
		.update(1, 0, floating_sampler, vertex_texture.view, SROO)
		.update(2, 0, floating_sampler, feature_texture.view, SROO)
		.update(3, 0, floating_sampler, bias_texture.view, SROO)
		.update(4, 0, floating_sampler, W0_texture.view, SROO)
		.update(5, 0, floating_sampler, W1_texture.view, SROO)
		.update(6, 0, floating_sampler, W2_texture.view, SROO)
		.update(7, 0, floating_sampler, W3_texture.view, SROO)
		.finalize();

	return dset;
}

vk::DescriptorSet environment_map(DeviceRenderContext &engine, const std::filesystem::path &path)
{
	vk::Sampler sampler = littlevk::SamplerAssembler(engine.device, engine.dal);

	auto tex = Texture::load("resources/environment.hdr");
	littlevk::Image dtex = engine.upload_texture(tex);

	vk::DescriptorSet dset = littlevk::bind(engine.device, engine.descriptor_pool)
		.allocate_descriptor_sets(*engine.environment.dsl).front();

	littlevk::bind(engine.device, dset, environment_dslbs)
		.update(0, 0, sampler, dtex.view, vk::ImageLayout::eShaderReadOnlyOptimal)
		.finalize();

	return dset;
}

std::ifstream::pos_type filesize(const std::string &filename)
{
	std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);
	return in.tellg();
}

int main(int argc, char *argv[])
{
	littlevk::config().enable_validation_layers = false;
	littlevk::config().enable_logging = false;

	if (argc < 2) {
		ulog_error("testbed", "Usage: testbed <ngf>\n");
		return EXIT_FAILURE;
	}

	std::string path = argv[1];

	// Load the neural geometry field
	auto ngf = NGF::load(path);
	auto hngf = HomogenizedNGF::from(ngf);

	// Configure renderer
	static const std::vector <const char *> extensions {
		VK_EXT_MESH_SHADER_EXTENSION_NAME,
		VK_EXT_ROBUSTNESS_2_EXTENSION_NAME,
		VK_KHR_MAINTENANCE_4_EXTENSION_NAME,
		VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME,
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
	};

	auto predicate = [](const vk::PhysicalDevice &phdev) {
		return littlevk::physical_device_able(phdev, extensions);
	};

	vk::PhysicalDevice phdev = littlevk::pick_physical_device(predicate);

	// Initialization
	DeviceRenderContext engine = DeviceRenderContext::from(phdev, extensions, ngf.feature_size);

	engine.camera_transform.position = glm::vec3 { 0, 0, 3 };
	engine.camera_transform.rotation = glm::vec3 { 0, glm::radians(180.0f), 0 };

	// Descriptor sets
	auto meshlet_dset = neural_geometry_image(engine, hngf);
	auto environment_dset = environment_map(engine, "resources/environment.hdr");

	// Frame data
	Options options;
	Statistics stats;

	stats.patch_count = ngf.patch_count;

	size_t frame = 0;
	size_t kbs = filesize(path)/1024;
	while (valid_window(engine)) {
		// Get events
		glfwPollEvents();

		// Handle input
		handle_key_input(engine.window.handle, engine.camera_transform);

		// Update camera state before passing to render hooks
		engine.camera.aspect = engine.aspect_ratio();

		MVP mvp;
		mvp.view = engine.camera.view_matrix(engine.camera_transform);
		mvp.proj = engine.camera.perspective_matrix();

		// Frame
		auto frame_info = new_frame(engine, frame);
		if (!frame_info)
			continue;

		auto [cmd, op] = *frame_info;

		glm::vec4 color(1.0f);
		if (options.key == "Depth")
			color = glm::vec4(0.0f);

		render_pass_begin(engine, cmd, op, color);

		if (options.key == "Shaded") {
			RayFrame rayframe = engine.camera.rayframe(engine.camera_transform);

			cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, engine.environment.handle);
			cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
					engine.environment.layout,
					0, { environment_dset }, nullptr);
			cmd.pushConstants <RayFrame> (engine.environment.layout,
				vk::ShaderStageFlagBits::eFragment,
				0, rayframe);
			cmd.bindVertexBuffers(0, { VK_NULL_HANDLE }, { 0 });
			cmd.draw(6, 1, 0, 0);
		}

		const auto &ppl = engine.primaries[options.key];
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, ppl.handle);
		cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, ppl.layout,
				0, { meshlet_dset }, nullptr);

		// Task/Mesh shader push constants
		int flags = 0;
		flags |= int(options.backface_culling);

		glm::vec3 viewing = glm::vec3(glm::inverse(mvp.view) * glm::vec4(0, 0, 1, 0));

		float time = glfwGetTime();

		TaskData task_data;
		task_data.model = Transform().matrix();
		task_data.view = mvp.view;
		task_data.proj = mvp.proj;
		task_data.flags = flags;
		task_data.resolution = options.resolution;
		task_data.viewing = viewing;
		task_data.time = time;

		cmd.pushConstants <TaskData> (ppl.layout,
			vk::ShaderStageFlagBits::eMeshEXT
				| vk::ShaderStageFlagBits::eTaskEXT,
			0, task_data);

		// Fragment shader push constants
		ShadingData shading_data;
		shading_data.viewing = viewing;
		shading_data.color = glm::vec3(0.59, 0.74, 0.76);

		cmd.pushConstants <ShadingData> (ppl.layout,
			vk::ShaderStageFlagBits::eFragment,
			sizeof(TaskData), shading_data);

		cmd.drawMeshTasksEXT(ngf.patch_count, 1, 1);

		// Interface
		auto path = imgui_pass(cmd, options, stats, kbs);
		if (path.size()) {
			auto ngf = NGF::load(path);
			auto hngf = HomogenizedNGF::from(ngf);
			meshlet_dset = neural_geometry_image(engine, hngf);
			kbs = filesize(path)/1024;
		}

		// End of render pass
		render_pass_end(engine, cmd);

		// Conclude the frame and submit
		end_frame(engine.graphics_queue, engine.sync, cmd, frame);

		// Present the frame
		present_frame(engine, op, frame);

		// Post frame
		frame = 1 - frame;
	}

	// Free the resources
	engine.window.drop();
	engine.dal.drop();
}
