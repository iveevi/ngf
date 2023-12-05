#include "common.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

// TODO: benchmark execution times, and also in CUDA
// TODO: group the complexes by proximity and batch the culling and drawing processes...

std::vector <std::array <uint32_t, 3>> nsc_indices(const std::vector <glm::vec3> &vertices, size_t complexe_count, size_t sample_rate)
{
	std::vector <std::array <uint32_t, 3>> tris;

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

std::vector <glm::vec3> vertex_normals(const std::vector <glm::vec3> &vertice, const std::vector <std::array <uint32_t, 3>> &indices)
{
	std::vector <glm::vec3> normals(vertice.size(), glm::vec3(0.0f));

	#pragma omp parallel
	for (size_t i = 0; i < indices.size(); i++) {
		glm::vec3 p0 = vertice[indices[i][0]];
		glm::vec3 p1 = vertice[indices[i][1]];
		glm::vec3 p2 = vertice[indices[i][2]];

		glm::vec3 n = glm::cross(p2 - p0, p1 - p0);

		normals[indices[i][0]] += n;
		normals[indices[i][1]] += n;
		normals[indices[i][2]] += n;
	}

	for (size_t i = 0; i < normals.size(); i++)
		normals[i] = glm::normalize(normals[i]);

	return normals;
}

// Utilities
template <typename T, uint32_t N>
struct rolling_window {
	static_assert(std::is_floating_point_v <T>, "rolling_window only works with floating types");

	T data[N] = { 0 };
	T average = 0;
	uint32_t index = 0;

	T push(T value) {
		average -= data[index] / N;
		data[index] = value;
		average += value / N;
		index = (index + 1) % N;
		return average;
	}
};

int main(int argc, char *argv[])
{
	// Expect a filename
	if (argc < 3) {
		ulog_error("rasterizer", "./rasterizer <reference> <nsc binary>\n");
		return 1;
	}

	// Read the reference mesh
	geometry reference = loader(argv[1]).get(0);

	ulog_info("main", "Reference mesh has %d vertices and %d triangles\n", reference.vertices.size(), reference.indices.size());

	// Open the file
	FILE *file = fopen(argv[2], "rb");
	if (!file) {
		ulog_error("rasterizer", "Could not open file %s\n", argv[1]);
		return 1;
	}

	// Read neural subdivision complex
	read(file);

	constexpr uint32_t rate = 16;

	// TODO: verbose logging
	static_assert(rate <= MAXIMUM_SAMPLE_RATE, "rate > MAXIMUM_SAMPLE_RATE");

	// Configure renderer
	auto predicate = [](vk::PhysicalDevice phdev) {
		return littlevk::physical_device_able(phdev, {
			VK_KHR_SWAPCHAIN_EXTENSION_NAME
		});
	};

	vk::PhysicalDevice phdev;
	phdev = littlevk::pick_physical_device(predicate);

	Renderer renderer(phdev);

	// Create a Vulkan query pool
	vk::QueryPoolCreateInfo query_pool_create_info = {};
	query_pool_create_info.queryType = vk::QueryType::eTimestamp;
	query_pool_create_info.queryCount = 4;

	vk::QueryPool query_pool = renderer.device.createQueryPool(query_pool_create_info);

	// Evaluate the surface

	// std::vector <glm::vec3> vertices = eval_cuda(renderer.push_constants.proj, renderer.push_constants.view, rate);
	// std::vector <glm::vec3> vertices(g_sdc.complex_count * rate * rate);

	std::vector <glm::vec3> vertices = eval_cuda(rate);
	// std::vector <glm::vec3> vertices = eval(rate);
	std::vector <std::array <uint32_t, 3>> triangles = nsc_indices(vertices, g_sdc.complex_count, rate);
	std::vector <glm::vec3> normals = vertex_normals(vertices, triangles);

	// eval_cuda(rate);

	printf("vertices:  %d\n", vertices.size());
	printf("normals:   %d\n", normals.size());
	printf("triangles: %d\n", triangles.size());

	std::vector <glm::vec3> interleaved;
	for (size_t i = 0; i < vertices.size(); i++) {
		interleaved.push_back(vertices[i]);
		interleaved.push_back(normals[i]);
	}

	// Translate to a Vulkan mesh
	Transform model_transform = {};
	littlevk::Buffer vertex_buffer;
	littlevk::Buffer interleaved_buffer;
	littlevk::Buffer index_buffer;

	// TODO: make CUDA interop work (shared vertex buffer)

	vertex_buffer = littlevk::buffer(renderer.device,
			vertices,
			vk::BufferUsageFlagBits::eVertexBuffer,
			renderer.mem_props).unwrap(renderer.dal);

	interleaved_buffer = littlevk::buffer(renderer.device,
			interleaved,
			vk::BufferUsageFlagBits::eVertexBuffer,
			renderer.mem_props).unwrap(renderer.dal);

	index_buffer = littlevk::buffer(renderer.device,
			triangles,
			vk::BufferUsageFlagBits::eIndexBuffer,
			renderer.mem_props).unwrap(renderer.dal);

	enum RenderMode : int {
		Point,
		Wireframe,
		Solid,
		Normal,
		Shaded
	};

	RenderMode mode = Wireframe;
	RenderMode ref_mode = Shaded;

	uint32_t nsc_triangle_count = triangles.size();

	auto prerender_hook = [&](const vk::CommandBuffer &cmd) {
		cmd.resetQueryPool(query_pool, 0, 4);
	};

	renderer.prerender_hooks.push_back(prerender_hook);

	auto point_hook = [&](const vk::CommandBuffer &cmd) {
		if (mode != Point)
			return;

		auto *pc = &renderer.push_constants;
		pc->model = model_transform.matrix();
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, renderer.point.pipeline);
		cmd.pushConstants <Renderer::push_constants_struct> (renderer.point.pipeline_layout, vk::ShaderStageFlagBits::eVertex, 0, *pc);
		cmd.bindVertexBuffers(0, { vertex_buffer.buffer }, { 0 });
		cmd.bindIndexBuffer(index_buffer.buffer, 0, vk::IndexType::eUint32);
		cmd.draw(vertices.size(), 1, 0, 0);
	};

	auto wireframe_hook = [&](const vk::CommandBuffer &cmd) {
		if (mode != Wireframe)
			return;

		auto *pc = &renderer.push_constants;
		pc->model = model_transform.matrix();
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, renderer.wireframe.pipeline);
		cmd.pushConstants <Renderer::push_constants_struct> (renderer.wireframe.pipeline_layout, vk::ShaderStageFlagBits::eVertex, 0, *pc);
		cmd.bindVertexBuffers(0, { vertex_buffer.buffer }, { 0 });
		cmd.bindIndexBuffer(index_buffer.buffer, 0, vk::IndexType::eUint32);
		cmd.drawIndexed(nsc_triangle_count * 3, 1, 0, 0, 0);
	};

	auto solid_hook = [&](const vk::CommandBuffer &cmd) {
		if (mode != Solid)
			return;

		auto *pc = &renderer.push_constants;
		pc->model = model_transform.matrix();
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, renderer.solid.pipeline);
		cmd.pushConstants <Renderer::push_constants_struct> (renderer.solid.pipeline_layout, vk::ShaderStageFlagBits::eVertex, 0, *pc);
		cmd.bindVertexBuffers(0, { vertex_buffer.buffer }, { 0 });
		cmd.bindIndexBuffer(index_buffer.buffer, 0, vk::IndexType::eUint32);
		cmd.drawIndexed(nsc_triangle_count * 3, 1, 0, 0, 0);
	};

	auto normal_hook = [&](const vk::CommandBuffer &cmd) {
		if (mode != Normal)
			return;

		auto *pc = &renderer.push_constants;
		pc->model = model_transform.matrix();
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, renderer.normal.pipeline);
		cmd.pushConstants <Renderer::push_constants_struct> (renderer.normal.pipeline_layout, vk::ShaderStageFlagBits::eVertex, 0, *pc);
		cmd.bindVertexBuffers(0, { *interleaved_buffer }, { 0 });
		cmd.bindIndexBuffer(*index_buffer, 0, vk::IndexType::eUint32);
		cmd.drawIndexed(nsc_triangle_count * 3, 1, 0, 0, 0);
	};

	// TODO: monoid to return these hooks...
	auto shaded_hook = [&](const vk::CommandBuffer &cmd) {
		if (mode != Shaded)
			return;

		auto *pc = &renderer.push_constants;
		pc->model = model_transform.matrix();
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, renderer.shaded.pipeline);
		cmd.pushConstants <Renderer::push_constants_struct> (renderer.shaded.pipeline_layout, vk::ShaderStageFlagBits::eVertex, 0, *pc);
		cmd.bindVertexBuffers(0, { *interleaved_buffer }, { 0 });
		cmd.bindIndexBuffer(*index_buffer, 0, vk::IndexType::eUint32);
		cmd.drawIndexed(nsc_triangle_count * 3, 1, 0, 0, 0);
	};

	// Screen capture
	bool screenshot = false;

	littlevk::Buffer staging;
	auto screenshot_hook = [&](const vk::CommandBuffer &cmd, const vk::Image &image) {
		if (!screenshot)
			return;

		const vk::Extent2D &extent = renderer.window->extent;
		size_t expected_size = sizeof(uint32_t) * extent.width * extent.height; // TODO: make sure # of bytes for channels is correct

		printf("screenshot_hook: %zu %zu\n", staging.device_size(), expected_size);
		if (staging.device_size() != expected_size) {
			printf("Allocating staging buffer for screenshot\n");
			vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst;
			staging = littlevk::buffer(renderer.device,
				expected_size, usage,
				renderer.mem_props).unwrap(renderer.dal);
		}

		// Pipeline barrier
		littlevk::transition(cmd, image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferSrcOptimal);
		littlevk::copy_image_to_buffer(cmd, image, staging, extent, vk::ImageLayout::eTransferSrcOptimal);
		littlevk::transition(cmd, image, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::ePresentSrcKHR);
	};

	uint32_t screenshot_counter = 0;
	auto screenshot_save_hook = [&]() {
		if (!screenshot)
			return;

		const vk::Extent2D &extent = renderer.window->extent;

		// Wait to finish
		renderer.device.waitIdle();

		std::vector <uint32_t> pixels;
		pixels.resize(extent.width * extent.height);
		littlevk::download(renderer.device, staging, pixels);

		// Convert from BGRA to RGBA
		for (size_t i = 0; i < pixels.size(); i++) {
			uint32_t &pixel = pixels[i];
			uint8_t *bytes = (uint8_t *) &pixel;
			std::swap(bytes[0], bytes[2]);
		}

		// stbi_write_png("screenshot.png", extent.width, extent.height, 4, pixels.data(), extent.width * sizeof(uint32_t));
		std::string filename = "screenshot_" + std::to_string(screenshot_counter++) + ".png";
		stbi_write_png(filename.c_str(), extent.width, extent.height, 4, pixels.data(), extent.width * sizeof(uint32_t));

		screenshot = false;
	};

	renderer.postrender_hooks.push_back(screenshot_hook);
	renderer.postsubmit_hooks.push_back(screenshot_save_hook);

	// User interface
	rolling_window <float, 60> frame_time_window;

	auto ui_hook = [&](const vk::CommandBuffer &cmd) {
		float frame_time = frame_time_window.push(ImGui::GetIO().DeltaTime);

		ImGui::Begin("Info");

		ImGui::Text("Frame time %.2f ms", frame_time * 1000.0f);
		ImGui::Text("Framerate  %.2f fps", 1.0f / frame_time);
		ImGui::Separator();

		// TODO: radio buttons
		ImGui::Text("Mode:");
		ImGui::RadioButton("Point##nsc", (int *) &mode, Point);
		ImGui::RadioButton("Wireframe##nsc", (int *) &mode, Wireframe);
		ImGui::RadioButton("Solid##nsc",     (int *) &mode, Solid);
		ImGui::RadioButton("Normal##nsc",    (int *) &mode, Normal);
		ImGui::RadioButton("Shaded##nsc",    (int *) &mode, Shaded);
		ImGui::Separator();

		ImGui::Text("Reference:");
		ImGui::RadioButton("Normal##ref",    (int *) &ref_mode, Normal);
		ImGui::RadioButton("Shaded##ref",    (int *) &ref_mode, Shaded);
		ImGui::Separator();

		if (ImGui::Button("Screenshot framebuffer")) {
			screenshot = true;
			printf("Screenshot requested\n");
		}

		ImGui::End();
	};

	renderer.hooks.push_back(ui_hook);

	// Timing hooks
	auto query_start_hook = [&](const vk::CommandBuffer &cmd) {
		cmd.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, query_pool, 0);
	};

	auto query_end_hook = [&](const vk::CommandBuffer &cmd) {
		cmd.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, query_pool, 1);
	};

	// Order the hooks
	renderer.hooks.push_back(query_start_hook);

	renderer.hooks.push_back(point_hook);
	renderer.hooks.push_back(wireframe_hook);
	renderer.hooks.push_back(solid_hook);
	renderer.hooks.push_back(normal_hook);
	renderer.hooks.push_back(shaded_hook);

	renderer.hooks.push_back(query_end_hook);

	// Retrieve the query results
	std::array <uint64_t, 2> query_results;

	vk::PhysicalDeviceProperties properties = renderer.phdev.getProperties();
	uint64_t timestamp_period = properties.limits.timestampPeriod;

	auto query_retrieval_hook = [&]() {
		renderer.device.getQueryPoolResults(query_pool, 0, 2,
			query_results.size() * sizeof(uint64_t),
			query_results.data(), sizeof(uint64_t),
			vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);

		float time = (query_results[1] - query_results[0]) * timestamp_period/1e6f;
		// printf("Neural Subdivision Complexes time: %.2f ms\n", time);
	};

	renderer.postsubmit_hooks.push_back(query_retrieval_hook);

	// Translate to a Vulkan mesh
	Transform ref_model_transform {
		.position = glm::vec3 { 2.0f, 0.0f, 0.0f },
	};

	littlevk::Buffer ref_vertex_buffer;
	littlevk::Buffer ref_index_buffer;

	ref_vertex_buffer = littlevk::buffer(renderer.device,
			interleave_attributes(reference),
			vk::BufferUsageFlagBits::eVertexBuffer,
			renderer.mem_props).unwrap(renderer.dal);

	ref_index_buffer = littlevk::buffer(renderer.device,
			reference.indices,
			vk::BufferUsageFlagBits::eIndexBuffer,
			renderer.mem_props).unwrap(renderer.dal);

	auto ref_normal_hook = [&](const vk::CommandBuffer &cmd) {
		if (ref_mode != Normal)
			return;

		auto *pc = &renderer.push_constants;
		pc->model = ref_model_transform.matrix();
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, renderer.normal.pipeline);
		cmd.pushConstants <Renderer::push_constants_struct> (renderer.normal.pipeline_layout, vk::ShaderStageFlagBits::eVertex, 0, *pc);
		cmd.bindVertexBuffers(0, { *ref_vertex_buffer }, { 0 });
		cmd.bindIndexBuffer(*ref_index_buffer, 0, vk::IndexType::eUint32);
		cmd.drawIndexed(reference.indices.size() * 3, 1, 0, 0, 0);
	};

	auto ref_shaded_hook = [&](const vk::CommandBuffer &cmd) {
		if (ref_mode != Shaded)
			return;

		auto *pc = &renderer.push_constants;
		pc->model = ref_model_transform.matrix();
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, renderer.shaded.pipeline);
		cmd.pushConstants <Renderer::push_constants_struct> (renderer.shaded.pipeline_layout, vk::ShaderStageFlagBits::eVertex, 0, *pc);
		cmd.bindVertexBuffers(0, { *ref_vertex_buffer }, { 0 });
		cmd.bindIndexBuffer(*ref_index_buffer, 0, vk::IndexType::eUint32);
		cmd.drawIndexed(reference.indices.size() * 3, 1, 0, 0, 0);
	};

	// More query hooks
	auto ref_query_start_hook = [&](const vk::CommandBuffer &cmd) {
		cmd.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, query_pool, 2);
	};

	auto ref_query_end_hook = [&](const vk::CommandBuffer &cmd) {
		cmd.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, query_pool, 3);
	};

	// renderer.hooks.push_back(ref_query_start_hook);
	//
	// renderer.hooks.push_back(ref_normal_hook);
	// renderer.hooks.push_back(ref_shaded_hook);
	//
	// renderer.hooks.push_back(ref_query_end_hook);

	// Retrieve the query results
	std::array <uint64_t, 2> ref_query_results;

	auto ref_query_retrieval_hook = [&]() {
		renderer.device.getQueryPoolResults(query_pool, 2, 2,
			ref_query_results.size() * sizeof(uint64_t),
			ref_query_results.data(), sizeof(uint64_t),
			vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);

		float time = (ref_query_results[1] - ref_query_results[0]) * timestamp_period/1e6f;
		// printf("Reference time: %.2f ms\n", time);
	};

	// renderer.postsubmit_hooks.push_back(ref_query_retrieval_hook);

	// TODO: time each call...
	while (!renderer.should_close()) {
		// printf("Difference in vertex count: %lu vs %lu\n", vertices.size(), reference.vertices.size());

		std::vector <glm::vec3> vertices = eval_cuda(rate);

		std::vector <glm::vec3> interleaved;
		for (size_t i = 0; i < vertices.size(); i++) {
			interleaved.push_back(vertices[i]);
			interleaved.push_back(normals[i]);
		}

		littlevk::upload(renderer.device, vertex_buffer, vertices);
		littlevk::upload(renderer.device, interleaved_buffer, interleaved);

		renderer.render();
		renderer.poll();
	}

	renderer.destroy();
}
