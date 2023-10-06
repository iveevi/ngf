#include "common.hpp"

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

	sdc_read(file);
	dnn_read(file);

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

	Renderer renderer;
	renderer.from(phdev);

	// Evaluate the surface
	// std::vector <glm::vec3> vertices = eval_cuda(renderer.push_constants.proj, renderer.push_constants.view, rate);
	std::vector <glm::vec3> vertices(g_sdc.complex_count * rate * rate);
	std::vector <std::array <uint32_t, 3>> indices = nsc_indices(vertices, g_sdc.complex_count, rate);
	std::vector <glm::vec3> normals = vertex_normals(vertices, indices);

	printf("vertices: %d\n", vertices.size());
	printf("normals: %d\n", normals.size());
	printf("indices: %d\n", indices.size());

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

	// TODO: make CUDA interop work

	vertex_buffer = littlevk::buffer(renderer.device,
			vertices,
			vk::BufferUsageFlagBits::eVertexBuffer,
			renderer.mem_props).unwrap(renderer.dal);

	interleaved_buffer = littlevk::buffer(renderer.device,
			interleaved,
			vk::BufferUsageFlagBits::eVertexBuffer,
			renderer.mem_props).unwrap(renderer.dal);

	index_buffer = littlevk::buffer(renderer.device,
			indices,
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

	uint32_t nsc_triangle_count = vertices.size() / 3;

	auto point_hook = [&](const vk::CommandBuffer &cmd) {
		if (mode != Point)
			return;

		auto *pc = &renderer.push_constants;
		pc->model = model_transform.matrix();
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, renderer.point.pipeline);
		cmd.pushConstants <Renderer::push_constants_struct> (renderer.point.pipeline_layout, vk::ShaderStageFlagBits::eVertex, 0, *pc);
		cmd.bindVertexBuffers(0, { vertex_buffer.buffer }, { 0 });
		cmd.bindIndexBuffer(index_buffer.buffer, 0, vk::IndexType::eUint32);
		// cmd.drawIndexed(nsc_triangle_count * 3, 1, 0, 0, 0);
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

	auto ui_hook = [&](const vk::CommandBuffer &cmd) {
		float frame_time = ImGui::GetIO().DeltaTime;
		ImGui::Begin("Info");

		ImGui::Text("Frame time: %f ms", frame_time * 1000.0f);
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

		ImGui::End();
	};

	renderer.hooks.push_back(point_hook);
	renderer.hooks.push_back(wireframe_hook);
	renderer.hooks.push_back(solid_hook);
	renderer.hooks.push_back(normal_hook);
	renderer.hooks.push_back(shaded_hook);
	renderer.hooks.push_back(ui_hook);

	// Translate to a Vulkan mesh
	Transform ref_model_transform {
		.position = glm::vec3 { 1.0f, 0.0f, 4.0f },
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

	renderer.hooks.push_back(ref_normal_hook);
	renderer.hooks.push_back(ref_shaded_hook);

	while (!renderer.should_close()) {
		auto [vertices, triangles, count] = eval_cuda(renderer.push_constants.proj, renderer.push_constants.view, rate);
	
		std::vector <glm::vec3> normals = vertex_normals(vertices, triangles);

		std::vector <glm::vec3> interleaved;
		for (size_t i = 0; i < vertices.size(); i++) {
			interleaved.push_back(vertices[i]);
			interleaved.push_back(normals[i]);
		}

		littlevk::upload(renderer.device, interleaved_buffer, interleaved);
		littlevk::upload(renderer.device, vertex_buffer, vertices);
		littlevk::upload(renderer.device, index_buffer, triangles);
		nsc_triangle_count = count;

		renderer.render();
		renderer.poll();
	}

	renderer.destroy();
}
