#include "pipeline.hpp"
#include "util.hpp"

#ifndef SHADERS_DIRECTORY
#define SHADERS_DIRECTORY ""
#endif

static constexpr vk::VertexInputBindingDescription vertex_binding {
	0, sizeof(Vertex), vk::VertexInputRate::eVertex,
};

static constexpr std::array <vk::VertexInputAttributeDescription, 2> vertex_attributes {
	vk::VertexInputAttributeDescription {
		0, 0, vk::Format::eR32G32B32Sfloat, 0
	},

	vk::VertexInputAttributeDescription {
		1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, normal)
	}
};

Pipeline ppl_normals
(
		const vk::Device      &device,
		const vk::RenderPass  &rp,
		const vk::Extent2D    &extent,
		littlevk::Deallocator *dal
)
{
	Pipeline ppl;

	// Read shader source
	std::string vertex_source = readfile(SHADERS_DIRECTORY "/mesh.vert.glsl");
	std::string fragment_source = readfile(SHADERS_DIRECTORY "/normals.frag.glsl");

	// Compile shader modules
	vk::ShaderModule vertex_module = littlevk::shader::compile(
		device, vertex_source,
		vk::ShaderStageFlagBits::eVertex,
		{ SHADERS_DIRECTORY }
	).unwrap(dal);

	vk::ShaderModule fragment_module = littlevk::shader::compile(
		device, fragment_source,
		vk::ShaderStageFlagBits::eFragment,
		{ SHADERS_DIRECTORY }
	).unwrap(dal);

	std::vector <vk::PipelineShaderStageCreateInfo> shader_stages {
		vk::PipelineShaderStageCreateInfo {
			{}, vk::ShaderStageFlagBits::eVertex, vertex_module, "main"
		},
		vk::PipelineShaderStageCreateInfo {
			{}, vk::ShaderStageFlagBits::eFragment, fragment_module, "main"
		}
	};

	// Create the pipeline
	vk::PushConstantRange push_constant_range {
		vk::ShaderStageFlagBits::eVertex,
		0, sizeof(BasePushConstants)
	};

	ppl.layout = littlevk::pipeline_layout(
		device,
		vk::PipelineLayoutCreateInfo {
			{}, {}, push_constant_range
		}
	).unwrap(dal);

	littlevk::pipeline::GraphicsCreateInfo pipeline_info;
	pipeline_info.subpass = 0;
	pipeline_info.shader_stages = shader_stages;
	pipeline_info.vertex_binding = vertex_binding;
	pipeline_info.vertex_attributes = vertex_attributes;
	pipeline_info.extent = extent;
	pipeline_info.pipeline_layout = ppl.layout;
	pipeline_info.render_pass = rp;
	pipeline_info.fill_mode = vk::PolygonMode::eFill;
	pipeline_info.cull_mode = vk::CullModeFlagBits::eNone;
	pipeline_info.dynamic_viewport = true;

	ppl.pipeline = littlevk::pipeline::compile(device, pipeline_info).unwrap(dal);

	return ppl;
}

Pipeline ppl_ngf
(
		const vk::Device      &device,
		const vk::RenderPass  &rp,
		const vk::Extent2D    &extent,
		littlevk::Deallocator *dal
)
{
	Pipeline ppl;

	// Read shader source
	std::string task_source = readfile(SHADERS_DIRECTORY "/ngf.task.glsl");
	std::string mesh_source = readfile(SHADERS_DIRECTORY "/ngf.mesh.glsl");
	std::string fragment_source = readfile(SHADERS_DIRECTORY "/ngf.frag.glsl");

	// Compile shader modules
	vk::ShaderModule task_module = littlevk::shader::compile(
		device, task_source,
		vk::ShaderStageFlagBits::eTaskEXT,
		{ SHADERS_DIRECTORY }
	).unwrap(dal);

	vk::ShaderModule mesh_module = littlevk::shader::compile(
		device, mesh_source,
		vk::ShaderStageFlagBits::eMeshEXT,
		{ SHADERS_DIRECTORY }
	).unwrap(dal);

	vk::ShaderModule fragment_module = littlevk::shader::compile(
		device, fragment_source,
		vk::ShaderStageFlagBits::eFragment,
		{ SHADERS_DIRECTORY }
	).unwrap(dal);

	printf("task module: %p\n", task_module);
	printf("mesh module: %p\n", mesh_module);
	printf("fragment module: %p\n", fragment_module);

	std::vector <vk::PipelineShaderStageCreateInfo> shader_stages {
		vk::PipelineShaderStageCreateInfo {
			{}, vk::ShaderStageFlagBits::eTaskEXT, task_module, "main"
		},
		vk::PipelineShaderStageCreateInfo {
			{}, vk::ShaderStageFlagBits::eMeshEXT, mesh_module, "main"
		},
		vk::PipelineShaderStageCreateInfo {
			{}, vk::ShaderStageFlagBits::eFragment, fragment_module, "main"
		}
	};

	// Create the pipeline
	// TODO: put mvp and stuff here later
	vk::PushConstantRange mesh_task_pc {
		vk::ShaderStageFlagBits::eMeshEXT | vk::ShaderStageFlagBits::eTaskEXT,
		0, sizeof(NGFPushConstants)
	};

	vk::PushConstantRange shading_pc {
		vk::ShaderStageFlagBits::eFragment,
		sizeof(NGFPushConstants), sizeof(ShadingPushConstants)
	};

	// TODO: inline this stuff
	std::array <vk::PushConstantRange, 2> push_constants {
		mesh_task_pc, shading_pc
	};

	// Buffer bindings
	vk::DescriptorSetLayoutBinding points {};
	points.binding = 0;
	points.descriptorCount = 1;
	points.descriptorType = vk::DescriptorType::eStorageBuffer;
	points.stageFlags = vk::ShaderStageFlagBits::eMeshEXT | vk::ShaderStageFlagBits::eTaskEXT;

	vk::DescriptorSetLayoutBinding features {};
	features.binding = 1;
	features.descriptorCount = 1;
	features.descriptorType = vk::DescriptorType::eStorageBuffer;
	features.stageFlags = vk::ShaderStageFlagBits::eMeshEXT;

	vk::DescriptorSetLayoutBinding patches {};
	patches.binding = 2;
	patches.descriptorCount = 1;
	patches.descriptorType = vk::DescriptorType::eStorageBuffer;
	patches.stageFlags = vk::ShaderStageFlagBits::eMeshEXT | vk::ShaderStageFlagBits::eTaskEXT;

	vk::DescriptorSetLayoutBinding network {};
	network.binding = 3;
	network.descriptorCount = 1;
	network.descriptorType = vk::DescriptorType::eStorageBuffer;
	network.stageFlags = vk::ShaderStageFlagBits::eMeshEXT;

	std::array <vk::DescriptorSetLayoutBinding, 4> bindings {
		points, features, patches, network
	};

	vk::DescriptorSetLayoutCreateInfo dsl_info {};
	dsl_info.bindingCount = bindings.size();
	dsl_info.pBindings = bindings.data();

	ppl.dsl = device.createDescriptorSetLayout(dsl_info);
	ppl.layout = littlevk::pipeline_layout(
		device,
		vk::PipelineLayoutCreateInfo {
			{}, ppl.dsl, push_constants
		}
	).unwrap(dal);

	littlevk::pipeline::GraphicsCreateInfo pipeline_info;
	pipeline_info.subpass = 0;
	pipeline_info.shader_stages = shader_stages;
	pipeline_info.extent = extent;
	pipeline_info.pipeline_layout = ppl.layout;
	pipeline_info.render_pass = rp;
	pipeline_info.fill_mode = vk::PolygonMode::eFill;
	pipeline_info.cull_mode = vk::CullModeFlagBits::eNone;
	pipeline_info.dynamic_viewport = true;

	ppl.pipeline = littlevk::pipeline::compile(device, pipeline_info).unwrap(dal);

	return ppl;
}
