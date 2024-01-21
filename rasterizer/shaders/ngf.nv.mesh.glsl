#version 460

#extension GL_GOOGLE_include_directive: require
#extension GL_NV_mesh_shader: require
#extension GL_EXT_debug_printf : require
#extension GL_EXT_control_flow_attributes : require

#include "payload.h"

layout (local_size_x = 32) in;

layout (triangles, max_vertices = 64, max_primitives = 98) out;

// Inputs
taskNV in Data {
	Payload payload;
} IN;

layout (push_constant) uniform NGFPushConstants {
	mat4 model;
	mat4 view;
	mat4 proj;

	vec2 extent;
	float time;
};

layout (binding = 0) readonly buffer Points
{
	vec3 data[];
} points;

layout (binding = 1) readonly buffer Features
{
	uint dimension;
	float data[];
} features;

layout (binding = 2) readonly buffer Patches
{
	ivec4 data[];
} patches;

// Local execution data
// TODO: enable arbitrary feature size
//   there was some bug when passing
//   feature size through PC or RD SSBO
const uint ENCODING_LEVELS = 10;
const uint FEATURE_SIZE    = 20;
const uint FFIN            = FEATURE_SIZE + 3 * (2 * ENCODING_LEVELS + 1);

// Neural network weights
layout (binding = 3) readonly buffer Layers
{
	float bias3[3];
	float weight3[64 * 3];

	float bias2[64];
	float weight2[64 * 64];

	float bias1[64];
	float weight1[64 * 64];

	float bias0[64];
	float weight0[64 * FFIN];
} layers;

// Outputs
out gl_MeshPerVertexNV {
	vec4  gl_Position;
} gl_MeshVerticesNV[];

layout (location = 0) out vec3 position[];
layout (location = 2) out flat uint pindex[];

// shared vec3 vertices[WORK_GROUP_SIZE * WORK_GROUP_SIZE];

vec4 project(vec3 p)
{
	vec4 pp = proj * view * model * vec4(p, 1.0f);
	pp.y = -pp.y;
	pp.z = (pp.z + pp.w) / 2.0;
	return pp;
}

float leaky_relu(float x)
{
	return max(x, 0.01f * x);
}

// Eval variables
float hidden0[64];
float hidden1[64];
float hidden2[64];

vec3 eval(ivec4 complex, float u, float v)
{
	vec3 v0 = points.data[complex.x];
	vec3 v1 = points.data[complex.y];
	vec3 v2 = points.data[complex.z];
	vec3 v3 = points.data[complex.w];

	// TODO: skip this and fma? or lerp/mix
	v0 = v0 * (1 - u) * (1 - v);
	v1 = v1 * (1 - u) * v;
	v2 = v2 * u * v;
	v3 = v3 * u * (1 - v);

	vec3 vertex = v0 + v1 + v2 + v3;

	// TODO: vectorize this calculation by packing (and padding) features into vec4s
	float feature[FEATURE_SIZE];
	for (uint i = 0; i < FEATURE_SIZE; i++) {
		float f0 = features.data[complex.x * FEATURE_SIZE + i];
		float f1 = features.data[complex.y * FEATURE_SIZE + i];
		float f2 = features.data[complex.z * FEATURE_SIZE + i];
		float f3 = features.data[complex.w * FEATURE_SIZE + i];

		f0 *= (1 - u) * (1 - v);
		f1 *= (1 - u) * v;
		f2 *= u * v;
		f3 *= u * (1 - v);

		feature[i] = f0 + f1 + f2 + f3;
	}

	// Positional encoding
	float ffin[FFIN];

	uint k = 0;

	[[unroll]]
	for (uint i = 0; i < FEATURE_SIZE; i++)
		ffin[k++] = feature[i];

	ffin[k++] = vertex.x;
	ffin[k++] = vertex.y;
	ffin[k++] = vertex.z;
	for (uint i = 0; i < ENCODING_LEVELS; i++) {
		float p = pow(2, i);
		vec3 sin_v = sin(p * vertex);
		vec3 cos_v = cos(p * vertex);

		ffin[k++] = sin_v.x;
		ffin[k++] = sin_v.y;
		ffin[k++] = sin_v.z;

		ffin[k++] = cos_v.x;
		ffin[k++] = cos_v.y;
		ffin[k++] = cos_v.z;
	}

	// Network evaluation
	uint isize;
	uint osize;

	// Layer 0
	isize = FFIN;
	osize = 64;

	for (uint i = 0; i < osize; i++) {
		float sum = 0.0f;
		for (uint j = 0; j < isize; j++)
			sum += layers.weight0[i * isize + j] * ffin[j];
		hidden0[i] = leaky_relu(sum + layers.bias0[i]);
	}

	// Layer 1
	isize = 64;
	osize = 64;

	for (uint i = 0; i < osize; i++) {
		float sum = 0.0f;
		for (uint j = 0; j < isize; j++)
			sum += layers.weight1[i * isize + j] * hidden0[j];
		hidden1[i] = leaky_relu(sum + layers.bias1[i]);
	}

	// Layer 2
	isize = 64;
	osize = 64;

	for (uint i = 0; i < osize; i++) {
		float sum = 0.0f;
		for (uint j = 0; j < isize; j++)
			sum += layers.weight2[i * isize + j] * hidden1[j];
		hidden2[i] = leaky_relu(sum + layers.bias2[i]);
	}

	// Layer 3
	float D[3];

	isize = 64;
	osize = 3;

	[[unroll]]
	for (uint i = 0; i < 3; i++) {
		float sum = 0.0f;
		for (uint j = 0; j < isize; j++)
			sum += layers.weight3[i * isize + j] * hidden2[j];
		D[i] = sum + layers.bias3[i];
	}

	vertex.x += D[0];
	vertex.y += D[1];
	vertex.z += D[2];

	return vertex;
}

void main()
{
	// int qwidth = 7;
	// int qheight = 7;
	// const uint MAX_QSIZE = WORK_GROUP_SIZE - 1;
	//
	// // TODO: need dyanmic QSIZE/stride depending on resolution
	// uvec2 offset = gl_LocalInvocationID.xy + (MAX_QSIZE * gl_WorkGroupID.xy);
	// if (offset.x > payload.resolution || offset.y > payload.resolution)
	// 	return;
	//
	// uvec2 offset_triangles = MAX_QSIZE * gl_WorkGroupID.xy;
	//
	// uint total_qsize = payload.resolution - 1;
	// uint qwidth = min(MAX_QSIZE, total_qsize - offset_triangles.x);
	// uint qheight = min(MAX_QSIZE, total_qsize - offset_triangles.y);
	//
	// uint vwidth = qwidth + 1;
	// uint vheight = qheight + 1;
	// SetMeshOutputsEXT(vwidth * vheight, 2 * qwidth * qheight);

	// TODO: skip if over the needed vertex res
	// debugPrintfEXT("work group (%d, %d), triangle batch is (%d, %d) out of total (%d, %d)\n",
	// 	gl_WorkGroupID.x, gl_WorkGroupID.y,
	// 	qwidth, qheight,
	// 	total_qsize, total_qsize);

	vec2 offset = vec2(0.0);
	vec2 uv = offset/float(IN.payload.resolution - 1);
	vec3 v = eval(patches.data[IN.payload.pindex], uv.x, uv.y);

	// if (gl_LocalInvocationIndex < 4)
	gl_MeshVerticesNV[gl_LocalInvocationIndex].gl_Position = project(v);

	if (gl_LocalInvocationIndex == 0) {
		debugPrintfEXT("generating a primitive %d\n", gl_LocalInvocationIndex);
		gl_PrimitiveCountNV = 2;

		gl_PrimitiveIndicesNV[0] = 0;
		gl_PrimitiveIndicesNV[1] = 1;
		gl_PrimitiveIndicesNV[2] = 2;
		gl_PrimitiveIndicesNV[3] = 1;
		gl_PrimitiveIndicesNV[4] = 2;
		gl_PrimitiveIndicesNV[5] = 3;
	}

	// gl_MeshVerticesEXT[gl_LocalInvocationIndex].gl_Position = project(v);
	// // vertices[gl_LocalInvocationIndex] = v;
	//
	// // Send position to fragment shader for normal vector calculations
	// position[gl_LocalInvocationIndex] = vec3(model * vec4(v, 1.0f));
	// pindex[gl_LocalInvocationIndex] = payload.pindex;
	//
	// // barrier();
	// if (gl_LocalInvocationID.x < qwidth && gl_LocalInvocationID.y < qheight) {
	// 	uint primitive_index = 2 * (gl_LocalInvocationID.x + qwidth * gl_LocalInvocationID.y);
	//
	// 	// TODO: diagonal cutting? switch to NV mesh shaders for read access
	// 	gl_PrimitiveTriangleIndicesEXT[primitive_index] = uvec3(gl_LocalInvocationIndex, gl_LocalInvocationIndex + 1, gl_LocalInvocationIndex + WORK_GROUP_SIZE);
	// 	gl_PrimitiveTriangleIndicesEXT[primitive_index + 1] = uvec3(gl_LocalInvocationIndex + WORK_GROUP_SIZE + 1, gl_LocalInvocationIndex + 1, gl_LocalInvocationIndex + WORK_GROUP_SIZE);
	// }
}
