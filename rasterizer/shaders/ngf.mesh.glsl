#version 450

#extension GL_GOOGLE_include_directive: require
#extension GL_EXT_mesh_shader: require
#extension GL_EXT_debug_printf : require

#include "payload.h"

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// NOTE: each mesh shader does one half of a patch
layout (triangles, max_vertices = 256, max_primitives = 15 * 15) out;

// Inputs
taskPayloadSharedEXT Payload payload;

layout (push_constant) uniform PushConstants {
	mat4 model;
	mat4 view;
	mat4 proj;
	uint feature_size;
};

layout (binding = 0) readonly buffer Points
{
	vec3 data[];
} points;

layout (binding = 1) readonly buffer Features
{
	float data[];
} features;

layout (binding = 2) readonly buffer Patches
{
	ivec4 data[];
} patches;

// Local execution data
const uint ENCODING_LEVELS = 10;
const uint FEATURE_SIZE    = 5; // TODO: pass as Pushconstnts...
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
layout (location = 1) out vec3 normals[];

vec4 project(vec3 p)
{
	vec4 pp = proj * view * model * vec4(p, 1.0f);
	pp.y = -pp.y;
	pp.z = (pp.z + pp.w) / 2.0;
	return pp;
}

ivec4 complex;

// Interpolating
vec3  vertex;
float feature[50];

void lerp(float u, float v)
{
	// Lerp the vertices
	vec3 v0 = points.data[complex.x];
	vec3 v1 = points.data[complex.y];
	vec3 v2 = points.data[complex.z];
	vec3 v3 = points.data[complex.w];

	// TODO: skip this and fma?
	v0 = v0 * (1 - u) * (1 - v);
	v1 = v1 * (1 - u) * v;
	v2 = v2 * u * v;
	v3 = v3 * u * (1 - v);

	vertex = v0 + v1 + v2 + v3;

	// TODO: vectorize this calculation by packing (and padding) features into vec4s
	for (uint i = 0; i < feature_size; i++) {
		float f0 = features.data[complex.x * feature_size + i];
		float f1 = features.data[complex.y * feature_size + i];
		float f2 = features.data[complex.z * feature_size + i];
		float f3 = features.data[complex.w * feature_size + i];

		f0 *= (1 - u) * (1 - v);
		f1 *= (1 - u) * v;
		f2 *= u * v;
		f3 *= u * (1 - v);

		feature[i] = f0 + f1 + f2 + f3;
	}
}

// Encoding
// TODO: again, pad with vec4s...
float ffin[128];

void encode()
{
	uint k = 0;
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
}

// TODO: cooperative matices per work group
// TODO: matrix x matrix operations shared by all processes
// TODO: alternatively run network for all vertices in task shader and then payload vertices into here

// Network evaluation
float hidden0[64];
float hidden1[64];
float hidden2[64];

float leaky_relu(float x)
{
	return max(x, 0.01f * x);
}

void mlp()
{
	uint isize;
	uint osize;

	// uint FFIN = feature_size + 3 * (2 * ENCODING_LEVELS + 1);
	debugPrintfEXT("Feature size %d, %d\n", feature_size, FEATURE_SIZE);

	// Layer 0
	isize = FFIN;
	osize = 64;

	for (uint i = 0; i < osize; i++) {
		float sum = 0.0f;
		for (uint j = 0; j < isize; j++)
			sum += layers.weight0[i * isize + j] * ffin[j];
		hidden0[i] = sum;
	}

	for (uint i = 0; i < osize; i++) {
		hidden0[i] += layers.bias0[i];
		hidden0[i] = leaky_relu(hidden0[i]);
	}

	// Layer 1
	isize = 64;
	osize = 64;

	for (uint i = 0; i < osize; i++) {
		float sum = 0.0f;
		for (uint j = 0; j < isize; j++)
			sum += layers.weight1[i * isize + j] * hidden0[j];
		hidden1[i] = sum;
	}

	for (uint i = 0; i < osize; i++) {
		hidden1[i] += layers.bias1[i];
		hidden1[i] = leaky_relu(hidden1[i]);
	}

	// Layer 2
	isize = 64;
	osize = 64;

	for (uint i = 0; i < osize; i++) {
		float sum = 0.0f;
		for (uint j = 0; j < isize; j++)
			sum += layers.weight2[i * isize + j] * hidden1[j];
		hidden2[i] = sum;
	}

	for (uint i = 0; i < osize; i++) {
		hidden2[i] += layers.bias2[i];
		hidden2[i] = leaky_relu(hidden2[i]);
	}

	// Layer 3
	float D[3];

	isize = 64;
	osize = 3;

	for (uint i = 0; i < osize; i++) {
		float sum = 0.0f;
		for (uint j = 0; j < isize; j++)
			sum += layers.weight3[i * isize + j] * hidden2[j];
		D[i] = sum + layers.bias3[i];
	}

	vertex.x += D[0];
	vertex.y += D[1];
	vertex.z += D[2];
}

void main()
{
	SetMeshOutputsEXT(4 * 8 * 8, 2 * 8 * 8);

	vec2 uv = gl_LocalInvocationID.xy/8.0f;

	complex = patches.data[payload.pindex];

	// TODO: should depend on the tessellation resolution
	const float stride = 1/8.0;

	uint vertex_base = 4 * gl_LocalInvocationIndex;
	uint primitive_index = 2 * gl_LocalInvocationIndex;

	lerp(uv.x, uv.y);
	encode();
	mlp();
	vec3 v0 = vertex;

	lerp(uv.x + stride, uv.y);
	encode();
	mlp();
	vec3 v1 = vertex;

	lerp(uv.x, uv.y + stride);
	encode();
	mlp();
	vec3 v2 = vertex;

	lerp(uv.x + stride, uv.y + stride);
	encode();
	mlp();
	vec3 v3 = vertex;

	gl_MeshVerticesEXT[vertex_base + 0].gl_Position = project(v0);
	gl_MeshVerticesEXT[vertex_base + 1].gl_Position = project(v1);
	gl_MeshVerticesEXT[vertex_base + 2].gl_Position = project(v2);
	gl_MeshVerticesEXT[vertex_base + 3].gl_Position = project(v3);

	vec3 n = normalize(cross(v2 - v0, v1 - v0));
	vec3 x = vec3(layers.bias3[0], layers.bias3[1], layers.bias3[2]);

	gl_PrimitiveTriangleIndicesEXT[primitive_index + 0] = uvec3(vertex_base + 0, vertex_base + 1, vertex_base + 2);
	gl_PrimitiveTriangleIndicesEXT[primitive_index + 1] = uvec3(vertex_base + 2, vertex_base + 3, vertex_base + 1);

	normals[vertex_base + 0] = n;
	normals[vertex_base + 1] = n;
	normals[vertex_base + 2] = n;
	normals[vertex_base + 3] = n;
}
