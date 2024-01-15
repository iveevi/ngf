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
};

layout (binding = 0) readonly buffer Points
{
	vec3 data[];
} points;

layout (binding = 2) readonly buffer Patches
{
	ivec4 data[];
} patches;

// Outputs
layout (location = 1) out vec3 normals[];

vec4 project(vec3 p)
{
	vec4 pp = proj * view * model * vec4(p, 1.0f);
	pp.y = -pp.y;
	pp.z = (pp.z + pp.w) / 2.0;
	return pp;
}

vec3 lerp(vec3 v0, vec3 v1, vec3 v2, vec3 v3, float u, float v)
{
	v0 = v0 * (1 - u) * (1 - v);
	v1 = v1 * (1 - u) * v;
	v2 = v2 * u * (1 - v);
	v3 = v3 * u * v;
	return v0 + v1 + v2 + v3;
}

void main()
{
	SetMeshOutputsEXT(4 * 8 * 8, 2 * 8 * 8);
	// if (gl_LocalInvocationIndex != 0)
	// 	return;

	vec2 uv = gl_LocalInvocationID.xy/8.0f;

	ivec4 c = patches.data[payload.pindex];

	// TODO: lerp from an image
	vec3 v0 = points.data[c.x];
	vec3 v1 = points.data[c.y];
	vec3 v2 = points.data[c.w];
	vec3 v3 = points.data[c.z];

	vec3 corners[4];

	float stride = 1/8.0;
	corners[0] = lerp(v0, v1, v2, v3, uv.x, uv.y);
	corners[1] = lerp(v0, v1, v2, v3, uv.x + stride, uv.y);
	corners[2] = lerp(v0, v1, v2, v3, uv.x, uv.y + stride);
	corners[3] = lerp(v0, v1, v2, v3, uv.x + stride, uv.y + stride);

	vec3 n = normalize(cross(v2 - v0, v1 - v0));

	uint vertexBaseIndex = 4 * gl_LocalInvocationIndex;
	gl_MeshVerticesEXT[vertexBaseIndex + 0].gl_Position = project(corners[0]);
	gl_MeshVerticesEXT[vertexBaseIndex + 1].gl_Position = project(corners[1]);
	gl_MeshVerticesEXT[vertexBaseIndex + 2].gl_Position = project(corners[2]);
	gl_MeshVerticesEXT[vertexBaseIndex + 3].gl_Position = project(corners[3]);

	uint primitiveBaseIndex = 2 * gl_LocalInvocationIndex;
	gl_PrimitiveTriangleIndicesEXT[primitiveBaseIndex + 0] = uvec3(vertexBaseIndex + 0, vertexBaseIndex + 1, vertexBaseIndex + 2);
	gl_PrimitiveTriangleIndicesEXT[primitiveBaseIndex + 1] = uvec3(vertexBaseIndex + 2, vertexBaseIndex + 3, vertexBaseIndex + 1);

	normals[vertexBaseIndex + 0] = n;
	normals[vertexBaseIndex + 1] = n;
	normals[vertexBaseIndex + 2] = n;
	normals[vertexBaseIndex + 3] = n;
}
