#version 450

#extension GL_EXT_mesh_shader: require

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

// NOTE: each mesh shader does one half of a patch
layout(triangles, max_vertices = 256, max_primitives = 15 * 15) out;

void main()
{
	vec2 uv = gl_.xy;
	vec2 v0 = uv;
	vec2 v1 = uv + vec2(0.5, 0);
	vec2 v2 = uv + vec2(0.5, 0.5);
	vec2 v3 = uv + vec2(0, 0.5);

	SetMeshOutputsEXT(4, 2);

	uint vertexBaseIndex = 0; // 4 * gl_LocalInvocationIndex;
	gl_MeshVerticesEXT[vertexBaseIndex + 0].gl_Position = vec4(v0, 0.0f, 1.0f);
	gl_MeshVerticesEXT[vertexBaseIndex + 1].gl_Position = vec4(v1, 0.0f, 1.0f);
	gl_MeshVerticesEXT[vertexBaseIndex + 2].gl_Position = vec4(v2, 0.0f, 1.0f);
	gl_MeshVerticesEXT[vertexBaseIndex + 3].gl_Position = vec4(v3, 0.0f, 1.0f);

	uint primitiveBaseIndex = 0; // 2 * gl_LocalInvocationIndex;
	gl_PrimitiveTriangleIndicesEXT[primitiveBaseIndex + 0] = uvec3(vertexBaseIndex + 0, vertexBaseIndex + 1, vertexBaseIndex + 2);
	gl_PrimitiveTriangleIndicesEXT[primitiveBaseIndex + 1] = uvec3( vertexBaseIndex + 2, vertexBaseIndex + 3, vertexBaseIndex + 0 );
}
