#version 450

layout (location = 0) out vec2 uv;

void main()
{
	int index = gl_VertexIndex;

	vec2 uvs[6] = vec2[6](
		vec2(0, 0),
		vec2(0, 1),
		vec2(1, 1),

		vec2(0, 0),
		vec2(1, 1),
		vec2(1, 0)
	);

	uv = uvs[index % 6];

	vec2 position = 2 * uv - 1;

	gl_Position = vec4(position, 0, 1);
}
