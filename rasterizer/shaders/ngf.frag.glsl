#version 450

layout (location = 0) in vec3 position;
layout (location = 0) out vec4 fragment;

void main()
{
	vec3 dU = dFdx(position);
	vec3 dV = dFdy(position);
	vec3 N = normalize(cross(dU, dV));

	// TODO: basic shading
	fragment = vec4(0.5 + 0.5 * N, 0);
}
