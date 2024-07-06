#version 450

layout (push_constant) uniform ShadingPushConstants {
	layout(offset = 208)
	vec3 viewing;
	vec3 color;
};

layout (location = 0) in vec3 position;

layout (location = 0) out vec4 fragment;

void main()
{
	vec3 dU = dFdx(position);
	vec3 dV = dFdyFine(position);
	vec3 N = normalize(cross(dU, dV));

	fragment = vec4(0.5 + 0.5 * N, 1.0f);
}
