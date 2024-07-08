#version 450

layout (push_constant) uniform ShadingPushConstants {
	layout(offset = 224)
	vec3 viewing;
	vec3 color;
};

layout (location = 0) in vec3 position;
layout (location = 2) in flat uint pindex;

layout (location = 0) out vec4 fragment;

// Spherical harmonics lighting
mat4 R = mat4(
	-0.034424, -0.021329, 0.086425, -0.146638,
	-0.021329, 0.034424, 0.004396, -0.050819,
	0.086425, 0.004396, 0.088365, -0.377601,
	-0.146638, -0.050819, -0.377601, 1.618500
);

mat4 G = mat4(
	-0.032890, -0.013668, 0.066403, -0.107776,
	-0.013668, 0.032890, -0.012273, 0.013852,
	0.066403, -0.012273, -0.021086, -0.223067,
	-0.107776, 0.013852, -0.223067, 1.598757
);

mat4 B = mat4(
	-0.035777, -0.008999, 0.051376, -0.087520,
	-0.008999, 0.035777, -0.034691, 0.030949,
	0.051376, -0.034691, -0.010211, -0.081895,
	-0.087520, 0.030949, -0.081895, 1.402876
);

// Tone mapping
vec3 aces(vec3 x)
{
	const float a = 2.51;
	const float b = 0.03;
	const float c = 2.43;
	const float d = 0.59;
	const float e = 0.14;
	return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main()
{
	vec3 light_direction = normalize(vec3(1, -1, -1));

	vec3 dU = dFdx(position);
	vec3 dV = dFdyFine(position);
	vec3 N = normalize(cross(dU, dV));

	vec3 diffuse = 0.1 * color * vec3(max(0, dot(N, light_direction)));

	vec3 H = normalize(-viewing + light_direction);
	vec3 specular = 0.4 * vec3(pow(max(0, dot(N, H)), 128));

	vec4 n = vec4(N, 1);
	float r = dot(n, R * n);
	float g = dot(n, G * n);
	float b = dot(n, B * n);

	vec3 ambient = 0.3 * color * vec3(r, g, b);
	vec3 color = diffuse + specular + ambient;
	fragment = vec4(aces(color), 1);
}
