#version 450

layout (push_constant) uniform ShadingPushConstants {
	layout(offset = 208)
	vec3 viewing;
	vec3 color;
	uint mode;
};

layout (location = 0) in vec3 position;
layout (location = 2) in flat uint pindex;

layout (location = 0) out vec4 fragment;

const uint COLORS = 16;
const vec3 WHEEL[16] = vec3[](
	vec3(0.880, 0.320, 0.320),
	vec3(0.880, 0.530, 0.320),
	vec3(0.880, 0.740, 0.320),
	vec3(0.810, 0.880, 0.320),
	vec3(0.600, 0.880, 0.320),
	vec3(0.390, 0.880, 0.320),
	vec3(0.320, 0.880, 0.460),
	vec3(0.320, 0.880, 0.670),
	vec3(0.320, 0.880, 0.880),
	vec3(0.320, 0.670, 0.880),
	vec3(0.320, 0.460, 0.880),
	vec3(0.390, 0.320, 0.880),
	vec3(0.600, 0.320, 0.880),
	vec3(0.810, 0.320, 0.880),
	vec3(0.880, 0.320, 0.740),
	vec3(0.880, 0.320, 0.530)
);

void main()
{
	vec3 light_direction = normalize(vec3(1, 1, 1));

	vec3 dU = dFdx(position);
	vec3 dV = dFdyFine(position);
	vec3 N = normalize(cross(dU, dV));

	if (mode == 2) {
		vec3 diffuse = color * vec3(max(0, dot(N, light_direction)));
		vec3 ambient = color * 0.1f;
		// fragment = vec4(diffuse + ambient, 0);
		// vec3 specular = vec3(pow(max(0, dot(-viewing, N)), 16));
		// fragment = vec4(specular, 0);

		vec3 H = normalize(-viewing + light_direction);
		vec3 specular = vec3(pow(max(0, dot(N, H)), 16));
		fragment = vec4(diffuse + specular + ambient, 0);
		fragment = pow(fragment, vec4(1/2.2));
	} else if (mode == 1) {
		// Normals
		fragment = vec4(0.5 + 0.5 * N, 1.0f);
	} else {
		// Patches
		fragment = vec4(WHEEL[pindex % COLORS], 1.0f);
	}
}
