#version 450

layout (push_constant) uniform ShadingPushConstants {
	layout(offset = 208)
	vec3 viewing;
	vec3 color;
};

layout (location = 0) in vec3 position;
layout (location = 2) in flat uint pindex;

layout (location = 0) out vec4 fragment;

const uint COLORS = 16;

// Color wheel for patches
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
	fragment = vec4(WHEEL[pindex % COLORS], 1.0f);
}
