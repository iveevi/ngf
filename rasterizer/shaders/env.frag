#version 450

#define PI 3.14159265

// layout (input_attachment_index = 0, binding = 0) uniform subpassInput depth;

layout (location = 0) in vec2 uv;

layout (push_constant) uniform PushConstants {
	vec3 origin;
	vec3 lower_left;
	vec3 horizontal;
	vec3 vertical;
};

layout (binding = 0) uniform sampler2D environment_texture;

layout (location = 0) out vec4 fragment;

void main()
{
	// if (subpassLoad(depth).x < 1)
	// 	discard;

	vec3 n = normalize(lower_left + uv.x * horizontal + (1 - uv.y) * vertical - origin);
	float phi = mod(PI + atan(n.z, n.x), 2 * PI);
	float theta = PI - acos(n.y);
	vec2 euv = vec2(phi, theta)/vec2(2 * PI, PI);
	vec3 env = texture(environment_texture, euv).xyz;
	fragment = vec4(env, 1);

	// TODO: some tone mapping here later...
}
