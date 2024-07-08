#version 450

layout (location = 0) out vec4 fragment;

void main()
{
	float near = 0.1f;
	float far = 1000.0f;

	float d = gl_FragCoord.z;
	float linearized = (near * far)/(far + d * (near - far));

	linearized = log(linearized + 1);

	fragment = vec4(vec3(linearized), 1);
}
