#version 450

layout (location = 0) in vec3 normal;

layout (location = 0) out vec4 fragment;

void main()
{
    fragment = vec4(0.5 + 0.5 * normal, 1.0);
}
