#version 460

#extension GL_GOOGLE_include_directive: require
#extension GL_EXT_mesh_shader: require
#extension GL_EXT_debug_printf : require

#include "payload.h"

layout (push_constant) uniform PushConstants {
	mat4 model;
	mat4 view;
	mat4 proj;
	float time;
};

taskPayloadSharedEXT Payload payload;

// TODO: counter for number of primitives sent

void main()
{
	// TODO: compute distance and set resolution
	payload.pindex = gl_GlobalInvocationID.x;

	// max resolution is 15, each work group is necessarily 8 x 8 and might overflow...
	payload.resolution = 2 + int(7 * sin(time) + 7);

	uint groups = (payload.resolution - 1 + 6)/7;
	EmitMeshTasksEXT(groups, groups, 1);
}
