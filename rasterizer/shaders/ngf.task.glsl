#version 450

#extension GL_GOOGLE_include_directive: require
#extension GL_EXT_mesh_shader: require
#extension GL_EXT_debug_printf : require

#include "payload.h"

taskPayloadSharedEXT Payload payload;

void main()
{
	// TODO: dyanmic LOD
	// debugPrintfEXT("TASK ID %d\n", gl_GlobalInvocationID.x);
	payload.pindex = gl_GlobalInvocationID.x;
	EmitMeshTasksEXT(1, 1, 1);
}
