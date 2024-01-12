#version 450

#extension GL_EXT_mesh_shader: require
// #extension GL_KHR_shader_subgroup_ballot: require

void main()
{
	// uvec4 validVotes = subgroupBallot(true);
	// uint validCount = subgroupBallotBitCount(validVotes);
	EmitMeshTasksEXT(4, 4, 1);
}
