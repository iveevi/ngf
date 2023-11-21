#include "common.hpp"

std::vector <float> interleave_attributes(const geometry &geometry)
{
	std::vector <float> attributes;

	for (uint32_t i = 0; i < geometry.vertices.size(); i++) {
		attributes.push_back(geometry.vertices[i].x);
		attributes.push_back(geometry.vertices[i].y);
		attributes.push_back(geometry.vertices[i].z);

		attributes.push_back(geometry.normals[i].x);
		attributes.push_back(geometry.normals[i].y);
		attributes.push_back(geometry.normals[i].z);
	}

	return attributes;
}

loader::loader(const std::filesystem::path &path)
{
	Assimp::Importer importer;
	ulog_assert(std::filesystem::exists(path), "loader", "File \"%s\" does not exist\n", path.c_str());

	// Read scene
	const aiScene *scene;
	scene = importer.ReadFile(path, aiProcess_GenNormals | aiProcess_Triangulate);

	// Check if the scene was loaded
	if ((!scene | scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) || !scene->mRootNode) {
		ulog_error("loader", "Assimp error: \"%s\"\n", importer.GetErrorString());
		return;
	}

	process_node(scene->mRootNode, scene, path.parent_path());
}

void loader::process_node(aiNode *node, const aiScene *scene, const std::string &directory)
{
	// Process all the node's meshes (if any)
	for (uint32_t i = 0; i < node->mNumMeshes; i++) {
		aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
		process_mesh(mesh, scene, directory);
	}

	// Recusively process all the node's children
	for (uint32_t i = 0; i < node->mNumChildren; i++)
		process_node(node->mChildren[i], scene, directory);

}

void loader::process_mesh(aiMesh *mesh, const aiScene *scene, const std::string &dir)
{

	std::vector <glm::vec3> vertices;
	std::vector <glm::vec3> normals;
        std::vector <glm::uvec3> indices;

	// Process all the mesh's vertices
	for (uint32_t i = 0; i < mesh->mNumVertices; i++) {
		vertices.push_back({
			mesh->mVertices[i].x,
			mesh->mVertices[i].y,
			mesh->mVertices[i].z
		});

		if (mesh->HasNormals()) {
			normals.push_back({
				mesh->mNormals[i].x,
				mesh->mNormals[i].y,
				mesh->mNormals[i].z
			});
		} else {
			normals.push_back({ 0.0f, 0.0f, 0.0f });
		}
	}

	// Process all the mesh's triangles
	for (uint32_t i = 0; i < mesh->mNumFaces; i++) {
		aiFace face = mesh->mFaces[i];
		ulog_assert(face.mNumIndices == 3, "process_mesh", "Only triangles are supported, got %d-sided polygon instead\n", face.mNumIndices);
		indices.push_back({
			face.mIndices[0],
			face.mIndices[1],
			face.mIndices[2]
		});
	}

	meshes.push_back({ vertices, normals, indices });
}

const geometry &loader::get(uint32_t i) const
{
	return meshes[i];
}
