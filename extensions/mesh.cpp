#include <vector>
#include <filesystem>

#include <glm/glm.hpp>

#include <assimp/scene.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>

#include "common.hpp"
#include "util.hpp"

struct Mesh {
	std::vector <glm::vec3> vertices;
	std::vector <glm::vec3> normals;
	std::vector <glm::vec2> uvs;
        std::vector <glm::ivec3> triangles;
};

static Mesh assimp_process_mesh(aiMesh *m, const aiScene *scene, const std::string &dir)
{
	std::vector <glm::vec3> vertices;
	std::vector <glm::vec3> normals;
	std::vector <glm::vec2> uvs;
        std::vector <glm::ivec3> triangles;

	// Process all the mesh's vertices
	for (uint32_t i = 0; i < m->mNumVertices; i++) {
		vertices.push_back({
			m->mVertices[i].x,
			m->mVertices[i].y,
			m->mVertices[i].z
		});

		if (m->HasNormals()) {
			normals.push_back({
				m->mNormals[i].x,
				m->mNormals[i].y,
				m->mNormals[i].z
			});
		} else {
			normals.push_back({ 0.0f, 0.0f, 0.0f });
		}

		if (m->HasTextureCoords(i)) {
			uvs.push_back({
				m->mTextureCoords[i]->x,
				m->mTextureCoords[i]->y,
			});
		} else {
			uvs.push_back({ 0.0f , 0.0f });
		}
	}

	// Process all the mesh's triangles
	for (uint32_t i = 0; i < m->mNumFaces; i++) {
		aiFace face = m->mFaces[i];
                triangles.push_back({
			face.mIndices[0],
			face.mIndices[1],
			face.mIndices[2]
		});
	}

	return Mesh { vertices, normals, uvs, triangles };
}

static std::vector <Mesh> assimp_process_node(aiNode *node, const aiScene *scene, const std::string &directory)
{
	std::vector <Mesh> meshes;

	// Process all the node's meshes (if any)
	for (uint32_t i = 0; i < node->mNumMeshes; i++) {
		aiMesh *m = scene->mMeshes[node->mMeshes[i]];
		Mesh pm = assimp_process_mesh(m, scene, directory);
		meshes.push_back(pm);
	}

	// Recusively process all the node's children
	for (uint32_t i = 0; i < node->mNumChildren; i++) {
		auto pn = assimp_process_node(node->mChildren[i], scene, directory);
		meshes.insert(meshes.begin(), pn.begin(), pn.end());
	}

	return meshes;
}

std::tuple <torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
load_mesh(const std::string &s)
{
	Assimp::Importer importer;
	std::filesystem::path path = s;

        // Read scene
	const aiScene *scene;
	scene = importer.ReadFile(path, aiProcess_GenNormals | aiProcess_Triangulate);

	// Check if the scene was loaded
	if ((!scene | (scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE)) || !scene->mRootNode) {
		printf("Assimp error: \"%s\"\n", importer.GetErrorString());
		return {};
	}

	std::vector <Mesh> meshes = assimp_process_node(scene->mRootNode, scene, path.parent_path());
	if (meshes.size() == 0) {
		printf("NO MESHES FOUND!\n");
		return {};
	}

	Mesh m = meshes.front();
	return std::make_tuple
	(
		vector_to_tensor <glm::vec3, torch::kFloat32, 3> (m.vertices),
		vector_to_tensor <glm::vec3, torch::kFloat32, 3> (m.normals),
		vector_to_tensor <glm::vec2, torch::kFloat32, 2> (m.uvs),
		vector_to_tensor <glm::ivec3, torch::kInt32, 3> (m.triangles)
	);

	// return assimp_process_node(scene->mRootNode, scene, path.parent_path());
}
