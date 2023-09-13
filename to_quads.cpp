#include <filesystem>
#include <iostream>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "microlog.h"

using quad = glm::uvec4;

struct quad_mesh {
	std::vector <glm::vec3> vertices;
	std::vector <glm::vec3> normals;
	std::vector <quad> quads;
};

struct loader {
	std::vector <quad_mesh> meshes;

	loader(const std::filesystem::path &path) {
		Assimp::Importer importer;

		// Read scene
		const aiScene *scene;
		scene = importer.ReadFile(path, aiProcess_GenNormals);

		// Check if the scene was loaded
		if ((!scene | scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) || !scene->mRootNode) {
			ulog_error("loader", "Assimp error: \"%s\"\n", importer.GetErrorString());
			return;
		}

		process_node(scene->mRootNode, scene, path.parent_path());
	}

	void process_node(aiNode *node, const aiScene *scene, const std::string &directory) {
		// Process all the node's meshes (if any)
		for (uint32_t i = 0; i < node->mNumMeshes; i++) {
			aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
			process_mesh(mesh, scene, directory);
		}

		// Recusively process all the node's children
		for (uint32_t i = 0; i < node->mNumChildren; i++)
			process_node(node->mChildren[i], scene, directory);

	}

	void process_mesh(aiMesh *, const aiScene *, const std::string &);

	const quad_mesh &get(uint32_t i) const {
		return meshes[i];
	}
};

void loader::process_mesh(aiMesh *mesh, const aiScene *scene, const std::string &dir)
{
	// quad_mesh data
	std::vector <glm::vec3> vertices;
	std::vector <glm::vec3> normals;
        std::vector <quad> quads;

	// Process all the mesh's vertices
	for (uint32_t i = 0; i < mesh->mNumVertices; i++) {
		vertices.push_back({
			mesh->mVertices[i].x,
			mesh->mVertices[i].y,
			mesh->mVertices[i].z
		});

		glm::vec3 n {
			mesh->mNormals[i].x,
			mesh->mNormals[i].y,
			mesh->mNormals[i].z
		};

		normals.push_back(glm::normalize(n));
	}

	// Process all the mesh's triangles
	for (uint32_t i = 0; i < mesh->mNumFaces; i++) {
		aiFace face = mesh->mFaces[i];
		ulog_assert(face.mNumIndices == 4, "process_mesh", "Only quads are supported, got %d-sided polygon instead\n", face.mNumIndices);
		quads.push_back({
			face.mIndices[0],
			face.mIndices[1],
			face.mIndices[2],
			face.mIndices[3]
		});
	}

	meshes.push_back({ vertices, normals, quads });
	ulog_info("loader", "Loaded quad mesh with %d vertices and %d quads\n", vertices.size(), quads.size());
}

int main(int argc, char *argv[])
{
	if (argc < 2) {
		printf("Usage: %s <path>\n", argv[0]);
		return 1;
	}

	std::filesystem::path path = argv[1];

	loader l(path);
	quad_mesh m = l.get(0);

	std::filesystem::path out_path = path.stem().concat(".quads");

	std::vector <quad> quads = m.quads;
	std::vector <glm::vec3> vertices;
	std::unordered_map <glm::vec3, uint32_t> vertex_map;
	for (const quad &q : quads) {
		for (uint32_t i = 0; i < 4; i++) {
			glm::vec3 v = m.vertices[q[i]];
			if (vertex_map.find(v) == vertex_map.end()) {
				vertex_map[v] = vertices.size();
				vertices.push_back(v);
			}
		}
	}

	for (quad &q : quads) {
		for (uint32_t i = 0; i < 4; i++) {
			glm::vec3 v = m.vertices[q[i]];
			q[i] = vertex_map[v];
		}
	}

	FILE *out = fopen(out_path.c_str(), "wb");
	if (!out) {
		ulog_error("main", "Failed to open output file \"%s\"\n", out_path.c_str());
		return 1;
	}

	uint32_t num_quads = quads.size();
	uint32_t num_vertices = vertices.size();

	fwrite(&num_quads, sizeof(uint32_t), 1, out);
	fwrite(&num_vertices, sizeof(uint32_t), 1, out);

	fwrite(quads.data(), sizeof(quad), num_quads, out);
	fwrite(vertices.data(), sizeof(glm::vec3), num_vertices, out);

	ulog_info("main", "Wrote %d quads and %d vertices to \"%s\"\n", num_quads, num_vertices, out_path.c_str());
	return 0;
}
