#include <assimp/scene.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>

#include "mesh.hpp"
#include "microlog.h"

std::vector <glm::vec3> smooth_normals(const Mesh &m)
{
	std::vector <glm::vec3> normals;
	normals.resize(m.vertices.size(), glm::vec3 { 0.0f });

	for (const glm::uvec3 &t : m.triangles) {
		const glm::vec3 &v0 = m.vertices[t.x];
		const glm::vec3 &v1 = m.vertices[t.y];
		const glm::vec3 &v2 = m.vertices[t.z];

		glm::vec3 cross = glm::cross(v0 - v1, v0 - v2);

		normals[t.x] += cross;
		normals[t.y] += cross;
		normals[t.z] += cross;
	}

	for (glm::vec3 &n : normals)
		n = glm::normalize(n);

	return normals;
}

std::vector <float> interleave_attributes(const Mesh &m)
{
	std::vector <float> attributes;
	for (size_t i = 0; i < m.vertices.size(); i++) {
		attributes.push_back(m.vertices[i].x);
		attributes.push_back(m.vertices[i].y);
		attributes.push_back(m.vertices[i].z);

		attributes.push_back(m.normals[i].x);
		attributes.push_back(m.normals[i].y);
		attributes.push_back(m.normals[i].z);
	}

	return attributes;
}

static Mesh assimp_process_mesh(aiMesh *m, const aiScene *scene, const std::string &dir)
{
	std::vector <glm::vec3> vertices;
	std::vector <glm::vec3> normals;
        std::vector <glm::uvec3> triangles;

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
	}

	// Process all the mesh's triangles
	for (uint32_t i = 0; i < m->mNumFaces; i++) {
		aiFace face = m->mFaces[i];
                ulog_assert(face.mNumIndices == 3, "process_mesh",
                            "Only triangles are supported, got %d-sided "
                            "polygon instead\n",
                            face.mNumIndices);

                triangles.push_back({
			face.mIndices[0],
			face.mIndices[1],
			face.mIndices[2]
		});
	}

	return Mesh { vertices, normals, triangles };
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

std::vector <Mesh> Mesh::load(const std::filesystem::path &path)
{
	Assimp::Importer importer;
        ulog_assert
	(
	 	std::filesystem::exists(path),
		"loader",
                "File \"%s\" does not exist\n", path.c_str()
	);

        // Read scene
	const aiScene *scene;
	scene = importer.ReadFile(path, aiProcess_GenNormals | aiProcess_Triangulate);

	// Check if the scene was loaded
	if ((!scene | scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) || !scene->mRootNode) {
		ulog_error("loader", "Assimp error: \"%s\"\n", importer.GetErrorString());
		return {};
	}

	return assimp_process_node(scene->mRootNode, scene, path.parent_path());
}

// #include <iostream>
// #include <glm/glm.hpp>
// #include <glm/gtx/string_cast.hpp>

Mesh Mesh::normalize(const Mesh &m)
{
	glm::vec3 min(FLT_MAX);
	glm::vec3 max(-FLT_MAX);

	for (const glm::vec3 &v : m.vertices) {
		min = glm::min(v, min);
		max = glm::max(v, max);
	}

	glm::vec3 d = max - min;
	glm::vec3 center = (max + min)/2.0f;

	Mesh nm = m;

	float scale = glm::length(d)/2.0f;
	for (glm::vec3 &v : nm.vertices)
		v = (v - center)/scale;

	// std::cout << "min, max, scale and center:"
	// 	<< glm::to_string(min) << "  "
	// 	<< glm::to_string(max) << "  "
	// 	<< scale << "  "
	// 	<< glm::to_string(center) << "\n";

	return nm;
}
