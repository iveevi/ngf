#pragma once

#include <filesystem>
#include <type_traits>
#include <vector>

#include <glm/glm.hpp>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/Exporter.hpp>

#include "microlog.h"

enum {
	eTriangle,
	eQuad
};

template <size_t primitive>
struct geometry {
	static_assert(primitive == eTriangle || primitive == eQuad, "Invalid primitive type");

	std::vector <glm::vec3> vertices;

	typename std::conditional <primitive == eTriangle, std::vector <glm::uvec3>, void *>::type triangles;
	typename std::conditional <primitive == eQuad, std::vector <glm::uvec4>, void *>::type quads;

	geometry() = default;
};

template <size_t primitive>
std::vector <geometry <primitive>> load_geometry(const std::filesystem::path &path)
{
	std::vector <geometry <primitive>> result;

	// Check that the file exists
	ulog_assert(std::filesystem::exists(path), "load_geometry", "file %s does not exist\n", path.c_str());

	auto process_mesh = [&](aiMesh *mesh, const aiScene *scene, const std::string &dir) {
		std::vector <glm::vec3> vertices;
		std::vector <glm::vec3> normals;

		std::vector <glm::uvec3> triangles;
		std::vector <glm::uvec4> quads;

		// Process all the mesh's vertices
		for (uint32_t i = 0; i < mesh->mNumVertices; i++) {
			vertices.push_back({
				mesh->mVertices[i].x,
				mesh->mVertices[i].y,
				mesh->mVertices[i].z
			});
		}

		if constexpr (primitive == eTriangle) {
			// Process all the mesh's triangles
			for (uint32_t i = 0; i < mesh->mNumFaces; i++) {
				aiFace face = mesh->mFaces[i];
				ulog_assert(face.mNumIndices == 3, "process_mesh", "Only triangles are supported, got %d-sided polygon instead\n", face.mNumIndices);
				triangles.push_back({
					face.mIndices[0],
					face.mIndices[1],
					face.mIndices[2]
				});
			}
		} else {
			// Process all the mesh's quads
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
		}

		if constexpr (primitive == eTriangle)
			result.push_back({ vertices, triangles, {} });
		else
			result.push_back({ vertices, {}, quads });
	};

	std::function <void (aiNode *, const aiScene *, const std::string &)> process_node;
	process_node = [&](aiNode *node, const aiScene *scene, const std::string &directory) {
		// Process all the node's meshes (if any)
		for (uint32_t i = 0; i < node->mNumMeshes; i++) {
			aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
			process_mesh(mesh, scene, directory);
		}

		// Recusively process all the node's children
		for (uint32_t i = 0; i < node->mNumChildren; i++)
			process_node(node->mChildren[i], scene, directory);

	};

	// Load all objects
	Assimp::Importer importer;

	// Read scene
	const aiScene *scene;
	if constexpr (primitive == eTriangle)
		scene = importer.ReadFile(path, aiProcess_Triangulate);
	else
		scene = importer.ReadFile(path, 0);

	// Check if the scene was loaded
	if ((!scene | scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) || !scene->mRootNode) {
		ulog_error("loader", "assimp error: \"%s\"\n", importer.GetErrorString());
		return result;
	}

	process_node(scene->mRootNode, scene, path.parent_path());
	return result;
}

template <size_t primitive>
void write_geometry(const geometry <primitive> &mesh, const std::filesystem::path &path)
{
	aiScene scene;

	scene.mRootNode = new aiNode();

	scene.mMaterials = new aiMaterial*[ 1 ];
	scene.mMaterials[ 0 ] = nullptr;
	scene.mNumMaterials = 1;

	scene.mMaterials[ 0 ] = new aiMaterial();

	scene.mMeshes = new aiMesh*[ 1 ];
	scene.mMeshes[ 0 ] = nullptr;
	scene.mNumMeshes = 1;

	scene.mMeshes[ 0 ] = new aiMesh();
	scene.mMeshes[ 0 ]->mMaterialIndex = 0;

	scene.mRootNode->mMeshes = new unsigned int[ 1 ];
	scene.mRootNode->mMeshes[ 0 ] = 0;
	scene.mRootNode->mNumMeshes = 1;

	auto pMesh = scene.mMeshes[ 0 ];

	pMesh->mVertices = new aiVector3D[mesh.vertices.size()];
	pMesh->mNormals = new aiVector3D[mesh.vertices.size()];
	pMesh->mNumVertices = mesh.vertices.size();
	pMesh->mNumUVComponents[0] = 0;

	for (uint32_t i = 0; i < mesh.vertices.size(); i++)
		pMesh->mVertices[i] = { mesh.vertices[i].x, mesh.vertices[i].y, mesh.vertices[i].z };

	if constexpr (primitive == eTriangle) {
		pMesh->mFaces = new aiFace[mesh.triangles.size()];
		pMesh->mNumFaces = mesh.triangles.size();

		for (uint32_t i = 0; i < mesh.triangles.size(); i++) {
			aiFace &face = pMesh->mFaces[i];
			face.mIndices = new unsigned int[3];
			face.mNumIndices = 3;

			glm::uvec3 t = mesh.triangles[i];
			face.mIndices[0] = t[0];
			face.mIndices[1] = t[1];
			face.mIndices[2] = t[2];
		}
	} else {
		pMesh->mFaces = new aiFace[mesh.quads.size()];
		pMesh->mNumFaces = mesh.quads.size();

		for (uint32_t i = 0; i < mesh.quads.size(); i++) {
			aiFace &face = pMesh->mFaces[i];
			face.mIndices = new unsigned int[4];
			face.mNumIndices = 4;

			glm::uvec4 t = mesh.quads[i];
			face.mIndices[0] = t[0];
			face.mIndices[1] = t[1];
			face.mIndices[2] = t[2];
			face.mIndices[3] = t[3];
		}
	}

	Assimp::Exporter exporter;
	std::string ext = path.extension().string().substr(1);
	exporter.Export(&scene, ext, path.string());

	ulog_info("write_geometry", "exported result to %s (%s)\n", path.c_str(), ext.c_str());
}

// TODO: an application to apply displacement maps...
