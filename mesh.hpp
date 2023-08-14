#pragma once

// Standard headers
#include <filesystem>
#include <vector>
#include <stack>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>

// GLM headers
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>
#include <glm/gtx/string_cast.hpp>

// Assimp headers
#include <assimp/Importer.hpp>
#include <assimp/Exporter.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

// Object loading
using Triangle = std::array <uint32_t, 3>;

struct Mesh {
	std::vector <glm::vec3> vertices;
	std::vector <glm::vec3> normals;
	std::vector <Triangle> triangles;
};

// Vertex deduplication
inline Mesh deduplicate(const Mesh &ref)
{
	std::unordered_map <glm::vec3, uint32_t> existing;

	Mesh fixed;
	auto add_uniquely = [&](int32_t i) -> uint32_t {
		glm::vec3 v = ref.vertices[i];
		if (existing.find(v) == existing.end()) {
			int32_t csize = fixed.vertices.size();
			fixed.vertices.push_back(v);
			fixed.normals.push_back(ref.normals[i]);

			existing[v] = csize;
			return csize;
		}

		return existing[v];
	};

	for (const Triangle &t : ref.triangles) {
		fixed.triangles.push_back(Triangle {
			add_uniquely(t[0]),
			add_uniquely(t[1]),
			add_uniquely(t[2])
		});
	}

	return fixed;
}

// Recompute normals
inline void recompute_normals(Mesh &ref)
{
	ref.normals.clear();
	ref.normals.resize(ref.vertices.size(), glm::vec3 { 0.0f });

	for (const Triangle &t : ref.triangles) {
		glm::vec3 a = ref.vertices[t[0]];
		glm::vec3 b = ref.vertices[t[1]];
		glm::vec3 c = ref.vertices[t[2]];

		glm::vec3 n = glm::normalize(glm::cross(b - a, c - a));

		ref.normals[t[0]] += n;
		ref.normals[t[1]] += n;
		ref.normals[t[2]] += n;
	}

	for (glm::vec3 &n : ref.normals)
		n = glm::normalize(n);
}

inline Mesh remesh(const Mesh &ref, const std::unordered_set <uint32_t> &complex)
{
	Mesh out;
	for (uint32_t i : complex) {
		uint32_t csize = out.vertices.size();

		assert(i >= 0 && i < ref.triangles.size());
		Triangle t = ref.triangles[i];

		out.triangles.push_back(Triangle {
			csize + 0,
			csize + 1,
			csize + 2
		});

		assert(t[0] >= 0 && t[0] < ref.vertices.size());
		assert(t[1] >= 0 && t[1] < ref.vertices.size());
		assert(t[2] >= 0 && t[2] < ref.vertices.size());

		out.vertices.push_back(ref.vertices[t[0]]);
		out.vertices.push_back(ref.vertices[t[1]]);
		out.vertices.push_back(ref.vertices[t[2]]);

		out.normals.push_back(ref.normals[t[0]]);
		out.normals.push_back(ref.normals[t[1]]);
		out.normals.push_back(ref.normals[t[2]]);
	}

	return deduplicate(out);
}

// Vertex adjacency graph
using VertexGraph = std::unordered_map <uint32_t, std::unordered_set <uint32_t>>;

// Edge graph
struct Edge {
	uint32_t a = 0;
	uint32_t b = 0;

	Edge() = default;
	Edge(uint32_t a_, uint32_t b_) : a(a_), b(b_) {
		if (a > b)
			std::swap(a, b);
	}

	bool has(uint32_t v) const {
		return (a == v) || (b == v);
	}

	bool operator==(const Edge &e) const {
		return (a == e.a) && (b == e.b);
	}

	bool operator!=(const Edge &e) const {
		return (a != e.a) || (b != e.b);
	}

	bool operator<(const Edge &e) const {
		return (a < e.a) || ((a == e.a) && (b < e.b));
	}
};

template <>
struct std::hash <Edge> {
	static constexpr std::hash <uint32_t> hasher = std::hash <uint32_t> {};

	uint32_t operator()(const Edge &e) const {
		uint32_t d = e.b - e.a;
		uint32_t ha = hasher(e.a);
		uint32_t hd = hasher(d);

		return ha ^ (hd << 1);
	}
};

using EdgeGraph = std::unordered_map <Edge, std::unordered_set <uint32_t>>;
using EdgeMap = std::unordered_map <Edge, std::unordered_set <uint32_t>>;

// Dual graph
using DualGraph = std::unordered_map <uint32_t, std::unordered_set <uint32_t>>;

inline std::tuple <VertexGraph, EdgeGraph, DualGraph> build_graphs(const Mesh &ref)
{
	// Populate vertex graph
	VertexGraph vertex_graph;

	for (uint32_t i = 0; i < ref.triangles.size(); i++) {
		uint32_t i0 = ref.triangles[i][0];
		uint32_t i1 = ref.triangles[i][1];
		uint32_t i2 = ref.triangles[i][2];

		vertex_graph[i0].insert(i1);
		vertex_graph[i0].insert(i2);

		vertex_graph[i1].insert(i0);
		vertex_graph[i1].insert(i2);

		vertex_graph[i2].insert(i0);
		vertex_graph[i2].insert(i1);
	}

	// Populate edge graph
	EdgeGraph edge_graph;

	auto add_edge = [&](uint32_t a, uint32_t b, uint32_t f) {
		if (a > b)
			std::swap(a, b);

		Edge e { a, b };
		edge_graph[e].insert(f);
	};

	for (uint32_t i = 0; i < ref.triangles.size(); i++) {
		uint32_t i0 = ref.triangles[i][0];
		uint32_t i1 = ref.triangles[i][1];
		uint32_t i2 = ref.triangles[i][2];

		add_edge(i0, i1, i);
		add_edge(i1, i2, i);
		add_edge(i2, i0, i);
	}

	// Populate dual graph
	DualGraph dual_graph;

	auto add_dual = [&](uint32_t a, uint32_t b, uint32_t f) {
		if (a > b)
			std::swap(a, b);

		Edge e { a, b };

		auto &set = dual_graph[f];
		auto adj = edge_graph[e];
		adj.erase(f);

		set.merge(adj);
	};

	for (uint32_t i = 0; i < ref.triangles.size(); i++) {
		uint32_t i0 = ref.triangles[i][0];
		uint32_t i1 = ref.triangles[i][1];
		uint32_t i2 = ref.triangles[i][2];

		add_dual(i0, i1, i);
		add_dual(i1, i2, i);
		add_dual(i2, i0, i);
	}

	return { vertex_graph, edge_graph, dual_graph };
}

#ifdef MESH_LOAD_SAVE

// Mesh loading and saving
Mesh process_mesh(aiMesh *mesh, const aiScene *scene, const std::string &dir)
{
	// Mesh data
	std::vector <glm::vec3> vertices;
	std::vector <glm::vec3> normals;
        std::vector <Triangle> triangles;

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
	std::stack <uint32_t> buffer;
	for (uint32_t i = 0; i < mesh->mNumFaces; i++) {
		aiFace face = mesh->mFaces[i];
		for (uint32_t j = 0; j < face.mNumIndices; j++) {
			buffer.push(face.mIndices[j]);
			if (buffer.size() >= 3) {
				uint32_t i0 = buffer.top(); buffer.pop();
				uint32_t i1 = buffer.top(); buffer.pop();
				uint32_t i2 = buffer.top(); buffer.pop();
				triangles.push_back({ i0, i1, i2 });
			}

			// triangles.push_back(face.mIndices[j]);
		}
	}

	return { vertices, normals, triangles };
}

Mesh process_node(aiNode *node, const aiScene *scene, const std::string &dir)
{
	for (uint32_t i = 0; i < node->mNumMeshes; i++) {
		aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
                Mesh processed_mesh = process_mesh(mesh, scene, dir);
		if (processed_mesh.triangles.size() > 0)
			return processed_mesh;
	}

	// Recusively process all the node's children
	for (uint32_t i = 0; i < node->mNumChildren; i++) {
		Mesh processed_mesh = process_node(node->mChildren[i], scene, dir);
		if (processed_mesh.triangles.size() > 0)
			return processed_mesh;
	}

	return {};
}

Mesh load_mesh(const std::filesystem::path &path)
{
	Assimp::Importer importer;

	// Read scene
	const aiScene *scene = importer.ReadFile(
		path, aiProcess_Triangulate
			| aiProcess_GenNormals
			| aiProcess_FlipUVs
	);

	// Check if the scene was loaded
	if (!scene | scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE
			|| !scene->mRootNode) {
		fprintf(stderr, "Assimp error: \"%s\"\n", importer.GetErrorString());
		return {};
	}

	// Process the scene (root node)
	return process_node(scene->mRootNode, scene, path.string());
}

void save_mesh(const Mesh &mesh, const std::filesystem::path &path)
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

	for (int i = 0; i < mesh.vertices.size(); i++) {
		pMesh->mVertices[i] = { mesh.vertices[i].x, mesh.vertices[i].y, mesh.vertices[i].z };
		pMesh->mNormals[i] = { mesh.normals[i].x, mesh.normals[i].y, mesh.normals[i].z };
	}

	pMesh->mFaces = new aiFace[mesh.triangles.size()];
	pMesh->mNumFaces = mesh.triangles.size();

	for (int i = 0; i < mesh.triangles.size(); i++) {
		aiFace &face = pMesh->mFaces[i];
		face.mIndices = new unsigned int[3];
		face.mNumIndices = 3;

		Triangle t = mesh.triangles[i];
		face.mIndices[0] = t[0];
		face.mIndices[1] = t[1];
		face.mIndices[2] = t[2];
	}

	Assimp::Exporter exporter;
	exporter.Export(&scene, "obj", path.string());
}

#endif
