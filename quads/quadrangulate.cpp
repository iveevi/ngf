#include <filesystem>
#include <random>
#include <unordered_map>
#include <unordered_set>

#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/Exporter.hpp>

#include "microlog.h"

struct ordered_pair {
	uint32_t a;
	uint32_t b;

	ordered_pair(uint32_t a_, uint32_t b_) : a(a_), b(b_) {
		if (a > b)
			std::swap(a, b);
	}

	bool operator==(const ordered_pair &other) const {
		return a == other.a && b == other.b;
	}

	// bool operator<(const ordered_pair &other) const {
	// 	return a < other.a || (a == other.a && b < other.b);
	// }

	static size_t hash(const ordered_pair &p) {
		return std::hash <uint32_t> ()(p.a) ^ std::hash <uint32_t> ()(p.b);
	}
};

template <typename T>
using ordered_pair_map = std::unordered_map <ordered_pair, T, decltype(&ordered_pair::hash)>;

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

geometry <eTriangle> compact(const geometry <eTriangle> &ref)
{
	std::unordered_map <glm::vec3, uint32_t> existing;

	geometry <eTriangle> fixed;
	auto add_uniquely = [&](int32_t i) ->size_t {
		glm::vec3 v = ref.vertices[i];
		if (existing.find(v) == existing.end()) {
			int32_t csize = fixed.vertices.size();
			fixed.vertices.push_back(v);

			existing[v] = csize;
			return csize;
		}

		return existing[v];
	};

	for (const glm::uvec3 &t : ref.triangles) {
		fixed.triangles.push_back(glm::uvec3 {
			add_uniquely(t[0]),
			add_uniquely(t[1]),
			add_uniquely(t[2])
		});
	}

	return fixed;
}

template <size_t primitive>
geometry <primitive> sanitize(const geometry <primitive> &ref)
{
	// Remove degenerate primitives
	geometry <primitive> fixed;
	fixed.vertices = ref.vertices;

	if constexpr (primitive == eTriangle) {
		uint32_t count = 0;
		for (const glm::uvec3 &t : ref.triangles) {
			bool ok = (t[0] != t[1]) && (t[0] != t[2]) && (t[1] != t[2]);
			if (ok)
				fixed.triangles.push_back(t);
			else
				count++;
		}

		if (count)
			ulog_warning("sanitize", "found %u degenerate triangles\n", count);
		else
			ulog_info("sanitize", "no degenerate triangles found\n");
		// TODO: ulog_ok
	} else if constexpr (primitive == eQuad) {
		uint32_t count = 0;
		for (const glm::uvec4 &q : ref.quads) {
			bool ok = (q[0] != q[1]) && (q[0] != q[2]) && (q[0] != q[3])
				&& (q[1] != q[2]) && (q[1] != q[3])
				&& (q[2] != q[3]);

			if (ok)
				fixed.quads.push_back(q);
			else
				count++;
		}

		if (count)
			ulog_warning("sanitize", "found %u degenerate quads\n", count);
		else
			ulog_info("sanitize", "no degenerate quads found\n");
	}

	return fixed;
}

geometry <eQuad> quadrangulate(const geometry <eTriangle> &ref)
{
	ordered_pair_map <glm::vec3> midpoints(0, ordered_pair::hash);

	std::unordered_map <uint32_t, glm::vec3> centroids(0);

	for (uint32_t t = 0; t < ref.triangles.size(); t++) {
		const glm::uvec3 &triangle = ref.triangles[t];

		for (uint32_t i = 0; i < 3; i++) {
			uint32_t a = triangle[i];
			uint32_t b = triangle[(i + 1) % 3];
			ordered_pair edge(a, b);
			if (midpoints.find(edge) == midpoints.end())
				midpoints[edge] = (ref.vertices[a] + ref.vertices[b]) / 2.0f;
		}

		glm::vec3 centroid = (ref.vertices[triangle[0]] + ref.vertices[triangle[1]] + ref.vertices[triangle[2]]) / 3.0f;
		centroids[t] = centroid;
	}

	// Create new vector of vertices
	ordered_pair_map <uint32_t> midpoint_indices(0, ordered_pair::hash);

	std::unordered_map <uint32_t, uint32_t> centroid_indices(0);

	std::vector <glm::vec3> vertices;
	for (const glm::vec3 &vertex : ref.vertices)
		vertices.push_back(vertex);

	for (const auto &pair : midpoints) {
		midpoint_indices[pair.first] = vertices.size();
		vertices.push_back(pair.second);
	}

	for (const auto &pair : centroids) {
		centroid_indices[pair.first] = vertices.size();
		vertices.push_back(pair.second);
	}

	// Create new vector of quads
	std::vector <glm::uvec4> quads;

	for (uint32_t t = 0; t < ref.triangles.size(); t++) {
		const glm::uvec3 &triangle = ref.triangles[t];

		ordered_pair edge0(triangle[0], triangle[1]);
		ordered_pair edge1(triangle[1], triangle[2]);
		ordered_pair edge2(triangle[2], triangle[0]);

		uint32_t m01 = midpoint_indices[edge0];
		uint32_t m12 = midpoint_indices[edge1];
		uint32_t m20 = midpoint_indices[edge2];
		uint32_t c = centroid_indices[t];

		quads.push_back(glm::uvec4(triangle[0], m01, c, m20));
		quads.push_back(glm::uvec4(triangle[1], m12, c, m01));
		quads.push_back(glm::uvec4(triangle[2], m20, c, m12));
	}

	geometry <eQuad> result;
	result.vertices = std::move(vertices);
	result.quads = std::move(quads);
	return result;
}

template <size_t primitive>
geometry <primitive> laplacian_smoothing(const geometry <primitive> &ref, float factor)
{
	std::unordered_map <uint32_t, std::unordered_set <uint32_t>> adjacency(0);

	if constexpr (primitive == eTriangle) {
		for (const auto &triangle : ref.triangles) {
			adjacency[triangle[0]].insert(triangle[1]);
			adjacency[triangle[0]].insert(triangle[2]);
			adjacency[triangle[1]].insert(triangle[0]);
			adjacency[triangle[1]].insert(triangle[2]);
			adjacency[triangle[2]].insert(triangle[0]);
			adjacency[triangle[2]].insert(triangle[1]);
		}
	} else {
		for (const auto &quad : ref.quads) {
			adjacency[quad[0]].insert(quad[1]);
			adjacency[quad[0]].insert(quad[3]);
			adjacency[quad[1]].insert(quad[0]);
			adjacency[quad[1]].insert(quad[2]);
			adjacency[quad[2]].insert(quad[1]);
			adjacency[quad[2]].insert(quad[3]);
			adjacency[quad[3]].insert(quad[0]);
			adjacency[quad[3]].insert(quad[2]);
		}
	}

	geometry <primitive> result;
	result.vertices.resize(ref.vertices.size());
	result.triangles = ref.triangles;
	result.quads = ref.quads;

	for (uint32_t i = 0; i < ref.vertices.size(); i++) {
		const glm::vec3 &vertex = ref.vertices[i];
		const std::unordered_set <uint32_t> &adjacent = adjacency[i];

		glm::vec3 sum(0.0f);
		for (uint32_t j : adjacent)
			sum += ref.vertices[j];

		result.vertices[i] = vertex + (sum/(float) adjacent.size() - vertex) * factor;
	}

	return result;
}

// Remove doublets in quadrilateral meshes
geometry <eQuad> cleanse_doublets(const geometry <eQuad> &ref)
{
	ordered_pair_map <std::vector <uint32_t>> edge_quads({}, ordered_pair::hash);

	for (uint32_t q = 0; q < ref.quads.size(); q++) {
		const glm::uvec4 &quad = ref.quads[q];

		ordered_pair edge0(quad[0], quad[1]);
		ordered_pair edge1(quad[1], quad[2]);
		ordered_pair edge2(quad[2], quad[3]);
		ordered_pair edge3(quad[3], quad[0]);

		edge_quads[edge0].push_back(q);
		edge_quads[edge1].push_back(q);
		edge_quads[edge2].push_back(q);
		edge_quads[edge3].push_back(q);
	}

	ordered_pair_map <std::vector <ordered_pair>> shared_edges({}, ordered_pair::hash);

	for (const auto &[e, quads] : edge_quads) {
		// Only consider valid manifold edges
		if (quads.size() == 2) {
			auto it = quads.begin();
			uint32_t q0 = *it;
			uint32_t q1 = *std::next(it);
			ordered_pair qpair = ordered_pair(q0, q1);
			shared_edges[qpair].push_back(e);
		}
	}

	struct doublet {
		uint32_t a;
		uint32_t b;
		uint32_t c;

		bool operator==(const doublet &d) const {
			return (a == d.a && b == d.b && c == d.c) || (a == d.b && b == d.a && c == d.c);
		}

		static size_t hash(const doublet &d) {
			return std::hash <uint32_t> ()(d.a) ^ std::hash <uint32_t> ()(d.b) ^ std::hash <uint32_t> ()(d.c);
		}
	};

	// TODO: regular vector?
	std::unordered_set <doublet, decltype(&doublet::hash)> doublets(0, &doublet::hash);

	for (const auto &[qpair, edges] : shared_edges) {
		ulog_assert(edges.size() <= 2, "Invalid manifold boundary", "quads %u and %u share %d\n", qpair.a, qpair.b, edges.size());
		if (edges.size() < 2)
			continue;

		auto it = edges.begin();
		ordered_pair edge0 = *it;
		ordered_pair edge1 = *(++it);

		doublet d;
		if (edge0.a == edge1.a)
			d = { edge0.b, edge0.a, edge1.b };
		else if (edge0.a == edge1.b)
			d = { edge0.b, edge0.a, edge1.a };
		else if (edge0.b == edge1.a)
			d = { edge0.a, edge0.b, edge1.b };
		else if (edge0.b == edge1.b)
			d = { edge0.a, edge0.b, edge1.a };
		else
			ulog_error("cleanse_doublets", "Invalid shared edge pair (%u, %u) and (%u, %u)\n", edge0.a, edge0.b, edge1.a, edge1.b);

		doublets.insert(d);
	}

	ulog_info("cleanse_doublets", "found %u doublets\n", doublets.size());

	// Constructing new primitives
	std::vector <glm::uvec4> quads = ref.quads;
	std::vector <bool> valid(ref.quads.size(), true);

	for (const doublet &d : doublets) {
		ordered_pair edge0(d.a, d.b);
		ordered_pair edge1(d.b, d.c);

		// printf("dissovling doublet (%u, %u, %u) -> edges (%u, %u) and (%u, %u)\n", d.a, d.b, d.c, edge0.a, edge0.b, edge1.a, edge1.b);

		const auto &equads = edge_quads[edge0];
		ulog_assert(edge_quads[edge0].size() == 2, "Invalid doublet", "quads %u and %u share %d\n", quads[0], quads[1], quads.size());
		ulog_assert(edge_quads[edge1].size() == 2, "Invalid doublet", "quads %u and %u share %d\n", quads[0], quads[1], quads.size());

		uint32_t q0 = equads[0];
		uint32_t q1 = equads[1];

		if (!valid[q0] || !valid[q1]) {
			ulog_warning("cleanse_doublets", "skipping (no longer) invalid doublet\n");
			continue;
		}

		// Dissolve doublet
		valid[q0] = false;
		valid[q1] = false;

		// New quad; find the two other vertices not in the doublet
		const glm::uvec4 &quad0 = ref.quads[q0];
		const glm::uvec4 &quad1 = ref.quads[q1];

		uint32_t v0 = 0;
		uint32_t v1 = 0;

		for (uint32_t i = 0; i < 4; i++) {
			if (quad0[i] != d.a && quad0[i] != d.b && quad0[i] != d.c)
				v0 = quad0[i];

			if (quad1[i] != d.a && quad1[i] != d.b && quad1[i] != d.c)
				v1 = quad1[i];
		}

		// printf("new quad (%u, %u, %u, %u)\n", d.a, v0, d.c, v1);
		// printf("  > original q0 (%u, %u, %u, %u)\n", quad0[0], quad0[1], quad0[2], quad0[3]);
		// printf("  > original q1 (%u, %u, %u, %u)\n", quad1[0], quad1[1], quad1[2], quad1[3]);

		quads.push_back({ d.a, v0, d.c, v1 });
	}

	// Reconstruct primitives
	std::vector <glm::uvec4> clean_quads;
	for (uint32_t i = 0; i < valid.size(); i++) {
		if (!valid[i])
			continue;

		clean_quads.push_back(quads[i]);
	}

	ulog_info("cleanse_doublets", "cleaned %lu quads\n", valid.size() - clean_quads.size());
	for (uint32_t i = valid.size(); i < quads.size(); i++)
		clean_quads.push_back(quads[i]);

	geometry <eQuad> result = ref;
	result.quads = clean_quads;

	return result;
}

struct loader {
	std::vector <geometry <eTriangle>> meshes;

	loader(const std::filesystem::path &path) {
		Assimp::Importer importer;

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

	const geometry <eTriangle> &get(uint32_t i) const {
		return meshes[i];
	}
};

void loader::process_mesh(aiMesh *mesh, const aiScene *scene, const std::string &dir)
{

	std::vector <glm::vec3> vertices;
	std::vector <glm::vec3> normals;
	std::vector <glm::uvec3> triangles;

	// Process all the mesh's vertices
	for (uint32_t i = 0; i < mesh->mNumVertices; i++) {
		vertices.push_back({
			mesh->mVertices[i].x,
			mesh->mVertices[i].y,
			mesh->mVertices[i].z
		});
	}

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

	meshes.push_back({ vertices, triangles, {} });
}

template <size_t primitive>
void write_obj(const geometry <primitive> &mesh, const std::filesystem::path &path)
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

	for (uint32_t i = 0; i < mesh.vertices.size(); i++) {
		pMesh->mVertices[i] = { mesh.vertices[i].x, mesh.vertices[i].y, mesh.vertices[i].z };
		// pMesh->mNormals[i] = { mesh.normals[i].x, mesh.normals[i].y, mesh.normals[i].z };
	}

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
	exporter.Export(&scene, "obj", path.string());

	ulog_info("write_obj", "exported result to %s\n", path.c_str());
}

int main(int argc, char *argv[])
{
	if (argc != 2) {
		ulog_error("quadrangulate", "Usage: %s <input>\n", argv[0]);
		return 1;
	}

	std::filesystem::path input { argv[1] };

	geometry <eTriangle> ref = loader(input).get(0);
	ulog_info("quadrangulate", "loaded %s, mesh has %d vertices and %d triangles\n", input.c_str(), ref.vertices.size(), ref.triangles.size());

	ref = compact(ref);
	ref = sanitize(ref);
	ulog_info("quadrangulate", "compacted %s, resulting mesh has %d vertices and %d triangles\n", input.c_str(), ref.vertices.size(), ref.triangles.size());

	geometry <eQuad> result = quadrangulate(ref);
	ulog_info("quadrangulate", "quadrangulated %s, resulting mesh has %d vertices and %d quads\n", input.c_str(), result.vertices.size(), result.quads.size());

	for (int i = 0; i < 3; i++)
		result = laplacian_smoothing(result, 0.9f);

	result = sanitize(result);
	result = cleanse_doublets(result);

	ulog_info("quadrangulate", "smoothed %s, resulting mesh has %d vertices and %d quads\n", input.c_str(), result.vertices.size(), result.quads.size());

	write_obj(result, (input.stem().string() + "_quad.obj"));
}
