#include <array>
#include <chrono>
#include <cstdint>
#include <stack>
#include <unordered_set>
#include <random>

#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <littlevk/littlevk.hpp>

#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/point_cloud.h>
#include <polyscope/curve_network.h>

#include "casdf/casdf.hpp"
#include "microlog.h"

using quad = std::array <uint32_t, 4>;

struct quad_mesh {
	std::vector <glm::vec3> vertices;
	std::vector <glm::vec3> normals;
	std::vector <quad> quads;
};

std::pair <quad_mesh, std::unordered_map <uint32_t, uint32_t>> deduplicate(const quad_mesh &qm)
{
	std::unordered_map <glm::vec3, uint32_t> vertex_map;

	std::vector <glm::vec3> new_vertices;
	std::vector <glm::vec3> new_normals;
	std::vector <quad> new_quads;

	for (const quad &q : qm.quads) {
		quad new_q;
		for (uint32_t i = 0; i < 4; i++) {
			uint32_t v = q[i];
			const glm::vec3 &vertex = qm.vertices[v];
			const glm::vec3 &normal = qm.normals[v];

			auto it = vertex_map.find(vertex);
			if (it == vertex_map.end()) {
				uint32_t new_v = new_vertices.size();
				vertex_map[vertex] = new_v;

				new_vertices.push_back(vertex);
				new_normals.push_back(normal);

				new_q[i] = new_v;
			} else {
				new_q[i] = it->second;
			}
		}

		new_quads.push_back(new_q);
	}

	// return { new_vertices, new_normals, new_quads };
	quad_mesh result { new_vertices, new_normals, new_quads };

	std::unordered_map <uint32_t, uint32_t> remap;
	for (uint32_t i = 0; i < qm.vertices.size(); i++)
		remap[i] = vertex_map[qm.vertices[i]];

	return { result, remap };
}

// Lossy vertex deduplication
inline std::pair <quad_mesh, std::unordered_map <uint32_t, uint32_t>> deduplicate(const quad_mesh &qm, float threshold, float scale = 1e4f)
{
	auto hasher = [&](const glm::vec3 &v) -> size_t {
		uint64_t x = v.x * scale;
		uint64_t y = v.y * scale;
		uint64_t z = v.z * scale;

		std::hash <uint64_t> h;
		return h(x) ^ h(y) ^ h(z);
	};

	auto eq = [&](const glm::vec3 &a, const glm::vec3 &b) -> bool {
		return glm::distance(a, b) < threshold;
	};

	std::unordered_map <glm::vec3, uint32_t, decltype(hasher), decltype(eq)> existing(0, hasher, eq);

	quad_mesh fixed;
	auto add_uniquely = [&](int32_t i) -> uint32_t {
		glm::vec3 v = qm.vertices[i];

		if (existing.find(v) == existing.end()) {
			int32_t csize = fixed.vertices.size();
			fixed.vertices.push_back(v);
			fixed.normals.push_back(qm.normals[i]);

			existing[v] = csize;
			return csize;
		}

		return existing[v];
	};

	for (const quad &q : qm.quads) {
		fixed.quads.push_back({
			add_uniquely(q[0]),
			add_uniquely(q[1]),
			add_uniquely(q[2]),
			add_uniquely(q[3])
		});
	}

	std::unordered_map <uint32_t, uint32_t> remap;
	for (size_t i = 0; i < qm.vertices.size(); i++)
		remap[i] = add_uniquely(i);

	return { fixed, remap };
}

struct subdivision_complex {
	std::vector <uint32_t> vertices;
	std::vector <uint32_t> quads;
	uint32_t size;
};

std::vector <glm::vec3> subdivide(const quad_mesh &source, const subdivision_complex &sdc)
{
	ULOG_ASSERT(sdc.vertices.size() == sdc.size * sdc.size);

	std::vector <glm::vec3> base;
	base.reserve(sdc.vertices.size());

	for (uint32_t v : sdc.vertices)
		base.push_back(source.vertices[v]);

	std::vector <glm::vec3> result;

	uint32_t new_size = 2 * sdc.size;
	result.resize(new_size * new_size);

	// Bilerp each new vertex
	for (uint32_t i = 0; i < new_size; i++) {
		for (uint32_t j = 0; j < new_size; j++) {
			float u = (float) i / (new_size - 1);
			float v = (float) j / (new_size - 1);

			float lu = u * (sdc.size - 1);
			float lv = v * (sdc.size - 1);

			uint32_t u0 = std::floor(lu);
			uint32_t u1 = std::ceil(lu);

			uint32_t v0 = std::floor(lv);
			uint32_t v1 = std::ceil(lv);

			glm::vec3 p00 = base[u0 * sdc.size + v0];
			glm::vec3 p10 = base[u1 * sdc.size + v0];
			glm::vec3 p01 = base[u0 * sdc.size + v1];
			glm::vec3 p11 = base[u1 * sdc.size + v1];

			lu -= u0;
			lv -= v0;

			glm::vec3 p = p00 * (1.0f - lu) * (1.0f - lv) +
				p10 * lu * (1.0f - lv) +
				p01 * (1.0f - lu) * lv +
				p11 * lu * lv;

			result[i * new_size + j] = p;
		}
	}

	return result;
}

struct ordered_pair {
	uint32_t a, b;

	bool from(uint32_t a_, uint32_t b_) {
		if (a_ > b_) {
			a = b_;
			b = a_;
			return true;
		}

		a = a_;
		b = b_;
		return false;
	}

	bool operator==(const ordered_pair &other) const {
		return a == other.a && b == other.b;
	}

	struct hash {
		size_t operator()(const ordered_pair &p) const {
			std::hash <uint32_t> h;
			return h(p.a) ^ h(p.b);
		}
	};
};

struct list_hash {
	size_t operator()(const std::vector <uint32_t> &v) const {
		std::hash <uint32_t> h;
		size_t result = 0;

		for (uint32_t i : v)
			result ^= h(i);

		return result;
	}
};

std::pair <quad_mesh, std::vector <subdivision_complex>> subdivide(const quad_mesh &source, const std::vector <subdivision_complex> &sdcs)
{
	std::vector <std::vector <glm::vec3>> new_patches;

	std::unordered_map <uint32_t, std::set <std::pair <uint32_t, uint32_t>>> sdc_corner_map;
	std::unordered_map <ordered_pair, std::set <std::pair <uint32_t, std::vector <uint32_t>>>, ordered_pair::hash> sdc_boundary_map;

	uint32_t size = sdcs[0].size;
	uint32_t new_size = 2 * sdcs[0].size;

	for (uint32_t i = 0; i < sdcs.size(); i++) {
		const subdivision_complex &s = sdcs[i];
		auto new_vertices = subdivide(source, s);
		ULOG_ASSERT(new_vertices.size() == new_size * new_size);
		new_patches.push_back(new_vertices);

		uint32_t i00 = 0;
		uint32_t i10 = new_size - 1;
		uint32_t i01 = (new_size - 1) * new_size;
		uint32_t i11 = new_size * new_size - 1;

		uint32_t c00 = s.vertices[0];
		uint32_t c10 = s.vertices[size - 1];
		uint32_t c01 = s.vertices[size * (size - 1)];
		uint32_t c11 = s.vertices[size * size - 1];

		sdc_corner_map[c00].insert({ i, i00 });
		sdc_corner_map[c10].insert({ i, i10 });
		sdc_corner_map[c01].insert({ i, i01 });
		sdc_corner_map[c11].insert({ i, i11 });

		ordered_pair p;
		bool reversed;

		std::vector <uint32_t> b00_10;
		std::vector <uint32_t> b00_01;
		std::vector <uint32_t> b10_11;
		std::vector <uint32_t> b01_11;

		// 00 -> 10
		reversed = p.from(c00, c10);
		if (reversed) {
			for (uint32_t i = new_size - 2; i >= 1; i--)
				b00_10.push_back(i);
		} else {
			for (uint32_t i = 1; i <= new_size - 2; i++)
				b00_10.push_back(i);
		}

		sdc_boundary_map[p].insert({ i, b00_10 });

		// 00 -> 01
		reversed = p.from(c00, c01);
		if (reversed) {
			for (uint32_t i = new_size * (new_size - 2); i >= new_size; i -= new_size)
				b00_01.push_back(i);
		} else {
			for (uint32_t i = new_size; i <= new_size * (new_size - 2); i += new_size)
				b00_01.push_back(i);
		}

		sdc_boundary_map[p].insert({ i, b00_01 });

		// 10 -> 11
		reversed = p.from(c10, c11);
		if (reversed) {
			for (uint32_t i = new_size - 2; i >= 1; i--)
				b10_11.push_back(i * new_size + new_size - 1);
		} else {
			for (uint32_t i = 1; i <= new_size - 2; i++)
				b10_11.push_back(i * new_size + new_size - 1);
		}

		sdc_boundary_map[p].insert({ i, b10_11 });

		// 01 -> 11
		reversed = p.from(c01, c11);
		if (reversed) {
			for (uint32_t i = new_size - 2; i >= 1; i--)
				b01_11.push_back((new_size - 1) * new_size + i);
		} else {
			for (uint32_t i = 1; i <= new_size - 2; i++)
				b01_11.push_back((new_size - 1) * new_size + i);
		}

		sdc_boundary_map[p].insert({ i, b01_11 });
	}

	quad_mesh new_source;

	std::vector <subdivision_complex> new_sdcs(
		sdcs.size(),
		subdivision_complex {
			std::vector <uint32_t> (new_size * new_size, 0),
			std::vector <uint32_t> ((new_size - 1) * (new_size - 1), 0),
			new_size
		}
	);

	// Resolve all the corners
	for (const auto &[_, pr] : sdc_corner_map) {
		auto source_corner = pr.begin();
		glm::vec3 p = new_patches[source_corner->first][source_corner->second];

		uint32_t new_corner = new_source.vertices.size();
		new_source.vertices.push_back(p);

		for (const auto &sdc_corner : pr)
			new_sdcs[sdc_corner.first].vertices[sdc_corner.second] = new_corner;
	}

	// Resolve all boundaries
	for (const auto &[p, bs] : sdc_boundary_map) {
		const auto &ref = *bs.begin();

		uint32_t offset = new_source.vertices.size();
		for (uint32_t i = 0; i < ref.second.size(); i++) {
			glm::vec3 v = new_patches[ref.first][ref.second[i]];
			new_source.vertices.push_back(v);
		}

		for (const auto &b : bs) {
			for (uint32_t i = 0; i < b.second.size(); i++) {
				uint32_t i0 = b.second[i];
				uint32_t new_vertex = offset + i;
				new_sdcs[b.first].vertices[i0] = new_vertex;
			}
		}
	}

	// Fill the remaining interiors
	for (uint32_t i = 0; i < sdcs.size(); i++) {
		for (uint32_t x = 1; x < new_size - 1; x++) {
			for (uint32_t y = 1; y < new_size - 1; y++) {
				uint32_t i00 = x * new_size + y;

				uint32_t new_vertex = new_source.vertices.size();
				new_source.vertices.push_back(new_patches[i][i00]);
				new_sdcs[i].vertices[i00] = new_vertex;
			}
		}
	}

	// Fill the faces
	for (uint32_t i = 0; i < sdcs.size(); i++) {
		for (uint32_t x = 0; x < new_size - 1; x++) {
			for (uint32_t y = 0; y < new_size - 1; y++) {
				uint32_t i00 = x * new_size + y;
				uint32_t i10 = i00 + 1;
				uint32_t i01 = (x + 1) * new_size + y;
				uint32_t i11 = i01 + 1;

				uint32_t new_quad = new_source.quads.size();
				new_source.quads.push_back({
					new_sdcs[i].vertices[i00],
					new_sdcs[i].vertices[i10],
					new_sdcs[i].vertices[i11],
					new_sdcs[i].vertices[i01]
				});

				new_sdcs[i].quads[x * (new_size - 1) + y] = new_quad;
			}
		}
	}

	return { new_source, new_sdcs };
}

void save(const quad_mesh &qm, const std::vector <subdivision_complex> &sdcs, const std::string &target_file, const std::filesystem::path &path)
{

	std::ofstream fout(path, std::ios::binary);

	uint32_t target_file_size = target_file.size();
	fout.write((char *) &target_file_size, sizeof(uint32_t));
	fout.write((char *) target_file.data(), target_file_size);

	std::vector <std::array <uint32_t, 4>> quad_corners;
	for (const auto &s : sdcs) {
		quad_corners.push_back({
			s.vertices[0],
			s.vertices[s.size - 1],
			s.vertices[s.size * (s.size - 1)],
			s.vertices[s.size * s.size - 1],
		});

		glm::vec3 c0 = qm.vertices[s.vertices[0]];
		glm::vec3 c1 = qm.vertices[s.vertices[s.size - 1]];
		glm::vec3 c2 = qm.vertices[s.vertices[s.size * (s.size - 1)]];
		glm::vec3 c3 = qm.vertices[s.vertices[s.size * s.size - 1]];

		printf("quad: (%f, %f, %f) (%f, %f, %f) (%f, %f, %f) (%f, %f, %f)\n",
			c0.x, c0.y, c0.z,
			c1.x, c1.y, c1.z,
			c2.x, c2.y, c2.z,
			c3.x, c3.y, c3.z);
	}

	std::unordered_map <uint32_t, uint32_t> corner_map;
	auto add_unique_corners = [&](uint32_t c) -> uint32_t {
		if (corner_map.count(c))
			return corner_map[c];

		uint32_t csize = corner_map.size();
		corner_map[c] = csize;
		return csize;
	};

	std::vector <uint32_t> normalized_complexes;

	for (const std::array <uint32_t, 4> &c : quad_corners) {
		normalized_complexes.push_back(add_unique_corners(c[0]));
		normalized_complexes.push_back(add_unique_corners(c[1]));
		normalized_complexes.push_back(add_unique_corners(c[2]));
		normalized_complexes.push_back(add_unique_corners(c[3]));
	}

	printf("Normalized complexes: %lu\n", normalized_complexes.size()/12);
	for (uint32_t i = 0; i < normalized_complexes.size(); i += 4)
		printf("%u %u %u %u\n", normalized_complexes[i], normalized_complexes[i + 1], normalized_complexes[i + 2], normalized_complexes[i + 3]);

	std::vector <glm::vec3> corners;
	corners.resize(corner_map.size());

	// std::vector <uint32_t> corner_indices;
	for (const auto &p : corner_map) {
		// corners.push_back(opt.vertices[p.first]);
		corners[p.second] = qm.vertices[p.first];
		// printf("%u: (%f, %f, %f)\n", p.second, corners.back().x, corners.back().y, corners.back().z);
	}

	printf("Corner count: %u\n", corner_map.size());
	for (uint32_t i = 0; i < corners.size(); i++)
		printf("%u: (%f, %f, %f)\n", i, corners[i].x, corners[i].y, corners[i].z);

	uint32_t corner_count = corner_map.size();
	fout.write((char *) &corner_count, sizeof(uint32_t));
	fout.write((char *) corners.data(), corners.size() * sizeof(glm::vec3));

	uint32_t quad_count = sdcs.size();
	fout.write((char *) &quad_count, sizeof(uint32_t));
	fout.write((char *) normalized_complexes.data(), normalized_complexes.size() * sizeof(uint32_t));

	for (const auto &s : sdcs) {
		uint32_t size = s.size;
		fout.write((char *) &size, sizeof(uint32_t));

		std::vector <glm::vec3> vertices;
		for (uint32_t v : s.vertices)
			vertices.push_back(qm.vertices[v]);

		uint32_t vertex_count = vertices.size();
		fout.write((char *) &vertex_count, sizeof(uint32_t));
		fout.write((char *) vertices.data(), vertices.size() * sizeof(glm::vec3));
	}
}

template <size_t V>
struct loader {
	static_assert(V == 3 || V == 4);
	using value_type = std::conditional_t <V == 3, geometry, quad_mesh>;

	std::vector <value_type> meshes;

	loader(const std::filesystem::path &path) {
		Assimp::Importer importer;

		// Read scene
		const aiScene *scene;

		if constexpr (V == 3)
			scene = importer.ReadFile(path, aiProcess_GenNormals | aiProcess_Triangulate);
		else if constexpr (V == 4)
			scene = importer.ReadFile(path, aiProcess_GenNormals);

		// Check if the scene was loaded
		if ((!scene | scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) || !scene->mRootNode) {
			ulog_error("loader", "Assimp error: \"%s\"\n", importer.GetErrorString());
			return;
		}

		process_node(scene->mRootNode, scene, path.parent_path());
		ulog_info("loader", "Loaded %d meshes from %s\n", meshes.size(), path.c_str());
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

	const value_type &get(uint32_t i) const {
		return meshes[i];
	}
};

template <>
void loader <3> ::process_mesh(aiMesh *mesh, const aiScene *scene, const std::string &dir)
{

	std::vector <glm::vec3> vertices;
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
	// std::stack <uint32_t> buffer;
	for (uint32_t i = 0; i < mesh->mNumFaces; i++) {
		aiFace face = mesh->mFaces[i];
		ulog_assert(face.mNumIndices == 3, "process_mesh", "Only triangles are supported, got %d-sided polygon instead\n", face.mNumIndices);
		triangles.push_back({
			face.mIndices[0],
			face.mIndices[1],
			face.mIndices[2]
		});
	}

	meshes.push_back({ vertices, triangles });
	ulog_info("loader", "Loaded triangle mesh with %d vertices and %d triangles\n", vertices.size(), triangles.size());
}

template <>
void loader <4> ::process_mesh(aiMesh *mesh, const aiScene *scene, const std::string &dir)
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

std::unordered_map <uint32_t, std::unordered_set <uint32_t>> vertex_graph(const quad_mesh &qm)
{
	std::unordered_map <uint32_t, std::unordered_set <uint32_t>> graph;

	for (const quad &q : qm.quads) {
		for (uint32_t i = 0; i < 4; i++) {
			uint32_t v0 = q[i];
			uint32_t v1 = q[(i + 1) % 4];

			graph[v0].insert(v1);
			graph[v1].insert(v0);
		}
	}

	return graph;
}

std::unordered_map <uint32_t, std::unordered_set <uint32_t>> dense_vertex_graph(const quad_mesh &qm)
{
	std::unordered_map <uint32_t, std::unordered_set <uint32_t>> graph;

	for (const quad &q : qm.quads) {
		uint32_t v0 = q[0];
		uint32_t v1 = q[1];
		uint32_t v2 = q[2];
		uint32_t v3 = q[3];

		graph[v0].insert(v1);
		graph[v0].insert(v2);
		graph[v0].insert(v3);

		graph[v1].insert(v0);
		graph[v1].insert(v2);
		graph[v1].insert(v3);

		graph[v2].insert(v0);
		graph[v2].insert(v1);
		graph[v2].insert(v3);

		graph[v3].insert(v0);
		graph[v3].insert(v1);
		graph[v3].insert(v2);
	}

	return graph;
}

geometry geometry_from_quad_mesh(const quad_mesh &qm)
{
	// Split each quad into two triangles
	std::vector <glm::vec3> vertices = qm.vertices;
	std::vector <glm::uvec3> triangles;

	for (const quad &q : qm.quads) {
		// Comparsing diagonals
		float d02 = glm::distance(vertices[q[0]], vertices[q[2]]);
		float d13 = glm::distance(vertices[q[1]], vertices[q[3]]);

		if (d02 < d13) {
			// 0 -- 2
			triangles.push_back({ q[0], q[1], q[2] });
			triangles.push_back({ q[0], q[2], q[3] });
		} else {
			// 1 -- 3
			triangles.push_back({ q[0], q[1], q[3] });
			triangles.push_back({ q[1], q[2], q[3] });
		}
	}

	return { vertices, triangles };
}

quad_mesh smooth(const quad_mesh &qm, float factor)
{
	// Compute vertex graph
	auto graph = vertex_graph(qm);

	// Compute new vertices
	std::vector <glm::vec3> new_vertices;
	for (uint32_t i = 0; i < qm.vertices.size(); i++) {
		const glm::vec3 &v = qm.vertices[i];

		glm::vec3 new_v { 0.0f };
		for (uint32_t j : graph[i])
			new_v += qm.vertices[j];

		new_v /= graph[i].size();
		new_vertices.push_back(v * (1.0f - factor) + new_v * factor);
	}

	return { new_vertices, qm.normals, qm.quads };
}

std::pair <quad_mesh, std::vector <float>> interpolate(const quad_mesh &qm, size_t N)
{
	// Subdivide each quad into N x N quads
	std::vector <glm::vec3> vertices;
	std::vector <glm::vec3> normals;
	std::vector <quad> quads;
	std::vector <float> weights;

	for (const quad &q : qm.quads) {
		const glm::vec3 &v0 = qm.vertices[q[0]];
		const glm::vec3 &v1 = qm.vertices[q[1]];
		const glm::vec3 &v2 = qm.vertices[q[2]];
		const glm::vec3 &v3 = qm.vertices[q[3]];

		const glm::vec3 &n0 = qm.normals[q[0]];
		const glm::vec3 &n1 = qm.normals[q[1]];
		const glm::vec3 &n2 = qm.normals[q[2]];
		const glm::vec3 &n3 = qm.normals[q[3]];
		glm::vec3 navg = (n0 + n1 + n2 + n3) / 4.0f;

		uint32_t base = vertices.size();
		for (uint32_t i = 0; i < N * N; i++) {
			uint32_t x = i % N;
			uint32_t y = i / N;

			float fx = (float) x / (float) (N - 1);
			float fy = (float) y / (float) (N - 1);

			glm::vec3 v = (1.0f - fx) * (1.0f - fy) * v0 + fx * (1.0f - fy) * v1 + fx * fy * v2 + (1.0f - fx) * fy * v3;
			glm::vec3 n = (1.0f - fx) * (1.0f - fy) * n0 + fx * (1.0f - fy) * n1 + fx * fy * n2 + (1.0f - fx) * fy * n3;
			n = glm::normalize(n);

			float d = (1 - 2 * fabs(0.5 - fx)) *  (1 - 2 * fabs(0.5 - fy));

			vertices.push_back(v);
			normals.push_back(n);
			weights.push_back(d);
		}

		for (uint32_t i = 0; i < N - 1; i++) {
			for (uint32_t j = 0; j < N - 1; j++) {
				uint32_t a = i + j * N + base;
				uint32_t b = i + 1 + j * N + base;
				uint32_t c = i + 1 + (j + 1) * N + base;
				uint32_t d = i + (j + 1) * N + base;

				quads.push_back({ a, b, c, d });
			}
		}
	}

	quad_mesh result = { vertices, normals, quads };
	return { result, weights };
}

struct live_progress {
	// TODO: Plus time barriers
	double total = 0.0f;
	double previous_progress = 0.0f;

	std::chrono::time_point <std::chrono::steady_clock> start;

	live_progress() {
		start = std::chrono::steady_clock::now();
	}

	void update(double progress) {
		double now = std::chrono::duration_cast <std::chrono::duration <double>> (std::chrono::steady_clock::now() - start).count();
		double delta = now - total;

		total += delta;
		previous_progress = progress;

		double projected_time = delta / progress;

		printf("\033[2K");
		printf("\rProgress: %.2f%% (%.2fs)", progress * 100.0f, delta);
		fflush(stdout);
	}
};

// TODO: iterator based formulation...

void optimization_phase(const geometry &source, cas_grid &cas, quad_mesh &opt)
{
	auto graph = dense_vertex_graph(opt);

	geometry gopt = geometry_from_quad_mesh(opt);

	cumesh opt_cumesh = cumesh_alloc(gopt);
	cumesh source_cumesh = cumesh_alloc(source);

	std::vector <glm::vec3> closest;
	std::vector <glm::vec3> bary;
	std::vector <float> distances;
	std::vector <uint32_t> indices;

	closest.resize(opt.vertices.size());
	bary.resize(opt.vertices.size());
	distances.resize(opt.vertices.size());
	indices.resize(opt.vertices.size());

	constexpr uint32_t sample_count = 5000;

	sample_result source_samples = sample_result_alloc(sample_count, eCUDA);
	sample_result host_source_samples = sample_result_alloc(sample_count, eCPU);

	closest_point_kinfo source_to_opt_kinfo = closest_point_kinfo_alloc(sample_count, eCUDA);
	closest_point_kinfo host_source_to_opt_kinfo = closest_point_kinfo_alloc(sample_count, eCPU);

	auto start = std::chrono::steady_clock::now();

	float total_time0 = 0.0f;
	float total_time1 = 0.0f;
	float total_time2 = 0.0f;

	for (uint32_t i = 0; i < 1000; i++) {
		std::chrono::time_point <std::chrono::steady_clock> t0;
		std::chrono::time_point <std::chrono::steady_clock> t1;

		// Using iterative closest point
		t0 = std::chrono::steady_clock::now();
		float rate = cas.precache_query(opt.vertices);
		cas.query(opt.vertices, closest, bary, distances, indices);
		t1 = std::chrono::steady_clock::now();

		float time0 = std::chrono::duration_cast <std::chrono::duration <float>> (t1 - t0).count();

		float mse = 0.0f;
		for (uint32_t j = 0; j < opt.vertices.size(); j++) {
			glm::vec3 dv = closest[j] - opt.vertices[j];
			opt.vertices[j] += dv * 0.1f;
			mse += glm::dot(dv, dv);
		}

		// And random sampling
		t0 = std::chrono::steady_clock::now();
		auto now = std::chrono::steady_clock::now();
		float time = std::chrono::duration_cast <std::chrono::duration <float>> (now - start).count();
		sample(source_samples, source_cumesh, time);
		memcpy(host_source_samples, source_samples);
		cudaMemcpy(source_to_opt_kinfo.points, source_samples.points, sizeof(glm::vec3) * sample_count, cudaMemcpyDeviceToDevice);
		t1 = std::chrono::steady_clock::now();

		float time1 = std::chrono::duration_cast <std::chrono::duration <float>> (t1 - t0).count();

		t0 = std::chrono::steady_clock::now();
		gopt = geometry_from_quad_mesh(opt);
		cumesh_reload(opt_cumesh, gopt);
		brute_closest_point(opt_cumesh, source_to_opt_kinfo);
		memcpy(host_source_to_opt_kinfo, source_to_opt_kinfo);
		t1 = std::chrono::steady_clock::now();

		float time2 = std::chrono::duration_cast <std::chrono::duration <float>> (t1 - t0).count();

		for (uint32_t j = 0; j < sample_count; j++) {
			glm::vec3 w = host_source_samples.points[j];
			glm::vec3 v = host_source_to_opt_kinfo.closest[j];
			glm::vec3 b = host_source_to_opt_kinfo.bary[j];
			uint32_t t = host_source_to_opt_kinfo.triangles[j];

			const glm::uvec3 &tri = gopt.triangles[t];

			glm::vec3 v0 = opt.vertices[tri.x];
			glm::vec3 v1 = opt.vertices[tri.y];
			glm::vec3 v2 = opt.vertices[tri.z];

			glm::vec3 delta = w - v;

			b = glm::clamp(b, 0.0f, 1.0f);
			glm::vec3 gv0 = b.x * delta;
			glm::vec3 gv1 = b.y * delta;
			glm::vec3 gv2 = b.z * delta;

			opt.vertices[tri.x] += gv0 * 0.1f;
			opt.vertices[tri.y] += gv1 * 0.1f;
			opt.vertices[tri.z] += gv2 * 0.1f;

			mse += glm::dot(gv0, gv0) + glm::dot(gv1, gv1) + glm::dot(gv2, gv2);

			uint32_t tsampled = host_source_samples.indices[j];
		}

		if (i % 10 == 0 && i < 700)
			opt = smooth(opt, 0.5f);

		// Average edge length
		float edge_length = 0.0f;

		for (const auto &q : opt.quads) {
			const glm::vec3 &v0 = opt.vertices[q[0]];
			const glm::vec3 &v1 = opt.vertices[q[1]];
			const glm::vec3 &v2 = opt.vertices[q[2]];
			const glm::vec3 &v3 = opt.vertices[q[3]];

			float e = 0.0f;
			e += glm::length(v1 - v0);
			e += glm::length(v2 - v0);
			e += glm::length(v3 - v0);
			e += glm::length(v3 - v1);
			e += glm::length(v2 - v1);
			e += glm::length(v3 - v2);

			edge_length += e/6.0f;
		}

		edge_length /= opt.quads.size();

		// Edge length gradients for critically close vertices (0.5)
		std::vector <glm::vec3> edge_gradients(opt.vertices.size(), glm::vec3(0.0f));

		for (const auto &[vi, adj] : graph) {
			const glm::vec3 &v = opt.vertices[vi];

			for (const auto &vj : adj) {
				const glm::vec3 &w = opt.vertices[vj];

				float d = glm::length(v - w);
				if (d < 0.5 * edge_length) {
					glm::vec3 g = glm::normalize(v - w) * (edge_length - d);
					edge_gradients[vi] += g;
				}
			}
		}

		// Apply edge gradients
		for (uint32_t j = 0; j < opt.vertices.size(); j++)
			opt.vertices[j] += 0.1f * edge_gradients[j];

		// Clear line
		total_time0 += time0;
		total_time1 += time1;
		total_time2 += time2;

		float sum = total_time0 + total_time1 + total_time2;
		float frac0 = total_time0 / sum;
		float frac1 = total_time1 / sum;
		float frac2 = total_time2 / sum;

		float avg_time0 = total_time0 / (i + 1);
		float avg_time1 = total_time1 / (i + 1);
		float avg_time2 = total_time2 / (i + 1);

		// TODO: tiny progress indicator as part of micro log
		printf("\033[2K");
		printf("Iteration %d, error %.5f, cache hit rate %.2f%%, t0 (CPU query) = %.2f, t1 (Sampling) = %.2f, t2 (GPU query) = %.2f\r",
			i, mse, (1 - rate) * 100.0f,
			frac0 * 100.0f, frac1 * 100.0f, frac2 * 100.0f);
		fflush(stdout);
	}

	// Compute smallest edge length
	float min_edge_length = std::numeric_limits <float> ::max();

	for (const auto &q : opt.quads) {
		const glm::vec3 &v0 = opt.vertices[q[0]];
		const glm::vec3 &v1 = opt.vertices[q[1]];
		const glm::vec3 &v2 = opt.vertices[q[2]];
		const glm::vec3 &v3 = opt.vertices[q[3]];

		float d01 = glm::length(v1 - v0);
		float d02 = glm::length(v2 - v0);
		float d03 = glm::length(v3 - v0);
		float d13 = glm::length(v3 - v1);
		float d12 = glm::length(v2 - v1);
		float d23 = glm::length(v3 - v2);

		min_edge_length = std::min(min_edge_length, d01);
		min_edge_length = std::min(min_edge_length, d02);
		min_edge_length = std::min(min_edge_length, d03);
		min_edge_length = std::min(min_edge_length, d13);
		min_edge_length = std::min(min_edge_length, d12);
		min_edge_length = std::min(min_edge_length, d23);
	}

	// TODO: stream the mesh data to a file so the progress can be viewed in realtime
}

static std::pair <glm::vec3, glm::vec3> bound(const geometry &g)
{
	glm::vec3 max = g.vertices[0];
	glm::vec3 min = g.vertices[0];
	for (const glm::vec3 &v : g.vertices) {
		max = glm::max(max, v);
		min = glm::min(min, v);
	}

	return { max, min };
}

static inline void triangle_closest_point(const glm::vec3 &v0, const glm::vec3 &v1, const glm::vec3 &v2, const glm::vec3 &p, glm::vec3 *closest, glm::vec3 *bary, float *distance)
{
	glm::vec3 B = v0;
	glm::vec3 E1 = v1 - v0;
	glm::vec3 E2 = v2 - v0;
	glm::vec3 D = B - p;

	float a = glm::dot(E1, E1);
	float b = glm::dot(E1, E2);
	float c = glm::dot(E2, E2);
	float d = glm::dot(E1, D);
	float e = glm::dot(E2, D);
	float f = glm::dot(D, D);

	float det = a * c - b * b;
	float s = b * e - c * d;
	float t = b * d - a * e;

	if (s + t <= det) {
		if (s < 0.0f) {
			if (t < 0.0f) {
				if (d < 0.0f) {
					s = glm::clamp(-d / a, 0.0f, 1.0f);
					t = 0.0f;
				} else {
					s = 0.0f;
					t = glm::clamp(-e / c, 0.0f, 1.0f);
				}
			} else {
				s = 0.0f;
				t = glm::clamp(-e / c, 0.0f, 1.0f);
			}
		} else if (t < 0.0f) {
			s = glm::clamp(-d / a, 0.0f, 1.0f);
			t = 0.0f;
		} else {
			float invDet = 1.0f / det;
			s *= invDet;
			t *= invDet;
		}
	} else {
		if (s < 0.0f) {
			float tmp0 = b + d;
			float tmp1 = c + e;
			if (tmp1 > tmp0) {
				float numer = tmp1 - tmp0;
				float denom = a - 2 * b + c;
				s = glm::clamp(numer / denom, 0.0f, 1.0f);
				t = 1 - s;
			} else {
				t = glm::clamp(-e / c, 0.0f, 1.0f);
				s = 0.0f;
			}
		} else if (t < 0.0f) {
			if (a + d > b + e) {
				float numer = c + e - b - d;
				float denom = a - 2 * b + c;
				s = glm::clamp(numer / denom, 0.0f, 1.0f);
				t = 1 - s;
			} else {
				s = glm::clamp(-e / c, 0.0f, 1.0f);
				t = 0.0f;
			}
		} else {
			float numer = c + e - b - d;
			float denom = a - 2 * b + c;
			s = glm::clamp(numer / denom, 0.0f, 1.0f);
			t = 1.0f - s;
		}
	}

	*closest = B + s * E1 + t * E2;
	*bary = glm::vec3(1.0f - s - t, s, t);
	*distance = glm::length(*closest - p);
}

int main(int argc, char *argv[])
{
	// Load arguments
	if (argc != 4) {
		printf("Usage: %s <simplified quad mesh> <source mesh> <iterations>\n", argv[0]);
		return 1;
	}

	std::filesystem::path quad_mesh_path = std::filesystem::weakly_canonical(argv[1]);
	if (!std::filesystem::exists(quad_mesh_path)) {
		printf("File %s does not exist\n", argv[1]);
		return 1;
	}

	std::filesystem::path source_mesh_path = std::filesystem::weakly_canonical(argv[2]);
	if (!std::filesystem::exists(source_mesh_path)) {
		printf("File %s does not exist\n", argv[2]);
		return 1;
	}

	uint32_t iterations = std::atoi(argv[3]);
	if (iterations <= 0) {
		printf("Invalid number of iterations\n");
		return 1;
	}

	// Load mesh
	static_assert(std::is_same_v <loader <3> ::value_type, geometry>);
	static_assert(std::is_same_v <loader <4> ::value_type, quad_mesh>);

	loader <4> qloader(quad_mesh_path);
	loader <3> sloader(source_mesh_path);

	// quad_mesh mesh = load(quad_mesh_path);
	quad_mesh qm = qloader.get(0);
	qm = deduplicate(qm).first;

	geometry source = sloader.get(0);

	// A little optimization
	std::vector <subdivision_complex> sdcs;
	for (uint32_t i = 0; i < qm.quads.size(); i++)
		sdcs.push_back({ { qm.quads[i][0], qm.quads[i][1], qm.quads[i][3], qm.quads[i][2] }, { i }, 2 });

	cas_grid cas(source, 128);

	namespace ps = polyscope;

	ps::init();

	ps::registerSurfaceMesh("base quad mesh", qm.vertices, qm.quads);

	for (int i = 0; i < iterations; i++) {
		if (i > 0)
			std::tie(qm, sdcs) = subdivide(qm, sdcs);

		optimization_phase(source, cas, qm);
		ps::registerSurfaceMesh("quad mesh" + std::to_string(i), qm.vertices, qm.quads);
	}

	std::vector <std::array <uint32_t, 3>> stris;
	for (auto &iv : source.triangles)
		stris.push_back({ iv.x, iv.y, iv.z });

	ps::registerSurfaceMesh("source", source.vertices, stris);

	std::vector <glm::vec3> colors(qm.quads.size(), glm::vec3(0.0f));

	constexpr glm::vec3 color_wheel[] = {
		{ 0.750, 0.250, 0.250 },
		{ 0.750, 0.500, 0.250 },
		{ 0.750, 0.750, 0.250 },
		{ 0.500, 0.750, 0.250 },
		{ 0.250, 0.750, 0.250 },
		{ 0.250, 0.750, 0.500 },
		{ 0.250, 0.750, 0.750 },
		{ 0.250, 0.500, 0.750 },
		{ 0.250, 0.250, 0.750 },
		{ 0.500, 0.250, 0.750 },
		{ 0.750, 0.250, 0.750 },
		{ 0.750, 0.250, 0.500 }
	};

	auto m = ps::registerSurfaceMesh("mesh", qm.vertices, qm.quads);
	for (uint32_t i = 0; i < sdcs.size(); i++) {
		uint32_t c = i % (sizeof(color_wheel) / sizeof(color_wheel[0]));
		for (uint32_t j : sdcs[i].quads)
			colors[j] = color_wheel[c];
	}

	m->addFaceColorQuantity("coding", colors);

	ps::show();

	// Save data
	std::filesystem::path bin_path = source_mesh_path.stem().string() + ".sdc";
	save(qm, sdcs, source_mesh_path, bin_path);
}
