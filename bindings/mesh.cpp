#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cstdint>
#include <filesystem>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <queue>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

namespace py = pybind11;

using vertex_graph = std::unordered_map <uint32_t, std::vector <uint32_t>>;

struct mesh {
	py::array_t <float> vertices;
	py::array_t <uint32_t> triangles;

	mesh(const std::vector <float> &v = {}, const std::vector <uint32_t> &t = {}) {
		// Allocate memory
		vertices = py::array_t <float> (v.size());
		triangles = py::array_t <uint32_t> (t.size());

		// Copy data
		std::memcpy(vertices.mutable_data(), v.data(), v.size() * sizeof(float));
		std::memcpy(triangles.mutable_data(), t.data(), t.size() * sizeof(uint32_t));

		// Reshape appropriately
		std::vector <long> v_shape = { long(v.size())/3, 3 };
		std::vector <long> t_shape = { long(t.size())/3, 3 };

		vertices = vertices.reshape(v_shape);
		triangles = triangles.reshape(t_shape);

		printf("vertices: %d, triangles: %d\n", vertices.shape(0), triangles.shape(0));
	}

	// mesh(const py::array_t <float> &v, const py::array_t <uint32_t> &t)
	// 	: vertices(v), triangles(t) {
	// 	// Make sure the arrays are (N, 3) and float32 and uint32
	// 	if (vertices.ndim() != 2 || vertices.shape(1) != 3 || vertices.dtype() != py::dtype("float32"))
	// 		throw std::runtime_error("vertices must be (N, 3) and float32");
	//
	// 	if (triangles.ndim() != 2 || triangles.shape(1) != 3 || triangles.dtype() != py::dtype("uint32"))
	// 		throw std::runtime_error("triangles must be (N, 3) and uint32");
	// }

	// Deduplicating vertices
	mesh deduplicate() {
		std::unordered_map <glm::vec3, uint32_t> existing;

		std::vector <float> new_vertices;
		std::vector <uint32_t> new_triangles;

		auto add_unique = [&](const glm::vec3 &v) {
			auto it = existing.find(v);
			if (it == existing.end()) {
				uint32_t index = new_vertices.size() / 3;
				existing[v] = index;
				new_vertices.push_back(v.x);
				new_vertices.push_back(v.y);
				new_vertices.push_back(v.z);
				return index;
			}

			return it->second;
		};

		float *vertices_ptr = vertices.mutable_data();
		uint32_t *triangles_ptr = triangles.mutable_data();

		for (uint32_t i = 0; i < triangles.size(); i++) {
			float *v = vertices_ptr + triangles_ptr[i] * 3;
			glm::vec3 v3 = { v[0], v[1], v[2] };
			new_triangles.push_back(add_unique(v3));
		}

		return { new_vertices, new_triangles };
	}

	// Build a vertex graph
	vertex_graph vgraph() const {
		vertex_graph graph;

		const uint32_t *triangles_ptr = triangles.data();

		for (uint32_t i = 0; i < triangles.shape(0); i++) {
			uint32_t i0 = triangles_ptr[i * 3 + 0];
			uint32_t i1 = triangles_ptr[i * 3 + 1];
			uint32_t i2 = triangles_ptr[i * 3 + 2];

			graph[i0].push_back(i1);
			graph[i0].push_back(i2);
			graph[i1].push_back(i0);
			graph[i1].push_back(i2);
			graph[i2].push_back(i0);
			graph[i2].push_back(i1);
		}

		return graph;
	}

	// Finding indices of critical points of functions
	py::array_t <uint32_t> critical(const py::array_t <float> &function) const {
		assert(function.dtype() == py::dtype("float32"));

		vertex_graph graph = vgraph();
		printf("graph size: %d\n", graph.size());
		printf("function size: %d\n", function.size());
		assert(function.size() == vertices.shape(0));

		const float *function_ptr = function.data();

		std::vector <uint32_t> critical;
		for (uint32_t i = 0; i < vertices.shape(0); i++) {
			float f = function_ptr[i];

			bool is_max = true;
			bool is_min = true;

			for (uint32_t n : graph[i]) {
				float fn = function_ptr[n];

				if (fn > f)
					is_max = false;
				if (fn < f)
					is_min = false;
			}

			if (is_max || is_min)
				critical.push_back(i);
		}

		printf("critical size: %d\n", critical.size());
		py::array_t <uint32_t> result(critical.size());
		std::memcpy(result.mutable_data(), critical.data(), critical.size() * sizeof(uint32_t));

		return result;
	}
};

// Closest point function
void triangle_closest_point(const glm::vec3 &v0, const glm::vec3 &v1, const glm::vec3 &v2, const glm::vec3 &p, glm::vec3 *closest, glm::vec3 *bary, float *distance)
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

// Accelerating mesh distance/point queries
struct cas {
	using triangle = std::array <uint32_t, 3>;

	std::vector <glm::vec3> vertices;
	std::vector <triangle> triangles;

	glm::vec3 min;
	glm::vec3 max;
	
	uint32_t resolution;
	glm::vec3 bin_size;

	using query_bin = std::vector <uint32_t>;
	std::vector <query_bin> overlapping_triangles;
	std::vector <query_bin> query_triangles;

	// Construct from mesh
	cas(const mesh &ref_, uint32_t resolution_)
			: resolution(resolution_) {
		// Copy mesh
		uint32_t vertex_count = ref_.vertices.shape(0);
		uint32_t triangle_count = ref_.triangles.shape(0);

		vertices.resize(vertex_count);
		triangles.resize(triangle_count);

		std::memcpy(vertices.data(), ref_.vertices.data(), vertex_count * sizeof(glm::vec3));
		std::memcpy(triangles.data(), ref_.triangles.data(), triangle_count * sizeof(triangle));

		// Allocate and fill the bins
		uint32_t size = resolution * resolution * resolution;
		overlapping_triangles.resize(size);
		query_triangles.resize(size);

		// Put triangles into bins
		min = glm::vec3(std::numeric_limits <float> ::max());
		max = glm::vec3(std::numeric_limits <float> ::lowest());

		for (const glm::vec3 &v : vertices) {
			min = glm::min(min, v);
			max = glm::max(max, v);
		}

		glm::vec3 extent = max - min;
		bin_size = extent / (float) resolution;

		for (size_t i = 0; i < triangles.size(); i++) {
			const triangle &tri = triangles[i];

			// Triangle belongs to all bins it intersects
			glm::vec3 v0 = vertices[tri[0]];
			glm::vec3 v1 = vertices[tri[1]];
			glm::vec3 v2 = vertices[tri[2]];

			glm::vec3 tri_min = glm::min(glm::min(v0, v1), v2);
			glm::vec3 tri_max = glm::max(glm::max(v0, v1), v2);

			glm::vec3 min_bin = glm::clamp((tri_min - min) / bin_size, glm::vec3(0), glm::vec3(resolution - 1));
			glm::vec3 max_bin = glm::clamp((tri_max - min) / bin_size, glm::vec3(0), glm::vec3(resolution - 1));

			for (int x = min_bin.x; x <= max_bin.x; x++) {
				for (int y = min_bin.y; y <= max_bin.y; y++) {
					for (int z = min_bin.z; z <= max_bin.z; z++) {
						int index = x + y * resolution + z * resolution * resolution;
						overlapping_triangles[index].push_back(i);
					}
				}
			}
		}

		// Print average load
		uint32_t total = 0;
		for (const query_bin &bin : overlapping_triangles)
			total += bin.size();

		printf("Average load: %f\n", (float) total / (float) size);
	}

	std::unordered_set <uint32_t> closest_triangles(const glm::vec3 &p) const {
		// Get the current bin
		glm::vec3 bin_flt = glm::clamp((p - min) / bin_size, glm::vec3(0), glm::vec3(resolution - 1));
		glm::ivec3 bin = glm::ivec3(bin_flt);
		uint32_t bin_index = bin.x + bin.y * resolution + bin.z * resolution * resolution;

		// Find the closest non-empty bins
		std::vector <glm::ivec3> closest_bins;

		if (!overlapping_triangles[bin_index].empty()) {
			closest_bins.push_back(bin);
		} else {
			std::vector <glm::ivec3> plausible_bins;
			std::queue <glm::ivec3> queue;

			std::unordered_set <glm::ivec3> visited;
			bool stop = false;

			queue.push(bin);
			while (!queue.empty()) {
				glm::ivec3 current = queue.front();
				queue.pop();

				// If visited, continue
				if (visited.find(current) != visited.end())
					continue;

				visited.insert(current);

				// If non-empty, add to plausible bins and continue
				uint32_t current_index = current.x + current.y * resolution + current.z * resolution * resolution;
				if (!overlapping_triangles[current_index].empty()) {
					plausible_bins.push_back(current);

					// Also set the stop flag to stop adding neighbors
					stop = true;
					continue;
				}

				if (stop)
					continue;

				int dx[] = { -1, 0, 0, 1, 0, 0 };
				int dy[] = { 0, -1, 0, 0, 1, 0 };
				int dz[] = { 0, 0, -1, 0, 0, 1 };

				// Add all neighbors to queue...
				for (int i = 0; i < 6; i++) {
					glm::ivec3 next = current + glm::ivec3(dx[i], dy[i], dz[i]);
					if (next.x < 0 || next.x >= resolution ||
						next.y < 0 || next.y >= resolution ||
						next.z < 0 || next.z >= resolution)
						continue;

					// ...if not visited
					if (visited.find(next) == visited.end())
						queue.push(next);
				}
			}

			// Sort plausible bins by distance
			std::sort(plausible_bins.begin(), plausible_bins.end(),
				[&](const glm::ivec3 &a, const glm::ivec3 &b) {
					return glm::distance(bin_flt, glm::vec3(a)) < glm::distance(bin_flt, glm::vec3(b));
				}
			);

			assert(!plausible_bins.empty());

			// Add first one always; stop adding when difference is larger than voxel size
			closest_bins.push_back(plausible_bins[0]);
			for (uint32_t i = 1; i < plausible_bins.size(); i++) {
				glm::vec3 a = glm::vec3(plausible_bins[i - 1]);
				glm::vec3 b = glm::vec3(plausible_bins[i]);

				if (glm::distance(a, b) > 1.1f)
					break;

				closest_bins.push_back(plausible_bins[i]);
			}
		}

		assert(!closest_bins.empty());

		// Within the final collection, make sure to search immediate neighbors
		std::unordered_set <uint32_t> final_bins;

		for (const glm::ivec3 &bin : closest_bins) {
			int dx[] = { 0, -1, 0, 0, 1, 0, 0 };
			int dy[] = { 0, 0, -1, 0, 0, 1, 0 };
			int dz[] = { 0, 0, 0, -1, 0, 0, 1 };

			for (int i = 0; i < 7; i++) {
				glm::ivec3 next = bin + glm::ivec3(dx[i], dy[i], dz[i]);
				if (next.x < 0 || next.x >= resolution ||
					next.y < 0 || next.y >= resolution ||
					next.z < 0 || next.z >= resolution)
					continue;

				uint32_t next_index = next.x + next.y * resolution + next.z * resolution * resolution;
				if (!overlapping_triangles[next_index].empty())
					final_bins.insert(next_index);
			}
		}

		std::unordered_set <uint32_t> final_triangles;
		for (uint32_t bin_index : final_bins) {
			for (uint32_t index : overlapping_triangles[bin_index])
				final_triangles.insert(index);
		}

		return final_triangles;
	}

	// Load the cached query triangles if not already loaded
	bool precache_query(const glm::vec3 &p) {
		// Check if the bin is already cached
		// uint32_t bin_index = to_index(p);
		
		glm::vec3 bin_flt = glm::clamp((p - min) / bin_size, glm::vec3(0), glm::vec3(resolution - 1));
		glm::ivec3 bin = glm::ivec3(bin_flt);
		uint32_t bin_index = bin.x + bin.y * resolution + bin.z * resolution * resolution;

		// printf("  Precaching bin %d\n", bin_index);
		// printf("  p = (%f, %f, %f)\n", p.x, p.y, p.z);
		// printf("  max = (%f, %f, %f)\n", max.x, max.y, max.z);
		// printf("  min = (%f, %f, %f)\n", min.x, min.y, min.z);
		assert(bin_index < query_triangles.size());

		if (!query_triangles[bin_index].empty())
			return false;

		// Otherwise, load the bin
		auto set = closest_triangles(p);
		query_triangles[bin_index] = query_bin(set.begin(), set.end());
		return true;
	}

	// Distance queries
	py::array_t <float> sdf(const py::array_t <float> &points) {
		// Ensure format
		assert(points.ndim() == 2);
		assert(points.shape(1) == 3);
		assert(points.dtype() == py::dtype("float32"));

		std::vector <float> distances(points.shape(0));
		std::vector <glm::vec3> points_vec(points.shape(0));

		// Copy points to vector
		auto points_info = points.request();
		float *points_ptr = (float *) points_info.ptr;

		std::copy(points_ptr, points_ptr + points.shape(0) * 3, (float *) points_vec.data());

		// First precache all the bins
		for (const glm::vec3 &p : points_vec)
			precache_query(p);

		// Then compute the distances
		for (uint32_t i = 0; i < points_vec.size(); i++) {
			const glm::vec3 &p = points_vec[i];

			glm::vec3 bin_flt = glm::clamp((p - min) / bin_size, glm::vec3(0), glm::vec3(resolution - 1));
			glm::ivec3 bin = glm::ivec3(bin_flt);
			uint32_t bin_index = bin.x + bin.y * resolution + bin.z * resolution * resolution;

			float min_distance = std::numeric_limits <float> ::infinity();
			for (uint32_t index : query_triangles[bin_index]) {
				const triangle &tri = triangles[index];

				const glm::vec3 &a = vertices[tri[0]];
				const glm::vec3 &b = vertices[tri[1]];
				const glm::vec3 &c = vertices[tri[2]];

				glm::vec3 closest_point;
				glm::vec3 bary;
				float distance;

				triangle_closest_point(a, b, c, p, &closest_point, &bary, &distance);

				if (distance < min_distance)
					min_distance = distance;
			}

			distances[i] = min_distance;
		}

		// Copy distances to numpy array
		py::array_t <float> distances_array(points.shape(0));
		auto distances_info = distances_array.request();

		float *distances_ptr = (float *) distances_info.ptr;
		std::copy(distances.begin(), distances.end(), distances_ptr);

		return distances_array;
	}
};

// Mesh loading and saving
mesh process_mesh(aiMesh *mesh, const aiScene *scene, const std::string &dir)
{
	// Mesh data
	std::vector <float> vertices;
        std::vector <uint32_t> triangles;

	// Process all the mesh's vertices
	for (uint32_t i = 0; i < mesh->mNumVertices; i++) {
		// vertices.push_back({
		// 	mesh->mVertices[i].x,
		// 	mesh->mVertices[i].y,
		// 	mesh->mVertices[i].z
		// });
		vertices.push_back(mesh->mVertices[i].x);
		vertices.push_back(mesh->mVertices[i].y);
		vertices.push_back(mesh->mVertices[i].z);
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

				// triangles.push_back({ i0, i1, i2 });
				triangles.push_back(i0);
				triangles.push_back(i1);
				triangles.push_back(i2);
			}
		}
	}

	printf("Loaded mesh with %lu vertices and %lu triangles\n", vertices.size()/3, triangles.size()/3);

	return { vertices, triangles };
}

mesh process_node(aiNode *node, const aiScene *scene, const std::string &dir)
{
	for (uint32_t i = 0; i < node->mNumMeshes; i++) {
		aiMesh *ai_mesh = scene->mMeshes[node->mMeshes[i]];
                mesh processed_mesh = process_mesh(ai_mesh, scene, dir);
		if (processed_mesh.triangles.size() > 0)
			return processed_mesh;
	}

	// Recusively process all the node's children
	for (uint32_t i = 0; i < node->mNumChildren; i++) {
		mesh processed_mesh = process_node(node->mChildren[i], scene, dir);
		if (processed_mesh.triangles.size() > 0)
			return processed_mesh;
	}

	return {};
}

mesh load(const py::str &path)
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
	return process_node(scene->mRootNode, scene, path);
}

PYBIND11_MODULE(mesh, m)
{
	pybind11::class_ <mesh> (m, "mesh")
		// .def(pybind11::init <const py::array_t <float> &, const py::array_t <uint32_t> &> ())
		.def(pybind11::init <> ())
		.def_readwrite("vertices", &mesh::vertices)
		.def_readwrite("triangles", &mesh::triangles)
		.def("deduplicate", &mesh::deduplicate)
		.def("critical", &mesh::critical);

	pybind11::class_ <cas> (m, "cas")
		.def(pybind11::init <const mesh &, uint32_t> ())
		.def("sdf", &cas::sdf);

	m.def("load", &load);
}
