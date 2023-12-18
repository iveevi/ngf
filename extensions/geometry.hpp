#include <cstdint>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>

struct ordered_pair {
	int32_t a, b;

	ordered_pair(int32_t a_ = 0, int32_t b_ = 0)
			: a(a_), b(b_) {
		if (a > b) {
			a = b_;
			b = a_;
		}
	}

	// TODO: remove?
	bool from(int32_t a_, uint32_t b_) {
		a = a_;
		b = b_;

		if (a > b) {
			a = b_;
			b = a_;
			return true;
		}

		// a = a_;
		// b = b_;
		return false;
	}

	bool operator==(const ordered_pair &other) const {
		return a == other.a && b == other.b;
	}

	bool operator<(const ordered_pair &other) const {
		return a < other.a || (a == other.a && b < other.b);
	}

	struct hash {
		size_t operator()(const ordered_pair &p) const {
			std::hash <int32_t> h;
			return h(p.a) ^ h(p.b);
		}
	};
};

struct geometry {
	std::vector <glm::vec3> vertices;
        std::vector <glm::vec3> normals;
	std::vector <glm::ivec3> triangles;

	geometry() = default;

	geometry(const torch::Tensor &torch_vertices, const torch::Tensor &torch_triangles) {
		// Expects:
		//   2D tensor of shape (N, 3) for vertices
		//   2D tensor of shape (N, 3) for normals
		//   2D tensor of shape (M, 3) for triangles
		assert(torch_vertices.dim() == 2 && torch_vertices.size(1) == 3);
		assert(torch_triangles.dim() == 2 && torch_triangles.size(1) == 3);

		// Ensure CPU tensors
		assert(torch_vertices.device().is_cpu());
		assert(torch_triangles.device().is_cpu());

		// Ensure float32 and uint32
		assert(torch_vertices.dtype() == torch::kFloat32);
		assert(torch_triangles.dtype() == torch::kInt32);

		vertices.resize(torch_vertices.size(0));
		triangles.resize(torch_triangles.size(0));

		float *vertices_ptr = torch_vertices.data_ptr <float> ();
		int32_t *triangles_ptr = torch_triangles.data_ptr <int32_t> ();

		memcpy(vertices.data(), vertices_ptr, sizeof(glm::vec3) * vertices.size());
		memcpy(triangles.data(), triangles_ptr, sizeof(glm::ivec3) * triangles.size());

		// Compute normals vectors
		normals.resize(torch_vertices.size(0), glm::vec3(0.0f));
		for (int i = 0; i < torch_triangles.size(0); i++) {
			glm::uvec3 triangle = *(glm::uvec3 *) (triangles_ptr + i * 3);
			glm::vec3 v0 = *(glm::vec3 *) (vertices_ptr + triangle[0] * 3);
			glm::vec3 v1 = *(glm::vec3 *) (vertices_ptr + triangle[1] * 3);
			glm::vec3 v2 = *(glm::vec3 *) (vertices_ptr + triangle[2] * 3);
			glm::vec3 normal = glm::cross(v1 - v0, v2 - v0);
			normals[triangle[0]] += normal;
			normals[triangle[1]] += normal;
			normals[triangle[2]] += normal;
		}

		for (glm::vec3 &n : normals)
			n = glm::normalize(n);
	}

	geometry(const torch::Tensor &torch_vertices, const torch::Tensor &torch_normals, const torch::Tensor &torch_triangles) {
		// Expects:
		//   2D tensor of shape (N, 3) for vertices
		//   2D tensor of shape (N, 3) for normals
		//   2D tensor of shape (M, 3) for triangles
		assert(torch_vertices.dim() == 2 && torch_vertices.size(1) == 3);
		assert(torch_normals.dim() == 2 && torch_normals.size(1) == 3);
		assert(torch_triangles.dim() == 2 && torch_triangles.size(1) == 3);

		// Ensure CPU tensors
		assert(torch_vertices.device().is_cpu());
		assert(torch_normals.device().is_cpu());
		assert(torch_triangles.device().is_cpu());

		// Ensure float32 and uint32
		assert(torch_vertices.dtype() == torch::kFloat32);
		assert(torch_normals.dtype() == torch::kFloat32);
		assert(torch_triangles.dtype() == torch::kInt32);

		vertices.resize(torch_vertices.size(0));
		normals.resize(torch_normals.size(0));
		triangles.resize(torch_triangles.size(0));

		float *vertices_ptr = torch_vertices.data_ptr <float> ();
		float *normals_ptr = torch_normals.data_ptr <float> ();
		int32_t *triangles_ptr = torch_triangles.data_ptr <int32_t> ();

		memcpy(vertices.data(), vertices_ptr, sizeof(glm::vec3) * vertices.size());
		memcpy(normals.data(), normals_ptr, sizeof(glm::vec3) * normals.size());
		memcpy(triangles.data(), triangles_ptr, sizeof(glm::ivec3) * triangles.size());
	}

	geometry deduplicate() const {
		std::unordered_map <glm::vec3, int32_t> existing;

		geometry fixed;

		auto add_uniquely = [&](int32_t i) -> int32_t {
			glm::vec3 v = vertices[i];
			if (existing.find(v) == existing.end()) {
				int32_t csize = fixed.vertices.size();
				fixed.vertices.push_back(v);
				fixed.normals.push_back(normals[i]);

				existing[v] = csize;
				return csize;
			}

			return existing[v];
		};

		for (const glm::ivec3 &t : triangles) {
			fixed.triangles.push_back(glm::ivec3 {
				add_uniquely(t.x),
				add_uniquely(t.y),
				add_uniquely(t.z)
			});
		}

		return fixed;
	}

	std::tuple <torch::Tensor, torch::Tensor, torch::Tensor> torched() const {
		// Return torch tensors for vertices, normals, and triangles (on CPU)
		torch::Tensor torch_vertices = torch::zeros({ (long) vertices.size(), 3 }, torch::kFloat32);
		torch::Tensor torch_normals = torch::zeros({ (long) normals.size(), 3 }, torch::kFloat32);
		torch::Tensor torch_triangles = torch::zeros({ (long) triangles.size(), 3 }, torch::kInt32);

		float *vertices_ptr = torch_vertices.data_ptr <float> ();
		float *normals_ptr = torch_normals.data_ptr <float> ();
		int32_t *triangles_ptr = torch_triangles.data_ptr <int32_t> ();

		memcpy(vertices_ptr, vertices.data(), sizeof(glm::vec3) * vertices.size());
		memcpy(normals_ptr, normals.data(), sizeof(glm::vec3) * normals.size());
		memcpy(triangles_ptr, triangles.data(), sizeof(glm::ivec3) * triangles.size());

		return std::make_tuple(torch_vertices.cuda(), torch_normals.cuda(), torch_triangles.cuda());
	}

	// Helper methods
	float area(size_t index) const {
		assert(index < triangles.size());
		const glm::ivec3 &triangle = triangles[index];
		const glm::vec3 &v0 = vertices[triangle[0]];
		const glm::vec3 &v1 = vertices[triangle[1]];
		const glm::vec3 &v2 = vertices[triangle[2]];
		return 0.5 * glm::length(glm::cross(v1 - v0, v2 - v0));
	}

	glm::vec3 centroid(size_t index) const {
		assert(index < triangles.size());
		const glm::ivec3 &triangle = triangles[index];
		const glm::vec3 &v0 = vertices[triangle[0]];
		const glm::vec3 &v1 = vertices[triangle[1]];
		const glm::vec3 &v2 = vertices[triangle[2]];
		return (v0 + v1 + v2) / 3.0f;
	}

	glm::vec3 face_normal(size_t index) const {
		assert(index < triangles.size());
		const glm::ivec3 &triangle = triangles[index];
		const glm::vec3 &v0 = vertices[triangle[0]];
		const glm::vec3 &v1 = vertices[triangle[1]];
		const glm::vec3 &v2 = vertices[triangle[2]];
		return glm::normalize(glm::cross(v1 - v0, v2 - v0));
	}

	// Graph structures
	using edge_graph = std::map <ordered_pair, std::unordered_set <int32_t>>;
	using dual_graph = std::unordered_map <int32_t, std::unordered_set <int32_t>>;

	edge_graph make_edge_graph() const {
		edge_graph egraph;

		auto add_edge = [&](int32_t a, int32_t  b, int32_t f) {
			if (a > b)
				std::swap(a, b);

			ordered_pair e { a, b };
			egraph[e].insert(f);
		};

		for (int32_t i = 0; i < triangles.size(); i++) {
			int32_t i0 = triangles[i][0];
			int32_t i1 = triangles[i][1];
			int32_t i2 = triangles[i][2];

			add_edge(i0, i1, i);
			add_edge(i1, i2, i);
			add_edge(i2, i0, i);
		}

		return egraph;
	}

	dual_graph make_dual_graph(const edge_graph &egraph) const {
		dual_graph dgraph;

		auto add_dual = [&](int32_t a, int32_t b, int32_t f) {
			if (a > b)
				std::swap(a, b);

			ordered_pair e { a, b };

			auto &set = dgraph[f];
			auto adj = egraph.at(e);
			adj.erase(f);

			set.merge(adj);
		};

		for (int32_t i = 0; i < triangles.size(); i++) {
			int32_t i0 = triangles[i][0];
			int32_t i1 = triangles[i][1];
			int32_t i2 = triangles[i][2];

			add_dual(i0, i1, i);
			add_dual(i1, i2, i);
			add_dual(i2, i0, i);
		}

		return dgraph;
	}
};

