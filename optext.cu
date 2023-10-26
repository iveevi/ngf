#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <stdio.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>

#include <torch/extension.h>

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

// Chartifying geometry into N clusters
std::vector <std::vector <int32_t>> cluster_once(const geometry &g, const geometry::dual_graph &dgraph, const std::vector <int32_t> &seeds)
{
	// Data structures
	struct face {
		int32_t i;
		float *costs = nullptr;

		bool operator<(const face &f) const {
			return (costs[i] < costs[f.i]) ||
				((std::abs(costs[i] - costs[f.i]) < 1e-6f) && (i < f.i));
		}

		bool operator>(const face &f) const {
			return (costs[i] > costs[f.i]) ||
				((std::abs(costs[i] - costs[f.i]) < 1e-6f) && (i > f.i));
		}

		bool operator==(const face &f) const {
			return i == f.i;
		}
	};

	struct : public std::priority_queue <face, std::vector <face>, std::greater <face>> {
		void rebuild() {
			std::make_heap(this->c.begin(), this->c.end(), this->comp);
		}
	} queue;

	// Initialization
	using cluster = std::vector <int32_t>;

	std::unordered_map <int32_t, int32_t> face_to_cluster;
	std::vector <cluster>                 clusters;
	std::vector <glm::vec3>               cluster_normals;
	std::vector <float>                   costs;

	for (int32_t s : seeds) {
		assert(s < g.triangles.size());
		face_to_cluster[s] = clusters.size();
		clusters.push_back(cluster { s });
		cluster_normals.push_back(g.face_normal(s));
	}

	costs.resize(g.triangles.size(), FLT_MAX);
	for (int32_t i = 0; i < g.triangles.size(); i++) {
		face face;
		face.i = i;
		face.costs = costs.data();

		if (std::find(seeds.begin(), seeds.end(), i) != seeds.end())
			costs[i] = 0.0;

		queue.push(face);
	}

	// Precompute centroids and normals
	std::vector <glm::vec3> centroids;
	std::vector <glm::vec3> normals;

	for (int32_t i = 0; i < g.triangles.size(); i++) {
		centroids.push_back(g.centroid(i));
		normals.push_back(g.face_normal(i));
	}

	// Grow the charts
	std::unordered_set <int32_t> visited;
	while (queue.size() > 0) {
		face face = queue.top();
		queue.pop();

		// printf("Remaining: %zu; current is %d with %f\n", queue.size(), face.i, costs[face.i]);

		if (std::isinf(costs[face.i]))
			break;

		int32_t c_index     = face_to_cluster[face.i];

		cluster &c          = clusters[c_index];
		glm::vec3 &c_normal = cluster_normals[c_index];

		for (int32_t neighbor : dgraph.at(face.i)) {
			if (visited.count(neighbor) > 0)
				continue;

			glm::vec3 n_normal = normals[neighbor];
			float dc           = glm::length(centroids[face.i] - centroids[neighbor]);
			float dn           = std::max(1 - glm::dot(c_normal, n_normal), 0.0f);
			float new_cost     = costs[face.i] + dn * dc;

			if (new_cost < costs[neighbor]) {
				float size = c.size();
				glm::vec3 new_normal = (c_normal * size + n_normal)/(size + 1.0f);

				face_to_cluster[neighbor] = c_index;
				cluster_normals[c_index]  = new_normal;
				costs[neighbor]           = new_cost;
				clusters[c_index].push_back(neighbor);

				queue.rebuild();
				// TODO: should be done after the loop... (any visited updates, that is)
				// visited.insert(neighbor);
			}
		}

		visited.insert(face.i);
	}

	return clusters;
}

std::vector <std::vector <int32_t>> cluster_geometry(const geometry &g, const std::vector <int32_t> &seeds, int32_t iterations)
{
	// Make the dual graph
	auto egraph = g.make_edge_graph();
	auto dgraph = g.make_dual_graph(egraph);

	std::vector <std::vector <int32_t>> clusters;
	std::vector <int32_t> next_seeds = seeds;

	for (int32_t i = 0; i < iterations; i++) {
		clusters = cluster_once(g, dgraph, next_seeds);
		printf("Iteration %d: %zu clusters\n", i, clusters.size());
		if (i == iterations - 1)
			break;

		// Find the central faces for each cluster,
		// i.e. the face closest to the centroid
		std::vector <glm::vec3> centroids;
		for (const auto &c : clusters) {
			glm::vec3 centroid(0.0f);
			float wsum = 0.0f;
			for (int32_t f : c) {
				float w = g.area(f);
				centroid += g.centroid(f) * w;
				wsum += w;
			}

			centroid /= wsum;
			centroids.push_back(centroid);
		}

		// For each cluster, find the closest face
		next_seeds.clear();
		for (int32_t i = 0; i < clusters.size(); i++) {
			const auto &c = clusters[i];
			const auto &centroid = centroids[i];

			float min_dist = FLT_MAX;
			int32_t min_face = -1;

			for (int32_t f : c) {
				float dist = glm::length(centroid - g.centroid(f));
				if (dist < min_dist) {
					min_dist = dist;
					min_face = f;
				}
			}

			next_seeds.push_back(min_face);
		}
	}

	return clusters;
}

// Closest point caching acceleration structure and arguments
struct dev_cached_grid {
	glm::vec3 min;
	glm::vec3 max;
	glm::vec3 bin_size;

	glm::vec3 *vertices = nullptr;
	glm::uvec3 *triangles = nullptr;

	uint32_t *query_triangles = nullptr;
	uint32_t *index0 = nullptr;
	uint32_t *index1 = nullptr;

	uint32_t vertex_count;
	uint32_t triangle_count;
	uint32_t resolution;
};

struct cached_grid {
	geometry ref;

	glm::vec3 min;
	glm::vec3 max;

	uint32_t resolution;
	glm::vec3 bin_size;

	using query_bin = std::vector <uint32_t>;
	std::vector <query_bin> overlapping_triangles;
	std::vector <query_bin> query_triangles;

	dev_cached_grid dev_cas;

	// Construct from mesh
	cached_grid(const geometry &, uint32_t);

	uint32_t to_index(const glm::ivec3 &bin) const;
	uint32_t to_index(const glm::vec3 &p) const;

	std::unordered_set <uint32_t> closest_triangles(const glm::vec3 &p) const;

	bool precache_query(const glm::vec3 &p);
	// float precache_query(const std::vector <glm::vec3> &points);

	float precache_query_vector(const torch::Tensor &);
	float precache_query_vector_device(const torch::Tensor &);

	// Returns closest point, barycentric coordinates, distance, and triangle index
	std::tuple <glm::vec3, glm::vec3, float, uint32_t> query(const glm::vec3 &p) const;

	void query_vector(const torch::Tensor &,
		torch::Tensor &,
		torch::Tensor &,
		torch::Tensor &,
		torch::Tensor &) const;

	void query_vector_device(const torch::Tensor &,
		torch::Tensor &,
		torch::Tensor &,
		torch::Tensor &,
		torch::Tensor &) const;

	void precache_device();
};

// Bounding box of mesh
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

// Closest point on triangle
__forceinline__ __host__ __device__
static void triangle_closest_point(const glm::vec3 &v0, const glm::vec3 &v1, const glm::vec3 &v2, const glm::vec3 &p, glm::vec3 *closest, glm::vec3 *bary, float *distance)
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

// Cached acceleration structure
cached_grid::cached_grid(const geometry &ref_, uint32_t resolution_)
		: ref(ref_), resolution(resolution_)
{
	uint32_t size = resolution * resolution * resolution;
	overlapping_triangles.resize(size);
	query_triangles.resize(size);

	// Put triangles into bins
	std::tie(max, min) = bound(ref);
	glm::vec3 extent = { max.x - min.x, max.y - min.y, max.z - min.z };
	bin_size = extent / (float) resolution;

	for (size_t i = 0; i < ref.triangles.size(); i++) {
		const glm::uvec3 &triangle = ref.triangles[i];

		// Triangle belongs to all bins it intersects
		glm::vec3 v0 = ref.vertices[triangle.x];
		glm::vec3 v1 = ref.vertices[triangle.y];
		glm::vec3 v2 = ref.vertices[triangle.z];

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
}

uint32_t cached_grid::to_index(const glm::ivec3 &bin) const
{
	return bin.x + bin.y * resolution + bin.z * resolution * resolution;
}

uint32_t cached_grid::to_index(const glm::vec3 &p) const
{
	glm::vec3 bin_flt = glm::clamp((p - min) / bin_size, glm::vec3(0), glm::vec3(resolution - 1));
	glm::ivec3 bin = glm::ivec3(bin_flt);
	return to_index(bin);
}

// Find the complete set of query triangles for a point
std::unordered_set <uint32_t> cached_grid::closest_triangles(const glm::vec3 &p) const
{
	// Get the current bin
	glm::vec3 bin_flt = glm::clamp((p - min) / bin_size, glm::vec3(0), glm::vec3(resolution - 1));
	glm::ivec3 bin = glm::ivec3(bin_flt);
	uint32_t bin_index = to_index(p);

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

			uint32_t next_index = to_index(next);
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
bool cached_grid::precache_query(const glm::vec3 &p)
{
	// Check if the bin is already cached
	uint32_t bin_index = to_index(p);
	if (!query_triangles[bin_index].empty())
		return false;

	// Otherwise, load the bin
	auto set = closest_triangles(p);
	query_triangles[bin_index] = query_bin(set.begin(), set.end());
	return true;
}

float cached_grid::precache_query_vector(const torch::Tensor &sources)
{
	// Ensure device and type and size
	assert(sources.dim() == 2 && sources.size(1) == 3);
	assert(sources.device().is_cpu());
	assert(sources.dtype() == torch::kFloat32);

	size_t size = sources.size(0);
	size_t any_count = 0;

	glm::vec3 *sources_ptr = (glm::vec3 *) sources.data_ptr <float> ();

	// #pragma omp parallel for reduction(+:any_count)
	for (uint32_t i = 0; i < size; i++) {
		any_count += precache_query(sources_ptr[i]);
	}

	return (float) any_count / (float) size;
}

float cached_grid::precache_query_vector_device(const torch::Tensor &sources)
{
	// Ensure device and type and size
	assert(sources.dim() == 2 && sources.size(1) == 3);
	assert(sources.device().is_cuda());
	assert(sources.dtype() == torch::kFloat32);

	size_t size = sources.size(0);
	size_t any_count = 0;

	glm::vec3 *sources_ptr_device = (glm::vec3 *) sources.data_ptr <float> ();
	glm::vec3 *sources_ptr = new glm::vec3[size];
	cudaMemcpy(sources_ptr, sources_ptr_device, size * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	// TODO: cuda check
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cerr << "(precache) CUDA error: " << cudaGetErrorString(err) << std::endl;
		exit(1);
	}

	// #pragma omp parallel for reduction(+:any_count)
	for (uint32_t i = 0; i < size; i++) {
		any_count += precache_query(sources_ptr[i]);
	}

	delete[] sources_ptr;
	return (float) any_count / (float) size;
}

// Single point query
std::tuple <glm::vec3, glm::vec3, float, uint32_t> cached_grid::query(const glm::vec3 &p) const
{
	// Assuming the point is precached already
	uint32_t bin_index = to_index(p);
	assert(bin_index < overlapping_triangles.size());

	const std::vector <uint32_t> &bin = query_triangles[bin_index];
	assert(bin.size() > 0);

	glm::vec3 closest = p;
	glm::vec3 barycentric;
	float distance = FLT_MAX;
	uint32_t triangle_index = 0;

	for (uint32_t index : bin) {
		const glm::uvec3 &tri = ref.triangles[index];
		glm::vec3 a = ref.vertices[tri[0]];
		glm::vec3 b = ref.vertices[tri[1]];
		glm::vec3 c = ref.vertices[tri[2]];

		glm::vec3 point;
		glm::vec3 bary;
		float dist;
		triangle_closest_point(a, b, c, p, &point, &bary, &dist);

		if (dist < distance) {
			closest = point;
			barycentric = bary;
			distance = dist;
			triangle_index = index;
		}
	}

	return std::make_tuple(closest, barycentric, distance, triangle_index);
}

void cached_grid::query_vector(const torch::Tensor &sources,
		torch::Tensor &closest,
		torch::Tensor &bary,
		torch::Tensor &distance,
		torch::Tensor &triangle_index) const
{
	// Check types, devices and sizes
	assert(sources.dim() == 2 && sources.size(1) == 3);
	assert(closest.dim() == 2 && closest.size(1) == 3);
	assert(bary.dim() == 2 && bary.size(1) == 3);
	assert(distance.dim() == 1);
	assert(triangle_index.dim() == 1);

	assert(sources.device().is_cpu());
	assert(closest.device().is_cpu());
	assert(bary.device().is_cpu());
	assert(distance.device().is_cpu());
	assert(triangle_index.device().is_cpu());

	assert(sources.dtype() == torch::kFloat32);
	assert(closest.dtype() == torch::kFloat32);
	assert(bary.dtype() == torch::kFloat32);
	assert(distance.dtype() == torch::kFloat32);
	assert(triangle_index.dtype() == torch::kInt32);

	assert(sources.size(0) == closest.size(0));
	assert(sources.size(0) == bary.size(0));
	assert(sources.size(0) == distance.size(0));
	assert(sources.size(0) == triangle_index.size(0));

	// Assuming all elements are precached already
	// and that the dst vector is already allocated
	size_t size = sources.size(0);

	glm::vec3 *sources_ptr = (glm::vec3 *) sources.data_ptr <float> ();
	glm::vec3 *closest_ptr = (glm::vec3 *) closest.data_ptr <float> ();
	glm::vec3 *bary_ptr = (glm::vec3 *) bary.data_ptr <float> ();
	float *distance_ptr = distance.data_ptr <float> ();
	int32_t *triangle_index_ptr = triangle_index.data_ptr <int32_t> ();

	#pragma omp parallel for
	for (uint32_t i = 0; i < size; i++) {
		uint32_t bin_index = to_index(sources_ptr[i]);
		auto [c, b, d, t] = query(sources_ptr[i]);

		closest_ptr[i] = c;
		bary_ptr[i] = b;
		distance_ptr[i] = d;
		triangle_index_ptr[i] = t;
	}
}

struct closest_point_kinfo {
	glm::vec3 *points;
	glm::vec3 *closest;
	glm::vec3 *bary;
	float     *distances;
	int32_t   *triangles;

	int32_t   point_count;
};

__global__
static void closest_point_kernel(dev_cached_grid cas, closest_point_kinfo kinfo)
{
	uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t stride = blockDim.x * gridDim.x;

	for (uint32_t i = tid; i < kinfo.point_count; i += stride) {
		glm::vec3 point = kinfo.points[i];
		glm::vec3 closest;
		uint32_t triangle;

		glm::vec3 bin_flt = glm::clamp((point - cas.min) / cas.bin_size,
				glm::vec3(0), glm::vec3(cas.resolution - 1));

		glm::ivec3 bin = glm::ivec3(bin_flt);
		uint32_t bin_index = bin.x + bin.y * cas.resolution + bin.z * cas.resolution * cas.resolution;

		uint32_t index0 = cas.index0[bin_index];
		uint32_t index1 = cas.index1[bin_index];

		glm::vec3 min_bary;
		float min_distance = FLT_MAX;

		for (uint32_t j = index0; j < index1; j++) {
			uint32_t triangle_index = cas.query_triangles[j];
			glm::uvec3 tri = cas.triangles[triangle_index];

			glm::vec3 v0 = cas.vertices[tri.x];
			glm::vec3 v1 = cas.vertices[tri.y];
			glm::vec3 v2 = cas.vertices[tri.z];

			// TODO: prune triangles that are too far away (based on bbox)?
			glm::vec3 candidate;
			glm::vec3 bary;
			float distance;

			triangle_closest_point(v0, v1, v2, point, &candidate, &bary, &distance);

			if (distance < min_distance) {
				closest = candidate;
				min_bary = bary;
				min_distance = distance;
				triangle = triangle_index;
			}
		}

		// TODO: barycentrics as well...
		kinfo.bary[i] = min_bary;
		kinfo.closest[i] = closest;
		kinfo.distances[i] = min_distance;
		kinfo.triangles[i] = triangle;
	}
}

void cached_grid::query_vector_device(const torch::Tensor &sources,
		torch::Tensor &closest,
		torch::Tensor &bary,
		torch::Tensor &distance,
		torch::Tensor &triangle_index) const
{
	// Check types, devices and sizes
	assert(sources.dim() == 2 && sources.size(1) == 3);
	assert(closest.dim() == 2 && closest.size(1) == 3);
	assert(bary.dim() == 2 && bary.size(1) == 3);
	assert(distance.dim() == 1);
	assert(triangle_index.dim() == 1);

	assert(sources.device().is_cuda());
	assert(closest.device().is_cuda());
	assert(bary.device().is_cuda());
	assert(distance.device().is_cuda());
	assert(triangle_index.device().is_cuda());

	assert(sources.dtype() == torch::kFloat32);
	assert(closest.dtype() == torch::kFloat32);
	assert(bary.dtype() == torch::kFloat32);
	assert(distance.dtype() == torch::kFloat32);
	assert(triangle_index.dtype() == torch::kInt32);

	assert(sources.size(0) == closest.size(0));
	assert(sources.size(0) == bary.size(0));
	assert(sources.size(0) == distance.size(0));
	assert(sources.size(0) == triangle_index.size(0));

	// Assuming all elements are precached already (in device as well)
	// and that the dst vector is already allocated
	size_t size = sources.size(0);

	closest_point_kinfo kinfo;
	kinfo.points = (glm::vec3 *) sources.data_ptr <float> ();
	kinfo.closest = (glm::vec3 *) closest.data_ptr <float> ();
	kinfo.bary = (glm::vec3 *) bary.data_ptr <float> ();
	kinfo.distances = distance.data_ptr <float> ();
	kinfo.triangles = triangle_index.data_ptr <int32_t> ();
	kinfo.point_count = size;

	dim3 block(256);
	dim3 grid((size + block.x - 1) / block.x);

	closest_point_kernel <<<grid, block>>> (dev_cas, kinfo);
	cudaDeviceSynchronize();
	// TODO: cuda check
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
		exit(1);
	}
}

void cached_grid::precache_device()
{
	dev_cas.min = min;
	dev_cas.max = max;
	dev_cas.bin_size = bin_size;

	dev_cas.resolution = resolution;
	dev_cas.vertex_count = ref.vertices.size();
	dev_cas.triangle_count = ref.triangles.size();

	std::vector <uint32_t> linear_query_triangles;
	std::vector <uint32_t> index0;
	std::vector <uint32_t> index1;

	uint32_t size = resolution * resolution * resolution;
	uint32_t offset = 0;

	for (uint32_t i = 0; i < size; i++) {
		uint32_t query_size = query_triangles[i].size();
		linear_query_triangles.insert(linear_query_triangles.end(),
				query_triangles[i].begin(),
				query_triangles[i].end());

		index0.push_back(offset);
		index1.push_back(offset + query_size);
		offset += query_size;
	}

	// Free old memory
	if (dev_cas.vertices != nullptr)
		cudaFree(dev_cas.vertices);

	if (dev_cas.triangles != nullptr)
		cudaFree(dev_cas.triangles);

	if (dev_cas.query_triangles != nullptr)
		cudaFree(dev_cas.query_triangles);

	if (dev_cas.index0 != nullptr)
		cudaFree(dev_cas.index0);

	if (dev_cas.index1 != nullptr)
		cudaFree(dev_cas.index1);

	// Allocate new memory
	// TODO: no need to keep reallocating
	cudaMalloc(&dev_cas.vertices, sizeof(glm::vec3) * ref.vertices.size());
	cudaMalloc(&dev_cas.triangles, sizeof(glm::uvec3) * ref.triangles.size());

	cudaMalloc(&dev_cas.query_triangles, sizeof(uint32_t) * linear_query_triangles.size());
	cudaMalloc(&dev_cas.index0, sizeof(uint32_t) * index0.size());
	cudaMalloc(&dev_cas.index1, sizeof(uint32_t) * index1.size());

	cudaMemcpy(dev_cas.vertices, ref.vertices.data(), sizeof(glm::vec3) * ref.vertices.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_cas.triangles, ref.triangles.data(), sizeof(glm::uvec3) * ref.triangles.size(), cudaMemcpyHostToDevice);

	cudaMemcpy(dev_cas.query_triangles, linear_query_triangles.data(), sizeof(uint32_t) * linear_query_triangles.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_cas.index0, index0.data(), sizeof(uint32_t) * index0.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_cas.index1, index1.data(), sizeof(uint32_t) * index1.size(), cudaMemcpyHostToDevice);
}

struct cumesh {
	const glm::vec3 *vertices;
	const glm::uvec3 *triangles;

	uint32_t vertex_count = 0;
	uint32_t triangle_count = 0;
};

__global__
static void barycentric_closest_point_kernel(cumesh cm, closest_point_kinfo kinfo)
{
	uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t stride = blockDim.x * gridDim.x;

	for (uint32_t i = tid; i < kinfo.point_count; i += stride) {
		glm::vec3 point = kinfo.points[i];
		glm::vec3 closest;
		glm::vec3 barycentrics;
		uint32_t triangle;

		float min_distance = FLT_MAX;
		for (uint32_t j = 0; j < cm.triangle_count; j++) {
			glm::uvec3 tri = cm.triangles[j];

			glm::vec3 v0 = cm.vertices[tri.x];
			glm::vec3 v1 = cm.vertices[tri.y];
			glm::vec3 v2 = cm.vertices[tri.z];

			glm::vec3 candidate;
			glm::vec3 bary;
			float distance;

			triangle_closest_point(v0, v1, v2, point, &candidate, &bary, &distance);

			if (distance < min_distance) {
				min_distance = distance;
				closest = candidate;
				barycentrics = bary;
				triangle = j;
			}
		}

		kinfo.bary[i] = barycentrics;
		kinfo.triangles[i] = triangle;
	}
}

void barycentric_closest_points(const torch::Tensor &vertices, const torch::Tensor &triangles, const torch::Tensor &sources, torch::Tensor &bary, torch::Tensor &indices)
{
	// Check types, devices and sizes
	assert(vertices.dim() == 2 && vertices.size(1) == 3);
	assert(triangles.dim() == 2 && triangles.size(1) == 3);
	assert(sources.dim() == 2 && sources.size(1) == 3);
	assert(bary.dim() == 2 && bary.size(1) == 3);
	assert(indices.dim() == 1);

	assert(vertices.device().is_cuda());
	assert(triangles.device().is_cuda());
	assert(sources.device().is_cuda());
	assert(bary.device().is_cuda());
	assert(indices.device().is_cuda());

	assert(vertices.dtype() == torch::kFloat32);
	assert(triangles.dtype() == torch::kInt32);
	assert(sources.dtype() == torch::kFloat32);
	assert(bary.dtype() == torch::kFloat32);
	assert(indices.dtype() == torch::kInt32);

	assert(sources.size(0) == bary.size(0));
	assert(sources.size(0) == indices.size(0));

	// Assuming all elements are precached already (in device as well)
	// and that the dst vector is already allocated
	size_t size = sources.size(0);

	cumesh cm;

	cm.vertices = (glm::vec3 *) vertices.data_ptr <float> ();
	cm.triangles = (glm::uvec3 *) triangles.data_ptr <int32_t> ();
	cm.vertex_count = vertices.size(0);
	cm.triangle_count = triangles.size(0);

	closest_point_kinfo kinfo;

	kinfo.points = (glm::vec3 *) sources.data_ptr <float> ();
	kinfo.bary = (glm::vec3 *) bary.data_ptr <float> ();
	kinfo.triangles = (int32_t *) indices.data_ptr <int32_t> ();
	kinfo.point_count = size;

	dim3 block(256);
	dim3 grid((size + block.x - 1) / block.x);

	barycentric_closest_point_kernel <<<grid, block>>> (cm, kinfo);

	// TODO: cuda check
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
		exit(1);
	}
}

__global__
void laplacian_smooth_kernel(glm::vec3 *result, glm::vec3 *vertices, uint32_t *graph, uint32_t count, uint32_t max_adj, float factor)
{
	uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= count)
		return;

	glm::vec3 sum = glm::vec3(0.0f);
	uint32_t adj_count = graph[tid * max_adj];
	uint32_t *adj = graph + tid * max_adj + 1;
	for (uint32_t i = 0; i < adj_count; i++)
		sum += vertices[adj[i]];
	sum /= (float) adj_count;
	if (adj_count == 0)
		sum = vertices[tid];

	result[tid] = vertices[tid] + (sum - vertices[tid]) * factor;
}

struct vertex_graph {
	std::unordered_map <uint32_t, std::unordered_set <uint32_t>> graph;
	uint32_t *dev_graph;

	int32_t max;
	int32_t max_adj;

	// TODO: CUDA version...
	vertex_graph(const torch::Tensor &triangles) {
		assert(triangles.dim() == 2 && triangles.size(1) == 3);
		assert(triangles.dtype() == torch::kInt32);
		assert(triangles.device().is_cpu());

		uint32_t triangle_count = triangles.size(0);

		max = 0;
		for (uint32_t i = 0; i < triangle_count; i++) {
			int32_t v0 = triangles[i][0].item().to <int32_t> ();
			int32_t v1 = triangles[i][1].item().to <int32_t> ();
			int32_t v2 = triangles[i][2].item().to <int32_t> ();

			graph[v0].insert(v1);
			graph[v0].insert(v2);

			graph[v1].insert(v0);
			graph[v1].insert(v2);

			graph[v2].insert(v0);
			graph[v2].insert(v1);

			max = std::max(max, std::max(v0, std::max(v1, v2)));
		}

		max_adj = 0;
		for (auto &kv : graph)
			max_adj = std::max(max_adj, (int32_t) kv.second.size());

		// Allocate a device graph
		uint32_t graph_size = max * (max_adj + 1);
		cudaMalloc(&dev_graph, graph_size * sizeof(uint32_t));

		std::vector <uint32_t> host_graph(graph_size, 0);
		for (auto &kv : graph) {
			uint32_t i = kv.first;
			uint32_t j = 0;
			assert(i * max_adj + j < graph_size);
			host_graph[i * max_adj + j++] = kv.second.size();
			for (auto &adj : kv.second) {
				assert(i * max_adj + j < graph_size);
				host_graph[i * max_adj + j++] = adj;
			}
		}

		cudaMemcpy(dev_graph, host_graph.data(), graph_size * sizeof(uint32_t), cudaMemcpyHostToDevice);
	}

	torch::Tensor smooth(const torch::Tensor &vertices, float factor) const {
		assert(vertices.dim() == 2 && vertices.size(1) == 3);
		assert(vertices.dtype() == torch::kFloat32);
		assert(vertices.device().is_cpu());
		assert(max < vertices.size(0));

		torch::Tensor result = torch::zeros_like(vertices);

		glm::vec3 *v = (glm::vec3 *) vertices.data_ptr <float> ();
		glm::vec3 *r = (glm::vec3 *) result.data_ptr <float> ();

		for (uint32_t i = 0; i <= max; i++) {
			if (graph.find(i) == graph.end())
				continue;

			glm::vec3 sum = glm::vec3(0.0f);
			for (auto j : graph.at(i))
				sum += v[j];
			sum /= (float) graph.at(i).size();

			r[i] = (1.0f - factor) * v[i] + factor * sum;
		}

		return result;
	}

	torch::Tensor smooth_device(const torch::Tensor &vertices, float factor) const {
		assert(vertices.dim() == 2 && vertices.size(1) == 3);
		assert(vertices.dtype() == torch::kFloat32);
		assert(vertices.device().is_cuda());
		assert(max < vertices.size(0));

		torch::Tensor result = torch::zeros_like(vertices);

		glm::vec3 *v = (glm::vec3 *) vertices.data_ptr <float> ();
		glm::vec3 *r = (glm::vec3 *) result.data_ptr <float> ();

		dim3 block(256);
		dim3 grid((vertices.size(0) + block.x - 1) / block.x);

		laplacian_smooth_kernel <<<grid, block>>> (r, v, dev_graph, vertices.size(0), max_adj, factor);

		return result;
	}
};

__global__
void triangulate_shorted_kernel(const glm::vec3 *__restrict__ vertices, glm::ivec3 *__restrict__ triangles, size_t sample_rate)
{
	size_t i = blockIdx.x;
	size_t j = threadIdx.x;
	size_t k = threadIdx.y;

	size_t offset = i * sample_rate * sample_rate;

	size_t a = offset + j * sample_rate + k;
	size_t b = a + 1;
	size_t c = offset + (j + 1) * sample_rate + k;
	size_t d = c + 1;

	const glm::vec3 &va = vertices[a];
	const glm::vec3 &vb = vertices[b];
	const glm::vec3 &vc = vertices[c];
	const glm::vec3 &vd = vertices[d];

	float d0 = glm::distance(va, vd);
	float d1 = glm::distance(vb, vc);

	size_t toffset = 2 * i * (sample_rate - 1) * (sample_rate - 1);
	size_t tindex = toffset + 2 * (j * (sample_rate - 1) + k);
	if (d0 < d1) {
		triangles[tindex] = glm::ivec3(a, d, b);
		triangles[tindex + 1] = glm::ivec3(a, c, d);
	} else {
		triangles[tindex] = glm::ivec3(a, c, b);
		triangles[tindex + 1] = glm::ivec3(b, c, d);
	}
}

torch::Tensor triangulate_shorted(const torch::Tensor &vertices, size_t complex_count, size_t sample_rate)
{
	assert(vertices.dtype() == torch::kFloat32);
	assert(vertices.dim() == 2 && vertices.size(1) == 3);
	assert(vertices.is_cuda());

	long triangle_count = 2 * complex_count * (sample_rate - 1) * (sample_rate - 1);

	auto options = torch::TensorOptions()
		.dtype(torch::kInt32)
		.device(torch::kCUDA, 0);

	torch::Tensor out = torch::zeros({ triangle_count, 3 }, options);

	glm::vec3 *vertices_ptr = (glm::vec3 *) vertices.data_ptr <float> ();
	glm::ivec3 *out_ptr = (glm::ivec3 *) out.data_ptr <int32_t> ();

	dim3 block(sample_rate - 1, sample_rate - 1);
	dim3 grid(complex_count);

	triangulate_shorted_kernel <<< grid, block >>> (vertices_ptr, out_ptr, sample_rate);

	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
		exit(1);
	}

	return out;
}

__global__
void remapper_kernel(int32_t *map, glm::ivec3 *__restrict__ triangles, size_t size)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	size_t stride = blockDim.x * gridDim.x;

	for (size_t i = tid; i < size; i += stride) {
		triangles[i].x = map[triangles[i].x];
		triangles[i].y = map[triangles[i].y];
		triangles[i].z = map[triangles[i].z];
	}
}

struct remapper : std::unordered_map <int32_t, int32_t> {
	// CUDA map
	int32_t *dev_map = nullptr; // index -> value

	explicit remapper(const std::unordered_map <int32_t, int32_t> &map)
			: std::unordered_map <int32_t, int32_t> (map) {
		// Make sure that all values are present
		// i.e. from 1 to map size
		for (int32_t i = 0; i < map.size(); i++)
			assert(this->find(i) != this->end());

		// Allocate a device map
		std::vector <int32_t> host_map(map.size());
		for (auto &kv : map)
			host_map[kv.first] = kv.second;

		cudaMalloc(&dev_map, map.size() * sizeof(int32_t));
		cudaMemcpy(dev_map, host_map.data(), map.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
	}

	torch::Tensor remap(const torch::Tensor &indices) const {
		assert(indices.dtype() == torch::kInt32);
		assert(indices.dim() == 2 && indices.size(1) == 3);
		assert(indices.is_cpu());

		torch::Tensor out = torch::zeros_like(indices);
		int32_t *out_ptr = out.data_ptr <int32_t> ();
		int32_t *indices_ptr = indices.data_ptr <int32_t> ();

		for (int32_t i = 0; i < indices.numel(); i++) {
			auto it = this->find(indices_ptr[i]);
			assert(it != this->end());
			out_ptr[i] = it->second;
		}

		return out;
	}

	torch::Tensor remap_device(const torch::Tensor &indices) const {
		assert(indices.dtype() == torch::kInt32);
		assert(indices.dim() == 2 && indices.size(1) == 3);
		assert(indices.is_cuda());

		torch::Tensor out = indices.clone();
		glm::ivec3 *out_ptr = (glm::ivec3 *) out.data_ptr <int32_t> ();

		dim3 block(256);
		dim3 grid((indices.size(0) + block.x - 1) / block.x);

		remapper_kernel <<< grid, block >>> (dev_map, out_ptr, indices.size(0));

		cudaDeviceSynchronize();
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
			exit(1);
		}

		return out;
	}

	torch::Tensor scatter(const torch::Tensor &vertices) const {
		assert(vertices.dtype() == torch::kFloat32);
		assert(vertices.dim() == 2 && vertices.size(1) == 3);
		assert(vertices.is_cpu());

		torch::Tensor out = torch::zeros_like(vertices);
		glm::vec3 *out_ptr = (glm::vec3 *) out.data_ptr <float> ();
		glm::vec3 *vertices_ptr = (glm::vec3 *) vertices.data_ptr <float> ();

		for (int32_t i = 0; i < vertices.size(0); i++) {
			auto it = this->find(i);
			assert(it != this->end());
			out_ptr[i] = vertices_ptr[it->second];
		}

		return out;
	}
};

remapper generate_remapper(const torch::Tensor &complexes,
		std::unordered_map <int32_t, std::set <int32_t>> &cmap,
		int64_t vertex_count,
		int64_t sample_rate)
{
	assert(complexes.is_cpu());
	assert(complexes.dtype() == torch::kInt32);
	assert(complexes.dim() == 2 && complexes.size(1) == 4);

	std::vector <glm::ivec4> cs(complexes.size(0));
	// printf("cs: %lu\n", cs.size());
	int32_t *ptr = complexes.data_ptr <int32_t> ();
	std::memcpy(cs.data(), ptr, complexes.size(0) * sizeof(glm::ivec4));

	// Mappings
	std::unordered_map <int32_t, int32_t> rcmap;
	for (const auto &[k, v] : cmap) {
		for (const auto &i : v)
			rcmap[i] = k;
	}

	std::unordered_map <int32_t, int32_t> remap;
	// remapper remap;
	for (size_t i = 0; i < vertex_count; i++)
		remap[i] = i;

	for (const auto &[_, s] : cmap) {
		int32_t new_vertex = *s.begin();
		for (const auto &v : s)
			remap[v] = new_vertex;
	}

	std::unordered_map <ordered_pair, std::set <std::pair <int32_t, std::vector <int32_t>>>, ordered_pair::hash> bmap;

	for (int32_t i = 0; i < cs.size(); i++) {
		int32_t i00 = i * sample_rate * sample_rate;
		int32_t i10 = i00 + (sample_rate - 1);
		int32_t i01 = i00 + (sample_rate - 1) * sample_rate;
		int32_t i11 = i00 + (sample_rate * sample_rate - 1);

		int32_t c00 = rcmap[i00];
		int32_t c10 = rcmap[i10];
		int32_t c01 = rcmap[i01];
		int32_t c11 = rcmap[i11];

		ordered_pair p;
		bool reversed;

		std::vector <int32_t> b00_10;
		std::vector <int32_t> b00_01;
		std::vector <int32_t> b10_11;
		std::vector <int32_t> b01_11;

		// 00 -> 10
		reversed = p.from(c00, c10);
		if (reversed) {
			for (int32_t i = sample_rate - 2; i >= 1; i--)
				b00_10.push_back(i + i00);
		} else {
			for (int32_t i = 1; i <= sample_rate - 2; i++)
				b00_10.push_back(i + i00);
		}

		bmap[p].insert({ i, b00_10 });

		// 00 -> 01
		reversed = p.from(c00, c01);
		if (reversed) {
			for (int32_t i = sample_rate * (sample_rate - 2); i >= sample_rate; i -= sample_rate)
				b00_01.push_back(i + i00);
		} else {
			for (int32_t i = sample_rate; i <= sample_rate * (sample_rate - 2); i += sample_rate)
				b00_01.push_back(i + i00);
		}

		bmap[p].insert({ i, b00_01 });

		// 10 -> 11
		reversed = p.from(c10, c11);
		if (reversed) {
			for (int32_t i = sample_rate - 2; i >= 1; i--)
				b10_11.push_back(i * sample_rate + sample_rate - 1 + i00);
		} else {
			for (int32_t i = 1; i <= sample_rate - 2; i++)
				b10_11.push_back(i * sample_rate + sample_rate - 1 + i00);
		}

		bmap[p].insert({ i, b10_11 });

		// 01 -> 11
		reversed = p.from(c01, c11);
		if (reversed) {
			for (int32_t i = sample_rate - 2; i >= 1; i--)
				b01_11.push_back((sample_rate - 1) * sample_rate + i + i00);
		} else {
			for (int32_t i = 1; i <= sample_rate - 2; i++)
				b01_11.push_back((sample_rate - 1) * sample_rate + i + i00);
		}

		bmap[p].insert({ i, b01_11 });
	}

	for (const auto &[p, bs] : bmap) {
		const auto &ref = *bs.begin();
		for (const auto &b : bs) {
			for (int32_t i = 0; i < b.second.size(); i++) {
				remap[b.second[i]] = ref.second[i];
			}
		}
	}

	return remapper(remap);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
        py::class_ <geometry> (m, "geometry")
                .def(py::init <const torch::Tensor &, const torch::Tensor &> ())
                .def(py::init <const torch::Tensor &, const torch::Tensor &, const torch::Tensor &> ())
		.def("torched", &geometry::torched)
		.def_readonly("vertices", &geometry::vertices)
		.def_readonly("normals", &geometry::normals)
		.def_readonly("triangles", &geometry::triangles);

	py::class_ <cached_grid> (m, "cached_grid")
		.def(py::init <const geometry &, uint32_t> ())
		.def("precache_query", &cached_grid::precache_query_vector)
		.def("precache_query_device", &cached_grid::precache_query_vector_device)
		.def("precache_device", &cached_grid::precache_device)
		.def("query", &cached_grid::query_vector)
		.def("query_device", &cached_grid::query_vector_device);

	py::class_ <vertex_graph> (m, "vertex_graph")
		.def(py::init <const torch::Tensor &> ())
		.def("smooth", &vertex_graph::smooth)
		.def("smooth_device", &vertex_graph::smooth_device);

	py::class_ <remapper> (m, "remapper")
		.def("remap", &remapper::remap, "Remap indices")
		.def("remap_device", &remapper::remap_device, "Remap indices")
		.def("scatter", &remapper::scatter, "Scatter vertices");

	m.def("cluster_geometry", &cluster_geometry);
	m.def("barycentric_closest_points", &barycentric_closest_points);
	m.def("triangulate_shorted", &triangulate_shorted);
	m.def("generate_remapper", &generate_remapper, "Generate remapper");
}
