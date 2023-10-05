#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>

#include <torch/extension.h>

struct geometry {
	std::vector <glm::vec3> vertices;
        std::vector <glm::vec3> normals;
	std::vector <glm::uvec3> triangles;

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
};

// Closest point caching acceleration structure and arguments
struct dev_cas_grid {
	glm::vec3 min;
	glm::vec3 max;
	glm::vec3 bin_size;

	glm::vec3 *vertices;
	glm::uvec3 *triangles;

	uint32_t *query_triangles;
	uint32_t *index0;
	uint32_t *index1;

	uint32_t vertex_count;
	uint32_t triangle_count;
	uint32_t resolution;
};

struct cas_grid {
	geometry ref;

	glm::vec3 min;
	glm::vec3 max;

	uint32_t resolution;
	glm::vec3 bin_size;

	using query_bin = std::vector <uint32_t>;
	std::vector <query_bin> overlapping_triangles;
	std::vector <query_bin> query_triangles;

	dev_cas_grid dev_cas;

	// Construct from mesh
	cas_grid(const geometry &, uint32_t);

	uint32_t to_index(const glm::ivec3 &bin) const;
	uint32_t to_index(const glm::vec3 &p) const;

	std::unordered_set <uint32_t> closest_triangles(const glm::vec3 &p) const;

	bool precache_query(const glm::vec3 &p);
	// float precache_query(const std::vector <glm::vec3> &points);

	float precache_query_vector(const torch::Tensor &);

	// Returns closest point, barycentric coordinates, distance, and triangle index
	std::tuple <glm::vec3, glm::vec3, float, uint32_t> query(const glm::vec3 &p) const;

	/*
	void query(const std::vector <glm::vec3> &,
			std::vector <glm::vec3> &,
			std::vector <glm::vec3> &,
			std::vector <float> &,
			std::vector <uint32_t> &) const; */

	void query_vector(const torch::Tensor &,
		torch::Tensor &,
		torch::Tensor &,
		torch::Tensor &,
		torch::Tensor &) const;

	void precache_device();
	// void query_device(closest_point_kinfo kinfo);
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
cas_grid::cas_grid(const geometry &ref_, uint32_t resolution_)
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

uint32_t cas_grid::to_index(const glm::ivec3 &bin) const
{
	return bin.x + bin.y * resolution + bin.z * resolution * resolution;
}

uint32_t cas_grid::to_index(const glm::vec3 &p) const
{
	glm::vec3 bin_flt = glm::clamp((p - min) / bin_size, glm::vec3(0), glm::vec3(resolution - 1));
	glm::ivec3 bin = glm::ivec3(bin_flt);
	return to_index(bin);
}

// Find the complete set of query triangles for a point
std::unordered_set <uint32_t> cas_grid::closest_triangles(const glm::vec3 &p) const
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
bool cas_grid::precache_query(const glm::vec3 &p)
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

// Precache a collection of query points

/*
float cas_grid::precache_query(const std::vector <glm::vec3> &points)
{
	uint32_t any_count = 0;
	for (const glm::vec3 &p : points)
		any_count += precache_query(p);

	return (float) any_count / (float) points.size();
} */

float cas_grid::precache_query_vector(const torch::Tensor &sources)
{
	// Ensure device and type and size
	assert(sources.dim() == 2 && sources.size(1) == 3);
	assert(sources.device().is_cpu());
	assert(sources.dtype() == torch::kFloat32);

	size_t size = sources.size(0);
	size_t any_count = 0;

	glm::vec3 *sources_ptr = (glm::vec3 *) sources.data_ptr <float> ();

	#pragma omp parallel for reduction(+:any_count)
	for (uint32_t i = 0; i < size; i++) {
		any_count += precache_query(sources_ptr[i]);
	}

	return (float) any_count / (float) size;
}

// Single point query
std::tuple <glm::vec3, glm::vec3, float, uint32_t> cas_grid::query(const glm::vec3 &p) const
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

// Host-side query

/*
void cas_grid::query(const std::vector <glm::vec3> &sources,
		std::vector <glm::vec3> &closest,
		std::vector <glm::vec3> &bary,
		std::vector <float> &distance,
		std::vector <uint32_t> &triangle_index) const
{
	// Assuming all elements are precached already
	// and that the dst vector is already allocated
	#pragma omp parallel for
	for (uint32_t i = 0; i < sources.size(); i++) {
		uint32_t bin_index = to_index(sources[i]);
		auto [c, b, d, t] = query(sources[i]);

		closest[i] = c;
		bary[i] = b;
		distance[i] = d;
		triangle_index[i] = t;
	}
} */

void cas_grid::query_vector(const torch::Tensor &sources,
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

void cas_grid::precache_device()
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

/*
void cas_grid::query_device(closest_point_kinfo kinfo)
{
	dim3 block(256);
	dim3 grid((kinfo.point_count + block.x - 1) / block.x);

	closest_point_kernel <<< grid, block >>> (dev_cas, kinfo);
} */

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
        py::class_ <geometry> (m, "geometry")
                .def(py::init <const torch::Tensor &, const torch::Tensor &, const torch::Tensor &> ())
		.def_readonly("vertices", &geometry::vertices)
		.def_readonly("normals", &geometry::normals)
		.def_readonly("triangles", &geometry::triangles);

	py::class_ <cas_grid> (m, "cas_grid")
		.def(py::init <const geometry &, uint32_t> ())
		.def("precache_query", &cas_grid::precache_query_vector)
		.def("query", &cas_grid::query_vector);
}