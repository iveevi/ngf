#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <stdio.h>
#include <unordered_set>
#include <vector>
#include <queue>

#include "common.hpp"

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

	bool good() const {
		return vertices != nullptr
			&& triangles != nullptr
			&& query_triangles != nullptr
			&& index0 != nullptr
			&& index1 != nullptr;
	}
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
	const glm::vec3 *__restrict__ points;
	glm::vec3       *__restrict__ closest;
	glm::vec3       *__restrict__ bary;
	float           *__restrict__ distances;
	int32_t         *__restrict__ triangles;

	int32_t                       point_count;
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

	// Make sure the dev cas is up to date
	if (!dev_cas.good())
		throw std::runtime_error("(query_device) dev_cas is not good");

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

	closest_point_kernel <<< grid, block >>> (dev_cas, kinfo);
	cudaDeviceSynchronize();
	// TODO: cuda check
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		throw std::runtime_error("(query_device) CUDA error: " + std::string(cudaGetErrorString(err)));
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

// TODO: const restrict...
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
	if (err != cudaSuccess)
		throw std::runtime_error("barycentric_closest_points: " + std::string(cudaGetErrorString(err)));
}

__global__
void laplacian_smooth_kernel(glm::vec3 *result, glm::vec3 *vertices, int32_t *graph, uint32_t count, uint32_t max_adj, float factor)
{
	int32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= count)
		return;

	glm::vec3 sum = glm::vec3(0.0f);
	int32_t adj_count = graph[tid * max_adj];
	int32_t *adj = graph + tid * max_adj + 1;
	for (int32_t i = 0; i < adj_count; i++)
		sum += vertices[adj[i]];
	sum /= (float) adj_count;
	if (adj_count == 0)
		sum = vertices[tid];

	result[tid] = vertices[tid] + (sum - vertices[tid]) * factor;
}

struct vertex_graph {
	std::unordered_map <int32_t, std::unordered_set <int32_t>> graph;

	int32_t *dev_graph = nullptr;
		
	int32_t max = 0;
	int32_t max_adj = 0;

	void allocate_device_graph() {
		max = 0;
		max_adj = 0;

		for (auto &kv : graph) {
			max_adj = std::max(max_adj, (int32_t) kv.second.size());
			max = std::max(max, kv.first);
		}

		// Allocate a device graph
		int32_t graph_size = max * (max_adj + 1);
		cudaMalloc(&dev_graph, graph_size * sizeof(int32_t));

		std::vector <uint32_t> host_graph(graph_size, 0);
		for (auto &kv : graph) {
			int32_t i = kv.first;
			int32_t j = 0;
			assert(i * max_adj + j < graph_size);
			host_graph[i * max_adj + j++] = kv.second.size();
			for (auto &adj : kv.second) {
				assert(i * max_adj + j < graph_size);
				host_graph[i * max_adj + j++] = adj;
			}
		}

		cudaMemcpy(dev_graph, host_graph.data(), graph_size * sizeof(int32_t), cudaMemcpyHostToDevice);
	}

	vertex_graph(const torch::Tensor &primitives) {
		assert(primitives.dim() == 2);
		assert(primitives.dtype() == torch::kInt32);
		assert(primitives.device().is_cpu());

		// if (primitives.size(1) == 3)
		// 	initialize_from_triangles(primitives);
		// else if (primitives.size(1) == 4)
		// 	initialize_from_quadrilaterals(primitives);
		if (primitives.size(1) == 4)
			initialize_from_quadrilaterals(primitives);
		else
			assert(false);
	}

	void initialize_from_triangles(const torch::Tensor &triangles) {
		assert(triangles.dim() == 2 && triangles.size(1) == 3);
		assert(triangles.dtype() == torch::kInt32);
		assert(triangles.device().is_cpu());

		int32_t triangle_count = triangles.size(0);

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
		}

		allocate_device_graph();
	}

	void initialize_from_quadrilaterals(const torch::Tensor &quads) {
		assert(quads.dim() == 2 && quads.size(1) == 4);
		assert(quads.dtype() == torch::kInt32);
		assert(quads.device().is_cpu());

		int32_t quad_count = quads.size(0);

		for (uint32_t i = 0; i < quad_count; i++) {
			int32_t v0 = quads[i][0].item().to <int32_t> ();
			int32_t v1 = quads[i][1].item().to <int32_t> ();
			int32_t v2 = quads[i][2].item().to <int32_t> ();
			int32_t v3 = quads[i][3].item().to <int32_t> ();

			graph[v0].insert(v1);
			graph[v0].insert(v3);

			graph[v1].insert(v0);
			graph[v1].insert(v2);

			graph[v2].insert(v1);
			graph[v2].insert(v3);

			graph[v3].insert(v0);
			graph[v3].insert(v2);
		}

		allocate_device_graph();
	}

	~vertex_graph() {
		if (dev_graph)
			cudaFree(dev_graph);
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

// torch::Tensor conformal_graph(const torch::Tensor &quads)
// {
// 	// Requries quads
// 	assert(quads.dim() == 2 && quads.size(1) == 4);
// 	assert(quads.dtype() == torch::kInt32);
// 	assert(quads.device().is_cpu());
//
// 	// First build the adjacency graph and record quad sharing
// 	std::unordered_map <int32_t, std::unordered_set <int32_t>> graph;
// 	std::unordered_map <int32_t, std::vector <glm::ivec4>> shared;
//
// 	uint32_t quad_count = quads.size(0);
//
// 	// Fill the graphs
// 	for (uint32_t i = 0; i < quad_count; i++) {
// 		int32_t v0 = quads[i][0].item().to <int32_t> ();
// 		int32_t v1 = quads[i][1].item().to <int32_t> ();
// 		int32_t v2 = quads[i][2].item().to <int32_t> ();
// 		int32_t v3 = quads[i][3].item().to <int32_t> ();
//
// 		graph[v0].insert(v1);
// 		graph[v0].insert(v3);
//
// 		graph[v1].insert(v0);
// 		graph[v1].insert(v2);
//
// 		graph[v2].insert(v1);
// 		graph[v2].insert(v3);
//
// 		graph[v3].insert(v0);
// 		graph[v3].insert(v2);
//
// 		glm::ivec4 quad = glm::ivec4(v0, v1, v2, v3);
// 		shared[v0].push_back(quad);
// 		shared[v1].push_back(quad);
// 		shared[v2].push_back(quad);
// 		shared[v3].push_back(quad);
// 	}
//
// 	// Collect all crossings; only vertices with valence 4, with opposite vertices not sharing quads
// 	std::vector <glm::ivec4> crossings;
//
// 	for (auto &kv : graph) {
// 		std::unordered_set <int32_t> &adj = kv.second;
// 		if (adj.size() != 4)
// 			continue;
//
// 		// Find the opposite vertices
// 		auto it = adj.begin();
// 		int32_t a = *it++;
// 		int32_t b = *it++;
// 		int32_t c = *it++;
// 		int32_t d = *it++;
//
// 		// printf("a: %d, b: %d, c: %d, d: %d\n", a, b, c, d);
// 		// printf("shared quads:\n");
// 		// for (auto &quad : shared[kv.first])
// 		// 	printf("  > %d %d %d %d\n", quad[0], quad[1], quad[2], quad[3]);
//
// 		// Expect that a-b and c-d are opposite (non-sharing), swap if otherwise
// 		int32_t opp_a = b;
//
// 		auto in_shared = [&shared](int32_t a, int32_t b) {
// 			for (auto &quad : shared[a])
// 				if (quad[0] == b || quad[1] == b || quad[2] == b || quad[3] == b)
// 					return true;
// 			return false;
// 		};
//
// 		if (in_shared(a, opp_a))
// 			opp_a = c;
// 		if (in_shared(a, opp_a))
// 			opp_a = d;
//
// 		assert(!in_shared(a, opp_a));
//
// 		// Then find the other two opposite vertices
// 		int32_t other_a = b;
// 		if (other_a == a || other_a == opp_a)
// 			other_a = c;
// 		if (other_a == a || other_a == opp_a)
// 			other_a = d;
//
// 		int32_t other_opp_a = b;
// 		if (other_opp_a == opp_a || other_opp_a == a || other_opp_a == other_a)
// 			other_opp_a = c;
// 		if (other_opp_a == opp_a || other_opp_a == a || other_opp_a == other_a)
// 			other_opp_a = d;
//
// 		assert(!in_shared(other_a, other_opp_a));
// 		assert(other_a != a && other_a != opp_a);
//
// 		// Finally, add the crossing
// 		crossings.push_back(glm::ivec4(a, opp_a, other_a, other_opp_a));
// 	}
//
// 	// Construct tensor of crossings
// 	torch::Tensor result = torch::zeros({ (long) crossings.size(), 4 }, torch::kInt32);
//
// 	for (uint32_t i = 0; i < crossings.size(); i++) {
// 		result[i][0] = crossings[i][0];
// 		result[i][1] = crossings[i][1];
// 		result[i][2] = crossings[i][2];
// 		result[i][3] = crossings[i][3];
// 	}
//
// 	return result;
// }

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
void remapper_kernel(const int32_t *__restrict__ map, glm::ivec3 *__restrict__ triangles, size_t size)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	size_t stride = blockDim.x * gridDim.x;

	for (size_t i = tid; i < size; i += stride) {
		triangles[i].x = map[triangles[i].x];
		triangles[i].y = map[triangles[i].y];
		triangles[i].z = map[triangles[i].z];
	}
}

__global__
void scatter_kernel(const int32_t *__restrict__ map, const glm::vec3 *__restrict__ data, glm::vec3 *__restrict__ dst, size_t size)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	size_t stride = blockDim.x * gridDim.x;

	for (size_t i = tid; i < size; i += stride) {
		int32_t index = map[i];
		dst[i] = data[index];
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

	torch::Tensor scatter_device(const torch::Tensor &vertices) const {
		assert(vertices.dtype() == torch::kFloat32);
		assert(vertices.dim() == 2 && vertices.size(1) == 3);
		assert(vertices.is_cuda());

		torch::Tensor out = torch::zeros_like(vertices);
		glm::vec3 *out_ptr = (glm::vec3 *) out.data_ptr <float> ();
		glm::vec3 *vertices_ptr = (glm::vec3 *) vertices.data_ptr <float> ();

		dim3 block(256);
		dim3 grid((vertices.size(0) + block.x - 1) / block.x);

		scatter_kernel <<< grid, block >>> (dev_map, vertices_ptr, out_ptr, vertices.size(0));

		cudaDeviceSynchronize();
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
			exit(1);
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
		.def("deduplicate", &geometry::deduplicate)
		.def("torched", &geometry::torched)
		.def_readonly("vertices", &geometry::vertices)
		.def_readonly("normals", &geometry::normals)
		.def_readonly("triangles", &geometry::triangles)
		.def("__repr__", [](const geometry &g) {
			return "geometry(vertices=" + std::to_string(g.vertices.size())
				+ ", triangles=" + std::to_string(g.triangles.size()) + ")";
		});

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
		.def("scatter", &remapper::scatter, "Scatter vertex data")
		.def("scatter_device", &remapper::scatter_device, "Scatter vertex data");

	// m.def("conformal_graph", &conformal_graph);
	m.def("cluster_geometry", &cluster_geometry);
	m.def("barycentric_closest_points", &barycentric_closest_points);
	m.def("triangulate_shorted", &triangulate_shorted);
	m.def("generate_remapper", &generate_remapper, "Generate remapper");
}
