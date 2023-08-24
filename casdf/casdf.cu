#include <queue>

#include "casdf.hpp"
#include "microlog.h"

// Bounding box of mesh
// TODO: rearrange into cpp and cuda files
static std::pair <glm::vec3, glm::vec3> bound(const Mesh &mesh)
{
	glm::vec3 max = mesh.vertices[0];
	glm::vec3 min = mesh.vertices[0];
	for (const glm::vec3 &v : mesh.vertices) {
		max = glm::max(max, v);
		min = glm::min(min, v);
	}

	return { max, min };
}

// Closest point on triangle
__forceinline__ __host__ __device__
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

__forceinline__ __device__
glm::uvec3 pcg(glm::uvec3 v)
{
	v = v * 1664525u + 1013904223u;
	v.x += v.y * v.z;
	v.y += v.z * v.x;
	v.z += v.x * v.y;
	v ^= v >> 16u;
	v.x += v.y * v.z;
	v.y += v.z * v.x;
	v.z += v.x * v.y;
	return v;
}

__forceinline__ __device__
glm::vec3 pcg(glm::vec3 v)
{
	glm::uvec3 u = *(glm::uvec3 *) &v;
	u = pcg(u);
	u &= glm::uvec3(0x007fffffu);
	u |= glm::uvec3(0x3f800000u);
	return *(glm::vec3 *) &u;
}

// GPU kernels
__global__
static void brute_closest_point_kernel(cumesh cu_mesh, closest_point_kinfo kinfo)
{
	uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t stride = blockDim.x * gridDim.x;

	for (uint32_t i = tid; i < kinfo.point_count; i += stride) {
		glm::vec3 point = kinfo.points[i];
		glm::vec3 closest;
		glm::vec3 barycentrics;
		uint32_t triangle;

		float min_distance = FLT_MAX;
		for (uint32_t j = 0; j < cu_mesh.triangle_count; j++) {
			glm::uvec3 tri = cu_mesh.triangles[j];

			glm::vec3 v0 = cu_mesh.vertices[tri.x];
			glm::vec3 v1 = cu_mesh.vertices[tri.y];
			glm::vec3 v2 = cu_mesh.vertices[tri.z];

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

		kinfo.closest[i] = closest;
		kinfo.bary[i] = barycentrics;
		kinfo.triangles[i] = triangle;
	}
}

__global__
static void closest_point_kernel(dev_cas_grid cas, closest_point_kinfo kinfo)
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
				min_distance = distance;
				closest = candidate;
				triangle = triangle_index;
			}
		}

		kinfo.closest[i] = closest;
		kinfo.triangles[i] = triangle;
	}
}

__global__
void sample_kernel(sample_result result, cumesh mesh, float time)
{
	uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t stride = blockDim.x * gridDim.x;

	for (uint32_t i = tid; i < result.point_count; i += stride) {
		glm::uvec3 seed0 = mesh.triangles[i % mesh.triangle_count];
		glm::vec3 seed1 = mesh.vertices[i % mesh.vertex_count];

		uint32_t tint = *(uint32_t *) &time;
		seed0.x ^= tint;
		seed0.y ^= tint;
		seed0.z ^= tint;

		seed1.x *= __sinf(time);
		seed1.y *= __sinf(time);
		seed1.z *= __sinf(time);

		glm::uvec3 tri = pcg(seed0);
		glm::vec3 bary = pcg(seed1);

		uint32_t tindex = tri.x % mesh.triangle_count;
		tri = mesh.triangles[tindex];

		glm::vec3 v0 = mesh.vertices[tri.x];
		glm::vec3 v1 = mesh.vertices[tri.y];
		glm::vec3 v2 = mesh.vertices[tri.z];

		bary = glm::normalize(bary);
		bary.x = 1.0f - bary.y - bary.z;

		result.points[i] = bary.x * v0 + bary.y * v1 + bary.z * v2;
		result.barys[i] = bary;
		result.triangles[i] = tri;
	}
}

// Allocate cumeshes
cumesh cumesh_alloc(const Mesh &mesh)
{
	cumesh cu_mesh;
	cu_mesh.vertex_count = mesh.vertices.size();
	cu_mesh.triangle_count = mesh.triangles.size();

	cudaMalloc(&cu_mesh.vertices, sizeof(glm::vec3) * cu_mesh.vertex_count);
	cudaMalloc(&cu_mesh.triangles, sizeof(glm::uvec3) * cu_mesh.triangle_count);

	cudaMemcpy(cu_mesh.vertices, mesh.vertices.data(),
		sizeof(glm::vec3) * cu_mesh.vertex_count, cudaMemcpyHostToDevice);

	cudaMemcpy(cu_mesh.triangles, mesh.triangles.data(),
		sizeof(glm::uvec3) * cu_mesh.triangle_count, cudaMemcpyHostToDevice);

	return cu_mesh;
}

void cumesh_reload(cumesh cu_mesh, const Mesh &mesh)
{
	if (cu_mesh.vertex_count != mesh.vertices.size()) {
		cudaFree(cu_mesh.vertices);
		cudaMalloc(&cu_mesh.vertices, sizeof(glm::vec3) * mesh.vertices.size());
		cu_mesh.vertex_count = mesh.vertices.size();
	}

	if (cu_mesh.triangle_count != mesh.triangles.size()) {
		cudaFree(cu_mesh.triangles);
		cudaMalloc(&cu_mesh.triangles, sizeof(glm::uvec3) * mesh.triangles.size());
		cu_mesh.triangle_count = mesh.triangles.size();
	}

	cudaMemcpy(cu_mesh.vertices, mesh.vertices.data(),
		sizeof(glm::vec3) * mesh.vertices.size(), cudaMemcpyHostToDevice);

	cudaMemcpy(cu_mesh.triangles, mesh.triangles.data(),
		sizeof(glm::uvec3) * mesh.triangles.size(), cudaMemcpyHostToDevice);
}

// Allocating sample information
sample_result sample_result_alloc(uint32_t point_count)
{
	sample_result result;
	result.point_count = point_count;

	cudaMalloc(&result.points, sizeof(glm::vec3) * point_count);
	cudaMalloc(&result.barys, sizeof(glm::vec3) * point_count);
	cudaMalloc(&result.triangles, sizeof(glm::uvec3) * point_count);

	return result;
}

void sample(sample_result result, cumesh mesh, float time)
{
	dim3 block(128);
	dim3 grid((result.point_count + block.x - 1) / block.x);
	sample_kernel <<< grid, block >>> (result, mesh, time);
}

// Allocate kinfo
closest_point_kinfo closest_point_kinfo_alloc(uint32_t point_count)
{
	closest_point_kinfo kinfo;

	cudaMalloc(&kinfo.points, point_count * sizeof(glm::vec3));
	cudaMalloc(&kinfo.closest, point_count * sizeof(glm::vec3));
	cudaMalloc(&kinfo.bary, point_count * sizeof(glm::vec3));
	cudaMalloc(&kinfo.triangles, point_count * sizeof(uint32_t));

	kinfo.point_count = point_count;

	return kinfo;
}

// Brute force closest point
void brute_closest_point(cumesh cu_mesh, closest_point_kinfo kinfo)
{
	dim3 block(128);
	dim3 grid((kinfo.point_count + block.x - 1) / block.x);
	brute_closest_point_kernel <<< grid, block >>> (cu_mesh, kinfo);
}

// Construct from mesh
cas_grid::cas_grid(const Mesh &ref_, uint32_t resolution_)
		: ref(ref_), resolution(resolution_)
{
	printf("Constructing cas grid\n");
	uint32_t size = resolution * resolution * resolution;
	overlapping_triangles.resize(size);
	query_triangles.resize(size);

	// Put triangles into bins
	std::tie(max, min) = bound(ref);
	glm::vec3 extent = { max.x - min.x, max.y - min.y, max.z - min.z };
	printf("extent: %f %f %f\n", extent.x, extent.y, extent.z);
	bin_size = extent / (float) resolution;
	printf("min: %f %f %f\n", min.x, min.y, min.z);
	printf("max: %f %f %f\n", max.x, max.y, max.z);
	printf("extent: %f %f %f\n", extent.x, extent.y, extent.z);
	printf("resolution: %d\n", resolution);
	printf("expected extent: %f %f %f\n", (max.x - min.x) / resolution, (max.y - min.y) / resolution, (max.z - min.z) / resolution);

	for (size_t i = 0; i < ref.triangles.size(); i++) {
		const Triangle &triangle = ref.triangles[i];

		// Triangle belongs to all bins it intersects
		glm::vec3 v0 = ref.vertices[triangle[0]];
		glm::vec3 v1 = ref.vertices[triangle[1]];
		glm::vec3 v2 = ref.vertices[triangle[2]];

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
	// printf("  Precaching bin %d\n", bin_index);
	// printf("  p = (%f, %f, %f)\n", p.x, p.y, p.z);
	// printf("  max = (%f, %f, %f)\n", max.x, max.y, max.z);
	// printf("  min = (%f, %f, %f)\n", min.x, min.y, min.z);
	// printf("  bin size = (%f, %f, %f)\n", bin_size.x, bin_size.y, bin_size.z);
	ULOG_ASSERT(bin_index < query_triangles.size());

	if (!query_triangles[bin_index].empty())
		return false;

	// Otherwise, load the bin
	auto set = closest_triangles(p);
	query_triangles[bin_index] = query_bin(set.begin(), set.end());
	return true;
}

// Precache a collection of query points
bool cas_grid::precache_query(const std::vector <glm::vec3> &points)
{
	uint32_t any_count = 0;
	for (const glm::vec3 &p : points)
		any_count += precache_query(p);

	// printf("Cache hit rate: %.2f\n", 1.0f - (float) any_count / points.size());
	return any_count > 0;
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
	glm::vec3 bary;
	float distance = FLT_MAX;
	uint32_t triangle_index = 0;

	for (uint32_t index : bin) {
		const Triangle &tri = ref.triangles[index];
		glm::vec3 a = ref.vertices[tri[0]];
		glm::vec3 b = ref.vertices[tri[1]];
		glm::vec3 c = ref.vertices[tri[2]];

		glm::vec3 point;
		glm::vec3 bary;
		float dist;
		triangle_closest_point(a, b, c, p, &point, &bary, &dist);

		if (dist < distance) {
			closest = point;
			bary = bary;
			distance = dist;
			triangle_index = index;
		}
	}

	return std::make_tuple(closest, bary, distance, triangle_index);
}

// Host-side query
void cas_grid::query(const std::vector <glm::vec3> &sources,
		std::vector <glm::vec3> &closest,
		std::vector <glm::vec3> &bary,
		std::vector <float> &distance,
		std::vector <uint32_t> &triangle_index) const
{
	// Assuming all elements are precached already
	// and that the dst vector is already allocated
	ULOG_ASSERT(sources.size() == closest.size());
	ULOG_ASSERT(sources.size() == bary.size());
	ULOG_ASSERT(sources.size() == distance.size());
	ULOG_ASSERT(sources.size() == triangle_index.size());

	#pragma omp parallel for
	for (uint32_t i = 0; i < sources.size(); i++) {
		uint32_t bin_index = to_index(sources[i]);
		auto [c, b, d, t] = query(sources[i]);

		closest[i] = c;
		bary[i] = b;
		distance[i] = d;
		triangle_index[i] = t;
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

void cas_grid::query_device(closest_point_kinfo kinfo)
{
	dim3 block(256);
	dim3 grid((kinfo.point_count + block.x - 1) / block.x);

	closest_point_kernel <<< grid, block >>> (dev_cas, kinfo);
}
