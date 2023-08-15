#include <queue>

#include "closest_point.cuh"

// Bounding box of mesh
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

__global__
static void closest_point_kernel(dev_cas_grid cas, closest_point_kinfo kinfo)
{
	uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t stride = blockDim.x * gridDim.x;

	for (uint32_t i = tid; i < kinfo.point_count; i += stride) {
		glm::vec3 point = kinfo.points[i];
		glm::vec3 closest;
		
		glm::vec3 bin_flt = glm::clamp((point - cas.min) / cas.bin_size,
				glm::vec3(0), glm::vec3(cas.resolution - 1));

		glm::ivec3 bin = glm::ivec3(bin_flt);
		uint32_t bin_index = bin.x + bin.y * cas.resolution + bin.z * cas.resolution * cas.resolution;

		uint32_t index0 = cas.index0[bin_index];
		uint32_t index1 = cas.index1[bin_index];

		float min_dist = FLT_MAX;
		for (uint32_t j = index0; j < index1; j++) {
			uint32_t triangle_index = cas.query_triangles[j];
			glm::uvec3 triangle = cas.triangles[triangle_index];

			glm::vec3 v0 = cas.vertices[triangle.x];
			glm::vec3 v1 = cas.vertices[triangle.y];
			glm::vec3 v2 = cas.vertices[triangle.z];

			// TODO: prune triangles that are too far away (based on bbox)?
			glm::vec3 candidate = triangle_closest_point(v0, v1, v2, point);
			float dist = glm::distance(point, candidate);

			if (dist < min_dist) {
				min_dist = dist;
				closest = candidate;
			}
		}

		kinfo.closest[i] = closest;
	}
}

// Construct from mesh
cas_grid::cas_grid(const Mesh &ref_, uint32_t resolution_)
		: ref(ref_), resolution(resolution_)
{
	uint32_t size = resolution * resolution * resolution;
	overlapping_triangles.resize(size);
	query_triangles.resize(size);

	// Put triangles into bins
	std::tie(max, min) = bound(ref);
	glm::vec3 extent = max - min;
	bin_size = extent / (float) resolution;

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

	dev_cas.vertices = 0;
	dev_cas.triangles = 0;

	dev_cas.query_triangles = 0;
	dev_cas.index0 = 0;
	dev_cas.index1 = 0;
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
	assert(bin_index < query_triangles.size());

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

	printf("Cache hit rate: %f\n", 1 - (float) any_count / points.size());
	return any_count > 0;
}

// Single point query
glm::vec3 cas_grid::query(const glm::vec3 &p) const
{
	// Assuming the point is precached already
	uint32_t bin_index = to_index(p);
	assert(bin_index < overlapping_triangles.size());

	const std::vector <uint32_t> &bin = query_triangles[bin_index];
	assert(bin.size() > 0);

	float min_dist = FLT_MAX;
	
	glm::vec3 closest = p;
	for (uint32_t index : bin) {
		const Triangle &tri = ref.triangles[index];
		glm::vec3 a = ref.vertices[tri[0]];
		glm::vec3 b = ref.vertices[tri[1]];
		glm::vec3 c = ref.vertices[tri[2]];

		glm::vec3 point = triangle_closest_point(a, b, c, p);

		float dist = glm::distance(p, point);
		if (dist < min_dist) {
			min_dist = dist;
			closest = point;
		}
	}

	return closest;
}

// Host-side query
void cas_grid::query(const std::vector <glm::vec3> &sources, std::vector <glm::vec3> &dst) const
{
	// Assuming all elements are precached already
	// and that the dst vector is already allocated
	assert(sources.size() == dst.size());

	#pragma omp parallel for
	for (uint32_t i = 0; i < sources.size(); i++) {
		uint32_t bin_index = to_index(sources[i]);
		dst[i] = query(sources[i]);
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
