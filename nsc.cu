#include <iostream>
#include <random>

#include <omp.h>

#include <glm/glm.hpp>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include <littlevk/littlevk.hpp>

#define MESH_LOAD_SAVE
#include "closest_point.cuh"
#include "mesh.hpp"
#include "viewer.hpp"

inline bool ray_x_triangle(const Mesh &mesh, size_t tindex, const glm::vec3 &x, const glm::vec3 &d)
{
	const Triangle &tri = mesh.triangles[tindex];

	glm::vec3 v0 = mesh.vertices[tri[0]];
	glm::vec3 v1 = mesh.vertices[tri[1]];
	glm::vec3 v2 = mesh.vertices[tri[2]];

	glm::vec3 e1 = v1 - v0;
	glm::vec3 e2 = v2 - v0;
	glm::vec3 p = cross(d, e2);

	float a = dot(e1, p);
	if (std::abs(a) < 1e-6)
		return false;

	float f = 1.0 / a;
	glm::vec3 s = x - v0;
	float u = f * dot(s, p);

	if (u < 0.0 || u > 1.0)
		return false;

	glm::vec3 q = cross(s, e1);
	float v = f * dot(d, q);

	if (v < 0.0 || u + v > 1.0)
		return false;

	float t = f * dot(e2, q);
	return t > 1e-6;
}

// Acceleration structure for interior point queries
struct ipqas {
	using bin = std::vector <uint32_t>;

	Mesh ref;
	uint32_t resolution;

	std::vector <bin> bins_xy;
	std::vector <bin> bins_xz;
	std::vector <bin> bins_yz;

	glm::vec3 ext_min;
	glm::vec3 ext_max;

	ipqas(const Mesh &ref_, uint32_t resolution_)
			: ref(ref_), resolution(resolution_) {
		ext_min = glm::vec3(std::numeric_limits <float> ::max());
		ext_max = glm::vec3(std::numeric_limits <float> ::min());

		for (const glm::vec3 &vertex : ref.vertices) {
			ext_min = glm::min(ext_min, vertex);
			ext_max = glm::max(ext_max, vertex);
		}

		bins_xy.resize(resolution * resolution);
		bins_xz.resize(resolution * resolution);
		bins_yz.resize(resolution * resolution);

		for (size_t i = 0; i < ref.triangles.size(); i++) {
			const Triangle &tri = ref.triangles[i];

			// Find the bounding box of the triangle
			glm::vec3 min = glm::vec3(std::numeric_limits <float> ::max());
			glm::vec3 max = glm::vec3(std::numeric_limits <float> ::lowest());

			for (size_t j = 0; j < 3; ++j) {
				glm::vec3 v = ref.vertices[tri[j]];
				v = (v - ext_min) / (ext_max - ext_min);
				min = glm::min(min, v);
				max = glm::max(max, v);
			}

			// Find the bins that the triangle overlaps
			glm::vec3 min_bin = glm::floor(min * glm::vec3(resolution));
			glm::vec3 max_bin = glm::ceil(max * glm::vec3(resolution));

			for (size_t x = min_bin.x; x < max_bin.x; x++) {
				for (size_t y = min_bin.y; y < max_bin.y; y++)
					bins_xy[x + y * resolution].push_back(i);
			}

			for (size_t x = min_bin.x; x < max_bin.x; x++) {
				for (size_t z = min_bin.z; z < max_bin.z; z++)
					bins_xz[x + z * resolution].push_back(i);
			}

			for (size_t y = min_bin.y; y < max_bin.y; y++) {
				for (size_t z = min_bin.z; z < max_bin.z; z++)
					bins_yz[y + z * resolution].push_back(i);
			}
		}
	}

	bool query(const glm::vec3 &v) const {
		static constexpr glm::vec3 dx { 1, 0, 0 };
		static constexpr glm::vec3 dy { 0, 1, 0 };
		static constexpr glm::vec3 dz { 0, 0, 1 };

		glm::vec3 normed = (v - ext_min) / (ext_max - ext_min);
		glm::vec3 voxel = normed * glm::vec3(resolution);
		voxel = glm::clamp(voxel, glm::vec3(0), glm::vec3(resolution - 1));

		uint32_t x = voxel.x;
		uint32_t y = voxel.y;
		uint32_t z = voxel.z;

		uint32_t xy_count = 0;
		uint32_t xy_count_neg = 0;

		for (size_t tindex : bins_xy[x + y * resolution]) {
			if (ray_x_triangle(ref, tindex, v, dz))
				xy_count++;

			if (ray_x_triangle(ref, tindex, v, -dz))
				xy_count_neg++;
		}

		uint32_t xz_count = 0;
		uint32_t xz_count_neg = 0;

		for (size_t tindex : bins_xz[x + z * resolution]) {
			if (ray_x_triangle(ref, tindex, v, dy))
				xz_count++;

			if (ray_x_triangle(ref, tindex, v, -dy))
				xz_count_neg++;
		}

		uint32_t yz_count = 0;
		uint32_t yz_count_neg = 0;

		for (size_t tindex : bins_yz[y + z * resolution]) {
			if (ray_x_triangle(ref, tindex, v, dx))
				yz_count++;

			if (ray_x_triangle(ref, tindex, v, -dx))
				yz_count_neg++;
		}

		bool xy_in = (xy_count % 2 == 1) && (xy_count_neg % 2 == 1);
		bool xz_in = (xz_count % 2 == 1) && (xz_count_neg % 2 == 1);
		bool yz_in = (yz_count % 2 == 1) && (yz_count_neg % 2 == 1);

		return xy_in || xz_in || yz_in;
	}

	void dump() const {
		// TODO: microlog
		float load_xy = 0.0f;
		float load_xz = 0.0f;
		float load_yz = 0.0f;

		for (const auto &bin : bins_xy)
			load_xy += bin.size();

		for (const auto &bin : bins_xz)
			load_xz += bin.size();

		for (const auto &bin : bins_yz)
			load_yz += bin.size();

		printf("Average bin load: %f (xy), %f (xz), %f (yz)\n",
			load_xy / bins_xy.size(),
			load_xz / bins_xz.size(),
			load_yz / bins_yz.size());
	}
};

std::pair <float, glm::vec3> closest_point(const Mesh &ref, const glm::vec3 &p)
{
	std::vector <std::pair <float, glm::vec3>> distances;
	distances.resize(ref.triangles.size());

	#pragma omp parallel for
	for (size_t i = 0; i < ref.triangles.size(); ++i) {
		const Triangle &tri = ref.triangles[i];
		glm::vec3 v0 = ref.vertices[tri[0]];
		glm::vec3 v1 = ref.vertices[tri[1]];
		glm::vec3 v2 = ref.vertices[tri[2]];
		glm::vec3 closest;
		glm::vec3 bary;
		float distance;
		triangle_closest_point(v0, v1, v2, p, &closest, &bary, &distance);
		distances[i] = std::make_pair(distance, closest);
	}

	std::sort(distances.begin(), distances.end(),
		[](const auto &a, const auto &b) {
			return a.first < b.first;
		}
	);

	return distances[0];
}

struct cumesh {
	glm::vec3 *vertices;
	glm::uvec3 *triangles;

	uint32_t vertex_count = 0;
	uint32_t triangle_count = 0;
};

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

struct sample_result {
	glm::vec3 *points;
	glm::vec3 *barys;
	glm::uvec3 *triangles;
	uint32_t point_count;
};

sample_result sample_result_alloc(uint32_t point_count)
{
	sample_result result;
	result.point_count = point_count;

	cudaMalloc(&result.points, sizeof(glm::vec3) * point_count);
	cudaMalloc(&result.barys, sizeof(glm::vec3) * point_count);
	cudaMalloc(&result.triangles, sizeof(glm::uvec3) * point_count);

	return result;
}

__global__
void sample(sample_result result, cumesh mesh, float time)
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

__global__
static void robust_closest_point_kernel(cumesh cu_mesh, closest_point_kinfo kinfo)
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

static void robust_closest_point(cumesh cu_mesh, closest_point_kinfo kinfo)
{
	dim3 block(128);
	dim3 grid((kinfo.point_count + block.x - 1) / block.x);
	robust_closest_point_kernel <<< grid, block >>> (cu_mesh, kinfo);
}

// Optimizers
struct momentum {
	std::vector <glm::vec3> grad;
	std::vector <glm::vec3> v;
	float mu = 0.9f;

	momentum(size_t count) {
		grad.resize(count, glm::vec3 { 0.0f });
		v.resize(count, glm::vec3 { 0.0f });
	}

	void step(std::vector <glm::vec3> &x, float lr) {
		#pragma omp parallel for
		for (size_t i = 0; i < x.size(); i++) {
			v[i] = mu * v[i] + lr * grad[i];
			x[i] += v[i];
		}
	}
};

void add_voxel(Mesh &mesh, const std::pair <glm::vec3, glm::vec3> &bounds, size_t resolution, size_t i)
{
	uint32_t x = i % resolution;
	uint32_t y = (i / resolution) % resolution;
	uint32_t z = i / (resolution * resolution);

	// Load all eight corners of the voxel
	glm::vec3 corners[8];

	corners[0] = glm::vec3(x, y, z) / (float) resolution;
	corners[1] = glm::vec3(x + 1, y, z) / (float) resolution;
	corners[2] = glm::vec3(x, y + 1, z) / (float) resolution;
	corners[3] = glm::vec3(x + 1, y + 1, z) / (float) resolution;
	corners[4] = glm::vec3(x, y, z + 1) / (float) resolution;
	corners[5] = glm::vec3(x + 1, y, z + 1) / (float) resolution;
	corners[6] = glm::vec3(x, y + 1, z + 1) / (float) resolution;
	corners[7] = glm::vec3(x + 1, y + 1, z + 1) / (float) resolution;

	for (glm::vec3 &v : corners)
		v = v * (bounds.second - bounds.first) + bounds.first;
		// v = v * (ext_max - ext_min) + ext_min;

	// Fill the data in the mesh
	uint32_t base = mesh.vertices.size();

	for (const glm::vec3 &v : corners)
		mesh.vertices.push_back(v);

	mesh.triangles.push_back({ base + 0, base + 1, base + 2 });
	mesh.triangles.push_back({ base + 1, base + 3, base + 2 });
	mesh.triangles.push_back({ base + 0, base + 2, base + 4 });
	mesh.triangles.push_back({ base + 2, base + 6, base + 4 });
	mesh.triangles.push_back({ base + 0, base + 4, base + 1 });
	mesh.triangles.push_back({ base + 1, base + 4, base + 5 });
	mesh.triangles.push_back({ base + 1, base + 5, base + 3 });
	mesh.triangles.push_back({ base + 3, base + 5, base + 7 });
	mesh.triangles.push_back({ base + 2, base + 3, base + 6 });
	mesh.triangles.push_back({ base + 3, base + 7, base + 6 });
	mesh.triangles.push_back({ base + 4, base + 6, base + 5 });
	mesh.triangles.push_back({ base + 5, base + 6, base + 7 });
}

// Manage the subdivision of a particular complex
struct subdivision_complex {
	uint32_t size;
	std::vector <uint32_t> vertices;

	std::vector <glm::vec3> upscale(const Mesh &ref) const {
		assert(vertices.size() == size * size);

		std::vector <glm::vec3> base;
		base.reserve(vertices.size());

		for (uint32_t v : vertices)
			base.push_back(ref.vertices[v]);

		std::vector <glm::vec3> result;

		uint32_t new_size = 2 * size;
		result.resize(new_size * new_size);

		// Bilerp each new vertex
		for (uint32_t i = 0; i < new_size; i++) {
			for (uint32_t j = 0; j < new_size; j++) {
				float u = (float) i / (new_size - 1);
				float v = (float) j / (new_size - 1);

				float lu = u * (size - 1);
				float lv = v * (size - 1);

				uint32_t u0 = std::floor(lu);
				uint32_t u1 = std::ceil(lu);

				uint32_t v0 = std::floor(lv);
				uint32_t v1 = std::ceil(lv);

				glm::vec3 p00 = base[u0 * size + v0];
				glm::vec3 p10 = base[u1 * size + v0];
				glm::vec3 p01 = base[u0 * size + v1];
				glm::vec3 p11 = base[u1 * size + v1];

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
};

// Optimization state
using complex = std::array <uint32_t, 4>;

struct scm {
	Mesh ref;

	uint32_t size;

	std::vector <complex> complexes;
	std::vector <subdivision_complex> subdivision_complexes;

	scm(const Mesh &ref_, const std::vector <complex> &complexes_)
			: ref(ref_), complexes(complexes_), size(2) {
		subdivision_complexes.resize(complexes.size());

		for (size_t i = 0; i < complexes.size(); i++) {
			complex c = complexes[i];
			subdivision_complex &s = subdivision_complexes[i];

			s.size = 2;
			s.vertices.resize(4);

			s.vertices[0] = c[0];
			s.vertices[1] = c[1];
			s.vertices[2] = c[2];
			s.vertices[3] = c[3];
		}
	}

	scm(const Mesh &ref_, const std::vector <complex> &complexes_, const std::vector <subdivision_complex> &subdivision_states_, uint32_t size_)
			: ref(ref_), complexes(complexes_), subdivision_complexes(subdivision_states_), size(size_) {}

	scm upscale() const {
		Mesh new_ref;

		uint32_t new_size = 2 * size;
		printf("Upscaling from %d to %d\n", size, new_size);

		std::vector <complex> new_complexes;
		std::vector <subdivision_complex> new_subdivision_states;

		for (const auto &sdv : subdivision_complexes) {
			auto new_vertices = sdv.upscale(ref);

			subdivision_complex new_sdv;
			new_sdv.size = new_size;

			// Fill the vertices
			uint32_t offset = new_ref.vertices.size();
			for (const auto &v : new_vertices) {
				new_sdv.vertices.push_back(new_ref.vertices.size());
				new_ref.vertices.push_back(v);
				new_ref.normals.push_back(glm::vec3(0.0f));
			}

			// Fill the triangles
			for (uint32_t i = 0; i < new_size - 1; i++) {
				for (uint32_t j = 0; j < new_size - 1; j++) {
					uint32_t i00 = i * new_size + j;
					uint32_t i10 = (i + 1) * new_size + j;
					uint32_t i01 = i * new_size + j + 1;
					uint32_t i11 = (i + 1) * new_size + j + 1;

					Triangle t1 { offset + i00, offset + i10, offset + i11 };
					Triangle t2 { offset + i00, offset + i01, offset + i11 };

					new_ref.triangles.push_back(t1);
					new_ref.triangles.push_back(t2);
				}
			}

			complex new_c;
			new_c[0] = new_sdv.vertices[0];
			new_c[1] = new_sdv.vertices[new_size - 1];
			new_c[2] = new_sdv.vertices[new_size * (new_size - 1)];
			new_c[3] = new_sdv.vertices[new_size * new_size - 1];

			new_complexes.push_back(new_c);
			new_subdivision_states.push_back(new_sdv);
		}

		// TODO: deduplicate vertices
		auto res = deduplicate(new_ref, 1e-6f);
		printf("Before/after: %d/%d\n", new_ref.vertices.size(), res.first.vertices.size());
		new_ref = res.first;

		auto remap = res.second;
		for (auto &c : new_complexes) {
			for (auto &v : c)
				v = remap[v];
		}

		for (auto &sdv : new_subdivision_states) {
			for (auto &v : sdv.vertices)
				v = remap[v];
		}

		return scm(new_ref, new_complexes, new_subdivision_states, new_size);
	}

	void save(const std::filesystem::path &path) const {
		// TODO: microlog
		printf("Saved subdivison complexes to %s\n", path.string().c_str());

		std::ofstream fout(path, std::ios::binary);

		std::string source = std::filesystem::absolute(path).string();
		uint32_t source_size = source.size();

		fout.write((char *) &source_size, sizeof(uint32_t));
		fout.write(source.c_str(), source_size);

		std::unordered_map <uint32_t, uint32_t> complex_remap;
		auto add_unique_corners = [&](uint32_t c) -> uint32_t {
			if (complex_remap.count(c))
				return complex_remap[c];

			uint32_t csize = complex_remap.size();
			complex_remap[c] = csize;
			return csize;
		};

		std::vector <complex> normalized_complexes;
		for (uint32_t i = 0; i < complexes.size(); i++) {
			auto [c0, c1, c2, c3] = complexes[i];
			c0 = add_unique_corners(c0);
			c1 = add_unique_corners(c1);
			c2 = add_unique_corners(c2);
			c3 = add_unique_corners(c3);
			normalized_complexes.push_back({ c0, c1, c2, c3 });
		}

		// printf("Normalized %u complexes\n", normalized_complexes.size());
		for (uint32_t i = 0; i < normalized_complexes.size(); i++) {
			auto [c0, c1, c2, c3] = normalized_complexes[i];
			auto [o0, o1, o2, o3] = complexes[i];
			// printf("  %u: %u %u %u %u (was %u %u %u %u)\n", i, c0, c1, c2, c3, o0, o1, o2, o3);
		}

		uint32_t corner_count = complex_remap.size();
		fout.write((char *) &corner_count, sizeof(uint32_t));

		for (auto [c, csize] : complex_remap) {
			glm::vec3 v = ref.vertices[c];
			fout.write((char *) &v, sizeof(glm::vec3));
		}

		uint32_t complex_count = normalized_complexes.size();
		fout.write((char *) &complex_count, sizeof(uint32_t));

		for (const complex &c : normalized_complexes)
			fout.write((char *) &c, sizeof(complex));

		// Write the vertex data for all the subdivison complexes
		for (uint32_t i = 0; i < subdivision_complexes.size(); i++) {
			const subdivision_complex &sdv = subdivision_complexes[i];

			uint32_t size = sdv.size;
			fout.write((char *) &size, sizeof(uint32_t));

			std::vector <glm::vec3> vertices;
			for (size_t vi : sdv.vertices)
				vertices.push_back(ref.vertices[vi]);

			uint32_t vertex_count = vertices.size();
			fout.write((char *) &vertex_count, sizeof(uint32_t));
			fout.write((char *) vertices.data(), sizeof(glm::vec3) * vertices.size());
		}

		fout.close();
	}
};

__forceinline__ __host__ __device__
void triangle_area_gradient(glm::vec3 vs[3], glm::vec3 gs[3])
{
	gs[0] = glm::vec3 { 0.0f };
	gs[1] = glm::vec3 { 0.0f };
	gs[2] = glm::vec3 { 0.0f };

	// Consider different edge pairs
	for (uint32_t i = 0; i < 3; i++) {
		uint32_t j = (i + 1) % 3;
		uint32_t k = (i + 2) % 3;

		glm::vec3 e0 = vs[j] - vs[i];
		glm::vec3 e1 = vs[k] - vs[i];
		float A = glm::length(glm::cross(e0, e1))/2.0f;

		float x1 = e0.x;
		float x2 = e0.y;
		float x3 = e0.z;

		float y1 = e1.x;
		float y2 = e1.y;
		float y3 = e1.z;

		float dA_dx1 = x1 * (y2 * y2 + y3 * y3) - y1 * (x2 * y2 + x3 * y3);
		float dA_dx2 = x2 * (y1 * y1 + y3 * y3) - y2 * (x1 * y1 + x3 * y3);
		float dA_dx3 = x3 * (y1 * y1 + y2 * y2) - y3 * (x1 * y1 + x2 * y2);

		float dA_dy1 = y1 * (x2 * x2 + x3 * x3) - x1 * (x2 * y2 + x3 * y3);
		float dA_dy2 = y2 * (x1 * x1 + x3 * x3) - x2 * (x1 * y1 + x3 * y3);
		float dA_dy3 = y3 * (x1 * x1 + x2 * x2) - x3 * (x1 * y1 + x2 * y2);

		gs[j] += glm::vec3 { dA_dx1, dA_dx2, dA_dx3 } / A;
		gs[k] += glm::vec3 { dA_dy1, dA_dy2, dA_dy3 } / A;
	}
}

int main(int argc, char *argv[])
{
	// Load arguments
	if (argc != 3) {
		printf("Usage: %s <filename> <resolution>\n", argv[0]);
		return 1;
	}

	std::filesystem::path path = std::filesystem::weakly_canonical(argv[1]);
	size_t resolution = std::atoi(argv[2]);

	// Load mesh
	Mesh mesh = load_mesh(path);
	printf("Loaded mesh with %lu vertices and %lu triangles\n", mesh.vertices.size(), mesh.triangles.size());
	printf("Resolution: %lu\n", resolution);

	// Construct the voxel mesh
	ipqas interior_query_as(mesh, resolution);
	interior_query_as.dump();

	glm::vec3 ext_min = interior_query_as.ext_min;
	glm::vec3 ext_max = interior_query_as.ext_max;

	uint32_t voxel_count = 0;
	std::vector <uint32_t> voxels;
	voxels.resize(resolution * resolution * resolution);

	#pragma omp parallel for reduction(+:voxel_count)
	for (uint32_t i = 0; i < voxels.size(); i++) {
		uint32_t x = i % resolution;
		uint32_t y = (i / resolution) % resolution;
		uint32_t z = i / (resolution * resolution);

		glm::vec3 center = glm::vec3(x + 0.5f, y + 0.5f, z + 0.5f) / (float) resolution;
		center = center * (ext_max - ext_min) + ext_min;

		if (interior_query_as.query(center)) {
			voxels[i] = 1;
			voxel_count++;
		}
	}

	printf("Voxel count: %u/%u\n", voxel_count, voxels.size());

	// Create a mesh out of the voxels
	Mesh voxel_mesh;
	for (size_t i = 0; i < voxels.size(); i++) {
		if (voxels[i] == 0)
			continue;

		// add_voxel(voxel_mesh, i);
		add_voxel(voxel_mesh, { ext_min, ext_max }, resolution, i);
	}

	voxel_mesh.normals.resize(voxel_mesh.vertices.size());
	voxel_mesh = deduplicate(voxel_mesh).first;

	printf("Voxel mesh: %u vertices, %u triangles\n", voxel_mesh.vertices.size(), voxel_mesh.triangles.size());

	// Adjaceny for the voxel elements
	std::unordered_map <uint32_t, std::vector<uint32_t>> voxel_adjacency;

	for (size_t i = 0; i < voxels.size(); i++) {
		if (voxels[i] == 0)
			continue;

		// Find all neighbors that are valid
		int dx[6] = { -1, 1, 0, 0, 0, 0 };
		int dy[6] = { 0, 0, -1, 1, 0, 0 };
		int dz[6] = { 0, 0, 0, 0, -1, 1 };

		int32_t x = i % resolution;
		int32_t y = (i / resolution) % resolution;
		int32_t z = i / (resolution * resolution);

		for (int j = 0; j < 6; j++) {
			int32_t nx = x + dx[j];
			int32_t ny = y + dy[j];
			int32_t nz = z + dz[j];

			if (nx < 0 || nx >= resolution || ny < 0 || ny >= resolution || nz < 0 || nz >= resolution)
				continue;

			int32_t nindex = nx + ny * resolution + nz * resolution * resolution;

			if (voxels[nindex] == 0)
				continue;

			voxel_adjacency[i].push_back(nindex);
		}
	}

	float adj_avg = 0.0f;
	for (const auto &p : voxel_adjacency)
		adj_avg += p.second.size();

	printf("Average adjacency: %f\n", adj_avg / (float) voxel_adjacency.size());

	// Remove voxels with only one neighbor
	for (auto it = voxel_adjacency.begin(); it != voxel_adjacency.end(); ) {
		if (it->second.size() <= 1) {
			printf("Removing voxel %u\n", it->first);
			it = voxel_adjacency.erase(it);
		} else
			it++;
	}

	if (voxel_adjacency.size() == 0) {
		printf("No voxels left after removing single-neighbor voxels\n");
		return -1;
	}

	// Find all the connected components and select the largest one
	std::vector <std::unordered_set <uint32_t>> components;

	std::unordered_set <uint32_t> remaining;
	for (auto [v, _] : voxel_adjacency)
		remaining.insert(v);

	while (remaining.size() > 0) {
		uint32_t seed = *remaining.begin();

		std::unordered_set <uint32_t> component;
		std::queue <uint32_t> queue;

		queue.push(seed);
		component.insert(seed);
		remaining.erase(seed);

		while (!queue.empty()) {
			uint32_t v = queue.front();
			queue.pop();

			for (uint32_t n : voxel_adjacency[v]) {
				if (remaining.count(n) == 0)
					continue;

				queue.push(n);
				component.insert(n);
				remaining.erase(n);
			}
		}

		components.push_back(component);
	}

	printf("Found %u components\n", components.size());
	for (size_t i = 0; i < components.size(); i++)
		printf("  component %u: %u voxels\n", i, components[i].size());

	std::sort(components.begin(), components.end(),
		[](const auto &a, const auto &b) {
			return a.size() > b.size();
		}
	);

	std::unordered_set <uint32_t> working_set = components[0];

	// Extract the voxel mesh
	Mesh reduced_mesh;
	for (size_t i = 0; i < voxels.size(); i++) {
		if (working_set.count(i) == 0)
			continue;

		// add_voxel(reduced_mesh, i);
		add_voxel(reduced_mesh, { ext_min, ext_max }, resolution, i);
	}

	reduced_mesh.normals.resize(reduced_mesh.vertices.size());
	reduced_mesh = deduplicate(reduced_mesh).first;

	// Remove all the duplicate triangles
	auto triangle_eq = [](const Triangle &t1, const Triangle &t2) {
		// Any permutation of the vertices is considered equal
		bool perm1 = (t1[0] == t2[0] && t1[1] == t2[1] && t1[2] == t2[2]);
		bool perm2 = (t1[0] == t2[0] && t1[1] == t2[2] && t1[2] == t2[1]);
		bool perm3 = (t1[0] == t2[1] && t1[1] == t2[0] && t1[2] == t2[2]);
		bool perm4 = (t1[0] == t2[1] && t1[1] == t2[2] && t1[2] == t2[0]);
		bool perm5 = (t1[0] == t2[2] && t1[1] == t2[0] && t1[2] == t2[1]);
		bool perm6 = (t1[0] == t2[2] && t1[1] == t2[1] && t1[2] == t2[0]);
		return perm1 || perm2 || perm3 || perm4 || perm5 || perm6;
	};

	auto triangle_hash = [](const Triangle &t) {
		std::hash <uint32_t> hasher;
		return hasher(t[0]) ^ hasher(t[1]) ^ hasher(t[2]);
	};

	std::unordered_set <Triangle, decltype(triangle_hash), decltype(triangle_eq)> triangles(0, triangle_hash, triangle_eq);
	std::unordered_set <Triangle, decltype(triangle_hash), decltype(triangle_eq)> triangles_to_remove(0, triangle_hash, triangle_eq);

	for (const auto &t : reduced_mesh.triangles) {
		if (triangles.count(t) > 0)
			triangles_to_remove.insert(t);
		else
			triangles.insert(t);
	}

	printf("Number of triangles to remove: %u\n", triangles_to_remove.size());

	Mesh skin_mesh = reduced_mesh;

	std::vector <Triangle> new_triangles;
	for (const auto &t : skin_mesh.triangles) {
		if (triangles_to_remove.count(t) == 0)
			new_triangles.push_back(t);
	}

	skin_mesh.triangles = new_triangles;
	skin_mesh = deduplicate(skin_mesh).first;

	// glm::vec3 color_wheel[] = {
	// 	{ 0.910, 0.490, 0.490 },
	// 	{ 0.910, 0.700, 0.490 },
	// 	{ 0.910, 0.910, 0.490 },
	// 	{ 0.700, 0.910, 0.490 },
	// 	{ 0.490, 0.910, 0.490 },
	// 	{ 0.490, 0.910, 0.700 },
	// 	{ 0.490, 0.910, 0.910 },
	// 	{ 0.490, 0.700, 0.910 },
	// 	{ 0.490, 0.490, 0.910 },
	// 	{ 0.700, 0.490, 0.910 },
	// 	{ 0.910, 0.490, 0.910 },
	// 	{ 0.910, 0.490, 0.700 }
	// };

	// Create the acceleration structure for point queries on the target mesh
	cas_grid cas = cas_grid(mesh, 128);
	cas.precache_query(skin_mesh.vertices);
	cas.precache_device();

	printf("CAS grid has been preloaded\n");

	// Precomputation setup for subdivision
	auto [vgraph, egraph, dual] = build_graphs(skin_mesh);

	// Find all complexes (4-cycles) with edges parallel to canonical axes
	auto complex_eq = [](const complex &c1, const complex &c2) {
		// Equal if they are the same under rotation
		for (size_t i = 0; i < 4; i++) {
			bool found = true;

			for (size_t j = 0; j < 4; j++) {
				if (c1[j] != c2[(i + j) % 4]) {
					found = false;
					break;
				}
			}

			if (found)
				return true;
		}

		// Also check the reverse
		for (size_t i = 0; i < 4; i++) {
			bool found = true;

			for (size_t j = 0; j < 4; j++) {
				if (c1[j] != c2[(i - j + 4) % 4]) {
					found = false;
					break;
				}
			}

			if (found)
				return true;
		}

		return false;
	};

	auto complex_hash = [](const complex &c) {
		std::hash <uint32_t> hasher;
		return hasher(c[0]) ^ hasher(c[1]) ^ hasher(c[2]) ^ hasher(c[3]);
	};

	std::unordered_set <complex, decltype(complex_hash), decltype(complex_eq)> complexes(0, complex_hash, complex_eq);

	// Remove non-axial edges
	auto simplified_graph = vgraph;

	for (auto &[v, adj] : simplified_graph) {
		glm::vec3 vv = skin_mesh.vertices[v];

		for (auto it = adj.begin(); it != adj.end(); ) {
			glm::vec3 va = skin_mesh.vertices[*it];
			glm::vec3 e = va - vv;

			uint32_t x_zero = std::abs(e.x) < 1e-6f;
			uint32_t y_zero = std::abs(e.y) < 1e-6f;
			uint32_t z_zero = std::abs(e.z) < 1e-6f;

			if (x_zero + y_zero + z_zero < 2)
				it = adj.erase(it);
			else
				it++;
		}
	}

	for (size_t v = 0; v < skin_mesh.vertices.size(); v++) {
		using cycle = std::array <uint32_t, 5>;
		using cycle_state = std::tuple <cycle, uint32_t, uint32_t>;

		std::vector <cycle> potential;
		std::queue <cycle_state> q;
		q.push({ { 0, 0, 0, 0, 0 }, v, 0 });

		while (!q.empty()) {
			auto [c, v, depth] = q.front();
			q.pop();

			if (depth == 5) {
				potential.push_back(c);
				continue;
			}

			c[depth] = v;
			for (uint32_t e : simplified_graph[v]) {
				// Fill the next index in the complex
				cycle_state next = { c, e, depth + 1 };
				q.push(next);
			}
		}

		// Find all cycles starting and ending with the same vertex
		for (const auto &c : potential) {
			if (c[0] != c[4])
				continue;

			// Cannot have duplicates in the middle
			bool ok = true;
			for (size_t i = 0; i < 4; i++) {
				for (size_t j = i + 1; j < 4; j++) {
					if (c[i] == c[j]) {
						ok = false;
						break;
					}
				}

				if (!ok)
					break;
			}

			if (!ok)
				continue;

			complex c2 = { c[0], c[1], c[2], c[3] };
			complexes.insert(c2);
		}
	}

	// Remove interior complexes (midpoint is inside the mesh)
	uint32_t removed = 0;
	for (auto it = complexes.begin(); it != complexes.end(); ) {
		const auto &c = *it;
		const auto &vertices = skin_mesh.vertices;
		glm::vec3 midpoint = 0.25f * (vertices[c[0]] + vertices[c[1]] + vertices[c[2]] + vertices[c[3]]);

		auto [d, v] = closest_point(skin_mesh, midpoint);
		if (d > 1e-3f) {
			printf("Removing complex with midpoint outside the mesh: %f\n", d);
			it = complexes.erase(it);
			removed++;
		} else {
			it++;
		}
	}

	printf("Number of (unique) complexes: %zu (%zu removed)\n", complexes.size(), removed);

	// Linearize these canonical complexes and record their subdivision state
	std::vector <complex> complexes_linearized(complexes.begin(), complexes.end());

	// Reorder the complexes from cyclic to array format
	for (auto &c : complexes_linearized)
		std::swap(c[2], c[3]);

	printf("Complexes: %u\n", complexes_linearized.size());
	for (const auto &c : complexes_linearized)
		printf("  %u %u %u %u\n", c[0], c[1], c[2], c[3]);

	scm opt(skin_mesh, complexes_linearized);
	printf("Optimizer ref details: %u vertices, %u faces\n", opt.ref.vertices.size(), opt.ref.triangles.size());

	Viewer viewer;
	viewer.add("mesh", mesh, Viewer::Mode::Shaded);
	viewer.add("ref", opt.ref, Viewer::Mode::Wireframe);
	viewer.add("skin", skin_mesh, Viewer::Mode::Wireframe);

	viewer.ref("ref")->color = { 0.3f, 0.3f, 0.8f };
	viewer.ref("skin")->color = { 0.3f, 0.8f, 0.3f };
	
	// Allocate optimization resources
	const uint32_t TARGET_SAMPLE_COUNT = 1000;

	closest_point_kinfo kinfo0 = closest_point_kinfo_alloc(opt.ref.vertices.size());
	closest_point_kinfo kinfo1 = closest_point_kinfo_alloc(TARGET_SAMPLE_COUNT);

	cumesh cu_target = cumesh_alloc(mesh);
	cumesh cu_opt = cumesh_alloc(opt.ref);

	sample_result cu_sample = sample_result_alloc(TARGET_SAMPLE_COUNT);

	// Optimization (while rendering)

	// kinfo.point_count = opt.ref.vertices.size();
	//
	// cudaMalloc(&kinfo.points, sizeof(glm::vec3) * opt.ref.vertices.size());
	// cudaMalloc(&kinfo.closest, sizeof(glm::vec3) * opt.ref.vertices.size());
	// cudaMalloc(&kinfo.bary, sizeof(glm::vec3) * opt.ref.vertices.size());
	// cudaMalloc(&kinfo.triangles, sizeof(uint32_t) * opt.ref.vertices.size());

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	std::tie(vgraph, egraph, dual) = build_graphs(opt.ref);

	// Compute target mesh surface normals
	std::vector <glm::vec3> surface_normals(mesh.triangles.size());

	for (uint32_t i = 0; i < mesh.triangles.size(); i++) {
		const auto &t = mesh.triangles[i];
		glm::vec3 a = mesh.vertices[t[0]];
		glm::vec3 b = mesh.vertices[t[1]];
		glm::vec3 c = mesh.vertices[t[2]];

		surface_normals[i] = glm::normalize(glm::cross(b - a, c - a));
	}

	// TODO: meshlet rendering, with different colors for each patch...
	momentum mopt(opt.ref.vertices.size());

	while (true) {
		GLFWwindow *window = viewer.window->handle;

		// Check window close state
		glfwPollEvents();
		if (glfwWindowShouldClose(window))
			break;

		// Refine on enter
		if (glfwGetKey(window, GLFW_KEY_ENTER) == GLFW_PRESS) {
			opt = opt.upscale();
			viewer.replace("ref", opt.ref);
			printf("Upscaled ref details: %u vertices, %u faces\n", opt.ref.vertices.size(), opt.ref.triangles.size());

			cudaFree(kinfo0.points);
			cudaFree(kinfo0.closest);

			kinfo0.point_count = opt.ref.vertices.size();
			cudaMalloc(&kinfo0.points, sizeof(glm::vec3) * opt.ref.vertices.size());
			cudaMalloc(&kinfo0.closest, sizeof(glm::vec3) * opt.ref.vertices.size());

			std::tie(vgraph, egraph, dual) = build_graphs(opt.ref);
		}

		// Optimize the skin mesh around the target (original) mesh
		std::vector <glm::vec3> host_closest(opt.ref.vertices.size());
		std::vector <uint32_t> host_triangles(opt.ref.vertices.size());

		{
			// TODO: memcpy async while caching
			// cudaMemcpy(kinfo0.points, opt.ref.vertices.data(),
			// 		sizeof(glm::vec3) * opt.ref.vertices.size(),
			// 		cudaMemcpyHostToDevice);
			cudaMemcpyAsync(kinfo0.points, opt.ref.vertices.data(),
					sizeof(glm::vec3) * opt.ref.vertices.size(),
					cudaMemcpyHostToDevice, stream);

			bool updated = cas.precache_query(opt.ref.vertices);
			if (updated)
				cas.precache_device();

			cudaStreamSynchronize(stream);

			// cas.query(opt.ref.vertices, host_closest);
			// TODO: provide the stream
			cas.query_device(kinfo0);
			cudaDeviceSynchronize();

			cudaMemcpy(host_closest.data(), kinfo0.closest,
					sizeof(glm::vec3) * opt.ref.vertices.size(),
					cudaMemcpyDeviceToHost);

			cudaMemcpy(host_triangles.data(), kinfo0.triangles,
					sizeof(uint32_t) * opt.ref.vertices.size(),
					cudaMemcpyDeviceToHost);

			mopt = momentum(opt.ref.vertices.size());
			cu_opt = cumesh_alloc(opt.ref);
		}

		std::vector <glm::vec3> gradients;
		gradients.reserve(opt.ref.vertices.size());

		for (uint32_t i = 0; i < opt.ref.vertices.size(); i++) {
			const glm::vec3 &v = opt.ref.vertices[i];
			const glm::vec3 &w = host_closest[i];
			gradients[i] = (w - v);
		}

		{
			double time = glfwGetTime();
			sample <<< 1, TARGET_SAMPLE_COUNT >>> (cu_sample, cu_target, time);
			cudaDeviceSynchronize();

			kinfo1.points = cu_sample.points;
			robust_closest_point(cu_opt, kinfo1);

			std::vector <glm::vec3> host_sample(TARGET_SAMPLE_COUNT);
			std::vector <glm::vec3> host_closest(TARGET_SAMPLE_COUNT);
			std::vector <glm::vec3> host_bary(TARGET_SAMPLE_COUNT);
			std::vector <uint32_t> host_triangles(TARGET_SAMPLE_COUNT);

			cudaMemcpy(host_sample.data(), kinfo1.points,
					sizeof(glm::vec3) * TARGET_SAMPLE_COUNT,
					cudaMemcpyDeviceToHost);

			cudaMemcpy(host_closest.data(), kinfo1.closest,
					sizeof(glm::vec3) * TARGET_SAMPLE_COUNT,
					cudaMemcpyDeviceToHost);

			cudaMemcpy(host_bary.data(), kinfo1.bary,
					sizeof(glm::vec3) * TARGET_SAMPLE_COUNT,
					cudaMemcpyDeviceToHost);

			cudaMemcpy(host_triangles.data(), kinfo1.triangles,
					sizeof(uint32_t) * TARGET_SAMPLE_COUNT,
					cudaMemcpyDeviceToHost);

			for (uint32_t i = 0; i < TARGET_SAMPLE_COUNT; i++) {
				const glm::vec3 &w = host_sample[i];
				const glm::vec3 &v = host_closest[i];
				const glm::vec3 &b = host_bary[i];

				const uint32_t &t = host_triangles[i];
				const Triangle &tri = opt.ref.triangles[t];

				glm::vec3 v0 = mesh.vertices[tri[0]];
				glm::vec3 v1 = mesh.vertices[tri[1]];
				glm::vec3 v2 = mesh.vertices[tri[2]];

				glm::vec3 delta = (w - v);

				glm::vec3 gv0 = b.x * delta;
				glm::vec3 gv1 = b.y * delta;
				glm::vec3 gv2 = b.z * delta;

				gradients[tri[0]] += gv0;
				gradients[tri[1]] += gv1;
				gradients[tri[2]] += gv2;
			}
		}

		// Compute average triangle area
		float total_area = 0.0f;
		for (uint32_t i = 0; i < opt.ref.triangles.size(); i++) {
			const Triangle &tri = opt.ref.triangles[i];
			const glm::vec3 &v0 = opt.ref.vertices[tri[0]];
			const glm::vec3 &v1 = opt.ref.vertices[tri[1]];
			const glm::vec3 &v2 = opt.ref.vertices[tri[2]];
			total_area += glm::length(glm::cross(v1 - v0, v2 - v0))/2.0f;
		}

		float avg_area = total_area / opt.ref.triangles.size();

		// Compute triangle area anti-distortion gradients
		{
			std::vector <glm::vec3> tri_gradients;
			tri_gradients.resize(opt.ref.vertices.size(), glm::vec3(0.0f));

			for (uint32_t i = 0; i < opt.ref.triangles.size(); i++) {
				const Triangle &tri = opt.ref.triangles[i];
			
				const glm::vec3 &v0 = opt.ref.vertices[tri[0]];
				const glm::vec3 &v1 = opt.ref.vertices[tri[1]];
				const glm::vec3 &v2 = opt.ref.vertices[tri[2]];

				float area = glm::length(glm::cross(v1 - v0, v2 - v0))/2.0f;

				glm::vec3 vs[3] = {v0, v1, v2};
				glm::vec3 gs[3] = {};

				triangle_area_gradient(vs, gs);

				float k = (avg_area - area);
				tri_gradients[tri[0]] += gs[0] * k;
				tri_gradients[tri[1]] += gs[1] * k;
				tri_gradients[tri[2]] += gs[2] * k;
			}

			// Project gradients onto the tangent plane
			for (uint32_t i = 0; i < opt.ref.vertices.size(); i++) {
				const glm::vec3 &v = opt.ref.vertices[i];
				const glm::vec3 &n = surface_normals[i];
				tri_gradients[i] -= glm::dot(tri_gradients[i], n) * n;
			}

			// Transfer gradients targetting an even distribution of triangle areas
			for (uint32_t i = 0; i < opt.ref.vertices.size(); i++)
				gradients[i] += tri_gradients[i];
		}

		// Edge anti-collapse/anti-foldover gradients
		{
			std::vector <glm::vec3> edge_gradients;
			edge_gradients.resize(opt.ref.vertices.size(), glm::vec3(0.0f));

			constexpr float ANGLE_THRESHOLD = glm::radians(25.0f);
			for (uint32_t i = 0; i < opt.ref.triangles.size(); i++) {
				const Triangle &tri = opt.ref.triangles[i];

				const glm::vec3 &v0 = opt.ref.vertices[tri[0]];
				const glm::vec3 &v1 = opt.ref.vertices[tri[1]];
				const glm::vec3 &v2 = opt.ref.vertices[tri[2]];
				glm::vec3 normal = glm::normalize(glm::cross(v1 - v0, v2 - v0));

				// Consider all pairs of edges
				for (uint32_t j = 0; j < 3; j++) {
					uint32_t m = (j + 1) % 3;
					uint32_t n = (j + 2) % 3;

					const glm::vec3 &v0 = opt.ref.vertices[tri[j]];
					const glm::vec3 &v1 = opt.ref.vertices[tri[m]];
					const glm::vec3 &v2 = opt.ref.vertices[tri[n]];

					glm::vec3 e0 = v1 - v0;
					glm::vec3 e1 = v2 - v0;

					float angle = glm::acos(glm::dot(e0, e1) / (glm::length(e0) * glm::length(e1)));

					glm::vec3 ex = v2 - v1;

					if (angle < ANGLE_THRESHOLD) {
						// Push apart
						float k = (ANGLE_THRESHOLD - angle);

						glm::vec3 g0 = glm::normalize(glm::cross(normal, e0));
						if (glm::dot(g0, ex) > 0.0f)
							g0 = -g0;

						glm::vec3 g1 = glm::normalize(glm::cross(normal, e1));
						if (glm::dot(g1, ex) > 0.0f)
							g1 = -g1;

						g0 *= k;
						g1 *= k;

						edge_gradients[tri[j]] += g0;
						edge_gradients[tri[m]] += g1;
					}
				}
			}

			// Transfer gradients targetting an even distribution of triangle areas
			// for (uint32_t i = 0; i < opt.ref.vertices.size(); i++)
			// 	gradients[i] += edge_gradients[i];
		}

		// TODO: adam optimizer...
		// TODO: apply gradients in cuda, using the imported vulkan buffer
		// mopt.step(gradients, 0.001f);
		for (uint32_t i = 0; i < opt.ref.vertices.size(); i++)
			opt.ref.vertices[i] += 0.01f * gradients[i];

		cumesh_reload(cu_opt, opt.ref);

		viewer.refresh("ref", opt.ref);
		viewer.render();
	}

	viewer.destroy();

	// Save the state of the subdivison complexes
	std::filesystem::path sdv_complexes = path.stem().string() + ".sdv";
	opt.save(sdv_complexes);
}
