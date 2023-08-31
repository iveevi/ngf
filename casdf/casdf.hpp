#pragma once

#include <cstdint>
#include <unordered_set>
#include <vector>

#include <glm/glm.hpp>

// TODO: python (pybind11) bindings
// TODO: optimize in a later project...

enum compute_api {
	eCPU,
	eCUDA
};

struct geometry {
	std::vector <glm::vec3> vertices;
	std::vector <glm::uvec3> triangles;
};

struct cumesh {
	glm::vec3 *vertices;
	glm::uvec3 *triangles;

	uint32_t vertex_count = 0;
	uint32_t triangle_count = 0;
};

cumesh cumesh_alloc(const geometry &);
void cumesh_reload(cumesh, const geometry &);

struct sample_result {
	glm::vec3 *points;
	glm::vec3 *barys;
	uint32_t *indices;
	uint32_t point_count;
	compute_api api;
};

sample_result sample_result_alloc(uint32_t, compute_api);

void sample(sample_result, const geometry &, float);
void sample(sample_result, const cumesh &, float);
void memcpy(sample_result, const sample_result &);

// Closest point caching acceleration structure and arguments
struct closest_point_kinfo {
	glm::vec3 *points;
	glm::vec3 *closest;
	glm::vec3 *bary;
	float *distances;
	uint32_t *triangles;
	uint32_t point_count;
	compute_api api;
};

closest_point_kinfo closest_point_kinfo_alloc(uint32_t, compute_api);
void memcpy(closest_point_kinfo, const closest_point_kinfo &);

void brute_closest_point(const geometry &, const closest_point_kinfo &);
void brute_closest_point(const cumesh &, const closest_point_kinfo &);

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
	float precache_query(const std::vector <glm::vec3> &points);

	// glm::vec3 query(const glm::vec3 &p) const;
	
	// Returns closest point, barycentric coordinates, distance, and triangle index
	std::tuple <glm::vec3, glm::vec3, float, uint32_t> query(const glm::vec3 &p) const;

	void query(const std::vector <glm::vec3> &,
			std::vector <glm::vec3> &,
			std::vector <glm::vec3> &,
			std::vector <float> &,
			std::vector <uint32_t> &) const;

	void precache_device();
	void query_device(closest_point_kinfo kinfo);
};

struct bin_vector {
	uint32_t *triangles;
	uint32_t *index0;
	uint32_t *index1;
};

struct dev_ipqas {
	bin_vector xy;
	bin_vector xz;
	bin_vector yz;

	uint32_t resolution;
	glm::vec3 min;
	glm::vec3 max;
};

struct ipqas {
	using bin = std::vector <uint32_t>;

	geometry ref;
	uint32_t resolution;

	std::vector <bin> bins_xy;
	std::vector <bin> bins_xz;
	std::vector <bin> bins_yz;

	glm::vec3 ext_min;
	glm::vec3 ext_max;

	ipqas(const geometry &ref_, uint32_t resolution_)
			: ref(ref_), resolution(resolution_) {
		ext_min = glm::vec3(std::numeric_limits <float> ::max());
		ext_max = glm::vec3(-std::numeric_limits <float> ::max());

		for (const glm::vec3 &vertex : ref.vertices) {
			ext_min = glm::min(ext_min, vertex);
			ext_max = glm::max(ext_max, vertex);
		}

		bins_xy.resize(resolution * resolution);
		bins_xz.resize(resolution * resolution);
		bins_yz.resize(resolution * resolution);

		for (uint32_t i = 0; i < ref.triangles.size(); i++) {
			const glm::uvec3 &tri = ref.triangles[i];

			// Find the bounding box of the triangle
			glm::vec3 min = glm::vec3(std::numeric_limits <float> ::max());
			glm::vec3 max = glm::vec3(std::numeric_limits <float> ::lowest());

			for (uint32_t j = 0; j < 3; ++j) {
				glm::vec3 v = ref.vertices[tri[j]];
				v = (v - ext_min) / (ext_max - ext_min);
				min = glm::min(min, v);
				max = glm::max(max, v);
			}

			// Find the bins that the triangle overlaps
			glm::vec3 min_bin = glm::floor(min * glm::vec3(resolution));
			glm::vec3 max_bin = glm::ceil(max * glm::vec3(resolution));

			for (uint32_t x = min_bin.x; x < max_bin.x; x++) {
				for (uint32_t y = min_bin.y; y < max_bin.y; y++)
					bins_xy[x + y * resolution].push_back(i);
			}

			for (uint32_t x = min_bin.x; x < max_bin.x; x++) {
				for (uint32_t z = min_bin.z; z < max_bin.z; z++)
					bins_xz[x + z * resolution].push_back(i);
			}

			for (uint32_t y = min_bin.y; y < max_bin.y; y++) {
				for (uint32_t z = min_bin.z; z < max_bin.z; z++)
					bins_yz[y + z * resolution].push_back(i);
			}
		}
	}

	float query(const glm::vec3 &) const;

	void query(const std::vector <glm::vec3> &vs, std::vector <float> &status) const {
		#pragma omp parallel for
		for (uint32_t i = 0; i < vs.size(); i++)
			status[i] *= query(vs[i]);
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

void sdf(const geometry &, ipqas &, closest_point_kinfo &, std::vector <float> &);

void sdf(cas_grid &, ipqas &, const std::vector <glm::vec3> &,
		std::vector <glm::vec3> &,
		std::vector <glm::vec3> &,
		std::vector <float> &,
		std::vector <uint32_t> &);
