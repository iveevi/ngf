#pragma once

#include <vector>

#include <glm/glm.hpp>

#include "mesh.hpp"

// TODO: python (pybind11) bindings
// TODO: optimize in a later project...

struct cumesh {
	glm::vec3 *vertices;
	glm::uvec3 *triangles;

	uint32_t vertex_count = 0;
	uint32_t triangle_count = 0;
};

cumesh cumesh_alloc(const Mesh &);
void cumesh_reload(cumesh, const Mesh &);

struct sample_result {
	glm::vec3 *points;
	glm::vec3 *barys;
	glm::uvec3 *triangles;
	uint32_t *indices;
	uint32_t *dup;
	uint32_t point_count;
};

sample_result sample_result_alloc(uint32_t);
void sample(sample_result, cumesh, float);

// Closest point caching acceleration structure and arguments
struct closest_point_kinfo {
	glm::vec3 *points;
	glm::vec3 *closest;
	glm::vec3 *bary;
	uint32_t *triangles;

	uint32_t point_count;
};

closest_point_kinfo closest_point_kinfo_alloc(uint32_t);
void brute_closest_point(cumesh, closest_point_kinfo); // TODO: cumesh method?

// inline closest_point_kinfo closest_point_kinfo_alloc(uint32_t point_count)
// {
// 	closest_point_kinfo kinfo;
//
// 	cudaMalloc(&kinfo.points, point_count * sizeof(glm::vec3));
// 	cudaMalloc(&kinfo.closest, point_count * sizeof(glm::vec3));
// 	cudaMalloc(&kinfo.bary, point_count * sizeof(glm::vec3));
// 	cudaMalloc(&kinfo.triangles, point_count * sizeof(uint32_t));
//
// 	kinfo.point_count = point_count;
//
// 	return kinfo;
// }

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
	Mesh ref;

	glm::vec3 min;
	glm::vec3 max;

	uint32_t resolution;
	glm::vec3 bin_size;

	using query_bin = std::vector <uint32_t>;
	std::vector <query_bin> overlapping_triangles;
	std::vector <query_bin> query_triangles;

	dev_cas_grid dev_cas;

	// Construct from mesh
	cas_grid(const Mesh &ref_, uint32_t resolution_);

	uint32_t to_index(const glm::ivec3 &bin) const;
	uint32_t to_index(const glm::vec3 &p) const;

	std::unordered_set <uint32_t> closest_triangles(const glm::vec3 &p) const;

	bool precache_query(const glm::vec3 &p);
	bool precache_query(const std::vector <glm::vec3> &points);

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
