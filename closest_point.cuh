#pragma once

#include <vector>

#include <glm/glm.hpp>

#include "mesh.hpp"

// Closest point caching acceleration structure and arguments
struct closest_point_kinfo {
	glm::vec3 *points;
	glm::vec3 *closest;

	uint32_t point_count;
};

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

	glm::vec3 query(const glm::vec3 &p) const;
	void query(const std::vector <glm::vec3> &sources, std::vector <glm::vec3> &dst) const;

	void precache_device();
	void query_device(closest_point_kinfo kinfo);
};
