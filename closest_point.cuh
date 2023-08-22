#pragma once

#include <vector>

#include <glm/glm.hpp>

#include "mesh.hpp"

// Closest point caching acceleration structure and arguments
struct closest_point_kinfo {
	glm::vec3 *points;
	glm::vec3 *closest;
	glm::vec3 *bary;
	uint32_t *triangles;

	uint32_t point_count;
};

inline closest_point_kinfo closest_point_kinfo_alloc(uint32_t point_count)
{
	closest_point_kinfo kinfo;

	cudaMalloc(&kinfo.points, point_count * sizeof(glm::vec3));
	cudaMalloc(&kinfo.closest, point_count * sizeof(glm::vec3));
	cudaMalloc(&kinfo.bary, point_count * sizeof(glm::vec3));
	cudaMalloc(&kinfo.triangles, point_count * sizeof(uint32_t));

	kinfo.point_count = point_count;

	return kinfo;
}

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
	uint32_t query_primitive(const glm::vec3 &p) const;

	void query(const std::vector <glm::vec3> &sources, std::vector <glm::vec3> &dst) const;

	void precache_device();
	void query_device(closest_point_kinfo kinfo);
};

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

