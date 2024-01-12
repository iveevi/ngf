#pragma once

#include <filesystem>
#include <vector>

#include <glm/glm.hpp>

struct Mesh {
	std::vector <glm::vec3> vertices;
	std::vector <glm::vec3> normals;
	std::vector <glm::uvec3> triangles;

	static std::vector <Mesh> load(const std::filesystem::path &);
	static Mesh normalize(const Mesh &);
};

std::vector <glm::vec3> smooth_normals(const Mesh &);
std::vector <float> interleave_attributes(const Mesh &);
