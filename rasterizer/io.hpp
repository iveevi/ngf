#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>

#include <glm/glm.hpp>

constexpr int32_t LAYERS = 4;

struct Tensor : std::vector <float> {
	int32_t width;
	int32_t height;
};

struct NGF {
	std::vector <glm::ivec4> patches;
	std::vector <glm::vec4> vertices;
	std::vector <float> features;

	uint32_t patch_count;
	uint32_t feature_size;

	std::array <Tensor, LAYERS> weights;
	std::array <Tensor, LAYERS> biases;

	static NGF load(const std::filesystem::path &);
};

struct Texture {
	int width;
	int height;
	int channels;
	std::vector <uint8_t> pixels;

	static Texture load(const std::filesystem::path &);
};
