#include <fstream>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#include "io.hpp"
#include "microlog.h"

NGF NGF::load(const std::filesystem::path &path)
{
	NGF ngf;

	std::ifstream fin(path);
	ulog_assert(fin.good(), "Bad ngf file %s\n", path.c_str());

	int32_t sizes[3];
	fin.read(reinterpret_cast <char *> (sizes), sizeof(sizes));
	ulog_info("ngf io", "%d patches, %d vertices, %d feature size\n", sizes[0], sizes[1], sizes[2]);

	std::vector <glm::ivec4> patches;
	std::vector <glm::vec3> vertices;
	std::vector <float> features;

	patches.resize(sizes[0]);
	vertices.resize(sizes[1]);
	features.resize(sizes[1] * sizes[2]);

	ngf.patch_count = sizes[0];
	ngf.feature_size = sizes[2];

	fin.read(reinterpret_cast <char *> (vertices.data()), vertices.size() * sizeof(glm::vec3));
	fin.read(reinterpret_cast <char *> (features.data()), features.size() * sizeof(float));
	fin.read(reinterpret_cast <char *> (patches.data()), patches.size() * sizeof(glm::ivec4));

	ulog_info("ngf io", "read patches data\n");

	std::array <Tensor, LAYERS> weights;
	for (int32_t i = 0; i < LAYERS; i++) {
		int32_t sizes[2];
		fin.read(reinterpret_cast <char *> (sizes), sizeof(sizes));
		ulog_info("ngf io", "weight matrix with size %d x %d\n", sizes[0], sizes[1]);

		Tensor w;
		w.width = sizes[0];
		w.height = sizes[1];
		w.resize(sizes[0] * sizes[1]);
		fin.read(reinterpret_cast <char *> (w.data()), w.size() * sizeof(float));

		weights[i] = w;
	}

	std::array <Tensor, LAYERS> biases;
	for (int32_t i = 0; i < LAYERS; i++) {
		int32_t size;
		fin.read(reinterpret_cast <char *> (&size), sizeof(size));
		ulog_info("ngf io", "bias vector with size %d\n", size);

		Tensor w;
		w.width = size;
		w.height = 1;
		w.resize(size);
		fin.read(reinterpret_cast <char *> (w.data()), w.size() * sizeof(float));

		biases[i] = w;
	}

	ngf.patches = patches;
	ngf.features = features;
	ngf.weights = weights;
	ngf.biases = biases;

	// TODO: put this elsewhere..
	// Need special care for vertices to align them properly
	ngf.vertices.resize(vertices.size());
	for (int32_t i = 0; i < vertices.size(); i++)
		ngf.vertices[i] = glm::vec4(vertices[i], 0.0f);

	return ngf;
}

Texture Texture::load(const std::filesystem::path &path)
{
	std::string tr = path.string();
	if (!std::filesystem::exists(path)) {
		ulog_error("load_texture", "load_texture: could not find path : %s\n", tr.c_str());
		return {};
	}

	int width;
	int height;
	int channels;

	stbi_set_flip_vertically_on_load(true);

	uint8_t *pixels = stbi_load(tr.c_str(), &width, &height, &channels, 4);

	std::vector <uint8_t> vector;
	vector.resize(width * height * 4);
	memcpy(vector.data(), pixels, vector.size() * sizeof(uint8_t));

	return Texture {
		.width = width,
		.height = height,
		.channels = channels,
		.pixels = vector
	};
}

