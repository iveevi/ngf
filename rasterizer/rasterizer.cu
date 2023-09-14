#include <array>
#include <fstream>
#include <iostream>
#include <vector>

#include <glm/glm.hpp>

#include <littlevk/littlevk.hpp>

#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/point_cloud.h>

struct neural_network {
	// Host buffers
	std::vector <float> Wm0;
	std::vector <float> Bs0;

	std::vector <float> Wm1;
	std::vector <float> Bs1;

	std::vector <float> Wm2;
	std::vector <float> Bs2;

	// Device (CUDA) buffers
	float *d_Wm0;
	float *d_Bs0;
	float *d_Wm1;
	float *d_Bs1;
	float *d_Wm2;
	float *d_Bs2;

	uint32_t W0;
	uint32_t H0;

	uint32_t W1;
	uint32_t H1;

	uint32_t W2;
	uint32_t H2;
} g_dnn;

void dnn_read(FILE *file)
{
	// Read neural network data
	printf("Neural Network:\n");

	uint32_t W0 = 0;
	uint32_t H0 = 0;

	fread((char *) &W0, sizeof(W0), 1, file);
	fread((char *) &H0, sizeof(H0), 1, file);
	printf("  > Layer 1: %d x %d + %d\n", W0, H0, W0);

	std::vector <float> Wm0(W0 * H0);
	std::vector <float> Bs0(W0);

	fread((char *) Wm0.data(), sizeof(float), W0 * H0, file);
	fread((char *) Bs0.data(), sizeof(float), W0, file);

	uint32_t W1 = 0;
	uint32_t H1 = 0;

	fread((char *) &W1, sizeof(W1), 1, file);
	fread((char *) &H1, sizeof(H1), 1, file);
	printf("  > Layer 2: %d x %d + %d\n", W1, H1, W1);

	std::vector <float> Wm1(W1 * H1);
	std::vector <float> Bs1(W1);

	fread((char *) Wm1.data(), sizeof(float), W1 * H1, file);
	fread((char *) Bs1.data(), sizeof(float), W1, file);

	uint32_t W2 = 0;
	uint32_t H2 = 0;

	fread((char *) &W2, sizeof(W2), 1, file);
	fread((char *) &H2, sizeof(H2), 1, file);
	printf("  > Layer 3: %d x %d + %d\n", W2, H2, W2);

	std::vector <float> Wm2(W2 * H2);
	std::vector <float> Bs2(W2);

	fread((char *) Wm2.data(), sizeof(float), W2 * H2, file);
	fread((char *) Bs2.data(), sizeof(float), W2, file);

	g_dnn.W0 = W0;
	g_dnn.H0 = H0;

	g_dnn.W1 = W1;
	g_dnn.H1 = H1;

	g_dnn.W2 = W2;
	g_dnn.H2 = H2;

	g_dnn.Wm0 = Wm0;
	g_dnn.Bs0 = Bs0;

	g_dnn.Wm1 = Wm1;
	g_dnn.Bs1 = Bs1;

	g_dnn.Wm2 = Wm2;
	g_dnn.Bs2 = Bs2;

	// Transfer to device
	cudaMalloc(&g_dnn.d_Wm0, W0 * H0 * sizeof(float));
	cudaMalloc(&g_dnn.d_Bs0, W0 * sizeof(float));

	cudaMalloc(&g_dnn.d_Wm1, W1 * H1 * sizeof(float));
	cudaMalloc(&g_dnn.d_Bs1, W1 * sizeof(float));

	cudaMalloc(&g_dnn.d_Wm2, W2 * H2 * sizeof(float));
	cudaMalloc(&g_dnn.d_Bs2, W2 * sizeof(float));

	cudaMemcpy(g_dnn.d_Wm0, Wm0.data(), W0 * H0 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(g_dnn.d_Bs0, Bs0.data(), W0 * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(g_dnn.d_Wm1, Wm1.data(), W1 * H1 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(g_dnn.d_Bs1, Bs1.data(), W1 * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(g_dnn.d_Wm2, Wm2.data(), W2 * H2 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(g_dnn.d_Bs2, Bs2.data(), W2 * sizeof(float), cudaMemcpyHostToDevice);
}

struct subdivision_complexes {
	// Host buffers
	std::vector <glm::uvec4> complexes;
	std::vector <glm::vec3>  vertices;
	std::vector <float>      features;

	// Device (CUDA) buffers
	glm::uvec4	         *d_complexes;
	glm::vec3	         *d_vertices;
	float		         *d_features;

	uint32_t	         complex_count;
	uint32_t	         vertex_count;
	uint32_t                 feature_count;
	uint32_t	         point_count;
	uint32_t	         matrix_size;
} g_sdc;

void sdc_read(FILE *file)
{
	// Read size data
	uint32_t sizes[5];
	fread((char *) sizes, sizeof(uint32_t), 5, file);

	g_sdc.complex_count = sizes[0];
	g_sdc.vertex_count = sizes[1];
	g_sdc.feature_count = sizes[2];
	g_sdc.point_count = sizes[3];
	g_sdc.matrix_size = sizes[4];

	printf("Neural Subdivision Complexes:\n");
	printf("  > %4d complexes\n", g_sdc.complex_count);
	printf("  > %4d vertices\n", g_sdc.vertex_count);
	printf("  > %4d encoding features\n", g_sdc.feature_count);
	printf("  > %4d point features\n", g_sdc.point_count);
	printf("  > %4d matrix size\n", g_sdc.matrix_size);

	// Read complexes data
	std::vector <glm::uvec4> complexes(g_sdc.complex_count);
	fread((char *) complexes.data(), sizeof(glm::uvec4), g_sdc.complex_count, file);

	// Read corner vertices and their features
	std::vector <glm::vec3> vertices(g_sdc.vertex_count);
	fread((char *) vertices.data(), sizeof(glm::vec3), g_sdc.vertex_count, file);

	std::vector <float> features(g_sdc.vertex_count * g_sdc.feature_count);
	fread((char *) features.data(), sizeof(float), g_sdc.vertex_count * g_sdc.feature_count, file);

	g_sdc.complexes = complexes;
	g_sdc.vertices = vertices;
	g_sdc.features = features;

	// Transfer to device
	cudaMalloc(&g_sdc.d_complexes, g_sdc.complex_count * sizeof(glm::uvec4));
	cudaMalloc(&g_sdc.d_vertices, g_sdc.vertex_count * sizeof(glm::vec3));
	cudaMalloc(&g_sdc.d_features, g_sdc.vertex_count * g_sdc.feature_count * sizeof(float));

	cudaMemcpy(g_sdc.d_complexes, complexes.data(), g_sdc.complex_count * sizeof(glm::uvec4), cudaMemcpyHostToDevice);
	cudaMemcpy(g_sdc.d_vertices, vertices.data(), g_sdc.vertex_count * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	cudaMemcpy(g_sdc.d_features, features.data(), g_sdc.vertex_count * g_sdc.feature_count * sizeof(float), cudaMemcpyHostToDevice);
}

// template <size_t rate>
// struct uniform_nsc_sample {
// 	float *lerped_inputs; // vector of (3 + POINT_COUNT)
// 	float *lerped_matrix; // vector of (3 * MATRIX_SIZE)
//
// 	static uniform_nsc_sample make() {
// 		uniform_nsc_sample sample;
//
// 		uint32_t lerped_inputs_size = (3 + g_sdc.point_count) * rate * rate * g_sdc.complex_count;
// 		uint32_t lerped_matrix_size = 3 * g_sdc.matrix_size * rate * rate * g_sdc.complex_count;
//
// 		std::vector <float> lerped_inputs(lerped_inputs_size);
// 		std::vector <float> lerped_matrix(lerped_matrix_size);
//
// 		uint32_t i = 0;
// 		for (uint32_t c = 0; c < g_sdc.complex_count; c++) {
// 			// Extract corner vertices and features of the complex
// 			glm::vec3 v00 = g_sdc.vertices[g_sdc.complexes[c].x];
// 			glm::vec3 v10 = g_sdc.vertices[g_sdc.complexes[c].y];
// 			glm::vec3 v01 = g_sdc.vertices[g_sdc.complexes[c].w];
// 			glm::vec3 v11 = g_sdc.vertices[g_sdc.complexes[c].z];
//
// 			std::vector <float> f00(g_sdc.feature_count);
//
// 			// Interpolation
// 		}
// 	}
// };

int main(int argc, char *argv[])
{
	// Expect a filename
	if (argc < 2) {
		printf("./rasterizer <nsc binary>\n");
		return 1;
	}

	// Open the file
	FILE *file = fopen(argv[1], "rb");
	if (!file) {
		fprintf(stderr, "Could not open file %s\n", argv[1]);
		return 1;
	}

	sdc_read(file);
	dnn_read(file);

	constexpr uint32_t rate = 5;

	uint32_t vertex_count = rate * rate * g_sdc.complex_count;
	uint32_t quad_count = (rate - 1) * (rate - 1) * g_sdc.complex_count;

	std::vector <glm::vec3> interpolated_base(vertex_count, { 0.0f, 0.0f, 0.0f });
	std::vector <std::array <uint32_t, 4>> interpolated_indices(quad_count, { 0, 0, 0, 0 });

	std::vector <float> interpolated_features(vertex_count * g_sdc.feature_count, 0.0f);

	uint32_t i = 0;
	uint32_t j = 0;

	for (uint32_t c = 0; c < g_sdc.complex_count; c++) {
		glm::vec3 v00 = g_sdc.vertices[g_sdc.complexes[c].x];
		glm::vec3 v10 = g_sdc.vertices[g_sdc.complexes[c].y];
		glm::vec3 v01 = g_sdc.vertices[g_sdc.complexes[c].w];
		glm::vec3 v11 = g_sdc.vertices[g_sdc.complexes[c].z];

		auto get_feature = [&](uint32_t v) {
			std::vector <float> f(g_sdc.feature_count);
			for (uint32_t i = 0; i < g_sdc.feature_count; i++)
				f[i] = g_sdc.features[v * g_sdc.feature_count + i];
			return f;
		};

		auto set_feature = [&](uint32_t v, std::vector <float> &f) {
			for (uint32_t i = 0; i < g_sdc.feature_count; i++)
				interpolated_features[v * g_sdc.feature_count + i] = f[i];
		};

		std::vector <float> f00 = get_feature(g_sdc.complexes[c].x);
		std::vector <float> f10 = get_feature(g_sdc.complexes[c].y);
		std::vector <float> f01 = get_feature(g_sdc.complexes[c].w);
		std::vector <float> f11 = get_feature(g_sdc.complexes[c].z);

		auto feature_lerp = [&](std::vector <float> &f0, std::vector <float> &f1, float k) {
			std::vector <float> f(g_sdc.feature_count);
			for (uint32_t i = 0; i < g_sdc.feature_count; i++)
				f[i] = glm::mix(f0[i], f1[i], k);
			return f;
		};

		for (uint32_t ix = 0; ix < rate; ix++) {
			for (uint32_t iy = 0; iy < rate; iy++) {
				float x = (float) ix / (rate - 1);
				float y = (float) iy / (rate - 1);

				glm::vec3 v0 = glm::mix(v00, v10, x);
				glm::vec3 v1 = glm::mix(v01, v11, x);
				glm::vec3 v = glm::mix(v0, v1, y);

				std::vector <float> f0 = feature_lerp(f00, f10, x);
				std::vector <float> f1 = feature_lerp(f01, f11, x);
				std::vector <float> f = feature_lerp(f0, f1, y);

				interpolated_base[i] = v;
				set_feature(i++, f);
			}
		}

		uint32_t base = c * rate * rate;
		for (uint32_t ix = 0; ix < rate - 1; ix++) {
			for (uint32_t iy = 0; iy < rate - 1; iy++) {
				uint32_t i0 = ix + rate * iy;
				uint32_t i1 = i0 + 1;
				uint32_t i2 = i0 + rate;
				uint32_t i3 = i2 + 1;

				interpolated_indices[j++] = { base + i0, base + i1, base + i3, base + i2 };
			}
		}
	}

	// Activation function
	auto elu = [](float x) {
		return x >= 0.0f ? x : glm::exp(x) - 1.0f;
	};

	// Evaluate the deep neural network on the interpolated features
	std::vector <float> intermediate_hidden1(vertex_count * g_dnn.W0);

	for (uint32_t i = 0; i < vertex_count; i++) {
		auto get_input = [&](uint32_t v) {
			std::vector <float> f(g_sdc.feature_count);
			for (uint32_t i = 0; i < g_sdc.feature_count; i++)
				f[i] = g_sdc.features[v * g_sdc.feature_count + i];
			return f;
		};

		std::vector <float> input = get_input(i);

		// Evaluate the first layer (matrix multiplication)
		std::vector <float> hidden(g_dnn.W0);

		const std::vector <float> &Wm = g_dnn.Wm0;
		for (uint32_t j = 0; j < g_dnn.W0; j++) {
			float sum = 0.0f;
			for (uint32_t k = 0; k < g_sdc.feature_count; k++)
				sum += input[k] * Wm[k * g_dnn.W0 + j];
			hidden[j] = sum;
		}

		// Add the bias
		const std::vector <float> &b = g_dnn.Bs0;
		// TODO: shared memory

		for (uint32_t j = 0; j < g_dnn.W0; j++)
			hidden[j] += b[j];

		// // Apply the activation function
		for (uint32_t j = 0; j < g_dnn.W0; j++)
			hidden[j] = elu(hidden[j]);

		// Store the result
		for (uint32_t j = 0; j < g_dnn.W0; j++)
			intermediate_hidden1[i * g_dnn.W0 + j] = hidden[j];
	}

	std::vector <float> intermediate_hidden2(vertex_count * g_dnn.W1);

	for (uint32_t i = 0; i < vertex_count; i++) {
		auto get_input = [&](uint32_t v) {
			std::vector <float> f(g_dnn.W0);
			for (uint32_t i = 0; i < g_dnn.W0; i++)
				f[i] = intermediate_hidden1[v * g_dnn.W0 + i];
			return f;
		};

		std::vector <float> input = get_input(i);

		// Evaluate the first layer (matrix multiplication)
		std::vector <float> hidden(g_dnn.W1);

		const std::vector <float> &Wm = g_dnn.Wm1;
		for (uint32_t j = 0; j < g_dnn.W1; j++) {
			float sum = 0.0f;
			for (uint32_t k = 0; k < g_dnn.W0; k++)
				sum += input[k] * Wm[k * g_dnn.W1 + j];
			hidden[j] = sum;
		}

		// Add the bias
		const std::vector <float> &b = g_dnn.Bs1;
		// TODO: shared memory

		for (uint32_t j = 0; j < g_dnn.W1; j++)
			hidden[j] += b[j];

		// // Apply the activation function
		for (uint32_t j = 0; j < g_dnn.W1; j++)
			hidden[j] = elu(hidden[j]);

		// Store the result
		for (uint32_t j = 0; j < g_dnn.W1; j++)
			intermediate_hidden2[i * g_dnn.W1 + j] = hidden[j];
	}

	std::vector <float> final_vertices(vertex_count * 3);

	for (uint32_t i = 0; i < vertex_count; i++) {
		auto get_input = [&](uint32_t v) {
			std::vector <float> f(g_dnn.W1);
			for (uint32_t i = 0; i < g_dnn.W1; i++)
				f[i] = intermediate_hidden2[v * g_dnn.W1 + i];
			return f;
		};

		std::vector <float> input = get_input(i);

		// Evaluate the first layer (matrix multiplication)
		std::vector <float> hidden(3);

		const std::vector <float> &Wm = g_dnn.Wm2;
		for (uint32_t j = 0; j < 3; j++) {
			float sum = 0.0f;
			for (uint32_t k = 0; k < g_dnn.W1; k++)
				sum += input[k] * Wm[k * 3 + j];
			hidden[j] = sum;
		}

		// Add the bias
		const std::vector <float> &b = g_dnn.Bs2;

		for (uint32_t j = 0; j < 3; j++)
			hidden[j] += b[j];

		// Store the result
		for (uint32_t j = 0; j < 3; j++)
			final_vertices[i * 3 + j] = hidden[j] + interpolated_base[i][j];
	}

	std::vector <glm::vec3> vertices(vertex_count);
	for (uint32_t i = 0; i < vertex_count; i++) {
		vertices[i] = glm::vec3(final_vertices[i * 3 + 0], final_vertices[i * 3 + 1], final_vertices[i * 3 + 2]);
		printf("%f %f %f\n", final_vertices[i * 3 + 0], final_vertices[i * 3 + 1], final_vertices[i * 3 + 2]);
	}

	namespace ps = polyscope;

	ps::init();
	// ps::registerSurfaceMesh("mesh", interpolated_base, interpolated_indices);
	ps::registerPointCloud("points", vertices);

	// auto cs = *reinterpret_cast <std::vector <std::array <uint32_t, 4>> *> (&g_sdc.complexes);
	// ps::registerSurfaceMesh("complexes", g_sdc.vertices, cs);
	ps::show();
}
