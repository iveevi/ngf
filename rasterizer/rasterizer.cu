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

// TODO: perform a batched version of this...
std::vector <float> matmul(const std::vector <float> W, const std::vector <float> &X, const std::vector <float> &B, size_t in, size_t out)
{
	// Check sizes
	if (W.size() != in * out) {
		printf("Error: matmul: W size mismatch\n");
		abort();
	}

	if (X.size() != in) {
		printf("Error: matmul: X size mismatch\n");
		abort();
	}

	if (B.size() != out) {
		printf("Error: matmul: B size mismatch\n");
		abort();
	}

	// Prepare result
	std::vector <float> Y(out);

	// Perform matrix multiplication and add bias
	for (size_t i = 0; i < out; i++) {
		float sum = 0.0f;

		for (size_t j = 0; j < in; j++)
			sum += W[i * in + j] * X[j];

		Y[i] = sum + B[i];
	}

	return Y;
}

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
	std::vector <glm::vec2>  normals;
	std::vector <float>      features;

	// Device (CUDA) buffers
	glm::uvec4	         *d_complexes;
	glm::vec3	         *d_vertices;
	float		         *d_features;

	uint32_t	         complex_count;
	uint32_t	         vertex_count;
	uint32_t                 feature_count;
} g_sdc;

void sdc_read(FILE *file)
{
	// Read size data
	uint32_t sizes[3];
	fread((char *) sizes, sizeof(uint32_t), 3, file);

	g_sdc.complex_count = sizes[0];
	g_sdc.vertex_count = sizes[1];
	g_sdc.feature_count = sizes[2];

	printf("Neural Subdivision Complexes:\n");
	printf("  > %4d complexes\n", g_sdc.complex_count);
	printf("  > %4d vertices\n", g_sdc.vertex_count);
	printf("  > %4d encoding features\n", g_sdc.feature_count);

	// Read complexes data
	std::vector <glm::uvec4> complexes(g_sdc.complex_count);
	fread((char *) complexes.data(), sizeof(glm::uvec4), g_sdc.complex_count, file);

	// Read corner vertices, normals, and their features
	std::vector <glm::vec3> vertices(g_sdc.vertex_count);
	fread((char *) vertices.data(), sizeof(glm::vec3), g_sdc.vertex_count, file);

	// Normals are spherical coordinates (omitting radius)
	std::vector <glm::vec2> normals(g_sdc.vertex_count);
	fread((char *) normals.data(), sizeof(glm::vec2), g_sdc.vertex_count, file);

	std::vector <float> features(g_sdc.vertex_count * g_sdc.feature_count);
	fread((char *) features.data(), sizeof(float), g_sdc.vertex_count * g_sdc.feature_count, file);

	g_sdc.complexes = complexes;
	g_sdc.vertices = vertices;
	g_sdc.normals = normals;
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

	constexpr uint32_t rate = 16;

	uint32_t vertex_count = rate * rate * g_sdc.complex_count;
	uint32_t quad_count = (rate - 1) * (rate - 1) * g_sdc.complex_count;

	std::vector <glm::vec3> lerped_P;
	std::vector <glm::vec2> lerped_N;
	std::vector <float>     lerped_E;

	lerped_P.resize(vertex_count);
	lerped_N.resize(vertex_count);
	lerped_E.resize(vertex_count * g_sdc.feature_count);

	uint32_t i = 0;
	for (uint32_t c = 0; c < g_sdc.complex_count; c++) {
		const glm::uvec4 &complex = g_sdc.complexes[c];

		glm::vec3 v00 = g_sdc.vertices[complex.x];
		glm::vec3 v10 = g_sdc.vertices[complex.y];
		glm::vec3 v01 = g_sdc.vertices[complex.w];
		glm::vec3 v11 = g_sdc.vertices[complex.z];

		glm::vec2 n00 = g_sdc.normals[complex.x];
		glm::vec2 n10 = g_sdc.normals[complex.y];
		glm::vec2 n01 = g_sdc.normals[complex.w];
		glm::vec2 n11 = g_sdc.normals[complex.z];

		auto get_feature = [&](uint32_t v) {
			std::vector <float> f(g_sdc.feature_count);
			for (uint32_t i = 0; i < g_sdc.feature_count; i++)
				f[i] = g_sdc.features[v * g_sdc.feature_count + i];
			return f;
		};

		auto set_feature = [&](uint32_t v, std::vector <float> &f) {
			for (uint32_t i = 0; i < g_sdc.feature_count; i++)
				lerped_E[v * g_sdc.feature_count + i] = f[i];
		};

		std::vector <float> f00 = get_feature(complex.x);
		std::vector <float> f10 = get_feature(complex.y);
		std::vector <float> f01 = get_feature(complex.w);
		std::vector <float> f11 = get_feature(complex.z);

		for (uint32_t ix = 0; ix < rate; ix++) {
			for (uint32_t iy = 0; iy < rate; iy++) {
				float u = (float) ix / (rate - 1);
				float v = (float) iy / (rate - 1);

				{
					glm::vec3 lp00 = v00 * u * v;
					glm::vec3 lp10 = v10 * (1.0f - u) * v;
					glm::vec3 lp01 = v01 * u * (1.0f - v);
					glm::vec3 lp11 = v11 * (1.0f - u) * (1.0f - v);
					lerped_P[i] = lp00 + lp10 + lp01 + lp11;
				}

				{
					glm::vec2 ln00 = n00 * u * v;
					glm::vec2 ln10 = n10 * (1.0f - u) * v;
					glm::vec2 ln01 = n01 * u * (1.0f - v);
					glm::vec2 ln11 = n11 * (1.0f - u) * (1.0f - v);
					lerped_N[i] = ln00 + ln10 + ln01 + ln11;
				}

				{
					std::vector <float> f(g_sdc.feature_count);
					for (uint32_t k = 0; k < g_sdc.feature_count; k++) {
						float f00k = f00[k] * u * v;
						float f10k = f10[k] * (1.0f - u) * v;
						float f01k = f01[k] * u * (1.0f - v);
						float f11k = f11[k] * (1.0f - u) * (1.0f - v);
						f[k] = f00k + f10k + f01k + f11k;
					}

					set_feature(i, f);
				}

				i++;
			}
		}
	}

	constexpr uint32_t lines = 3;

	// {
	// 	printf("Lerped base: %lu\n", lerped_P.size());
	// 	for (uint32_t i = 0; i < lines; i++)
	// 		printf("  > %u: %f %f %f\n", i, lerped_P[i].x, lerped_P[i].y, lerped_P[i].z);
	// 	printf("  ...\n");
	// 	for (uint32_t i = lerped_P.size() - lines; i < lerped_P.size(); i++)
	// 		printf("  > %u: %f %f %f\n", i, lerped_P[i].x, lerped_P[i].y, lerped_P[i].z);
	// }
	//
	// {
	// 	printf("Lerped normals: %lu\n", lerped_N.size());
	// 	for (uint32_t i = 0; i < lines; i++)
	// 		printf("  > %u: %f %f\n", i, lerped_N[i].x, lerped_N[i].y);
	// 	printf("  ...\n");
	// 	for (uint32_t i = lerped_N.size() - lines; i < lerped_N.size(); i++)
	// 		printf("  > %u: %f %f\n", i, lerped_N[i].x, lerped_N[i].y);
	// }
	//
	// {
	// 	int size = lerped_E.size()/g_sdc.feature_count;
	// 	printf("Lerped features: %lu\n", size);
	// 	for (uint32_t i = 0; i < lines; i++) {
	// 		printf("  > %u: [ ", i);
	// 		for (uint32_t k = 0; k < lines; k++)
	// 			printf("%f ", lerped_E[i * g_sdc.feature_count + k]);
	// 		printf(" ... ");
	// 		for (uint32_t k = g_sdc.feature_count - lines; k < g_sdc.feature_count; k++)
	// 			printf("%f ", lerped_E[i * g_sdc.feature_count + k]);
	// 		printf("]\n");
	// 	}
	// 	printf("  ...\n");
	// 	for (uint32_t i = size - lines; i < size; i++) {
	// 		printf("  > %u: [ ", i);
	// 		for (uint32_t k = 0; k < lines; k++)
	// 			printf("%f ", lerped_E[i * g_sdc.feature_count + k]);
	// 		printf(" ... ");
	// 		for (uint32_t k = g_sdc.feature_count - lines; k < g_sdc.feature_count; k++)
	// 			printf("%f ", lerped_E[i * g_sdc.feature_count + k]);
	// 		printf("]\n");
	// 	}
	// }

	// Construct the network input with embeddings
	// TODO: get these from the file as well...
	constexpr uint32_t L = 8;
	constexpr uint32_t K = 16;

	uint32_t embedded_size = g_sdc.feature_count + (2 * L + 1) * 3 + (2 * K + 2);
	printf("Embedded size: %u\n", embedded_size);

	if (embedded_size != g_dnn.H0) {
		printf("ERROR: embedded_size != g_dnn.W0 [%u != %u]\n", embedded_size, g_dnn.H0);
		abort();
	}

	std::vector <float> embedded;
	embedded.resize(vertex_count * embedded_size);

	for (uint32_t i = 0; i < vertex_count; i++) {
		float *vembedded = &embedded[i * embedded_size];

		// First copy the feature
		float *lfeature = &lerped_E[i * g_sdc.feature_count];
		for (uint32_t k = 0; k < g_sdc.feature_count; k++)
			vembedded[k] = lfeature[k];

		// Positional encoding
		const glm::vec3 &p = lerped_P[i];

		std::vector <float> pos_enc;
		pos_enc.push_back(p.x);
		pos_enc.push_back(p.y);
		pos_enc.push_back(p.z);

		for (uint32_t j = 1; j <= L; j++) {
			glm::vec3 sp = glm::sin(powf(2.0f, j) * p);
			glm::vec3 cp = glm::cos(powf(2.0f, j) * p);
			pos_enc.push_back(sp.x);
			pos_enc.push_back(sp.y);
			pos_enc.push_back(sp.z);
			pos_enc.push_back(cp.x);
			pos_enc.push_back(cp.y);
			pos_enc.push_back(cp.z);
		}

		if (pos_enc.size() != 3 * (2 * L + 1)) {
			printf("ERROR: pos_enc.size() != 3 * (2 * L + 1) [%lu != %u]\n", pos_enc.size(), 3 * (2 * L + 1));
			abort();
		}

		for (uint32_t k = 0; k < pos_enc.size(); k++)
			vembedded[g_sdc.feature_count + k] = pos_enc[k];

		// Normal vector one-blob encoding
		float n0 = lerped_N[i].x;
		float n1 = lerped_N[i].y;

		std::vector <float> one_blob0;
		std::vector <float> one_blob1;

		constexpr float sigma = 0.1;
		for (uint32_t j = 0; j < K; j++) {
			float t = float(j)/(K - 1);
			float k0 = expf(-powf(n0 - t, 2.0f)/(2.0f * sigma * sigma));
			float k1 = expf(-powf(n1 - t, 2.0f)/(2.0f * sigma * sigma));
			one_blob0.push_back(k0);
			one_blob1.push_back(k1);
		}

		std::vector <float> normal_enc;
		normal_enc.push_back(n0);
		normal_enc.push_back(n1);

		normal_enc.insert(normal_enc.end(), one_blob0.begin(), one_blob0.end());
		normal_enc.insert(normal_enc.end(), one_blob1.begin(), one_blob1.end());

		for (uint32_t k = 0; k < normal_enc.size(); k++)
			vembedded[g_sdc.feature_count + pos_enc.size() + k] = normal_enc[k];
	}

	printf("Embedded: %lu\n", embedded.size());

	// Print the first and last embedding in full
	printf("First embedding: [ ");
	for (uint32_t k = 0; k < embedded_size; k++) {
		if (k % 5 == 0)
			printf("\n                  ");
		printf("%.4e ", embedded[k]);
	}
	printf("\n]\n");

	// Evaluate the first network layer
	std::vector <float> hidden;
	printf("Layer with input of %lu and output of %u\n", embedded_size, g_dnn.W0);
	hidden.resize(vertex_count * g_dnn.H1);

	for (uint32_t i = 0; i < vertex_count; i++) {
		float *vembedded = &embedded[i * embedded_size];

		std::vector <float> X;
		X.insert(X.end(), vembedded, vembedded + embedded_size);

		std::vector <float> Y = matmul(g_dnn.Wm0, X, g_dnn.Bs0, embedded_size, g_dnn.W0);

		for (uint32_t k = 0; k < g_dnn.W0; k++)
			hidden[i * g_dnn.W0 + k] = Y[k];
	}

	printf("First hidden: [ ");
	for (uint32_t k = 0; k < g_dnn.W0; k++) {
		if (k % 8 == 0)
			printf("\n               ");
		printf("%.4f ", hidden[k]);
	}
	printf("\n]\n");

	// Apply activation
	for (uint32_t i = 0; i < hidden.size(); i++)
		hidden[i] = sinf(hidden[i]);

	printf("First hidden (+sin): [ ");
	for (uint32_t k = 0; k < g_dnn.W0; k++) {
		if (k % 8 == 0)
			printf("\n               ");
		printf("%.4f ", hidden[k]);
	}
	printf("\n]\n");

	// Second layer
	printf("Layer with input of %u and output of %u\n", g_dnn.W0, g_dnn.W1);
	hidden.resize(vertex_count * g_dnn.W1);

	for (uint32_t i = 0; i < vertex_count; i++) {
		float *vhidden = &hidden[i * g_dnn.W0];

		std::vector <float> X;
		X.insert(X.end(), vhidden, vhidden + g_dnn.W0);

		std::vector <float> Y = matmul(g_dnn.Wm1, X, g_dnn.Bs1, g_dnn.W0, g_dnn.W1);

		for (uint32_t k = 0; k < g_dnn.W1; k++)
			hidden[i * g_dnn.W1 + k] = Y[k];
	}

	printf("Second hidden: [ ");
	for (uint32_t k = 0; k < g_dnn.W1; k++) {
		if (k % 8 == 0)
			printf("\n               ");
		printf("%.4f ", hidden[k]);
	}
	printf("\n]\n");

	// Apply activation
	for (uint32_t i = 0; i < hidden.size(); i++)
		hidden[i] = sinf(hidden[i]);

	printf("Second hidden (+sin): [ ");
	for (uint32_t k = 0; k < g_dnn.W1; k++) {
		if (k % 8 == 0)
			printf("\n               ");
		printf("%.4f ", hidden[k]);
	}
	printf("\n]\n");

	// Last layer
	printf("Layer with input of %u and output of %u\n", g_dnn.W1, g_dnn.W2);
	if (g_dnn.W2 != 3) {
		printf("ERROR: W2 != 3\n");
		abort();
	}

	hidden.resize(vertex_count * g_dnn.W2);

	for (uint32_t i = 0; i < vertex_count; i++) {
		float *vhidden = &hidden[i * g_dnn.W1];

		std::vector <float> X;
		X.insert(X.end(), vhidden, vhidden + g_dnn.W1);

		std::vector <float> Y = matmul(g_dnn.Wm2, X, g_dnn.Bs2, g_dnn.W1, g_dnn.W2);

		for (uint32_t k = 0; k < g_dnn.W2; k++)
			hidden[i * g_dnn.W2 + k] = Y[k];
	}

	printf("Final hidden: [ ");
	for (uint32_t k = 0; k < g_dnn.W2; k++)
		printf("%.4f ", hidden[k]);
	printf("]\n");

	// Convert to vec3s and add to the base
	std::vector <glm::vec3> displacements(vertex_count);
	for (uint32_t i = 0; i < vertex_count; i++) {
		glm::vec3 disp = glm::vec3(hidden[i * g_dnn.W2 + 0], hidden[i * g_dnn.W2 + 1], hidden[i * g_dnn.W2 + 2]);
		displacements[i] = disp;
	}

	// Apply displacements
	std::vector <glm::vec3> final_P(vertex_count);
	for (uint32_t i = 0; i < vertex_count; i++)
		final_P[i] = lerped_P[i] + displacements[i];

	// Quads
	std::vector <std::array <uint32_t, 4>> quads;

	uint32_t f = 0;
	for (uint32_t c = 0; c < g_sdc.complex_count; c++) {
		uint32_t base = c * rate * rate;
		for (uint32_t ix = 0; ix < rate - 1; ix++) {
			for (uint32_t iy = 0; iy < rate - 1; iy++) {
				uint32_t i0 = ix + rate * iy;
				uint32_t i1 = i0 + 1;
				uint32_t i2 = i0 + rate;
				uint32_t i3 = i2 + 1;

				quads.push_back({ base + i0, base + i1, base + i3, base + i2 });
			}
		}
	}

	// Triangulate (shortest diagonal)
	std::vector <std::array <uint32_t, 3>> tris;

	uint32_t t = 0;
	for (uint32_t c = 0; c < g_sdc.complex_count; c++) {
		uint32_t base = c * rate * rate;
		for (uint32_t ix = 0; ix < rate - 1; ix++) {
			for (uint32_t iy = 0; iy < rate - 1; iy++) {
				uint32_t i0 = ix + rate * iy;
				uint32_t i1 = i0 + 1;
				uint32_t i2 = i0 + rate;
				uint32_t i3 = i2 + 1;

				glm::vec3 p0 = final_P[base + i0];
				glm::vec3 p1 = final_P[base + i1];
				glm::vec3 p2 = final_P[base + i2];
				glm::vec3 p3 = final_P[base + i3];

				float d03 = glm::length(p0 - p3);
				float d12 = glm::length(p1 - p2);

				if (d03 < d12) {
					tris.push_back({ base + i0, base + i1, base + i3 });
					tris.push_back({ base + i0, base + i3, base + i2 });
				} else {
					tris.push_back({ base + i0, base + i1, base + i2 });
					tris.push_back({ base + i1, base + i3, base + i2 });
				}
			}
		}
	}

	namespace ps = polyscope;

	ps::init();
	ps::registerSurfaceMesh("Quads", final_P, quads);
	ps::registerSurfaceMesh("Tris", final_P, tris);
	ps::show();
}
