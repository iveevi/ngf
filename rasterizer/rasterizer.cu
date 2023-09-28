#include <array>
#include <fstream>
#include <iostream>
#include <vector>

#include <glm/glm.hpp>

#include <omp.h>

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
std::vector <float> matmulvec(const std::vector <float> W, const std::vector <float> &X, const std::vector <float> &B, size_t in, size_t out)
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
	#pragma omp parallel
	for (size_t i = 0; i < out; i++) {
		float sum = 0.0f;

		for (size_t j = 0; j < in; j++)
			sum += W[i * in + j] * X[j];

		Y[i] = sum + B[i];
	}

	return Y;
}

// Batched version (true matrix multiplication)
// TODO: concat into a single matrix multiplication (e.g. remove the bias and add as the last column of the weight matrix)
std::vector <float> matmul(const std::vector <float> &W, const std::vector <float> &Xbatched, const std::vector <float> &B, size_t in, size_t out)
{
	// Check sizes
	if (W.size() != in * out) {
		printf("Error: matmul: W size mismatch\n");
		abort();
	}

	if (Xbatched.size() % in != 0) {
		printf("Error: matmul: X size mismatch\n");
		abort();
	}

	if (B.size() != out) {
		printf("Error: matmul: B size mismatch\n");
		abort();
	}

	// Prepare result
	size_t batch = Xbatched.size() / in;

	std::vector <float> Ybatched(out * batch);

	// Perform matrix multiplication and add bias
	#pragma omp parallel
	for (size_t b = 0; b < batch; b++) {
		for (size_t i = 0; i < out; i++) {
			float sum = 0.0f;

			for (size_t j = 0; j < in; j++)
				sum += W[i * in + j] * Xbatched[b * in + j];

			Ybatched[b * out + i] = sum + B[i];
		}
	}

	return Ybatched;
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

	// Corner feature vectors
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

// TODO: microlog...
std::vector <glm::vec3> eval(size_t sample_rate)
{
	uint32_t vertex_count = sample_rate * sample_rate * g_sdc.complex_count;
	// printf("Evaluating %d vertices...\n", vertex_count);

	std::vector <glm::vec3> lerped_P;
	std::vector <float>     lerped_E;

	lerped_P.resize(vertex_count);
	lerped_E.resize(vertex_count * g_sdc.feature_count);

	uint32_t i = 0;
	for (uint32_t c = 0; c < g_sdc.complex_count; c++) {
		const glm::uvec4 &complex = g_sdc.complexes[c];

		glm::vec3 v00 = g_sdc.vertices[complex.x];
		glm::vec3 v10 = g_sdc.vertices[complex.y];
		glm::vec3 v01 = g_sdc.vertices[complex.w];
		glm::vec3 v11 = g_sdc.vertices[complex.z];

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

		for (uint32_t ix = 0; ix < sample_rate; ix++) {
			for (uint32_t iy = 0; iy < sample_rate; iy++) {
				float u = (float) ix / (sample_rate - 1);
				float v = (float) iy / (sample_rate - 1);

				{
					glm::vec3 lp00 = v00 * u * v;
					glm::vec3 lp10 = v10 * (1.0f - u) * v;
					glm::vec3 lp01 = v01 * u * (1.0f - v);
					glm::vec3 lp11 = v11 * (1.0f - u) * (1.0f - v);
					lerped_P[i] = lp00 + lp10 + lp01 + lp11;
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

	// Construct the network input with embeddings
	// TODO: get these from the file as well...
	constexpr uint32_t L = 8;
	constexpr uint32_t K = 16;

	uint32_t embedded_size = g_sdc.feature_count + 3 * (2 * L + 1);
	// printf("Embedded size: %u\n", embedded_size);

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

		for (uint32_t j = 0; j < L; j++) {
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
	}

	// printf("Embedded: %lu\n", embedded.size());

	// Print the first and last embedding in full
	// printf("First embedding: [ ");
	// for (uint32_t k = 0; k < embedded_size; k++) {
	// 	if (k % 5 == 0)
	// 		printf("\n                  ");
	// 	printf("%.4e ", embedded[k]);
	// }
	// printf("\n]\n");

	// Evaluate the first network layer
	std::vector <float> hidden;
	// printf("Layer with input of %lu and output of %u\n", embedded_size, g_dnn.W0);
	// hidden.resize(vertex_count * g_dnn.H1);

	// for (uint32_t i = 0; i < vertex_count; i++) {
	// 	float *vembedded = &embedded[i * embedded_size];
	//
	// 	std::vector <float> X;
	// 	X.insert(X.end(), vembedded, vembedded + embedded_size);
	//
	// 	std::vector <float> Y = matmul(g_dnn.Wm0, X, g_dnn.Bs0, embedded_size, g_dnn.W0);
	//
	// 	for (uint32_t k = 0; k < g_dnn.W0; k++)
	// 		hidden[i * g_dnn.W0 + k] = Y[k];
	// }

	hidden = matmul(g_dnn.Wm0, embedded, g_dnn.Bs0, embedded_size, g_dnn.W0);
	if (hidden.size() != vertex_count * g_dnn.W0) {
		printf("ERROR: hidden.size() != vertex_count * g_dnn.W0 [%lu != %u]\n", hidden.size(), vertex_count * g_dnn.W0);
		abort();
	}

	// printf("First hidden: [ ");
	// for (uint32_t k = 0; k < g_dnn.W0; k++) {
	// 	if (k % 8 == 0)
	// 		printf("\n               ");
	// 	printf("%.4f ", hidden[k]);
	// }
	// printf("\n]\n");

	// Apply activation
	#pragma omp parallel
	for (uint32_t i = 0; i < hidden.size(); i++)
		hidden[i] = sinf(hidden[i]);

	// printf("First hidden (+sin): [ ");
	// for (uint32_t k = 0; k < g_dnn.W0; k++) {
	// 	if (k % 8 == 0)
	// 		printf("\n               ");
	// 	printf("%.4f ", hidden[k]);
	// }
	// printf("\n]\n");

	// Second layer
	// printf("Layer with input of %u and output of %u\n", g_dnn.W0, g_dnn.W1);
	// hidden.resize(vertex_count * g_dnn.W1);

	// for (uint32_t i = 0; i < vertex_count; i++) {
	// 	float *vhidden = &hidden[i * g_dnn.W0];
	//
	// 	std::vector <float> X;
	// 	X.insert(X.end(), vhidden, vhidden + g_dnn.W0);
	//
	// 	std::vector <float> Y = matmul(g_dnn.Wm1, X, g_dnn.Bs1, g_dnn.W0, g_dnn.W1);
	//
	// 	for (uint32_t k = 0; k < g_dnn.W1; k++)
	// 		hidden[i * g_dnn.W1 + k] = Y[k];
	// }

	hidden = matmul(g_dnn.Wm1, hidden, g_dnn.Bs1, g_dnn.W0, g_dnn.W1);
	if (hidden.size() != vertex_count * g_dnn.W1) {
		printf("ERROR: hidden.size() != vertex_count * g_dnn.W1 [%lu != %u]\n", hidden.size(), vertex_count * g_dnn.W1);
		abort();
	}

	// printf("Second hidden: [ ");
	// for (uint32_t k = 0; k < g_dnn.W1; k++) {
	// 	if (k % 8 == 0)
	// 		printf("\n               ");
	// 	printf("%.4f ", hidden[k]);
	// }
	// printf("\n]\n");

	// Apply activation
	#pragma omp parallel
	for (uint32_t i = 0; i < hidden.size(); i++)
		hidden[i] = sinf(hidden[i]);

	// printf("Second hidden (+sin): [ ");
	// for (uint32_t k = 0; k < g_dnn.W1; k++) {
	// 	if (k % 8 == 0)
	// 		printf("\n               ");
	// 	printf("%.4f ", hidden[k]);
	// }
	// printf("\n]\n");

	// Last layer
	// printf("Layer with input of %u and output of %u\n", g_dnn.W1, g_dnn.W2);
	if (g_dnn.W2 != 3) {
		printf("ERROR: W2 != 3\n");
		abort();
	}

	// hidden.resize(vertex_count * g_dnn.W2);

	// for (uint32_t i = 0; i < vertex_count; i++) {
	// 	float *vhidden = &hidden[i * g_dnn.W1];
	//
	// 	std::vector <float> X;
	// 	X.insert(X.end(), vhidden, vhidden + g_dnn.W1);
	//
	// 	std::vector <float> Y = matmul(g_dnn.Wm2, X, g_dnn.Bs2, g_dnn.W1, g_dnn.W2);
	//
	// 	for (uint32_t k = 0; k < g_dnn.W2; k++)
	// 		hidden[i * g_dnn.W2 + k] = Y[k];
	// }

	hidden = matmul(g_dnn.Wm2, hidden, g_dnn.Bs2, g_dnn.W1, g_dnn.W2);
	if (hidden.size() != vertex_count * g_dnn.W2) {
		printf("ERROR: hidden.size() != vertex_count * g_dnn.W2 [%lu != %u]\n", hidden.size(), vertex_count * g_dnn.W2);
		abort();
	}

	// printf("Final hidden: [ ");
	// for (uint32_t k = 0; k < g_dnn.W2; k++)
	// 	printf("%.4f ", hidden[k]);
	// printf("]\n");

	// Convert to vec3s and add to the base
	// std::vector <glm::vec3> displacements(vertex_count);
	// for (uint32_t i = 0; i < vertex_count; i++) {
	// 	glm::vec3 disp = glm::vec3(hidden[i * g_dnn.W2 + 0], hidden[i * g_dnn.W2 + 1], hidden[i * g_dnn.W2 + 2]);
	// 	displacements[i] = disp;
	// }

	glm::vec3 *displacements = (glm::vec3 *) hidden.data();

	// Apply displacements
	std::vector <glm::vec3> final_P(vertex_count);
	for (uint32_t i = 0; i < vertex_count; i++)
		final_P[i] = lerped_P[i] + displacements[i];

	return final_P;
}

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

	auto final_P = eval(rate);

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
