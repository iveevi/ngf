#include "common.hpp"

neural_network        g_dnn;
subdivision_complexes g_sdc;

// Combine weights and biases into a single matrix
std::vector <float> combine(const std::vector <float> &W, const std::vector <float> &B)
{
	size_t rows = B.size();
	size_t cols = W.size() / B.size();

	ulog_assert(W.size() == rows * cols, "combine", "W size mismatch\n");

	std::vector <float> Wc(rows * (cols + 1));
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++)
			Wc[i * (cols + 1) + j] = W[i * cols + j];
		Wc[i * (cols + 1) + cols] = B[i];
	}

	return Wc;
}

// TODO: Automatic resolution derived from the projected area of the complex

void dnn_read(FILE *file)
{
	// Read neural network data
	ulog_info("dnn_read", "Neural Network:\n");

	uint32_t W0 = 0;
	uint32_t H0 = 0;

	fread((char *) &W0, sizeof(W0), 1, file);
	fread((char *) &H0, sizeof(H0), 1, file);
	ulog_info("dnn_read", "  > Layer 1: %d x %d + %d\n", W0, H0, W0);
	ulog_assert(W0 <= DNN_INTERIM_SIZE, "dnn_read", "W0 too large\n");
	ulog_assert(H0 <= DNN_INTERIM_SIZE, "dnn_read", "H0 too large\n");

	std::vector <float> Wm0(W0 * H0);
	std::vector <float> Bs0(W0);

	fread((char *) Wm0.data(), sizeof(float), W0 * H0, file);
	fread((char *) Bs0.data(), sizeof(float), W0, file);

	// Combined layer (as one matrix)
	std::vector <float> Wm0c = combine(Wm0, Bs0);

	uint32_t W1 = 0;
	uint32_t H1 = 0;

	fread((char *) &W1, sizeof(W1), 1, file);
	fread((char *) &H1, sizeof(H1), 1, file);
	ulog_info("dnn_read", "  > Layer 2: %d x %d + %d\n", W1, H1, W1);
	ulog_assert(W1 <= DNN_INTERIM_SIZE, "dnn_read", "W1 too large\n");
	ulog_assert(H1 <= DNN_INTERIM_SIZE, "dnn_read", "H1 too large\n");

	std::vector <float> Wm1(W1 * H1);
	std::vector <float> Bs1(W1);

	fread((char *) Wm1.data(), sizeof(float), W1 * H1, file);
	fread((char *) Bs1.data(), sizeof(float), W1, file);

	std::vector <float> Wm1c = combine(Wm1, Bs1);

	uint32_t W2 = 0;
	uint32_t H2 = 0;

	fread((char *) &W2, sizeof(W2), 1, file);
	fread((char *) &H2, sizeof(H2), 1, file);
	ulog_info("dnn_read", "  > Layer 3: %d x %d + %d\n", W2, H2, W2);
	ulog_assert(W2 <= DNN_INTERIM_SIZE, "dnn_read", "W2 too large\n");
	ulog_assert(H2 <= DNN_INTERIM_SIZE, "dnn_read", "H2 too large\n");

	std::vector <float> Wm2(W2 * H2);
	std::vector <float> Bs2(W2);

	fread((char *) Wm2.data(), sizeof(float), W2 * H2, file);
	fread((char *) Bs2.data(), sizeof(float), W2, file);

	std::vector <float> Wm2c = combine(Wm2, Bs2);

	g_dnn.W0 = W0;
	g_dnn.H0 = H0;

	g_dnn.W1 = W1;
	g_dnn.H1 = H1;

	g_dnn.W2 = W2;
	g_dnn.H2 = H2;

	g_dnn.Wm0c = Wm0c;
	g_dnn.Wm1c = Wm1c;
	g_dnn.Wm2c = Wm2c;

	// Transfer to device
	cudaMalloc((void **) &g_dnn.d_Wm0c, Wm0c.size() * sizeof(float));
	cudaMalloc((void **) &g_dnn.d_Wm1c, Wm1c.size() * sizeof(float));
	cudaMalloc((void **) &g_dnn.d_Wm2c, Wm2c.size() * sizeof(float));

	cudaMemcpy(g_dnn.d_Wm0c, Wm0c.data(), Wm0c.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(g_dnn.d_Wm1c, Wm1c.data(), Wm1c.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(g_dnn.d_Wm2c, Wm2c.data(), Wm2c.size() * sizeof(float), cudaMemcpyHostToDevice);
}

void sdc_read(FILE *file)
{
	// Read size data
	uint32_t sizes[3];
	fread((char *) sizes, sizeof(uint32_t), 3, file);

	g_sdc.complex_count = sizes[0];
	g_sdc.vertex_count = sizes[1];
	g_sdc.feature_count = sizes[2];

	ulog_info("sdc_read", "Neural Subdivision Complexes:\n");
	ulog_info("sdc_read", "  > %4d complexes\n", g_sdc.complex_count);
	ulog_info("sdc_read", "  > %4d vertices\n", g_sdc.vertex_count);
	ulog_info("sdc_read", "  > %4d encoding features\n", g_sdc.feature_count);

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

	// Allocate the intermediate buffers
	// cudaMalloc((void **) &g_dnn.d_interim_one, DNN_INTERIM_SIZE * sizeof(float));
	// cudaMalloc((void **) &g_dnn.d_interim_two, DNN_INTERIM_SIZE * sizeof(float));

	// TODO: reduce to only a few complexes
	cudaMalloc((void **) &g_dnn.d_embedded, DNN_INTERIM_SIZE * sizeof(float) * sizes[0] * COMPLEX_BATCH_SIZE);
	cudaMalloc((void **) &g_dnn.d_interim_one, DNN_INTERIM_SIZE * sizeof(float) * sizes[0] * COMPLEX_BATCH_SIZE);
	cudaMalloc((void **) &g_dnn.d_interim_two, DNN_INTERIM_SIZE * sizeof(float) * sizes[0] * COMPLEX_BATCH_SIZE);
}
