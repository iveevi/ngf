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

void dnn_read(FILE *file)
{
	// Read neural network data
	ulog_info("dnn_read", "Neural Network:\n");

	// Prepare references
	struct layer_info {
		std::vector <float> Wm;
		std::vector <float> Bs;
		std::vector <float> Wm_c;

		uint32_t W;
		uint32_t H;
	};

	std::array <layer_info, 4> layers;

	// Read all the weight matrices
	for (uint32_t i = 0; i < layers.size(); i++) {
		uint32_t W = 0;
		uint32_t H = 0;

		fread((char *) &W, sizeof(W), 1, file);
		fread((char *) &H, sizeof(H), 1, file);

		ulog_info("dnn_read", "  > Layer %d: %d x %d\n", i, W, H, W);

		layers[i].Wm = std::vector <float> (W * H);
		layers[i].W  = W;
		layers[i].H  = H;
		fread((char *) layers[i].Wm.data(), sizeof(float), W * H, file);
	}

	// Read all the biases
	for (uint32_t i = 0; i < layers.size(); i++) {
		uint32_t W = 0;

		fread((char *) &W, sizeof(W), 1, file);

		// Verify that the dimensions match
		ulog_assert(W == layers[i].W, "dnn_read", "H (%d) != H (%d)\n", W, layers[i].W);

		layers[i].Bs = std::vector <float> (W);
		fread((char *) layers[i].Bs.data(), sizeof(float), W, file);

		// Combine weights and biases into a single matrix
		layers[i].Wm_c = combine(layers[i].Wm, layers[i].Bs);
	}

	// Verify sizes and transfer properties
	uint32_t ffwd_size = g_sdc.ffwd_size();
	for (uint32_t i = 0; i < layers.size(); i++) {
		auto &layer = layers[i];

		g_dnn.Wm_c[i] = layer.Wm_c;
		g_dnn.Wm[i]   = layer.Wm;
		g_dnn.Bs[i]   = layer.Bs;
		g_dnn.Ws[i]   = layer.W;
		g_dnn.Hs[i]   = layer.H;

		ulog_assert(layer.H == ffwd_size, "dnn_read", "H (%d) != ffwd_size (%d)\n", layer.H, ffwd_size);
                ulog_assert(layer.Wm_c.size() % (ffwd_size + 1) == 0,
                            "dnn_read",
                            "Wm_c.size() (%d) %% (ffwd_size + 1) (%d) != 0 "
                            "(incompatible weight matrix)\n",
                            layer.Wm_c.size(), ffwd_size + 1);

                ffwd_size = layer.W;
	}

	ulog_assert(ffwd_size == 3, "dnn_read", "ffwd_size (%d) != 3 (vertex size)\n", ffwd_size);

	// Allocating and copying to device
	for (uint32_t i = 0; i < layers.size(); i++) {
		auto &layer = layers[i];

		cudaMalloc(&g_dnn.d_Wm_c[i], layer.Wm_c.size() * sizeof(float));
		cudaMemcpy(g_dnn.d_Wm_c[i], layer.Wm_c.data(), layer.Wm_c.size() * sizeof(float), cudaMemcpyHostToDevice);

		CUDA_CHECK_SYNCED();
	}
}

void sdc_read(FILE *file)
{
	ulog_info("sdc_read", "Neural Subdivision Complexes:\n");

	// Read top level constants
	// float constants[3];
	fread((char *) g_dnn.constants.data(), sizeof(float), 3, file);

	// TODO: make host device...
	float s0 = g_dnn.constants[0];
	float s1 = g_dnn.constants[1];
	float s2 = g_dnn.constants[2];

	g_dnn.activations = {
		[s0](float x) {
			float softplus = log(1 + exp(s0 * x));
			float sin = sinf(s0 * x);
			float gauss = exp(-x * x / s0);
			return softplus * sin * gauss;
		},

		[s1](float x) {
			float softplus = log(1 + exp(s1 * x));
			float sin = sinf(s1 * x);
			float gauss = exp(-x * x / s1);
			return softplus * sin * gauss;
		},

		[s2](float x) {
			float softplus = log(1 + exp(s2 * x));
			float sin = sinf(s2 * x);
			float gauss = exp(-x * x / s2);
			return softplus * sin * gauss;
		}
	};

	ulog_info("sdc_read", "  > %f %f %f\n", s0, s1, s2);

	// Read size data
	uint32_t sizes[3];
	fread((char *) sizes, sizeof(uint32_t), 3, file);

	g_sdc.complex_count = sizes[0];
	g_sdc.vertex_count = sizes[1];
	g_sdc.feature_size = sizes[2];

	ulog_info("sdc_read", "Neural Subdivision Complexes:\n");
	ulog_info("sdc_read", "  > %4d complexes\n", g_sdc.complex_count);
	ulog_info("sdc_read", "  > %4d vertices\n", g_sdc.vertex_count);
	ulog_info("sdc_read", "  > %4d features [%2d]\n", g_sdc.vertex_count, g_sdc.feature_size);

	// Read complexes data
	std::vector <glm::ivec4> complexes(g_sdc.complex_count);
	fread((char *) complexes.data(), sizeof(glm::ivec4), g_sdc.complex_count, file);

	// Read corner vertices, normals, and their features
	std::vector <glm::vec3> vertices(g_sdc.vertex_count);
	fread((char *) vertices.data(), sizeof(glm::vec3), g_sdc.vertex_count, file);

	// Corner feature vectors
	std::vector <float> features(g_sdc.vertex_count * g_sdc.feature_size);
	fread((char *) features.data(), sizeof(float), g_sdc.vertex_count * g_sdc.feature_size, file);

	g_sdc.complexes = complexes;
	g_sdc.vertices = vertices;
	g_sdc.features = features;

	// Transfer to device
	cudaMalloc(&g_sdc.d_complexes, g_sdc.complex_count * sizeof(glm::ivec4));
	cudaMalloc(&g_sdc.d_vertices, g_sdc.vertex_count * sizeof(glm::vec3));
	cudaMalloc(&g_sdc.d_features, g_sdc.vertex_count * g_sdc.feature_size* sizeof(float));

	cudaMemcpy(g_sdc.d_complexes, complexes.data(), g_sdc.complex_count * sizeof(glm::ivec4), cudaMemcpyHostToDevice);
	cudaMemcpy(g_sdc.d_vertices, vertices.data(), g_sdc.vertex_count * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	cudaMemcpy(g_sdc.d_features, features.data(), g_sdc.vertex_count * g_sdc.feature_size * sizeof(float), cudaMemcpyHostToDevice);
}

void read(FILE *file)
{
	sdc_read(file);
	dnn_read(file);
}
