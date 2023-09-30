#include "common.hpp"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

constexpr size_t MAXIMUM_SAMPLE_RATE = 16;
constexpr size_t COMPLEX_BATCH_SIZE = MAXIMUM_SAMPLE_RATE * MAXIMUM_SAMPLE_RATE;
constexpr size_t DNN_INTERIM_SIZE = 128;

struct neural_network {
	// Host buffers
	std::vector <float> Wm0c;
	std::vector <float> Wm1c;
	std::vector <float> Wm2c;

	// Device (CUDA) buffers
	float *d_Wm0c;
	float *d_Wm1c;
	float *d_Wm2c;

	float *d_interim_one;
	float *d_interim_two;

	uint32_t W0;
	uint32_t H0;

	uint32_t W1;
	uint32_t H1;

	uint32_t W2;
	uint32_t H2;
} g_dnn;

// Matrix multiplication with bias
// TODO: benchmark execution times, and also in CUDA
template <float (*act)(float)>
std::vector <float> matmul_biased(const std::vector <float> &W,
		const std::vector <float> &Xbatched,
		size_t in, size_t out)
{
	// Check sizes
	ulog_assert(W.size() == (in + 1) * out, "matmul_biased", "W size mismatch\n");
	ulog_assert(Xbatched.size() % in == 0,  "matmul_biased", "X size mismatch\n");

	// Prepare result
	size_t batch = Xbatched.size() / in;

	std::vector <float> Ybatched(out * batch);

	// Perform matrix multiplication and add bias
	for (size_t b = 0; b < batch; b++) {
		const float *Xrow = &Xbatched[b * in];
		float *Yrow = &Ybatched[b * out];

		#pragma omp parallel
		for (size_t i = 0; i < out; i++) {
			const float *Wrow = &W[i * (in + 1)];
			float sum = Wrow[in];
			for (size_t j = 0; j < in; j++)
				sum += Wrow[j] * Xrow[j];

			if constexpr (act)
				Yrow[i] = act(sum);
			else
				Yrow[i] = sum;
		}
	}

	return Ybatched;
}

// TODO: batch by each (or several complexes; set some maximum cache/interim memory amt)

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

// Automatic resolution derived from the projected area of the complex

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
	cudaMalloc((void **) &g_dnn.d_interim_one, DNN_INTERIM_SIZE * sizeof(float) * sizes[0] * COMPLEX_BATCH_SIZE);
	cudaMalloc((void **) &g_dnn.d_interim_two, DNN_INTERIM_SIZE * sizeof(float) * sizes[0] * COMPLEX_BATCH_SIZE);
}

// CUDA error checking
#define CUDA_CHECK_ERROR()                                                     				\
	{                                                                            			\
    		cudaError_t e = cudaGetLastError();                                        		\
    		ulog_assert(e == cudaSuccess, "CUDA_CHECK_ERROR", "CUDA error %d: %s [%s:%d]\n", e, 	\
                	cudaGetErrorString(e), __FILE__, __LINE__);					\
  	}

// CUDA MLP input lerping and embedding for all complexes
struct EmbedInfo {
	uint32_t feature_count;
	glm::vec3  *vertices;
	float      *features;
};

__global__
void lerp_and_embed(float *embedded, EmbedInfo info, glm::uvec4 complex, uint32_t sample_rate)
{
	// Simple 2D work group
	uint32_t i = threadIdx.x;
	uint32_t j = threadIdx.y;
	if (i >= sample_rate || j >= sample_rate)
		return;

	glm::vec3 v00 = info.vertices[complex.x];
	glm::vec3 v10 = info.vertices[complex.y];
	glm::vec3 v01 = info.vertices[complex.z];
	glm::vec3 v11 = info.vertices[complex.w];

	float *f00 = info.features + complex.x * info.feature_count;
	float *f10 = info.features + complex.y * info.feature_count;
	float *f01 = info.features + complex.z * info.feature_count;
	float *f11 = info.features + complex.w * info.feature_count;

	float u = (float) i / (float) sample_rate;
	float v = (float) j / (float) sample_rate;

	// Compute the lerped feature and put into the embedded buffer
	constexpr uint32_t L = 8;

	uint32_t embedded_size = info.feature_count + 3 * (2 * L + 1);

	float *embedded_ptr = embedded + (i * sample_rate + j) * embedded_size;
	for (size_t k = 0; k < info.feature_count; k++) {
		float f00k = f00[k] * u * v;
		float f10k = f10[k] * (1.0f - u) * v;
		float f11k = f11[k] * u * (1.0f - v);
		float f01k = f01[k] * (1.0f - u) * (1.0f - v);
		embedded_ptr[k] = f00k + f10k + f11k + f01k;
	}

	// Lerp the vertex position
	glm::vec3 P = v00 * u * v
		      + v10 * (1.0f - u) * v
		      + v11 * u * (1.0f - v)
		      + v01 * (1.0f - u) * (1.0f - v);

	// Fill the rest of the embedded buffer with the positional encoding
	embedded_ptr += info.feature_count;

	embedded_ptr[0] = P.x;
	embedded_ptr[1] = P.y;
	embedded_ptr[2] = P.z;
	embedded_ptr += 3;

	for (uint32_t k = 0; k < L; k++) {
		float x = P.x * powf(2.0f, (float) k);
		float y = P.y * powf(2.0f, (float) k);
		float z = P.z * powf(2.0f, (float) k);

		embedded_ptr[0] = sinf(x);
		embedded_ptr[1] = sinf(y);
		embedded_ptr[2] = sinf(z);
		embedded_ptr[3] = cosf(x);
		embedded_ptr[4] = cosf(y);
		embedded_ptr[5] = cosf(z);
		embedded_ptr += 6;
	}
}

void lerp_and_embed(uint32_t sample_rate)
{
	constexpr uint32_t L = 8;

	// Always start with the first interim buffer
	float *d_embedded = g_dnn.d_interim_one;
	ulog_info("lerped_and_embed", "embedding %d complexes\n", g_sdc.complex_count);

	uint32_t embedded_size = g_sdc.feature_count + 3 * (2 * L + 1);

	EmbedInfo info = {
		.feature_count = g_sdc.feature_count,
		.vertices = g_sdc.d_vertices,
		.features = g_sdc.d_features,
	};

	for (size_t i = 0; i < g_sdc.complex_count; i++) {
		glm::uvec4 complex = g_sdc.complexes[i];

		// TODO: multiple grid_dims?
		dim3 block_dim(sample_rate, sample_rate);
		dim3 grid_dim(1, 1);

		lerp_and_embed <<< grid_dim, block_dim >>> (d_embedded, info, complex, sample_rate);

		d_embedded += sample_rate * sample_rate * embedded_size;
	}
		
	cudaDeviceSynchronize();
	CUDA_CHECK_ERROR();
}

// CUDA MLP evaluation
template <float (*act)(float)>
__global__
void matmul_biased(float *dst_base, float *W_base, float *X_base, uint32_t in, uint32_t out)
{
	// Assumes that X is a batch of dimension in x batch_size
	// Assumes that W is a matrix of dimension out x (in + 1)

	// Computes as many output features as there are threads (horizontal parallelism)
	uint32_t i = threadIdx.x;

	float *dst = dst_base + i * out;
	const float *X = &X_base[in * i];

	for (size_t i = 0; i < out; i++) {
		const float *W = &W_base[i * (in + 1)];
		float sum = W[in];
		for (size_t j = 0; j < in; j++)
			sum += W[j] * X[j];

		if constexpr (act == nullptr)
			dst[i] = sum;
		else
			dst[i] = act(sum);
	}
}

void eval_cuda(size_t sample_rate)
{
	constexpr uint32_t L = 8;

	uint32_t embedded_size = g_sdc.feature_count + 3 * (2 * L + 1);

	lerp_and_embed(sample_rate);
	cudaDeviceSynchronize();
	CUDA_CHECK_ERROR();
	
	{
		std::vector <float> buffer;
		buffer.resize(g_sdc.complex_count * sample_rate * sample_rate * embedded_size);
		cudaMemcpy(buffer.data(), g_dnn.d_interim_one, buffer.size() * sizeof(float), cudaMemcpyDeviceToHost);
		printf("embedding[0] =");
		for (size_t i = 0; i < embedded_size; i++)
			printf(" %f", buffer[i]);
		printf("\n");
	}

	// Evaluate the first layer
	ulog_info("eval_cuda", "evaluating first layer\n");

	dim3 block_dim(sample_rate * sample_rate);
	// dim3 grid_dim(g_sdc.complex_count);
	dim3 grid_dim(1, 1);

	float *d_out = g_dnn.d_interim_two;
	float *d_in = g_dnn.d_interim_one;

	// uint32_t embedded_size = g_sdc.feature_count + 3 * (2 * L + 1);
	for (size_t i = 0; i < g_sdc.complex_count; i++) {
		float *d_W = g_dnn.d_Wm0c;

		matmul_biased <sinf> <<< grid_dim, block_dim >>> (d_out, d_W, d_in, embedded_size, g_dnn.W0);

		d_out += sample_rate * sample_rate * g_dnn.W0;
		d_in += sample_rate * sample_rate * embedded_size;
	}

	cudaDeviceSynchronize();
	CUDA_CHECK_ERROR();

	// Evaluate the second layer
	ulog_info("eval_cuda", "evaluating second layer\n");

	d_out = g_dnn.d_interim_one;
	d_in = g_dnn.d_interim_two;

	for (size_t i = 0; i < g_sdc.complex_count; i++) {
		float *d_W = g_dnn.d_Wm1c;

		matmul_biased <sinf> <<< grid_dim, block_dim >>> (d_out, d_W, d_in, g_dnn.W0, g_dnn.W1);

		d_out += sample_rate * sample_rate * g_dnn.W1;
		d_in += sample_rate * sample_rate * g_dnn.W0;
	}

	// Evaluate the final layer
	ulog_info("eval_cuda", "evaluating final layer\n");

	d_out = g_dnn.d_interim_two;
	d_in = g_dnn.d_interim_one;

	for (size_t i = 0; i < g_sdc.complex_count; i++) {
		float *d_W = g_dnn.d_Wm2c;

		matmul_biased <nullptr> <<< grid_dim, block_dim >>> (d_out, d_W, d_in, g_dnn.W1, g_dnn.W2);

		d_out += sample_rate * sample_rate * g_dnn.W2;
		d_in += sample_rate * sample_rate * g_dnn.W1;
	}

	cudaDeviceSynchronize();
	CUDA_CHECK_ERROR();

	{
		std::vector <glm::vec3> buffer;
		buffer.resize(g_sdc.complex_count * sample_rate * sample_rate);
		cudaMemcpy(buffer.data(), g_dnn.d_interim_two, buffer.size() * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
		printf("displacements[0] = %f %f %f\n", buffer[0].x, buffer[0].y, buffer[0].z);
	}
}

// TODO: microlog...
std::vector <glm::vec3> eval(size_t sample_rate)
{
	uint32_t vertex_count = sample_rate * sample_rate * g_sdc.complex_count;
	ulog_info("eval", "Evaluating %d vertices\n", vertex_count);

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

	ulog_info("eval", "Lerped vertices and features\n");

	// Construct the network input with embeddings
	// TODO: get these from the file as well...
	constexpr uint32_t L = 8;

	uint32_t embedded_size = g_sdc.feature_count + 3 * (2 * L + 1);
	ulog_assert(embedded_size == g_dnn.H0, "eval", "embedded_size != g_dnn.H0 [%u != %u]\n", embedded_size, g_dnn.H0);

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

		ulog_assert(pos_enc.size() == 3 * (2 * L + 1), "eval", "pos_enc.size() != 3 * (2 * L + 1) [%lu != %u]\n", pos_enc.size(), 3 * (2 * L + 1));

		for (uint32_t k = 0; k < pos_enc.size(); k++)
			vembedded[g_sdc.feature_count + k] = pos_enc[k];
	}

	ulog_info("eval", "Constructed network input embeddings\n");

	printf("embedding[0]:");
	for (uint32_t i = 0; i < embedded_size; i++)
		printf(" %f", embedded[i]);
	printf("\n");

	// Evaluate the first network layer
	std::vector <float> hidden;

	hidden = matmul_biased <sinf> (g_dnn.Wm0c, embedded, embedded_size, g_dnn.W0);
	ulog_assert(hidden.size() == vertex_count * g_dnn.W0, "eval", "hidden.size() != vertex_count * g_dnn.W0 [%lu != %u]\n", hidden.size(), vertex_count * g_dnn.W0);
	ulog_info("eval", "Evaluated first network layer\n");

	hidden = matmul_biased <sinf> (g_dnn.Wm1c, hidden, g_dnn.W0, g_dnn.W1);
	ulog_assert(hidden.size() == vertex_count * g_dnn.W1, "eval", "hidden.size() != vertex_count * g_dnn.W1 [%lu != %u]\n", hidden.size(), vertex_count * g_dnn.W1);
	ulog_assert(g_dnn.W2 == 3, "eval", "W2 != 3");
	ulog_info("eval", "Evaluated second network layer\n");

	hidden = matmul_biased <nullptr> (g_dnn.Wm2c, hidden, g_dnn.W1, g_dnn.W2);
	ulog_assert(hidden.size() == vertex_count * g_dnn.W2, "eval", "hidden.size() != vertex_count * g_dnn.W2 [%lu != %u]\n", hidden.size(), vertex_count * g_dnn.W2);
	ulog_info("eval", "Evaluated final network layer\n");

	// Apply displacements
	glm::vec3 *displacements = (glm::vec3 *) hidden.data();

	printf("displacements[0]: %f %f %f\n", displacements[0].x, displacements[0].y, displacements[0].z);

	std::vector <glm::vec3> final_P(vertex_count);
	for (uint32_t i = 0; i < vertex_count; i++)
		final_P[i] = lerped_P[i] + displacements[i];

	return final_P;
}

// TODO: group the complexes by proximity and batch the culling and drawing processes...

std::vector <std::array <uint32_t, 3>> nsc_indices(const std::vector <glm::vec3> &vertices, size_t complexe_count, size_t sample_rate)
{
	std::vector <std::array <uint32_t, 3>> tris;

	for (uint32_t c = 0; c < g_sdc.complex_count; c++) {
		uint32_t base = c * sample_rate * sample_rate;
		for (uint32_t ix = 0; ix < sample_rate - 1; ix++) {
			for (uint32_t iy = 0; iy < sample_rate - 1; iy++) {
				uint32_t i0 = ix + sample_rate * iy;
				uint32_t i1 = i0 + 1;
				uint32_t i2 = i0 + sample_rate;
				uint32_t i3 = i2 + 1;

				glm::vec3 p0 = vertices[base + i0];
				glm::vec3 p1 = vertices[base + i1];
				glm::vec3 p2 = vertices[base + i2];
				glm::vec3 p3 = vertices[base + i3];

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

	return tris;
}

struct geometry {
	std::vector <glm::vec3> vertices;
	std::vector <glm::vec3> normals;
	std::vector <glm::uvec3> indices;
};

std::vector <float> interleave_attributes(const geometry &geometry)
{
	std::vector <float> attributes;

	for (uint32_t i = 0; i < geometry.vertices.size(); i++) {
		attributes.push_back(geometry.vertices[i].x);
		attributes.push_back(geometry.vertices[i].y);
		attributes.push_back(geometry.vertices[i].z);

		attributes.push_back(geometry.normals[i].x);
		attributes.push_back(geometry.normals[i].y);
		attributes.push_back(geometry.normals[i].z);
	}

	return attributes;
}

struct loader {
	std::vector <geometry> meshes;

	loader(const std::filesystem::path &path) {
		Assimp::Importer importer;

		// Read scene
		const aiScene *scene;
		scene = importer.ReadFile(path, aiProcess_GenNormals | aiProcess_Triangulate);

		// Check if the scene was loaded
		if ((!scene | scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) || !scene->mRootNode) {
			ulog_error("loader", "Assimp error: \"%s\"\n", importer.GetErrorString());
			return;
		}

		process_node(scene->mRootNode, scene, path.parent_path());
	}

	void process_node(aiNode *node, const aiScene *scene, const std::string &directory) {
		// Process all the node's meshes (if any)
		for (uint32_t i = 0; i < node->mNumMeshes; i++) {
			aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
			process_mesh(mesh, scene, directory);
		}

		// Recusively process all the node's children
		for (uint32_t i = 0; i < node->mNumChildren; i++)
			process_node(node->mChildren[i], scene, directory);

	}

	void process_mesh(aiMesh *, const aiScene *, const std::string &);

	const geometry &get(uint32_t i) const {
		return meshes[i];
	}
};

void loader::process_mesh(aiMesh *mesh, const aiScene *scene, const std::string &dir)
{

	std::vector <glm::vec3> vertices;
	std::vector <glm::vec3> normals;
        std::vector <glm::uvec3> indices;

	// Process all the mesh's vertices
	for (uint32_t i = 0; i < mesh->mNumVertices; i++) {
		vertices.push_back({
			mesh->mVertices[i].x,
			mesh->mVertices[i].y,
			mesh->mVertices[i].z
		});

		if (mesh->HasNormals()) {
			normals.push_back({
				mesh->mNormals[i].x,
				mesh->mNormals[i].y,
				mesh->mNormals[i].z
			});
		} else {
			normals.push_back({ 0.0f, 0.0f, 0.0f });
		}
	}

	// Process all the mesh's triangles
	for (uint32_t i = 0; i < mesh->mNumFaces; i++) {
		aiFace face = mesh->mFaces[i];
		ulog_assert(face.mNumIndices == 3, "process_mesh", "Only triangles are supported, got %d-sided polygon instead\n", face.mNumIndices);
		indices.push_back({
			face.mIndices[0],
			face.mIndices[1],
			face.mIndices[2]
		});
	}

	meshes.push_back({ vertices, normals, indices });
}

int main(int argc, char *argv[])
{
	// Expect a filename
	if (argc < 3) {
		ulog_error("rasterizer", "./rasterizer <reference> <nsc binary>\n");
		return 1;
	}

	// Read the reference mesh
	geometry reference = loader(argv[1]).get(0);

	ulog_info("main", "Reference mesh has %d vertices and %d triangles\n", reference.vertices.size(), reference.indices.size());

	// Open the file
	FILE *file = fopen(argv[2], "rb");
	if (!file) {
		ulog_error("rasterizer", "Could not open file %s\n", argv[1]);
		return 1;
	}

	sdc_read(file);
	dnn_read(file);

	constexpr uint32_t rate = 9;
	static_assert(rate <= MAXIMUM_SAMPLE_RATE, "rate > MAXIMUM_SAMPLE_RATE");

	// Configure renderer
	auto predicate = [](vk::PhysicalDevice phdev) {
		return littlevk::physical_device_able(phdev, {
			VK_KHR_SWAPCHAIN_EXTENSION_NAME
		});
	};

	vk::PhysicalDevice phdev;
	phdev = littlevk::pick_physical_device(predicate);

	Renderer renderer;
	renderer.from(phdev);

	// Evaluate the surface
	eval_cuda(rate);
	std::vector <glm::vec3> vertices = eval(rate);
	std::vector <std::array <uint32_t, 3>> indices = nsc_indices(vertices, g_sdc.complex_count, rate);

	// Translate to a Vulkan mesh
	Transform model_transform = {};
	littlevk::Buffer vertex_buffer;
	littlevk::Buffer index_buffer;

	vertex_buffer = littlevk::buffer(renderer.device,
			vertices,
			vk::BufferUsageFlagBits::eVertexBuffer,
			renderer.mem_props).unwrap(renderer.dal);

	index_buffer = littlevk::buffer(renderer.device,
			indices,
			vk::BufferUsageFlagBits::eIndexBuffer,
			renderer.mem_props).unwrap(renderer.dal);

	bool render_solid = true;
	bool render_wireframe = false;

	auto solid_hook = [&](const vk::CommandBuffer &cmd) {
		if (!render_solid)
			return;

		auto *pc = &renderer.push_constants;
		pc->model = model_transform.matrix();
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, renderer.solid.pipeline);
		cmd.pushConstants <Renderer::push_constants_struct> (renderer.solid.pipeline_layout, vk::ShaderStageFlagBits::eVertex, 0, *pc);
		cmd.bindVertexBuffers(0, { vertex_buffer.buffer }, { 0 });
		cmd.bindIndexBuffer(index_buffer.buffer, 0, vk::IndexType::eUint32);
		cmd.drawIndexed(indices.size() * 3, 1, 0, 0, 0);
	};

	auto wireframe_hook = [&](const vk::CommandBuffer &cmd) {
		if (!render_wireframe)
			return;

		auto *pc = &renderer.push_constants;
		pc->model = model_transform.matrix();
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, renderer.wireframe.pipeline);
		cmd.pushConstants <Renderer::push_constants_struct> (renderer.wireframe.pipeline_layout, vk::ShaderStageFlagBits::eVertex, 0, *pc);
		cmd.bindVertexBuffers(0, { vertex_buffer.buffer }, { 0 });
		cmd.bindIndexBuffer(index_buffer.buffer, 0, vk::IndexType::eUint32);
		cmd.drawIndexed(indices.size() * 3, 1, 0, 0, 0);
	};

	auto ui_hook = [&](const vk::CommandBuffer &cmd) {
		float frame_time = ImGui::GetIO().DeltaTime;
		ImGui::Begin("Info");
		ImGui::Text("Frame time: %f ms", frame_time * 1000.0f);

		if (ImGui::Checkbox("Solid", &render_solid))
			render_wireframe = !render_solid;
		if (ImGui::Checkbox("Wireframe", &render_wireframe))
			render_solid = !render_wireframe;

		ImGui::End();
	};

	renderer.hooks.push_back(solid_hook);
	renderer.hooks.push_back(wireframe_hook);
	renderer.hooks.push_back(ui_hook);

	// Translate to a Vulkan mesh
	Transform ref_model_transform {
		.position = glm::vec3 { 1.0f, 0.0f, 4.0f },
	};

	littlevk::Buffer ref_vertex_buffer;
	littlevk::Buffer ref_index_buffer;

	ref_vertex_buffer = littlevk::buffer(renderer.device,
			interleave_attributes(reference),
			vk::BufferUsageFlagBits::eVertexBuffer,
			renderer.mem_props).unwrap(renderer.dal);

	ref_index_buffer = littlevk::buffer(renderer.device,
			reference.indices,
			vk::BufferUsageFlagBits::eIndexBuffer,
			renderer.mem_props).unwrap(renderer.dal);

	auto render_hook = [&](const vk::CommandBuffer &cmd) {
		auto *pc = &renderer.push_constants;
		pc->model = ref_model_transform.matrix();
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, renderer.shaded.pipeline);
		cmd.pushConstants <Renderer::push_constants_struct> (renderer.shaded.pipeline_layout, vk::ShaderStageFlagBits::eVertex, 0, *pc);
		cmd.bindVertexBuffers(0, { *ref_vertex_buffer }, { 0 });
		cmd.bindIndexBuffer(*ref_index_buffer, 0, vk::IndexType::eUint32);
		cmd.drawIndexed(reference.indices.size() * 3, 1, 0, 0, 0);
	};

	renderer.hooks.push_back(render_hook);

	while (!renderer.should_close()) {
		renderer.render();
		renderer.poll();
	}
}
