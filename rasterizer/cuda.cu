#include "common.hpp"

// TODO: batch by each (or several complexes; set some maximum cache/interim memory amt)
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

	float u = (float) i / (float) (sample_rate - 1);
	float v = (float) j / (float) (sample_rate - 1);

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
	float *d_embedded = g_dnn.d_embedded;
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

__global__
void displace(float *embedded_base, float *displacement_base, uint32_t embedded_size, uint32_t feature_count, uint32_t vertex_count)
{
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t stride = blockDim.x * gridDim.x;

	// Writes to the displacement buffer
	for (uint32_t k = i; k < vertex_count; k += stride) {
		float *displacement = displacement_base + k * 3;
		float *embedded = embedded_base + k * embedded_size + feature_count;

		displacement[0] += embedded[0];
		displacement[1] += embedded[1];
		displacement[2] += embedded[2];
	}
}

std::vector <glm::vec3> eval_cuda(uint32_t sample_rate)
{
	constexpr uint32_t L = 8;

	uint32_t embedded_size = g_sdc.feature_count + 3 * (2 * L + 1);

	lerp_and_embed(sample_rate);
	cudaDeviceSynchronize();
	CUDA_CHECK_ERROR();

	// {
	// 	std::vector <float> buffer;
	// 	buffer.resize(g_sdc.complex_count * sample_rate * sample_rate * embedded_size);
	// 	cudaMemcpy(buffer.data(), g_dnn.d_embedded, buffer.size() * sizeof(float), cudaMemcpyDeviceToHost);
	// 	printf("embedding[0] =");
	// 	for (size_t i = 0; i < embedded_size; i++)
	// 		printf(" %f", buffer[i]);
	// 	printf("\n");
	// }

	// Evaluate the first layer
	ulog_info("eval_cuda", "evaluating first layer\n");

	dim3 block_dim(sample_rate * sample_rate);
	// dim3 grid_dim(g_sdc.complex_count);
	dim3 grid_dim(1, 1);

	float *d_out = g_dnn.d_interim_two;
	float *d_in = g_dnn.d_embedded;

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

	// {
	// 	std::vector <glm::vec3> buffer;
	// 	buffer.resize(g_sdc.complex_count * sample_rate * sample_rate);
	// 	cudaMemcpy(buffer.data(), g_dnn.d_interim_two, buffer.size() * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	// 	printf("displacements[0] = %f %f %f\n", buffer[0].x, buffer[0].y, buffer[0].z);
	// }

	// Combine with original vertices
	ulog_info("eval_cuda", "combining with original vertices\n");

	// TODO: whiy is this part slow? time all the parts...
	uint32_t vertex_count = sample_rate * sample_rate * g_sdc.complex_count;
	block_dim = 256;
	grid_dim = (vertex_count + block_dim.x - 1) / block_dim.x;

	displace <<< grid_dim, block_dim >>> (g_dnn.d_embedded, g_dnn.d_interim_two, embedded_size, g_sdc.feature_count, vertex_count);
	cudaDeviceSynchronize();
	CUDA_CHECK_ERROR();

	std::vector <glm::vec3> buffer;
	buffer.resize(g_sdc.complex_count * sample_rate * sample_rate);
	cudaMemcpy(buffer.data(), g_dnn.d_interim_two, buffer.size() * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	printf("displacements[0] = %f %f %f\n", buffer[0].x, buffer[0].y, buffer[0].z);

	return buffer;
}

