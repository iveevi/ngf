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
	uint32_t                feature_count;
	uint32_t                complex_count;
	const glm::vec3 *__restrict__ vertices;
	const float     *__restrict__ features;
};

__global__
void lerp_and_embed(float *__restrict__ embedded, EmbedInfo info, glm::uvec4 complex, uint32_t sample_rate)
{
	// TODO: shared memomory the features and vertices in the complex
	// Simple 2D work group
	uint32_t i = threadIdx.x;
	uint32_t j = threadIdx.y;
	uint32_t b = blockIdx.x;

	if (i >= sample_rate || j >= sample_rate) {
		constexpr uint32_t L   = 8;
		uint32_t embedded_size = info.feature_count + 3 * (2 * L + 1);
		uint32_t offset        = (i * sample_rate + j) * embedded_size;

		for (uint32_t k = 0; k < embedded_size; k++)
			embedded[offset + k] = 0.0f;

		return;
	}

	glm::vec3 v00 = info.vertices[complex.x];
	glm::vec3 v10 = info.vertices[complex.y];
	glm::vec3 v01 = info.vertices[complex.z];
	glm::vec3 v11 = info.vertices[complex.w];

	float u = (float) i / (float) (sample_rate - 1);
	float v = (float) j / (float) (sample_rate - 1);

	// Compute the lerped feature and put into the embedded buffer
	constexpr uint32_t L   = 8;
	uint32_t embedded_size = info.feature_count + 3 * (2 * L + 1);
	uint32_t offset        = (i * sample_rate + j) * embedded_size;

	for (size_t k = 0; k < info.feature_count; k++) {
		float f00k = info.features[complex.x * info.feature_count + k];
		float f10k = info.features[complex.y * info.feature_count + k];
		float f11k = info.features[complex.w * info.feature_count + k];
		float f01k = info.features[complex.z * info.feature_count + k];
		embedded[offset + k] = f00k * u * v
			+ f10k * (1.0f - u) * v
			+ f11k * u * (1.0f - v)
			+ f01k * (1.0f - u) * (1.0f - v);
	}

	// Lerp the vertex position
	glm::vec3 P = v00 * u * v
		      + v10 * (1.0f - u) * v
		      + v11 * u * (1.0f - v)
		      + v01 * (1.0f - u) * (1.0f - v);

	// Fill the rest of the embedded buffer with the positional encoding
	embedded[offset + info.feature_count + 0] = P.x;
	embedded[offset + info.feature_count + 1] = P.y;
	embedded[offset + info.feature_count + 2] = P.z;

	#pragma unroll
	for (uint32_t k = 0; k < L; k++) {
		float x = P.x * powf(2.0f, (float) k);
		float y = P.y * powf(2.0f, (float) k);
		float z = P.z * powf(2.0f, (float) k);

		embedded[offset + info.feature_count + 6 * k + 3] = sinf(x);
		embedded[offset + info.feature_count + 6 * k + 4] = sinf(y);
		embedded[offset + info.feature_count + 6 * k + 5] = sinf(z);
		embedded[offset + info.feature_count + 6 * k + 6] = cosf(x);
		embedded[offset + info.feature_count + 6 * k + 7] = cosf(y);
		embedded[offset + info.feature_count + 6 * k + 8] = cosf(z);
	}
}

__global__
void lerp_and_embed_parallel(float *__restrict__ embedded, EmbedInfo info, glm::uvec4 *complexes, uint32_t sample_rate)
{
	// TODO: shared memomory the features and vertices in the complex
	// Simple 2D work group
	uint32_t i = threadIdx.x;
	uint32_t j = threadIdx.y;
	uint32_t b = blockIdx.x;

	if (i >= sample_rate || j >= sample_rate)
		return;

	if (b >= info.complex_count)
		return;

	glm::uvec4 complex = complexes[b];

	glm::vec3 v00 = info.vertices[complex.x];
	glm::vec3 v10 = info.vertices[complex.y];
	glm::vec3 v01 = info.vertices[complex.z];
	glm::vec3 v11 = info.vertices[complex.w];

	float u = (float) i / (float) (sample_rate - 1);
	float v = (float) j / (float) (sample_rate - 1);

	// Compute the lerped feature and put into the embedded buffer
	constexpr uint32_t L   = 8;
	uint32_t embedded_size = info.feature_count + 3 * (2 * L + 1);
	uint32_t complex_offset = b * sample_rate * sample_rate;
	uint32_t offset        = (i * sample_rate + j) * embedded_size + complex_offset * embedded_size;

	for (size_t k = 0; k < info.feature_count; k++) {
		float f00k = info.features[complex.x * info.feature_count + k];
		float f10k = info.features[complex.y * info.feature_count + k];
		float f11k = info.features[complex.w * info.feature_count + k];
		float f01k = info.features[complex.z * info.feature_count + k];
		embedded[offset + k] = f00k * u * v
			+ f10k * (1.0f - u) * v
			+ f11k * u * (1.0f - v)
			+ f01k * (1.0f - u) * (1.0f - v);
	}

	// Lerp the vertex position
	glm::vec3 P = v00 * u * v
		      + v10 * (1.0f - u) * v
		      + v11 * u * (1.0f - v)
		      + v01 * (1.0f - u) * (1.0f - v);

	// Fill the rest of the embedded buffer with the positional encoding
	embedded[offset + info.feature_count + 0] = P.x;
	embedded[offset + info.feature_count + 1] = P.y;
	embedded[offset + info.feature_count + 2] = P.z;

	#pragma unroll
	for (uint32_t k = 0; k < L; k++) {
		float x = P.x * powf(2.0f, (float) k);
		float y = P.y * powf(2.0f, (float) k);
		float z = P.z * powf(2.0f, (float) k);

		embedded[offset + info.feature_count + 6 * k + 3] = sinf(x);
		embedded[offset + info.feature_count + 6 * k + 4] = sinf(y);
		embedded[offset + info.feature_count + 6 * k + 5] = sinf(z);
		embedded[offset + info.feature_count + 6 * k + 6] = cosf(x);
		embedded[offset + info.feature_count + 6 * k + 7] = cosf(y);
		embedded[offset + info.feature_count + 6 * k + 8] = cosf(z);
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
		.complex_count = g_sdc.complex_count,
		.vertices = g_sdc.d_vertices,
		.features = g_sdc.d_features,
	};

	dim3 block_dim(sample_rate, sample_rate);
	dim3 grid_dim(g_sdc.complex_count);

	lerp_and_embed_parallel <<< grid_dim, block_dim >>> (d_embedded, info, g_sdc.d_complexes, sample_rate);

	cudaDeviceSynchronize();
	CUDA_CHECK_ERROR();
}

template <float (*act)(float)>
__global__
void matmul_biased_parallel(float *__restrict__ dst_base, const float *__restrict__ W_base, const float *__restrict__ X_base, uint32_t in, uint32_t out, uint32_t sample_rate)
{
	// Assumes that X is a batch of dimension in x batch_size
	// Assumes that W is a matrix of dimension out x (in + 1)

	// TODO: try putting the matrix into closer shared memory instead of constant memory
	// or into constant cache...

	// Computes as many output features as there are threads (horizontal parallelism)
	uint32_t i = threadIdx.x;
	uint32_t j = blockIdx.x;

	// Register the input
	float X_local[64];
	const float *X = &X_base[in * i + j * sample_rate * sample_rate * in];
	for (size_t k = 0; k < in; k++)
		X_local[k] = X[k];

	// TODO: shared input...
	float *dst = dst_base + i * out + j * sample_rate * sample_rate * out;

	for (size_t n = 0; n < out; n++) {
		const float *W = &W_base[n * (in + 1)];

		float sum = W[in];
		for (size_t k = 0; k < in; k++)
			sum += W[k] * X_local[k];

		if constexpr (act == nullptr)
			dst[n] = sum;
		else
			dst[n] = act(sum);
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

#include <glm/gtx/string_cast.hpp>

eval_cuda_result eval_cuda(const glm::mat4 &proj, const glm::mat4 &view, uint32_t sample_rate)
{
	constexpr uint32_t L = 8;

	uint32_t embedded_size = g_sdc.feature_count + 3 * (2 * L + 1);

	// Evaluate sampling rates for each complex (min 2, max 16)
	glm::mat4 model(1.0f); // assuming idle model matrix

	std::vector <uint32_t> sample_rates(g_sdc.complex_count);

	float resolution = 1000.0f;
	for (uint32_t i = 0; i < g_sdc.complex_count; i++) {
		const glm::uvec4 &complex = g_sdc.complexes[i];

		glm::vec4 v0(g_sdc.vertices[complex.x], 1.0f);
		glm::vec4 v1(g_sdc.vertices[complex.y], 1.0f);
		glm::vec4 v2(g_sdc.vertices[complex.z], 1.0f);
		glm::vec4 v3(g_sdc.vertices[complex.w], 1.0f);

		// Get distnce from camera to the center of the quad
		// glm::vec3 center = (v0 + v1 + v2 + v3) / 4.0f;
		//
		// glm::vec3 center_view = glm::vec3(view[3][0], view[3][1], view[3][2]);
		// float distance = glm::length(center - center_view);
		//
		// // Get sampling rate as a function of distance
		// float rate = std::clamp(log2f(resolution/distance), 4.0f, 16.0f);

		// // Project vertices
		v0 = proj * view * model * v0;
		v1 = proj * view * model * v1;
		v2 = proj * view * model * v2;
		v3 = proj * view * model * v3;

		// Compute area of the resulting quad
		glm::vec2 pv0 = glm::vec2(v0) / v0.w;
		glm::vec2 pv1 = glm::vec2(v1) / v1.w;
		glm::vec2 pv2 = glm::vec2(v2) / v2.w;
		glm::vec2 pv3 = glm::vec2(v3) / v3.w;

		glm::vec2 e0 = pv1 - pv0;
		glm::vec2 e1 = pv2 - pv1;
		glm::vec2 e2 = pv0 - pv2;

		// Triangle 1 (0 -- 1 -- 2)
		float a0 = glm::length(e0) * glm::length(e1) * 0.5f;

		// Triangle 2 (0 -- 2 -- 3)
		float a1 = glm::length(e2) * glm::length(e0) * 0.5f;

		// Compute the sampling rate
		float area = a0 + a1;
		float pixels = area * resolution;
		// float pixels_per_triangle = pixels / (2.0f *
		// printf("area: %f, pixels: %f\n", area, pixels);

		// Find smallest resolution which results in at least 1-4 pixels per triangles
		uint32_t rate = 2;
		while (rate * rate < pixels && rate < 16)
			rate++;

		sample_rates[i] = rate;
		// printf("complex: %d pixels, rate %d\n", (int) pixels, rate);
	}

	// TODO: update log timer manaully here?
	// ulog_info("eval_cuda", "embedding %d complexes\n", g_sdc.complex_count);
	// lerp_and_embed(sample_rate);
	// cudaDeviceSynchronize();
	// CUDA_CHECK_ERROR();

	// Compute triangle indices (TODO: atomic additions from CUDA kernel)
	std::vector <std::array <uint32_t, 3>> triangles;
	for (uint32_t i = 0; i < g_sdc.complex_count; i++) {
		uint32_t rate = sample_rates[i];

		// Compute triangle indices
		// TODO: this is using uniform evaluation...
		uint32_t offset = i * sample_rate * sample_rate;
		for (uint32_t j = 0; j < rate - 1; j++) {
			for (uint32_t k = 0; k < rate - 1; k++) {
				uint32_t i0 = offset + j * rate + k;
				uint32_t i1 = i0 + 1;
				uint32_t i2 = offset + (j + 1) * rate + k;
				uint32_t i3 = i2 + 1;

				triangles.push_back({ i0, i1, i2 });
				triangles.push_back({ i1, i2, i3 });
			}
		}
	}

	{
		// Always start with the first interim buffer
		float *d_embedded = g_dnn.d_embedded;
		ulog_info("lerped_and_embed", "embedding %d complexes\n", g_sdc.complex_count);

		uint32_t embedded_size = g_sdc.feature_count + 3 * (2 * L + 1);

		EmbedInfo info = {
			.feature_count = g_sdc.feature_count,
			.complex_count = g_sdc.complex_count,
			.vertices = g_sdc.d_vertices,
			.features = g_sdc.d_features,
		};

		for (size_t i = 0; i < g_sdc.complex_count; i++) {
			glm::uvec4 complex = g_sdc.complexes[i];

			dim3 block_dim(sample_rates[i], sample_rates[i]);
			dim3 grid_dim(1, 1);

			lerp_and_embed <<< grid_dim, block_dim >>> (d_embedded, info, complex, sample_rates[i]);

			d_embedded += sample_rate * sample_rate * embedded_size;
		}

		cudaDeviceSynchronize();
		CUDA_CHECK_ERROR();
	}

	// Evaluate the first layer
	ulog_info("eval_cuda", "evaluating first layer\n");

	dim3 block_dim(sample_rate * sample_rate);
	dim3 grid_dim(g_sdc.complex_count);

	float *d_out = g_dnn.d_interim_two;
	float *d_in = g_dnn.d_embedded;

	matmul_biased_parallel <sinf> <<< grid_dim, block_dim >>> (d_out, g_dnn.d_Wm0c, d_in, embedded_size, g_dnn.W0, sample_rate);

	cudaDeviceSynchronize();
	CUDA_CHECK_ERROR();

	// Evaluate the second layer
	ulog_info("eval_cuda", "evaluating second layer\n");

	d_out = g_dnn.d_interim_one;
	d_in = g_dnn.d_interim_two;

	matmul_biased_parallel <sinf> <<< grid_dim, block_dim >>> (d_out, g_dnn.d_Wm1c, d_in, g_dnn.W0, g_dnn.W1, sample_rate);

	cudaDeviceSynchronize();
	CUDA_CHECK_ERROR();

	// Evaluate the final layer
	ulog_info("eval_cuda", "evaluating final layer\n");

	d_out = g_dnn.d_interim_two;
	d_in = g_dnn.d_interim_one;

	matmul_biased_parallel <nullptr> <<< grid_dim, block_dim >>> (d_out, g_dnn.d_Wm2c, d_in, g_dnn.W1, g_dnn.W2, sample_rate);

	cudaDeviceSynchronize();
	CUDA_CHECK_ERROR();

	// Combine with original vertices
	ulog_info("eval_cuda", "combining with original vertices\n");

	// TODO: whiy is this part slow? time all the parts...
	uint32_t vertex_count = sample_rate * sample_rate * g_sdc.complex_count;
	block_dim = 256;
	grid_dim = (vertex_count + block_dim.x - 1) / block_dim.x;

	printf("vertex_count = %u\n", vertex_count);
	printf("  block_dim = %u\n", block_dim.x);
	printf("  grid_dim = %u\n", grid_dim.x);

	displace <<< grid_dim, block_dim >>> (g_dnn.d_embedded, g_dnn.d_interim_two, embedded_size, g_sdc.feature_count, vertex_count);
	cudaDeviceSynchronize();
	CUDA_CHECK_ERROR();

	std::vector <glm::vec3> buffer;
	buffer.resize(g_sdc.complex_count * sample_rate * sample_rate);
	cudaMemcpy(buffer.data(), g_dnn.d_interim_two, buffer.size() * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	printf("displacements[0] = %f %f %f\n", buffer[0].x, buffer[0].y, buffer[0].z);

	ulog_info("eval_cuda", "finished evaluation\n");

	return { buffer, triangles, triangles.size() };
}

