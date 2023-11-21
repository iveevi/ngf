#include "common.hpp"

// CUDA interpolation kernel
// Batch size should be (complexes [grid.x], sample rate [block.x], sample rate [block.y])
struct interpolate_kernel_info {
	glm::vec3        *__restrict__ lp;
	float            *__restrict__ lf;

	const glm::ivec4 *__restrict__ complexes;
	const glm::vec3  *__restrict__ vertices;
	const float      *__restrict__ features;

	uint32_t                       sample_rate;
	uint32_t                       feature_size;
};

__global__
static void interpolate_kernel(interpolate_kernel_info kinfo)
{
	uint32_t cid = blockIdx.x;
	uint32_t ix  = threadIdx.x;
	uint32_t iy  = threadIdx.y;

	const glm::ivec4 complex = kinfo.complexes[cid];

	const glm::vec3 v00 = kinfo.vertices[complex.x];
	const glm::vec3 v10 = kinfo.vertices[complex.y];
	const glm::vec3 v01 = kinfo.vertices[complex.w];
	const glm::vec3 v11 = kinfo.vertices[complex.z];

	const float *f00 = &kinfo.features[complex.x * kinfo.feature_size];
	const float *f10 = &kinfo.features[complex.y * kinfo.feature_size];
	const float *f01 = &kinfo.features[complex.w * kinfo.feature_size];
	const float *f11 = &kinfo.features[complex.z * kinfo.feature_size];

	float u = (float) ix / (kinfo.sample_rate - 1);
	float v = (float) iy / (kinfo.sample_rate - 1);

	// Destination index
	uint32_t did = (cid * kinfo.sample_rate + ix) * kinfo.sample_rate + iy;

	// Vertex interpolation
	glm::vec3 lp00 = v00 * (1 - u) * (1 - v);
	glm::vec3 lp10 = v10 * u * (1 - v);
	glm::vec3 lp01 = v01 * (1 - u) * v;
	glm::vec3 lp11 = v11 * u * v;

	kinfo.lp[did] = lp00 + lp10 + lp01 + lp11;

	// Feature interpolation
	for (uint32_t k = 0; k < kinfo.feature_size; k++) {
		float f00k = f00[k] * (1 - u) * (1 - v);
		float f10k = f10[k] * u * (1 - v);
		float f01k = f01[k] * (1 - u) * v;
		float f11k = f11[k] * u * v;
		kinfo.lf[did * kinfo.feature_size + k] = f00k + f10k + f01k + f11k;
	}
}

// End to end evaluation
// Batch size should be (batch [grid.x], batch size [block.x]) [total to vertex count]
struct eval_kernel_info {
	glm::vec3   *__restrict__ lp;
	const float *__restrict__ lf;
	const float *__restrict__ Wm_c[4];

	float                     s0;
	float                     s1;
	float                     s2;

	uint32_t                  feature_size;
	uint32_t                  vertex_count;
};

__forceinline__ __device__
static void matmul(const float *__restrict__ Wm_c, const float *__restrict__ X, float *__restrict__ Y, uint32_t in, uint32_t out)
{
	for (uint32_t i = 0; i < out; i++) {
		float sum = Wm_c[i * (in + 1) + in];
		for (uint32_t j = 0; j < in; j++)
			sum += Wm_c[i * (in + 1) + j] * X[j];
		Y[i] = sum;
	}
}

__forceinline__ __device__
static void activate(float *__restrict__ Y, float s, uint32_t N)
{
	for (uint32_t i = 0; i < N; i++) {
		float x = Y[i];
		float softplus = log(1 + exp(s * x));
		float sin = sinf(s * x);
		float gauss = exp(-x * x / s);
		Y[i] = softplus * sin * gauss;
	}
}

__global__
static void eval_kernel(eval_kernel_info kinfo)
{
	uint32_t vid = blockIdx.x * blockDim.x + threadIdx.x;

	// Intermediary variables
	// TODO: or split into smaller blocks at a time?
	float embedding[83];
	float layer1[64];
	float layer2[64];
	float layer3[64];
	float layer4[3];

	// TODO: size verification?

	// Fill the embedding
	glm::vec3    p = kinfo.lp[vid];
	const float *f = &kinfo.lf[vid * kinfo.feature_size];

	uint32_t j = 0;
	for (uint32_t i = 0; i < kinfo.feature_size; i++)
		embedding[j++] = f[i];

	embedding[j++] = p.x;
	embedding[j++] = p.y;
	embedding[j++] = p.z;

	for (uint32_t L = 0; L < FREQUENCIES; L++) {
		float x = p.x * powf(2, L);
		float y = p.y * powf(2, L);
		float z = p.z * powf(2, L);

		embedding[j++] = sinf(x);
		embedding[j++] = sinf(y);
		embedding[j++] = sinf(z);
		embedding[j++] = cosf(x);
		embedding[j++] = cosf(y);
		embedding[j++] = cosf(z);
	}

	// Compute each of the layers
	matmul(kinfo.Wm_c[0], embedding, layer1, 83, 64);
	activate(layer1, kinfo.s0, 64);

	matmul(kinfo.Wm_c[1], layer1, layer2, 64, 64);
	activate(layer2, kinfo.s1, 64);

	matmul(kinfo.Wm_c[2], layer2, layer3, 64, 64);
	activate(layer3, kinfo.s2, 64);

	matmul(kinfo.Wm_c[3], layer3, layer4, 64, 3);

	// Add the displacement to the vertex
	kinfo.lp[vid] += glm::vec3(layer4[0], layer4[1], layer4[2]);
}

// Evaluate the neural network (on CUDA device)
std::vector <glm::vec3> eval_cuda(uint32_t sample_rate)
{
	// Cache context
	struct {
		glm::vec3 *lp = nullptr;
		float     *lf = nullptr;
	} static context;

	ulog_info("eval_cuda", "sample_rate = %u\n", sample_rate);

	// Allocate memory on CUDA device
	// glm::vec3 *lp;
	// float     *lf;

	uint32_t vertex_count = sample_rate * sample_rate * g_sdc.complex_count;
	if (context.lp == nullptr || context.lf == nullptr) {
		ulog_info("eval_cuda", "Allocating memory for context\n");

		cudaMalloc(&context.lp, vertex_count * sizeof(glm::vec3));
		cudaMalloc(&context.lf, vertex_count * g_sdc.feature_size * sizeof(float));
	}

	// Prepare for kernel and launch it
	interpolate_kernel_info iinfo;
	iinfo.lp = context.lp;
	iinfo.lf = context.lf;

	iinfo.complexes = g_sdc.d_complexes;
	iinfo.vertices  = g_sdc.d_vertices;
	iinfo.features  = g_sdc.d_features;

	iinfo.sample_rate = sample_rate;
	iinfo.feature_size = g_sdc.feature_size;

	{
		ulog_info("eval_cuda", "interpolate_kernel\n");

		// TODO: scoped timer
		dim3 grid  (g_sdc.complex_count);
		dim3 block (sample_rate, sample_rate);

		interpolate_kernel <<<grid, block>>> (iinfo);
		CUDA_CHECK_SYNCED();
	}

	// Evaluate the neural network
	eval_kernel_info einfo;
	einfo.lp = context.lp;
	einfo.lf = context.lf;

	einfo.Wm_c[0] = g_dnn.d_Wm_c[0];
	einfo.Wm_c[1] = g_dnn.d_Wm_c[1];
	einfo.Wm_c[2] = g_dnn.d_Wm_c[2];
	einfo.Wm_c[3] = g_dnn.d_Wm_c[3];

	einfo.s0 = g_dnn.constants[0];
	einfo.s1 = g_dnn.constants[1];
	einfo.s2 = g_dnn.constants[2];

	einfo.feature_size = g_sdc.feature_size;
	einfo.vertex_count = vertex_count;

	{
		ulog_info("eval_cuda", "eval_kernel\n");

		dim3 grid  (g_sdc.complex_count);
		dim3 block (sample_rate * sample_rate);

		eval_kernel <<<grid, block>>> (einfo);
		CUDA_CHECK_SYNCED();
	}

	// TODO: cache context
	ulog_info("eval_cuda", "finished computations, doing final memory transactions\n");

	std::vector <glm::vec3> result(vertex_count);
	cudaMemcpy(result.data(), context.lp, vertex_count * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	CUDA_CHECK_SYNCED();

	// cudaFree(lp);
	// cudaFree(lf);

	return result;
}
