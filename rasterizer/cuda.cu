#include "common.hpp"
#include "util.hpp"

// Batch size should be (batch [grid.x], rate, rate [block.x, block.y]) [total to vertex count]
__global__
static void interpolate_kernel(const float4 *__restrict__ flat, float *__restrict__ lpf, uint32_t sample_rate, uint32_t feature_size)
{
	// TODO: use texture memory to interpolate...
	uint32_t cid = blockIdx.x;
	uint32_t ix  = threadIdx.x;
	uint32_t iy  = threadIdx.y;

	float u = (float) ix / (sample_rate - 1);
	float v = (float) iy / (sample_rate - 1);

	uint32_t stride = feature_size + 3;
	uint32_t did    = (cid * sample_rate + ix) * sample_rate + iy;

	#pragma unroll
	for (uint32_t k = 0; k < 23; k++) {
		float4 fv = flat[cid * stride + k];
		float a = __fmaf_rn(fv.x, (1 - u) * (1 - v), fv.y * u * (1 - v));
		float b = __fmaf_rn(fv.w, (1 - u) * v, fv.z * u * v);
		lpf[did * stride + k] = __fmaf_rn(a, 1, b);
	}
}

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
static void activate(float *__restrict__ Y, uint32_t N)
{
	for (uint32_t i = 0; i < N; i++) {
		float x = Y[i];
		Y[i] = (x > 0) ? x : 0.01f * x;
	}
}

// Batch size should be (batch [grid.x], batch size [block.x]) [total to vertex count]
__constant__ float Wm_c0[84 * 64];
__constant__ float Wm_c1[65 * 64];
__constant__ float Wm_c2[65 * 64];
__constant__ float Wm_c3[65 * 3];

__global__
static void eval_kernel(float3 *__restrict__ lp,
		const float *__restrict__ lpf,
		uint32_t feature_size,
		uint32_t vertex_count)
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
	uint32_t stride = feature_size + 3;
	uint32_t base = vid * stride;

	float vx = lpf[base + 0];
	float vy = lpf[base + 1];
	float vz = lpf[base + 2];

	uint32_t j = 0;
	for (uint32_t i = 0; i < feature_size; i++)
		embedding[j++] = lpf[base + 3 + i];

	embedding[j++] = vx;
	embedding[j++] = vy;
	embedding[j++] = vz;

	float twok = 1.0f;

	#pragma unroll
	for (uint32_t L = 0; L < FREQUENCIES; L++) {
		float x = vx * twok;
		float y = vy * twok;
		float z = vz * twok;
		twok *= 2.0f;

		embedding[j++] = __sinf(x);
		embedding[j++] = __sinf(y);
		embedding[j++] = __sinf(z);
		embedding[j++] = __cosf(x);
		embedding[j++] = __cosf(y);
		embedding[j++] = __cosf(z);
	}

	// Compute each of the layers
	matmul(Wm_c0, embedding, layer1, 83, 64);
	activate(layer1, 64);

	matmul(Wm_c1, layer1, layer2, 64, 64);
	activate(layer2, 64);

	matmul(Wm_c2, layer2, layer3, 64, 64);
	activate(layer3, 64);

	matmul(Wm_c3, layer3, layer4, 64, 3);

	// Add the displacement to the vertex
	lp[vid] = make_float3(vx + layer4[0], vy + layer4[1], vz + layer4[2]);
}

// Evaluate the neural network (on CUDA device)
std::vector <glm::vec3> eval_cuda(uint32_t sample_rate)
{
	timer      clk;
	cuda_timer cu_clk;

	// Cache context
	struct {
		float3    *lp  = nullptr;
		float     *lpf = nullptr;
	} static context;

	// ulog_info("eval_cuda", "sample_rate = %u\n", sample_rate);

	// Allocate memory on CUDA device
	uint32_t vertex_count = sample_rate * sample_rate * g_sdc.complex_count;

	if (context.lp == nullptr || context.lpf == nullptr) {
		ulog_info("eval_cuda", "Allocating memory for context\n");

		cudaMalloc(&context.lp, vertex_count * sizeof(float3));
		cudaMalloc(&context.lpf, vertex_count * (g_sdc.feature_size + 3) * sizeof(float));
	}

	// Copy weights to the constant memory
	// TODO: store biases separately and evaluate together with bias (especially the last layer; or precompute the bias added to the vertices?)
	CUDA_CHECK(cudaMemcpyToSymbol(Wm_c0, g_dnn.d_Wm_c[0], g_dnn.Wm_c[0].size() * sizeof(float), 0, cudaMemcpyDeviceToDevice));
	CUDA_CHECK(cudaMemcpyToSymbol(Wm_c1, g_dnn.d_Wm_c[1], g_dnn.Wm_c[1].size() * sizeof(float), 0, cudaMemcpyDeviceToDevice));
	CUDA_CHECK(cudaMemcpyToSymbol(Wm_c2, g_dnn.d_Wm_c[2], g_dnn.Wm_c[2].size() * sizeof(float), 0, cudaMemcpyDeviceToDevice));
	CUDA_CHECK(cudaMemcpyToSymbol(Wm_c3, g_dnn.d_Wm_c[3], g_dnn.Wm_c[3].size() * sizeof(float), 0, cudaMemcpyDeviceToDevice));

	{
		clk.tick();

		// ulog_info("eval_cuda", "interpolate_kernel\n");

		// TODO: scoped timer
		dim3 grid  (g_sdc.complex_count);
		dim3 block (sample_rate, sample_rate);

		cu_clk.tick();
		interpolate_kernel <<< grid, block >>> (g_sdc.d_flat, context.lpf, sample_rate, g_sdc.feature_size);
		double cu_t = cu_clk.tock();

		// TODO: context stream
		// CUDA_CHECK_SYNCED();

		double t = clk.tock();
		// printf("interpolate_kernel: %.2f ms (%.2f ms)\n", t, cu_t);
	}

	{
		// TODO: try shared memory...
		clk.tick();

		// ulog_info("eval_cuda", "eval_kernel\n");

		dim3 grid  (g_sdc.complex_count);
		dim3 block (sample_rate * sample_rate);

		// TODO: compare with an implementation which does matrix multiplies...
		cu_clk.tick();
		eval_kernel <<< grid, block >>> (context.lp, context.lpf, g_sdc.feature_size, vertex_count);
		double cu_t = cu_clk.tock();

		double t = clk.tock();
		// printf("eval_kernel: %.2f ms (%.2f ms)\n", t, cu_t);
	}

	// TODO: cache context
	// ulog_info("eval_cuda", "finished computations, doing final memory transactions\n");

	// TODO: or ansyn copy (or none at all...)
	CUDA_CHECK_SYNCED();
	std::vector <glm::vec3> result(vertex_count);
	cudaMemcpy(result.data(), context.lp, vertex_count * sizeof(float3), cudaMemcpyDeviceToHost);
	CUDA_CHECK_SYNCED();

	// cudaFree(lp);
	// cudaFree(lf);

	return result;
}
