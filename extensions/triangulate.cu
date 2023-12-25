#include "common.hpp"

// TODO: microlog
__global__
void triangulate_shorted_kernel(const glm::vec3 *__restrict__ vertices, glm::ivec3 *__restrict__ triangles, size_t sample_rate)
{
	size_t i = blockIdx.x;
	size_t j = threadIdx.x;
	size_t k = threadIdx.y;

	size_t offset = i * sample_rate * sample_rate;

	size_t a = offset + j * sample_rate + k;
	size_t b = a + 1;
	size_t c = offset + (j + 1) * sample_rate + k;
	size_t d = c + 1;

	const glm::vec3 &va = vertices[a];
	const glm::vec3 &vb = vertices[b];
	const glm::vec3 &vc = vertices[c];
	const glm::vec3 &vd = vertices[d];

	float d0 = glm::distance(va, vd);
	float d1 = glm::distance(vb, vc);

	size_t toffset = 2 * i * (sample_rate - 1) * (sample_rate - 1);
	size_t tindex = toffset + 2 * (j * (sample_rate - 1) + k);
	if (d0 < d1) {
		triangles[tindex] = glm::ivec3(a, d, b);
		triangles[tindex + 1] = glm::ivec3(a, c, d);
	} else {
		triangles[tindex] = glm::ivec3(a, c, b);
		triangles[tindex + 1] = glm::ivec3(b, c, d);
	}
}

torch::Tensor triangulate_shorted(const torch::Tensor &vertices, size_t complex_count, size_t sample_rate)
{
	assert(vertices.dtype() == torch::kFloat32);
	assert(vertices.dim() == 2 && vertices.size(1) == 3);
	assert(vertices.is_cuda());

	long triangle_count = 2 * complex_count * (sample_rate - 1) * (sample_rate - 1);

	auto options = torch::TensorOptions()
		.dtype(torch::kInt32)
		.device(torch::kCUDA, 0);

	torch::Tensor out = torch::zeros({ triangle_count, 3 }, options);

	glm::vec3 *vertices_ptr = (glm::vec3 *) vertices.data_ptr <float> ();
	glm::ivec3 *out_ptr = (glm::ivec3 *) out.data_ptr <int32_t> ();

	dim3 block(sample_rate - 1, sample_rate - 1);
	dim3 grid(complex_count);

	triangulate_shorted_kernel <<< grid, block >>> (vertices_ptr, out_ptr, sample_rate);

	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
		exit(1);
	}

	return out;
}
