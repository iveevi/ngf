#include <glm/glm.hpp>

#include <torch/extension.h>

struct cumesh {
	glm::vec3 *vertices;
	glm::vec3 *normals;
	glm::uvec3 *triangles;

	size_t vertex_count;
	size_t triangle_count;
};

	__forceinline__ __host__ __device__
glm::uvec3 pcg(glm::uvec3 v)
{
	v = v * 1664525u + 1013904223u;
	v.x += v.y * v.z;
	v.y += v.z * v.x;
	v.z += v.x * v.y;
	v ^= v >> 16u;
	v.x += v.y * v.z;
	v.y += v.z * v.x;
	v.z += v.x * v.y;
	return v;
}

__forceinline__ __host__ __device__
glm::vec3 pcg(glm::vec3 v)
{
	glm::uvec3 u = *(glm::uvec3 *) &v;
	u = pcg(u);
	u &= glm::uvec3(0x007fffffu);
	u |= glm::uvec3(0x3f800000u);
	return *(glm::vec3 *) &u;
}

__global__
void sample_kernel(cumesh g, glm::vec3 *points, glm::vec3 *normals, int32_t *tris, glm::vec3 *bary, size_t N, float time)
{
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;

	for (size_t i = tid; i < N; i += stride) {
		glm::vec3 seed = glm::vec3(i, time, points[0].x * time);
		seed = pcg(seed);

		float indexf = seed.x - floorf(seed.x);
		size_t index = (size_t) (indexf * g.triangle_count);
		glm::uvec3 t = g.triangles[index];

		glm::vec3 v0 = g.vertices[t.x];
		glm::vec3 v1 = g.vertices[t.y];
		glm::vec3 v2 = g.vertices[t.z];

		glm::vec3 b = *(glm::vec3 *) &seed;
		b = pcg(b);
		b.x -= floorf(b.x);
		b.y -= floorf(b.y);

		if (b.x + b.y > 1.0f) {
			b.x = 1.0f - b.x;
			b.y = 1.0f - b.y;
		}
		b.z = 1.0f - b.x - b.y;

		glm::vec3 p = v0 * b.x + v1 * b.y + v2 * b.z;

		points[i] = p;
		normals[i] = g.normals[index];
		bary[i] = b;
		tris[i] = index;
	}
}

static cumesh translate(const torch::Tensor &vertices, const torch::Tensor &normals, const torch::Tensor &triangles)
{
	// Expects:
	//   2D tensor of shape (N, 3) for vertices
	//   2D tensor of shape (M, 4) for triangles
	assert(vertices.dim() == 2 && vertices.size(1) == 3);
	assert(normals.dim() == 2 && normals.size(1) == 3);
	assert(triangles.dim() == 2 && triangles.size(1) == 3);

	assert(triangles.size(0) == normals.size(0));

	// Ensure CUDA tensors
	assert(vertices.device().is_cuda());
	assert(normals.device().is_cuda());
	assert(triangles.device().is_cuda());

	// Ensure float32 and uint32
	assert(vertices.dtype() == torch::kFloat32);
	assert(normals.dtype() == torch::kFloat32);
	assert(triangles.dtype() == torch::kInt32);

	cumesh g;
	g.vertices = (glm::vec3 *) vertices.data_ptr();
	g.normals = (glm::vec3 *) normals.data_ptr();
	g.triangles = (glm::uvec3 *) triangles.data_ptr();
	g.vertex_count = vertices.size(0);
	g.triangle_count = triangles.size(0);
	return g;
}

auto torch_sample(torch::Tensor vertices, torch::Tensor normals, torch::Tensor triangles, size_t N, float time)
{
	cumesh g = translate(vertices, normals, triangles);

	torch::Tensor Ps = torch::zeros({ (signed long) N, 3 }, torch::kFloat32).cuda();
	torch::Tensor Ns = torch::zeros({ (signed long) N, 3 }, torch::kFloat32).cuda();
	torch::Tensor Ts = torch::zeros({ (signed long) N }, torch::kInt32).cuda();
	torch::Tensor Bs = torch::zeros({ (signed long) N, 3 }, torch::kFloat32).cuda();

	assert(Ps.device().is_cuda());
	assert(Ns.device().is_cuda());
	assert(Ts.device().is_cuda());
	assert(Bs.device().is_cuda());

	dim3 block(256);
	dim3 grid((N + block.x - 1) / block.x);

	sample_kernel <<< grid, block >>> (g,
		(glm::vec3 *) Ps.data_ptr(),
		(glm::vec3 *) Ns.data_ptr(),
		(int32_t *) Ts.data_ptr(),
		(glm::vec3 *) Bs.data_ptr(),
		N, time);

	return std::make_tuple(Ps, Ns, Ts, Bs);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("sample", &torch_sample, "Sample points from a mesh");
}
