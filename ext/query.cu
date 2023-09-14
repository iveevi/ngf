#include <glm/glm.hpp>

#include <torch/extension.h>

struct cumesh {
	glm::vec3 *vertices;
	glm::vec3 *normals;
	glm::uvec3 *triangles;

	size_t vertex_count;
	size_t triangle_count;
};

struct cumesh_quad {
	glm::vec3 *vertices;
	glm::uvec4 *quads;

	float *u;
	float *v;
	uint32_t *complexes;

	size_t vertex_count;
	size_t quad_count;
};

__forceinline__ __host__ __device__
void triangle_closest_point(const glm::vec3 &v0, const glm::vec3 &v1, const glm::vec3 &v2, const glm::vec3 &p, glm::vec3 *closest, glm::vec3 *bary, float *distance)
{
	glm::vec3 B = v0;
	glm::vec3 E1 = v1 - v0;
	glm::vec3 E2 = v2 - v0;
	glm::vec3 D = B - p;

	float a = glm::dot(E1, E1);
	float b = glm::dot(E1, E2);
	float c = glm::dot(E2, E2);
	float d = glm::dot(E1, D);
	float e = glm::dot(E2, D);
	float f = glm::dot(D, D);

	float det = a * c - b * b;
	float s = b * e - c * d;
	float t = b * d - a * e;

	if (s + t <= det) {
		if (s < 0.0f) {
			if (t < 0.0f) {
				if (d < 0.0f) {
					s = glm::clamp(-d / a, 0.0f, 1.0f);
					t = 0.0f;
				} else {
					s = 0.0f;
					t = glm::clamp(-e / c, 0.0f, 1.0f);
				}
			} else {
				s = 0.0f;
				t = glm::clamp(-e / c, 0.0f, 1.0f);
			}
		} else if (t < 0.0f) {
			s = glm::clamp(-d / a, 0.0f, 1.0f);
			t = 0.0f;
		} else {
			float invDet = 1.0f / det;
			s *= invDet;
			t *= invDet;
		}
	} else {
		if (s < 0.0f) {
			float tmp0 = b + d;
			float tmp1 = c + e;
			if (tmp1 > tmp0) {
				float numer = tmp1 - tmp0;
				float denom = a - 2 * b + c;
				s = glm::clamp(numer / denom, 0.0f, 1.0f);
				t = 1 - s;
			} else {
				t = glm::clamp(-e / c, 0.0f, 1.0f);
				s = 0.0f;
			}
		} else if (t < 0.0f) {
			if (a + d > b + e) {
				float numer = c + e - b - d;
				float denom = a - 2 * b + c;
				s = glm::clamp(numer / denom, 0.0f, 1.0f);
				t = 1 - s;
			} else {
				s = glm::clamp(-e / c, 0.0f, 1.0f);
				t = 0.0f;
			}
		} else {
			float numer = c + e - b - d;
			float denom = a - 2 * b + c;
			s = glm::clamp(numer / denom, 0.0f, 1.0f);
			t = 1.0f - s;
		}
	}

	*closest = B + s * E1 + t * E2;
	*bary = glm::vec3(1.0f - s - t, s, t);
	*distance = glm::length(*closest - p);
}

__global__
void closest_point(cumesh g, glm::vec3 *points, glm::vec3 *dst, glm::vec3 *normals, size_t N)
{
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;

	for (size_t i = tid; i < N; i += stride) {
		glm::vec3 p = points[i];

		glm::vec3 closest = g.vertices[0];
		// glm::vec3 normal = g.normals[0];
		glm::vec3 bary = glm::vec3(1.0f, 0.0f, 0.0f);
		float closest_dist = glm::distance(p, closest);
		size_t closest_triangle = 0;

		for (size_t j = 0; j < g.triangle_count; j++) {
			const glm::uvec3 &tri = g.triangles[j];
			glm::vec3 v0 = g.vertices[tri.x];
			glm::vec3 v1 = g.vertices[tri.y];
			glm::vec3 v2 = g.vertices[tri.z];

			glm::vec3 c;
			glm::vec3 b;
			float d;

			triangle_closest_point(v0, v1, v2, p, &c, &b, &d);
			if (d < closest_dist) {
				closest = c;
				bary = b;
				closest_dist = d;
				closest_triangle = j;
			}
		}

		dst[i] = closest;
		normals[i] = g.normals[closest_triangle];
	}
}

__global__
void closest_point_bary(cumesh g, glm::vec3 *points, int32_t *tris, glm::vec3 *bary, size_t N)
{
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;

	for (size_t i = tid; i < N; i += stride) {
		glm::vec3 p = points[i];

		glm::vec3 closest = g.vertices[0];
		glm::vec3 closest_bary = glm::vec3(1.0f, 0.0f, 0.0f);
		float closest_dist = glm::distance(p, closest);
		int32_t closest_triangle = 0;

		for (int32_t j = 0; j < g.triangle_count; j++) {
			const glm::uvec3 &tri = g.triangles[j];
			glm::vec3 v0 = g.vertices[tri.x];
			glm::vec3 v1 = g.vertices[tri.y];
			glm::vec3 v2 = g.vertices[tri.z];

			glm::vec3 c;
			glm::vec3 b;
			float d;

			triangle_closest_point(v0, v1, v2, p, &c, &b, &d);
			if (d < closest_dist) {
				closest = c;
				closest_bary = b;
				closest_dist = d;
				closest_triangle = j;
			}
		}

		tris[i] = closest_triangle;
		bary[i] = closest_bary;
	}
}

static cumesh translate(const torch::Tensor &vertices, const torch::Tensor &normals, const torch::Tensor &triangles)
{
	// Expects:
	//   2D tensor of shape (N, 3) for vertices
	//   2D tensor of shape (M, 3) for triangles
	assert(vertices.dim() == 2 && vertices.size(1) == 3);
	assert(normals.dim() == 2 && normals.size(1) == 3);
	assert(triangles.dim() == 2 && triangles.size(1) == 3);

	assert(normals.size(0) == triangles.size(0));

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

static cumesh translate(const torch::Tensor &vertices, const torch::Tensor &triangles)
{
	// Expects:
	//   2D tensor of shape (N, 3) for vertices
	//   2D tensor of shape (M, 3) for triangles
	assert(vertices.dim() == 2 && vertices.size(1) == 3);
	assert(triangles.dim() == 2 && triangles.size(1) == 3);

	// Ensure CUDA tensors
	assert(vertices.device().is_cuda());
	assert(triangles.device().is_cuda());

	// Ensure float32 and uint32
	assert(vertices.dtype() == torch::kFloat32);
	assert(triangles.dtype() == torch::kInt32);

	cumesh g;
	g.vertices = (glm::vec3 *) vertices.data_ptr();
	g.triangles = (glm::uvec3 *) triangles.data_ptr();
	g.vertex_count = vertices.size(0);
	g.triangle_count = triangles.size(0);
	return g;
}

auto torch_closest(const torch::Tensor &X, const torch::Tensor &vertices, const torch::Tensor &normals, const torch::Tensor &triangles)
{
	// Ensure X is a 2D tensor of shape (N, 3) in CUDA
	assert(X.dim() == 2 && X.size(1) == 3);
	assert(X.device().is_cuda());
	assert(X.dtype() == torch::kFloat32);

	torch::Tensor Y = torch::zeros_like(X);
	torch::Tensor N = torch::zeros_like(X);

	assert(Y.device().is_cuda());
	assert(N.device().is_cuda());

	cumesh g = translate(vertices, normals, triangles);

	glm::vec3 *points = (glm::vec3 *) X.data_ptr();

	glm::vec3 *dst = (glm::vec3 *) Y.data_ptr();
	glm::vec3 *nrm = (glm::vec3 *) N.data_ptr();

	size_t size = X.size(0);

	dim3 block(256);
	dim3 grid((size + block.x - 1) / block.x);
	closest_point <<< grid, block >>> (g, points, dst, nrm, size);
	cudaDeviceSynchronize();

	return std::make_tuple(Y, N);
}

auto torch_closest_bary(const torch::Tensor &X, const torch::Tensor &vertices, const torch::Tensor &triangles)
{
	// Ensure X is a 2D tensor of shape (N, 3) in CUDA
	assert(X.dim() == 2 && X.size(1) == 3);
	assert(X.device().is_cuda());
	assert(X.dtype() == torch::kFloat32);

	torch::Tensor T = torch::zeros({ X.size(0) }, torch::dtype(torch::kInt32)).cuda();
	torch::Tensor B = torch::zeros_like(X);

	assert(T.device().is_cuda());
	assert(B.device().is_cuda());

	cumesh g = translate(vertices, triangles);

	glm::vec3 *points = (glm::vec3 *) X.data_ptr();

	int32_t *tris = (int32_t *) T.data_ptr();
	glm::vec3 *bary = (glm::vec3 *) B.data_ptr();

	size_t size = X.size(0);

	dim3 block(256);
	dim3 grid((size + block.x - 1) / block.x);
	closest_point_bary <<< grid, block >>> (g, points, tris, bary, size);
	cudaDeviceSynchronize();

	return std::make_tuple(T, B);
}

struct query_accelerator {
	cumesh g;

	// Mapping from each complex to a continguous array of triangle indices
	int32_t **blocks;
	int32_t *block_sizes;
	int32_t complex_count;

	auto closest(const torch::Tensor &, int32_t);
};

__global__
void closest_point_qacc(query_accelerator qacc, glm::vec3 *points, glm::vec3 *dst, glm::vec3 *normals, size_t N, size_t sample_rate)
{
	// NOTE: assumes that 1 complex is every sample_rate * sample_rate points
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;

	for (size_t i = tid; i < N; i += stride) {
		glm::vec3 p = points[i];

		int32_t complex = i / (sample_rate * sample_rate);

		int32_t *block = qacc.blocks[complex];
		int32_t block_size = qacc.block_sizes[complex];

		glm::vec3 closest = qacc.g.vertices[0];
		glm::vec3 bary = glm::vec3(1.0f, 0.0f, 0.0f);
		float closest_dist = glm::distance(p, closest);
		size_t closest_triangle = 0;

		for (size_t j = 0; j < block_size; j++) {
			const glm::uvec3 &tri = qacc.g.triangles[block[j]];
			glm::vec3 v0 = qacc.g.vertices[tri.x];
			glm::vec3 v1 = qacc.g.vertices[tri.y];
			glm::vec3 v2 = qacc.g.vertices[tri.z];

			glm::vec3 c;
			glm::vec3 b;
			float d;

			triangle_closest_point(v0, v1, v2, p, &c, &b, &d);
			if (d < closest_dist) {
				closest = c;
				bary = b;
				closest_dist = d;
				closest_triangle = j;
			}
		}

		dst[i] = closest;
		normals[i] = qacc.g.normals[closest_triangle];
	}
}

auto query_accelerator::closest(const torch::Tensor &X, int32_t sample_rate)
{
	// Ensure X is a 2D tensor of shape (N, 3) in CUDA
	assert(X.dim() == 2 && X.size(1) == 3);
	assert(X.device().is_cuda());
	assert(X.dtype() == torch::kFloat32);

	torch::Tensor Y = torch::zeros_like(X);
	torch::Tensor N = torch::zeros_like(X);

	assert(Y.device().is_cuda());
	assert(N.device().is_cuda());

	glm::vec3 *points = (glm::vec3 *) X.data_ptr();

	glm::vec3 *dst = (glm::vec3 *) Y.data_ptr();
	glm::vec3 *nrm = (glm::vec3 *) N.data_ptr();

	size_t size = X.size(0);
	// printf("size = %zu, sample_rate = %d\n", size, sample_rate);

	dim3 block(256);
	dim3 grid((size + block.x - 1) / block.x);
	closest_point_qacc <<< grid, block >>> (*this, points, dst, nrm, size, sample_rate);
	cudaDeviceSynchronize();

	return std::make_tuple(Y, N);
}

query_accelerator torch_accelerator(const torch::Tensor &X, const torch::Tensor &V, const torch::Tensor &N, const torch::Tensor &T, int32_t sample_rate, float scale)
{
	// (V, T) is the reference mesh that we are partitioning

	// For construction, we need CPU tensors
	assert(X.is_cuda() && V.is_cuda() && N.is_cuda() && T.is_cuda());

	auto X_cpu = X.cpu();
	auto V_cpu = V.cpu();
	auto T_cpu = T.cpu();

	// X and V must be float32 (?, 3)
	assert(X.dim() == 2 && X.size(1) == 3);
	assert(V.dim() == 2 && V.size(1) == 3);
	assert(N.dim() == 2 && N.size(1) == 3);

	assert(X.dtype() == torch::kFloat32);
	assert(V.dtype() == torch::kFloat32);
	assert(N.dtype() == torch::kFloat32);

	// T must be int32 (?, 3)
	assert(T.dim() == 2 && T.size(1) == 3);
	assert(T.dtype() == torch::kInt32);

	printf("Sample rate: %d\n", sample_rate);
	int32_t complex_count = X.size(0)/(sample_rate * sample_rate);
	printf("Complex count: %d\n", complex_count);

	// Find bounds for each complex
	std::vector <glm::vec3> complex_min(complex_count, glm::vec3(FLT_MAX));
	std::vector <glm::vec3> complex_max(complex_count, glm::vec3(-FLT_MAX));

	glm::vec3 *X_ptr = (glm::vec3 *) X_cpu.data_ptr();
	for (uint32_t i = 0; i < X_cpu.size(0); i++) {
		int32_t complex = i/(sample_rate * sample_rate);
		complex_min[complex] = glm::min(complex_min[complex], X_ptr[i]);
		complex_max[complex] = glm::max(complex_max[complex], X_ptr[i]);
	}

	// Use the maximal bound box size
	glm::vec3 max_half_size(0.0f);
	for (uint32_t i = 0; i < complex_count; i++)
		max_half_size = glm::max(max_half_size, (complex_max[i] - complex_min[i])/2.0f);

	// printf("Bounds for each complex:\n");
	// for (uint32_t i = 0; i < complex_count; i++)
	// 	printf("  %d: (%f, %f, %f) - (%f, %f, %f)\n", i, complex_min[i].x, complex_min[i].y, complex_min[i].z, complex_max[i].x, complex_max[i].y, complex_max[i].z);

	// Collect all triangles whose bounding box overlaps with an expanded version of the bounding box of each complex
	std::vector <std::vector <int32_t>> complex_triangles(complex_count);

	glm::vec3 *V_ptr = (glm::vec3 *) V_cpu.data_ptr();
	glm::ivec3 *T_ptr = (glm::ivec3 *) T_cpu.data_ptr();

	for (uint32_t i = 0; i < T_cpu.size(0); i++) {
		const glm::ivec3 &t = T_ptr[i];
		const glm::vec3 v0 = V_ptr[t.x];
		const glm::vec3 v1 = V_ptr[t.y];
		const glm::vec3 v2 = V_ptr[t.z];

		glm::vec3 min = glm::min(glm::min(v0, v1), v2);
		glm::vec3 max = glm::max(glm::max(v0, v1), v2);

		for (uint32_t j = 0; j < complex_count; j++) {
			glm::vec3 center = (complex_min[j] + complex_max[j]) * 0.5f;
			glm::vec3 expanded_min = center - max_half_size * scale;
			glm::vec3 expanded_max = center + max_half_size * scale;

			if (min.x <= expanded_max.x && max.x >= expanded_min.x &&
				min.y <= expanded_max.y && max.y >= expanded_min.y &&
				min.z <= expanded_max.z && max.z >= expanded_min.z) {
				complex_triangles[j].push_back(i);
			}
		}
	}

	// printf("Load for each complex:\n");
	float total = 0.0f;
	int32_t max_load = 0.0f;
	int32_t min_load = INT_MAX;

	for (uint32_t i = 0; i < complex_count; i++) {
		int32_t load = complex_triangles[i].size();
		// printf("  %d: %d\n", i, load);
		max_load = glm::max(max_load, load);
		min_load = glm::min(min_load, load);
		total += load;
	}

	assert(min_load > 0);

	float avg = total / complex_count;
	printf("Average load: %f, min/max load: %d/%d\n", avg, min_load, max_load);
	printf("Theoretical speedup (average) %3.2f\n", T.size(0)/avg);
	printf("Theoretical speedup (worst)   %3.2f\n", T.size(0)/float(max_load));

	// TODO: there should be no empty complexes

	// Transfer to GPU
	int32_t **blocks = new int32_t *[complex_count];
	int32_t *block_sizes = new int32_t[complex_count];
	for (uint32_t i = 0; i < complex_count; i++) {
		const std::vector <int32_t> &triangles = complex_triangles[i];

		int32_t *triangles_gpu;
		cudaMalloc(&triangles_gpu, triangles.size() * sizeof(int32_t));
		cudaMemcpy(triangles_gpu, triangles.data(), triangles.size() * sizeof(int32_t), cudaMemcpyHostToDevice);

		blocks[i] = triangles_gpu;
		block_sizes[i] = triangles.size();
	}

	// Create the accelerator
	query_accelerator accelerator;
	accelerator.complex_count = complex_count;

	cudaMalloc(&accelerator.blocks, complex_count * sizeof(int32_t *));
	cudaMemcpy(accelerator.blocks, blocks, complex_count * sizeof(int32_t *), cudaMemcpyHostToDevice);

	cudaMalloc(&accelerator.block_sizes, complex_count * sizeof(int32_t));
	cudaMemcpy(accelerator.block_sizes, block_sizes, complex_count * sizeof(int32_t), cudaMemcpyHostToDevice);

	cumesh g = translate(V, N, T);
	accelerator.g = g;

	delete[] blocks;
	delete[] block_sizes;

	return accelerator;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("closest", &torch_closest, "Closest point on a mesh");
	m.def("closest_bary", &torch_closest_bary, "Closest point on a mesh with barycentric coordinates");
	m.def("accelerator", &torch_accelerator, "Construct an accelerator for a mesh");

	py::class_ <query_accelerator> (m, "query_accelerator")
		.def(py::init <> ())
		.def("closest", &query_accelerator::closest, "Closest point on a mesh", py::arg("X"), py::arg("sample_rate"));
}
