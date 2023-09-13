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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("closest", &torch_closest, "Closest point on a mesh");
	m.def("closest_bary", &torch_closest_bary, "Closest point on a mesh with barycentric coordinates");
}
