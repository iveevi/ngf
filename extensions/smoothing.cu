#include "common.hpp"

// TODO: can make faster by avoided strided access, e.g. index per vertex per adjancency...
__global__
void kernel_smooth
(
	const float3 *__restrict__ vertices,
	const int32_t *__restrict__ graph,
	float3 *__restrict__ result,
	uint32_t count,
	uint32_t bound,
	float factor
)
{
	int32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	int32_t stride = blockDim.x * gridDim.x;
	for (int32_t i = tid; i < count; i += stride) {
		float3 vertex = vertices[i];
		float x = 0;
		float y = 0;
		float z = 0;

		const int32_t *base = graph + i * bound;
		int32_t k = 0;
		while (base[k] >= 0) {
			float3 v = vertices[base[k++]];
			x += v.x;
			y += v.y;
			z += v.z;
		}

		if (k > 0)
			result[i] = make_float3(x/k, y/k, z/k);
		else
			result[i] = vertex;

//		x = (1 - factor) * vertex.x + factor * x/k;
//		y = (1 - factor) * vertex.y + factor * y/k;
//		z = (1 - factor) * vertex.z + factor * z/k;
//
//		result[i] = make_float3(x, y, z);
	}
}

Graph::Graph(const torch::Tensor &primitives, size_t vertices) : count(vertices)
{
	assert(primitives.dim() == 2);
	assert(primitives.dtype() == torch::kInt32);
	assert(primitives.device().is_cpu());

	if (primitives.size(1) == 3)
		initialize_from_triangles(primitives);
	else if (primitives.size(1) == 4)
		initialize_from_quadrilaterals(primitives);
	else
		assert(false);
}

Graph::~Graph()
{
	if (device)
		cudaFree(device);

	device = nullptr;
}

void Graph::allocate_device_graph()
{
	bound = 0;
	for (auto &kv : graph)
		bound = std::max(bound, (int32_t) kv.second.size());

	// Allocate a device graph
	int32_t graph_size = count * bound;
	cudaMalloc(&device, graph_size * sizeof(int32_t));

	std::vector <int32_t> host_graph(graph_size, -1);
	for (size_t i = 0; i < count; i++) {
		int32_t *base = host_graph.data() + i * bound;

		size_t j = 0;
		for (int32_t v : graph[i])
			base[j++] = v;
	}

	cudaMemcpy(device, host_graph.data(), graph_size * sizeof(int32_t), cudaMemcpyHostToDevice);
}

void Graph::initialize_from_triangles(const torch::Tensor &triangles)
{
	assert(triangles.dim() == 2 && triangles.size(1) == 3);
	assert(triangles.dtype() == torch::kInt32);
	assert(triangles.device().is_cpu());

	int32_t triangle_count = triangles.size(0);

	for (uint32_t i = 0; i < triangle_count; i++) {
		int32_t v0 = triangles[i][0].item().to <int32_t> ();
		int32_t v1 = triangles[i][1].item().to <int32_t> ();
		int32_t v2 = triangles[i][2].item().to <int32_t> ();

		graph[v0].insert(v1);
		graph[v0].insert(v2);

		graph[v1].insert(v0);
		graph[v1].insert(v2);

		graph[v2].insert(v0);
		graph[v2].insert(v1);
	}

	allocate_device_graph();
}

void Graph::initialize_from_quadrilaterals(const torch::Tensor &quads)
{
	assert(quads.dim() == 2 && quads.size(1) == 4);
	assert(quads.dtype() == torch::kInt32);
	assert(quads.device().is_cpu());

	int32_t quad_count = quads.size(0);

	for (uint32_t i = 0; i < quad_count; i++) {
		int32_t v0 = quads[i][0].item().to <int32_t> ();
		int32_t v1 = quads[i][1].item().to <int32_t> ();
		int32_t v2 = quads[i][2].item().to <int32_t> ();
		int32_t v3 = quads[i][3].item().to <int32_t> ();

		graph[v0].insert(v1);
		graph[v0].insert(v3);

		graph[v1].insert(v0);
		graph[v1].insert(v2);

		graph[v2].insert(v1);
		graph[v2].insert(v3);

		graph[v3].insert(v0);
		graph[v3].insert(v2);
	}

	allocate_device_graph();
}

torch::Tensor Graph::smooth(const torch::Tensor &vertices, float factor) const
{
	assert(vertices.dim() == 2 && vertices.size(1) == 3);
	assert(vertices.dtype() == torch::kFloat32);
	assert(vertices.device().is_cuda());
	assert(vertices.size(0) <= count);

	torch::Tensor result = torch::zeros_like(vertices);
	kernel_smooth <<< 64, 64 >>>
	(
		(float3 *) vertices.data_ptr <float> (), device,
		(float3 *) result.data_ptr <float> (),
		count, bound, factor
	);

	cudaDeviceSynchronize();
	return result;
}
