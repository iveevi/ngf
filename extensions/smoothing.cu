#include "common.hpp"

__global__
void laplacian_smooth_kernel(glm::vec3 *result, glm::vec3 *vertices, int32_t *graph, uint32_t count, uint32_t max_adj, float factor)
{
	int32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= count)
		return;

	glm::vec3 sum = glm::vec3(0.0f);
	int32_t adj_count = graph[tid * max_adj];
	int32_t *adj = graph + tid * max_adj + 1;
	for (int32_t i = 0; i < adj_count; i++)
		sum += vertices[adj[i]];
	sum /= (float) adj_count;
	if (adj_count == 0)
		sum = vertices[tid];

	result[tid] = vertices[tid] + (sum - vertices[tid]) * factor;
}

vertex_graph::vertex_graph(const torch::Tensor &primitives)
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

vertex_graph::~vertex_graph()
{
	if (dev_graph)
		cudaFree(dev_graph);
}

void vertex_graph::allocate_device_graph()
{
	max = 0;
	max_adj = 0;

	for (auto &kv : graph) {
		max_adj = std::max(max_adj, (int32_t) kv.second.size());
		max = std::max(max, kv.first);
	}

	// Allocate a device graph
	int32_t graph_size = max * (max_adj + 1);
	cudaMalloc(&dev_graph, graph_size * sizeof(int32_t));

	std::vector <uint32_t> host_graph(graph_size, 0);
	for (auto &kv : graph) {
		int32_t i = kv.first;
		int32_t j = 0;
		assert(i * max_adj + j < graph_size);
		host_graph[i * max_adj + j++] = kv.second.size();
		for (auto &adj : kv.second) {
			assert(i * max_adj + j < graph_size);
			host_graph[i * max_adj + j++] = adj;
		}
	}

	cudaMemcpy(dev_graph, host_graph.data(), graph_size * sizeof(int32_t), cudaMemcpyHostToDevice);
}

void vertex_graph::initialize_from_triangles(const torch::Tensor &triangles)
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

void vertex_graph::initialize_from_quadrilaterals(const torch::Tensor &quads)
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

torch::Tensor vertex_graph::smooth(const torch::Tensor &vertices, float factor) const
{
	assert(vertices.dim() == 2 && vertices.size(1) == 3);
	assert(vertices.dtype() == torch::kFloat32);
	assert(vertices.device().is_cpu());
	assert(max < vertices.size(0));

	torch::Tensor result = torch::zeros_like(vertices);

	glm::vec3 *v = (glm::vec3 *) vertices.data_ptr <float> ();
	glm::vec3 *r = (glm::vec3 *) result.data_ptr <float> ();

	for (uint32_t i = 0; i <= max; i++) {
		if (graph.find(i) == graph.end())
			continue;

		glm::vec3 sum = glm::vec3(0.0f);
		for (auto j : graph.at(i))
			sum += v[j];
		sum /= (float) graph.at(i).size();

		r[i] = (1.0f - factor) * v[i] + factor * sum;
	}

	return result;
}

torch::Tensor vertex_graph::smooth_device(const torch::Tensor &vertices, float factor) const
{
	assert(vertices.dim() == 2 && vertices.size(1) == 3);
	assert(vertices.dtype() == torch::kFloat32);
	assert(vertices.device().is_cuda());
	assert(max < vertices.size(0));

	torch::Tensor result = torch::zeros_like(vertices);

	glm::vec3 *v = (glm::vec3 *) vertices.data_ptr <float> ();
	glm::vec3 *r = (glm::vec3 *) result.data_ptr <float> ();

	dim3 block(256);
	dim3 grid((vertices.size(0) + block.x - 1) / block.x);

	laplacian_smooth_kernel <<<grid, block>>> (r, v, dev_graph, vertices.size(0), max_adj, factor);

	return result;
}
