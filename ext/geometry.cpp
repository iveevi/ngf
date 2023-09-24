#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>

#include <torch/extension.h>

struct geometry {
	std::vector <glm::vec3> vertices;
	std::vector <glm::ivec3> triangles;
};

static geometry translate(const torch::Tensor &vertices, const torch::Tensor &triangles)
{
	// Expects:
	//   2D tensor of shape (N, 3) for vertices
	//   2D tensor of shape (M, 4) for quads
	assert(vertices.dim() == 2 && vertices.size(1) == 3);
	assert(triangles.dim() == 2 && triangles.size(1) == 3);

	// Ensure CPU tensors
	assert(vertices.device().is_cpu());
	assert(triangles.device().is_cpu());

	// Ensure float32 and uint32
	assert(vertices.dtype() == torch::kFloat32);
	assert(triangles.dtype() == torch::kInt32);

	geometry g;
	g.vertices.resize(vertices.size(0));
	g.triangles.resize(triangles.size(0));

	float *vertices_ptr = vertices.data_ptr <float> ();
	int32_t *triangles_ptr = triangles.data_ptr <int32_t> ();

	std::memcpy(g.vertices.data(), vertices_ptr, sizeof(glm::vec3) * vertices.size(0));
	std::memcpy(g.triangles.data(), triangles_ptr, sizeof(glm::ivec3) * triangles.size(0));

	return g;
}

static auto vertex_graph(const geometry &ref)
{
	std::unordered_map <uint32_t, std::unordered_set <uint32_t>> graph;

	for (const glm::ivec3 &t : ref.triangles) {
		graph[t.x].insert(t.y);
		graph[t.x].insert(t.z);

		graph[t.y].insert(t.x);
		graph[t.y].insert(t.z);

		graph[t.z].insert(t.x);
		graph[t.z].insert(t.y);
	}

	return graph;
}

static geometry laplacian_smooth(const geometry &ref, float factor)
{
	geometry out = ref;
	auto graph = vertex_graph(ref);
	for (int i = 0; i < ref.vertices.size(); i++) {
		glm::vec3 v = ref.vertices[i];
		glm::vec3 sum = glm::vec3(0.0f);
		for (uint32_t j : graph[i])
			sum += ref.vertices[j];
		float n = (float) graph[i].size();
		glm::vec3 avg = n > 0.0f ? sum / n : v;
		out.vertices[i] = v + factor * (avg - v);
	}

	return out;
}

torch::Tensor torch_laplacian_smooth(const torch::Tensor &vertices, const torch::Tensor &triangles, float factor)
{
	geometry g = translate(vertices, triangles);
	g = laplacian_smooth(g, factor);

	torch::Tensor out_vertices = torch::zeros_like(vertices);
	float *out_ptr = out_vertices.data_ptr <float> ();
	std::memcpy(out_ptr, g.vertices.data(), sizeof(glm::vec3) * g.vertices.size());

	return out_vertices;
}

struct ordered_pair {
	int32_t a, b;

	bool from(int32_t a_, uint32_t b_) {
		if (a_ > b_) {
			a = b_;
			b = a_;
			return true;
		}

		a = a_;
		b = b_;
		return false;
	}

	bool operator==(const ordered_pair &other) const {
		return a == other.a && b == other.b;
	}

	struct hash {
		size_t operator()(const ordered_pair &p) const {
			std::hash <int32_t> h;
			return h(p.a) ^ h(p.b);
		}
	};
};

auto torch_sdc_weld(const torch::Tensor &complexes,
		std::unordered_map <int32_t, std::set <int32_t>> &cmap,
		int64_t vertex_count,
		int64_t sample_rate)
{
	assert(complexes.is_cpu());
	assert(complexes.dtype() == torch::kInt32);
	assert(complexes.dim() == 2 && complexes.size(1) == 4);

	std::vector <glm::ivec4> cs(complexes.size(0));
	printf("cs: %lu\n", cs.size());
	int32_t *ptr = complexes.data_ptr <int32_t> ();
	std::memcpy(cs.data(), ptr, complexes.size(0) * sizeof(glm::ivec4));

	// Mappings
	std::unordered_map <int32_t, int32_t> rcmap;
	for (const auto &[k, v] : cmap) {
		for (const auto &i : v)
			rcmap[i] = k;
	}

	std::unordered_map <int32_t, int32_t> remap;
	for (size_t i = 0; i < vertex_count; i++)
		remap[i] = i;

	for (const auto &[_, s] : cmap) {
		int32_t new_vertex = *s.begin();
		for (const auto &v : s)
			remap[v] = new_vertex;
	}

	std::unordered_map <ordered_pair, std::set <std::pair <int32_t, std::vector <int32_t>>>, ordered_pair::hash> bmap;

	for (int32_t i = 0; i < cs.size(); i++) {
		int32_t i00 = i * sample_rate * sample_rate;
		int32_t i10 = i00 + (sample_rate - 1);
		int32_t i01 = i00 + (sample_rate - 1) * sample_rate;
		int32_t i11 = i00 + (sample_rate * sample_rate - 1);

		int32_t c00 = rcmap[i00];
		int32_t c10 = rcmap[i10];
		int32_t c01 = rcmap[i01];
		int32_t c11 = rcmap[i11];

		ordered_pair p;
		bool reversed;

		std::vector <int32_t> b00_10;
		std::vector <int32_t> b00_01;
		std::vector <int32_t> b10_11;
		std::vector <int32_t> b01_11;

		// 00 -> 10
		reversed = p.from(c00, c10);
		if (reversed) {
			for (int32_t i = sample_rate - 2; i >= 1; i--)
				b00_10.push_back(i + i00);
		} else {
			for (int32_t i = 1; i <= sample_rate - 2; i++)
				b00_10.push_back(i + i00);
		}

		bmap[p].insert({ i, b00_10 });

		// 00 -> 01
		reversed = p.from(c00, c01);
		if (reversed) {
			for (int32_t i = sample_rate * (sample_rate - 2); i >= sample_rate; i -= sample_rate)
				b00_01.push_back(i + i00);
		} else {
			for (int32_t i = sample_rate; i <= sample_rate * (sample_rate - 2); i += sample_rate)
				b00_01.push_back(i + i00);
		}

		bmap[p].insert({ i, b00_01 });

		// 10 -> 11
		reversed = p.from(c10, c11);
		if (reversed) {
			for (int32_t i = sample_rate - 2; i >= 1; i--)
				b10_11.push_back(i * sample_rate + sample_rate - 1 + i00);
		} else {
			for (int32_t i = 1; i <= sample_rate - 2; i++)
				b10_11.push_back(i * sample_rate + sample_rate - 1 + i00);
		}

		bmap[p].insert({ i, b10_11 });

		// 01 -> 11
		reversed = p.from(c01, c11);
		if (reversed) {
			for (int32_t i = sample_rate - 2; i >= 1; i--)
				b01_11.push_back((sample_rate - 1) * sample_rate + i + i00);
		} else {
			for (int32_t i = 1; i <= sample_rate - 2; i++)
				b01_11.push_back((sample_rate - 1) * sample_rate + i + i00);
		}

		bmap[p].insert({ i, b01_11 });
	}

	for (const auto &[p, bs] : bmap) {
		const auto &ref = *bs.begin();
		for (const auto &b : bs) {
			for (int32_t i = 0; i < b.second.size(); i++) {
				remap[b.second[i]] = ref.second[i];
			}
		}
	}

	std::vector <glm::ivec3> faces;
	for (int32_t i = 0; i < cs.size(); i++) {
		int32_t voffset = i * sample_rate * sample_rate;
		for (int32_t x = 0; x < sample_rate - 1; x++) {
			for (int32_t y = 0; y < sample_rate - 1; y++) {
				int32_t i00 = x * sample_rate + y;
				int32_t i10 = i00 + 1;
				int32_t i01 = (x + 1) * sample_rate + y;
				int32_t i11 = i01 + 1;

				assert(remap.find(i00 + voffset) != remap.end());
				assert(remap.find(i10 + voffset) != remap.end());
				assert(remap.find(i01 + voffset) != remap.end());
				assert(remap.find(i11 + voffset) != remap.end());

				i00 = remap[i00 + voffset];
				i10 = remap[i10 + voffset];
				i01 = remap[i01 + voffset];
				i11 = remap[i11 + voffset];

				faces.push_back(glm::ivec3(i00, i11, i10));
				faces.push_back(glm::ivec3(i00, i01, i11));
				// faces.push_back(glm::ivec4(i00, i10, i11, i01));
			}
		}
	}

	// TODO: fix and align the normals (different function)

	torch::Tensor indices = torch::zeros({ (long) faces.size(), 3 }, torch::kInt32);
	ptr = indices.data_ptr <int32_t> ();
	std::memcpy(ptr, faces.data(), faces.size() * sizeof(glm::ivec3));

	return std::make_tuple(indices, remap);
}

auto torch_sdc_separate(const torch::Tensor &vertices, const std::unordered_map <int32_t, int32_t> &remap)
{
	assert(vertices.dtype() == torch::kFloat32);
	assert(vertices.dim() == 2 && vertices.size(1) == 3);
	assert(vertices.is_cpu());

	torch::Tensor expanded = torch::zeros_like(vertices);

	int32_t vsize = vertices.size(0);
	glm::vec3 *ptr = (glm::vec3 *) expanded.data_ptr <float> ();
	glm::vec3 *src = (glm::vec3 *) vertices.data_ptr <float> ();

	for (int32_t i = 0; i < vsize; i++) {
		auto it = remap.find(i);
		assert(it != remap.end());
		int32_t j = it->second;
		assert(j < vsize);
		ptr[i] = src[j];
	}

	return expanded;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("laplacian_smooth", &torch_laplacian_smooth, "Laplacian smoothing");
	m.def("sdc_weld", &torch_sdc_weld, "SDC welding");
	m.def("sdc_separate", &torch_sdc_separate, "SDC separating");
}
