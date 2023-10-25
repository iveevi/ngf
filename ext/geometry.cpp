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

struct remapper : std::unordered_map <int32_t, int32_t> {
	using std::unordered_map <int32_t, int32_t> ::unordered_map;

	torch::Tensor remap(const torch::Tensor &indices) const {
		assert(indices.dtype() == torch::kInt32);
		assert(indices.dim() == 2);
		assert(indices.is_cpu());

		torch::Tensor out = torch::zeros_like(indices);
		int32_t *out_ptr = out.data_ptr <int32_t> ();
		int32_t *indices_ptr = indices.data_ptr <int32_t> ();

		for (int32_t i = 0; i < indices.numel(); i++) {
			auto it = this->find(indices_ptr[i]);
			assert(it != this->end());
			out_ptr[i] = it->second;
		}

		return out;
	}

	torch::Tensor scatter(const torch::Tensor &vertices) const {
		assert(vertices.dtype() == torch::kFloat32);
		assert(vertices.dim() == 2 && vertices.size(1) == 3);
		assert(vertices.is_cpu());

		torch::Tensor out = torch::zeros_like(vertices);
		glm::vec3 *out_ptr = (glm::vec3 *) out.data_ptr <float> ();
		glm::vec3 *vertices_ptr = (glm::vec3 *) vertices.data_ptr <float> ();

		for (int32_t i = 0; i < vertices.size(0); i++) {
			auto it = this->find(i);
			assert(it != this->end());
			out_ptr[i] = vertices_ptr[it->second];
		}

		return out;
	}
};

auto generate_remapper(const torch::Tensor &complexes,
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

	// std::unordered_map <int32_t, int32_t> remap;
	remapper remap;
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

	return remap;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("generate_remapper", &generate_remapper, "Generate remapper");

	py::class_ <remapper> (m, "Remapper")
		.def(py::init <> ())
		.def("remap", &remapper::remap, "Remap indices")
		.def("scatter", &remapper::scatter, "Scatter vertices");
}
