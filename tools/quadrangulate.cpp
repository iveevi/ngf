#include <filesystem>
#include <random>
#include <unordered_map>
#include <unordered_set>

#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>

#include "geometry.hpp"
#include "microlog.h"

struct ordered_pair {
	uint32_t a;
	uint32_t b;

	ordered_pair(uint32_t a_, uint32_t b_) : a(a_), b(b_) {
		if (a > b)
			std::swap(a, b);
	}

	bool operator==(const ordered_pair &other) const {
		return a == other.a && b == other.b;
	}

	static size_t hash(const ordered_pair &p) {
		return std::hash <uint32_t> ()(p.a) ^ std::hash <uint32_t> ()(p.b);
	}
};

template <typename T>
using ordered_pair_map = std::unordered_map <ordered_pair, T, decltype(&ordered_pair::hash)>;

template <size_t primitive>
geometry <primitive> compact(const geometry <primitive> &ref)
{
	std::unordered_map <glm::vec3, uint32_t> existing;

	geometry <primitive> fixed;
	auto add_uniquely = [&](int32_t i) -> size_t {
		glm::vec3 v = ref.vertices[i];
		if (existing.find(v) == existing.end()) {
			int32_t csize = fixed.vertices.size();
			fixed.vertices.push_back(v);

			existing[v] = csize;
			return csize;
		}

		return existing[v];
	};

	const auto &primitives = [&]() -> const auto & {
		if constexpr (primitive == eTriangle)
			return ref.triangles;
		else if constexpr (primitive == eQuad)
			return ref.quads;
	} ();

	using primitive_type = typename std::conditional <primitive == eTriangle, glm::uvec3, glm::uvec4> ::type;
	for (const primitive_type &p : primitives) {
		if constexpr (primitive == eTriangle) {
			fixed.triangles.push_back(glm::uvec3 {
				add_uniquely(p[0]),
				add_uniquely(p[1]),
				add_uniquely(p[2])
			});
		} else if constexpr (primitive == eQuad) {
			fixed.quads.push_back(glm::uvec4 {
				add_uniquely(p[0]),
				add_uniquely(p[1]),
				add_uniquely(p[2]),
				add_uniquely(p[3])
			});
		}
	}

	return fixed;
}

template <size_t primitive>
geometry <primitive> sanitize(const geometry <primitive> &ref)
{
	// Remove degenerate primitives
	geometry <primitive> fixed;
	fixed.vertices = ref.vertices;

	if constexpr (primitive == eTriangle) {
		uint32_t count = 0;
		for (const glm::uvec3 &t : ref.triangles) {
			bool ok = (t[0] != t[1]) && (t[0] != t[2]) && (t[1] != t[2]);
			if (ok)
				fixed.triangles.push_back(t);
			else
				count++;
		}

		if (count)
			ulog_warning("sanitize", "found %u degenerate triangles\n", count);
		else
			ulog_info("sanitize", "no degenerate triangles found\n");
		// TODO: ulog_ok
	} else if constexpr (primitive == eQuad) {
		uint32_t count = 0;
		for (const glm::uvec4 &q : ref.quads) {
			bool ok = (q[0] != q[1]) && (q[0] != q[2]) && (q[0] != q[3])
				&& (q[1] != q[2]) && (q[1] != q[3])
				&& (q[2] != q[3]);

			if (ok)
				fixed.quads.push_back(q);
			else
				count++;
		}

		if (count)
			ulog_warning("sanitize", "found %u degenerate quads\n", count);
		else
			ulog_info("sanitize", "no degenerate quads found\n");
	}

	return fixed;
}

struct vertex_graph : std::unordered_map <uint32_t, std::vector <uint32_t>> {
	using std::unordered_map <uint32_t, std::vector <uint32_t>> ::unordered_map;

	template <size_t primitive>
	vertex_graph(const geometry <primitive> &ref) {
		const auto &primitives = [&]() -> const auto & {
			if constexpr (primitive == eTriangle)
				return ref.triangles;
			else if constexpr (primitive == eQuad)
				return ref.quads;
		} ();

		using primitive_type = typename std::conditional <primitive == eTriangle, glm::uvec3, glm::uvec4> ::type;
		for (const primitive_type &p : primitives) {
			if constexpr (primitive == eTriangle) {
				(*this)[p[0]].push_back(p[1]);
				(*this)[p[0]].push_back(p[2]);

				(*this)[p[1]].push_back(p[0]);
				(*this)[p[1]].push_back(p[2]);

				(*this)[p[2]].push_back(p[0]);
				(*this)[p[2]].push_back(p[1]);
			} else if constexpr (primitive == eQuad) {
				(*this)[p[0]].push_back(p[1]);
				(*this)[p[0]].push_back(p[3]);

				(*this)[p[1]].push_back(p[0]);
				(*this)[p[1]].push_back(p[2]);

				(*this)[p[2]].push_back(p[1]);
				(*this)[p[2]].push_back(p[3]);

				(*this)[p[3]].push_back(p[0]);
				(*this)[p[3]].push_back(p[2]);
			}
		}
	}
};

struct edge_graph : ordered_pair_map <std::vector <uint32_t>> {
	using base = ordered_pair_map <std::vector <uint32_t>>;
	using ordered_pair_map <std::vector <uint32_t>> ::ordered_pair_map;

	template <size_t primitive>
	edge_graph(const geometry <primitive> &ref) : base(0, ordered_pair::hash) {
		const auto &primitives = [&]() -> const auto & {
			if constexpr (primitive == eTriangle)
				return ref.triangles;
			else if constexpr (primitive == eQuad)
				return ref.quads;
		} ();

		using primitive_type = typename std::conditional <primitive == eTriangle, glm::uvec3, glm::uvec4> ::type;
		for (uint32_t i = 0; i < primitives.size(); i++) {
			const primitive_type &p = primitives[i];
			if constexpr (primitive == eTriangle) {
				ordered_pair op0(p[0], p[1]);
				ordered_pair op1(p[1], p[2]);
				ordered_pair op2(p[2], p[0]);

				(*this)[op0].push_back(i);
				(*this)[op1].push_back(i);
				(*this)[op2].push_back(i);
			} else if constexpr (primitive == eQuad) {
				ordered_pair op0(p[0], p[1]);
				ordered_pair op1(p[1], p[2]);
				ordered_pair op2(p[2], p[3]);
				ordered_pair op3(p[3], p[0]);

				(*this)[op0].push_back(i);
				(*this)[op1].push_back(i);
				(*this)[op2].push_back(i);
				(*this)[op3].push_back(i);
			}
		}
	}

	// Display distribution of edge adjacency
	void check() const {
		std::unordered_map <uint32_t, uint32_t> distribution;
		for (const auto &pair : *this)
			distribution[pair.second.size()]++;

		ulog_info("edge_graph", "distribution of edge adjacency:\n");
		for (const auto &pair : distribution)
			ulog_info("edge_graph", "%7u edges are adjacent to %u faces\n", pair.second, pair.first);
	}

	void check(uint32_t N) {
		std::unordered_map <uint32_t, uint32_t> distribution;
		for (const auto &pair : *this)
			distribution[pair.second.size()]++;

		printf("elements with %u adjacencies: %u\n", N, distribution[N]);
		for (const auto &pair : *this) {
			if (pair.second.size() == N) {
				printf("  %u-%u: ", pair.first.a, pair.first.b);
				for (uint32_t i : pair.second)
					printf("%u ", i);
				printf("\n");
			}
		}
	}
};

struct dual_graph : std::unordered_map <uint32_t, std::unordered_set <uint32_t>> {
	// TODO: record manifoldness per face?
	using std::unordered_map <uint32_t, std::unordered_set <uint32_t>> ::unordered_map;

	dual_graph(const edge_graph &egraph) {
		for (const auto &pair : egraph) {
			for (uint32_t i = 0; i < pair.second.size(); i++) {
				for (uint32_t j = i + 1; j < pair.second.size(); j++) {
					(*this)[pair.second[i]].insert(pair.second[j]);
					(*this)[pair.second[j]].insert(pair.second[i]);
				}
			}
		}
	}
};

template <size_t primitive>
geometry <primitive> laplacian_smoothing(const geometry <primitive> &ref, float factor)
{
	std::unordered_map <uint32_t, std::unordered_set <uint32_t>> adjacency(0);

	if constexpr (primitive == eTriangle) {
		for (const auto &triangle : ref.triangles) {
			adjacency[triangle[0]].insert(triangle[1]);
			adjacency[triangle[0]].insert(triangle[2]);
			adjacency[triangle[1]].insert(triangle[0]);
			adjacency[triangle[1]].insert(triangle[2]);
			adjacency[triangle[2]].insert(triangle[0]);
			adjacency[triangle[2]].insert(triangle[1]);
		}
	} else {
		for (const auto &quad : ref.quads) {
			adjacency[quad[0]].insert(quad[1]);
			adjacency[quad[0]].insert(quad[3]);
			adjacency[quad[1]].insert(quad[0]);
			adjacency[quad[1]].insert(quad[2]);
			adjacency[quad[2]].insert(quad[1]);
			adjacency[quad[2]].insert(quad[3]);
			adjacency[quad[3]].insert(quad[0]);
			adjacency[quad[3]].insert(quad[2]);
		}
	}

	geometry <primitive> result;
	result.vertices.resize(ref.vertices.size());
	result.triangles = ref.triangles;
	result.quads = ref.quads;

	for (uint32_t i = 0; i < ref.vertices.size(); i++) {
		const glm::vec3 &vertex = ref.vertices[i];
		const std::unordered_set <uint32_t> &adjacent = adjacency[i];

		glm::vec3 sum(0.0f);
		for (uint32_t j : adjacent)
			sum += ref.vertices[j];

		result.vertices[i] = vertex + (sum/(float) adjacent.size() - vertex) * factor;
	}

	return result;
}

// Compute primitive areas
template <size_t primitive>
std::vector <double> primitive_areas(const geometry <primitive> &ref)
{
	std::vector <double> areas;
	if constexpr (primitive == eTriangle)
		areas.resize(ref.triangles.size());
	else
		areas.resize(ref.quads.size());

	// TODO: openmp?
	for (uint32_t i = 0; i < areas.size(); i++) {
		if constexpr (primitive == eTriangle) {
			const glm::uvec3 &triangle = ref.triangles[i];
			const glm::dvec3 &a = ref.vertices[triangle[0]];
			const glm::dvec3 &b = ref.vertices[triangle[1]];
			const glm::dvec3 &c = ref.vertices[triangle[2]];
			glm::dvec3 cross = glm::cross(b - a, c - a);
			areas[i] = glm::length(cross) / 2.0;
		} else if constexpr (primitive == eQuad) {
			const glm::uvec4 &quad = ref.quads[i];
			const glm::dvec3 &a = ref.vertices[quad[0]];
			const glm::dvec3 &b = ref.vertices[quad[1]];
			const glm::dvec3 &c = ref.vertices[quad[2]];
			const glm::dvec3 &d = ref.vertices[quad[3]];
			glm::dvec3 cross1 = glm::cross(b - a, c - a);
			glm::dvec3 cross2 = glm::cross(c - a, d - a);
			areas[i] = (glm::length(cross1) + glm::length(cross2)) / 2.0;
		}
	}

	return areas;
}

geometry <eQuad> quadrangulate(const geometry <eTriangle> &ref)
{
	ordered_pair_map <glm::vec3> midpoints(0, ordered_pair::hash);

	std::unordered_map <uint32_t, glm::vec3> centroids(0);

	for (uint32_t t = 0; t < ref.triangles.size(); t++) {
		const glm::uvec3 &triangle = ref.triangles[t];

		for (uint32_t i = 0; i < 3; i++) {
			uint32_t a = triangle[i];
			uint32_t b = triangle[(i + 1) % 3];
			ordered_pair edge(a, b);
			if (midpoints.find(edge) == midpoints.end())
				midpoints[edge] = (ref.vertices[a] + ref.vertices[b]) / 2.0f;
		}

		glm::vec3 centroid = (ref.vertices[triangle[0]] + ref.vertices[triangle[1]] + ref.vertices[triangle[2]]) / 3.0f;
		centroids[t] = centroid;
	}

	// Create new vector of vertices
	ordered_pair_map <uint32_t> midpoint_indices(0, ordered_pair::hash);

	std::unordered_map <uint32_t, uint32_t> centroid_indices(0);

	std::vector <glm::vec3> vertices;
	for (const glm::vec3 &vertex : ref.vertices)
		vertices.push_back(vertex);

	for (const auto &pair : midpoints) {
		midpoint_indices[pair.first] = vertices.size();
		vertices.push_back(pair.second);
	}

	for (const auto &pair : centroids) {
		centroid_indices[pair.first] = vertices.size();
		vertices.push_back(pair.second);
	}

	// Create new vector of quads
	std::vector <glm::uvec4> quads;

	for (uint32_t t = 0; t < ref.triangles.size(); t++) {
		const glm::uvec3 &triangle = ref.triangles[t];

		ordered_pair edge0(triangle[0], triangle[1]);
		ordered_pair edge1(triangle[1], triangle[2]);
		ordered_pair edge2(triangle[2], triangle[0]);

		uint32_t m01 = midpoint_indices[edge0];
		uint32_t m12 = midpoint_indices[edge1];
		uint32_t m20 = midpoint_indices[edge2];
		uint32_t c = centroid_indices[t];

		quads.push_back(glm::uvec4(triangle[0], m01, c, m20));
		quads.push_back(glm::uvec4(triangle[1], m12, c, m01));
		quads.push_back(glm::uvec4(triangle[2], m20, c, m12));
	}

	geometry <eQuad> result;
	result.vertices = std::move(vertices);
	result.quads = std::move(quads);
	return result;
}

// Remove doublets in quadrilateral meshes
geometry <eQuad> cleanse_doublets(const geometry <eQuad> &ref)
{
	ordered_pair_map <std::vector <uint32_t>> edge_quads({}, ordered_pair::hash);

	for (uint32_t q = 0; q < ref.quads.size(); q++) {
		const glm::uvec4 &quad = ref.quads[q];

		ordered_pair edge0(quad[0], quad[1]);
		ordered_pair edge1(quad[1], quad[2]);
		ordered_pair edge2(quad[2], quad[3]);
		ordered_pair edge3(quad[3], quad[0]);

		edge_quads[edge0].push_back(q);
		edge_quads[edge1].push_back(q);
		edge_quads[edge2].push_back(q);
		edge_quads[edge3].push_back(q);
	}

	ordered_pair_map <std::vector <ordered_pair>> shared_edges({}, ordered_pair::hash);

	for (const auto &[e, quads] : edge_quads) {
		// Only consider valid manifold edges
		if (quads.size() == 2) {
			auto it = quads.begin();
			uint32_t q0 = *it;
			uint32_t q1 = *std::next(it);
			ordered_pair qpair = ordered_pair(q0, q1);
			shared_edges[qpair].push_back(e);
		}
	}

	struct doublet {
		uint32_t a;
		uint32_t b;
		uint32_t c;

		bool operator==(const doublet &d) const {
			return (a == d.a && b == d.b && c == d.c) || (a == d.b && b == d.a && c == d.c);
		}

		static size_t hash(const doublet &d) {
			return std::hash <uint32_t> ()(d.a) ^ std::hash <uint32_t> ()(d.b) ^ std::hash <uint32_t> ()(d.c);
		}
	};

	// TODO: regular vector?
	std::unordered_set <doublet, decltype(&doublet::hash)> doublets(0, &doublet::hash);

	for (const auto &[qpair, edges] : shared_edges) {
		ulog_assert(edges.size() <= 2, "Invalid manifold boundary", "quads %u and %u share %d\n", qpair.a, qpair.b, edges.size());
		if (edges.size() < 2)
			continue;

		auto it = edges.begin();
		ordered_pair edge0 = *it;
		ordered_pair edge1 = *(++it);

		doublet d;
		if (edge0.a == edge1.a)
			d = { edge0.b, edge0.a, edge1.b };
		else if (edge0.a == edge1.b)
			d = { edge0.b, edge0.a, edge1.a };
		else if (edge0.b == edge1.a)
			d = { edge0.a, edge0.b, edge1.b };
		else if (edge0.b == edge1.b)
			d = { edge0.a, edge0.b, edge1.a };
		else
			ulog_error("cleanse_doublets", "Invalid shared edge pair (%u, %u) and (%u, %u)\n", edge0.a, edge0.b, edge1.a, edge1.b);

		doublets.insert(d);
	}

	ulog_info("cleanse_doublets", "found %u doublets\n", doublets.size());

	// Constructing new primitives
	std::vector <glm::uvec4> quads = ref.quads;
	std::vector <bool> valid(ref.quads.size(), true);

	for (const doublet &d : doublets) {
		ordered_pair edge0(d.a, d.b);
		ordered_pair edge1(d.b, d.c);

		const auto &equads = edge_quads[edge0];
		ulog_assert(edge_quads[edge0].size() == 2, "Invalid doublet", "quads %u and %u share %d\n", quads[0], quads[1], quads.size());
		ulog_assert(edge_quads[edge1].size() == 2, "Invalid doublet", "quads %u and %u share %d\n", quads[0], quads[1], quads.size());

		uint32_t q0 = equads[0];
		uint32_t q1 = equads[1];

		if (!valid[q0] || !valid[q1]) {
			ulog_warning("cleanse_doublets", "skipping (no longer) invalid doublet\n");
			continue;
		}

		// Dissolve doublet
		valid[q0] = false;
		valid[q1] = false;

		// New quad; find the two other vertices not in the doublet
		const glm::uvec4 &quad0 = ref.quads[q0];
		const glm::uvec4 &quad1 = ref.quads[q1];

		uint32_t v0 = 0;
		uint32_t v1 = 0;

		for (uint32_t i = 0; i < 4; i++) {
			if (quad0[i] != d.a && quad0[i] != d.b && quad0[i] != d.c)
				v0 = quad0[i];

			if (quad1[i] != d.a && quad1[i] != d.b && quad1[i] != d.c)
				v1 = quad1[i];
		}

		quads.push_back({ d.a, v0, d.c, v1 });
	}

	// Reconstruct primitives
	std::vector <glm::uvec4> clean_quads;
	for (uint32_t i = 0; i < valid.size(); i++) {
		if (!valid[i])
			continue;

		clean_quads.push_back(quads[i]);
	}

	ulog_info("cleanse_doublets", "cleaned %lu quads\n", valid.size() - clean_quads.size());
	for (uint32_t i = valid.size(); i < quads.size(); i++)
		clean_quads.push_back(quads[i]);

	geometry <eQuad> result = ref;
	result.quads = clean_quads;

	return result;
}

// template <size_t primitive>
// struct geomety_optimization_state {
// 	vertex_graph vgraph;
// 	edge_graph edge_graph;
// 	dual_graph dgraph;
// 	geometry <primitive> ref;
// 	std::vector <bool> valid;
// };

// Perform a diagonal collapse on a specific quad
// TODO: use an optimization state object
geometry <eQuad> diagonal_collapse(const geometry <eQuad> &ref, const vertex_graph &vgraph, uint32_t quad)
{
	geometry <eQuad> result = ref;

	glm::uvec4 q = ref.quads[quad];
	// std::swap(q[1], q[2]);

	glm::vec3 v0 = ref.vertices[q[0]];
	glm::vec3 v1 = ref.vertices[q[1]];
	glm::vec3 v2 = ref.vertices[q[2]];
	glm::vec3 v3 = ref.vertices[q[3]];

	// Compute the two diagonals
	glm::vec3 d0 = v2 - v0;
	glm::vec3 d1 = v3 - v1;

	float l0 = glm::length(d0);
	float l1 = glm::length(d1);

	// Compute and add the new vertex (midpoint of the diagonal)
	glm::vec3 mid = (l0 < l1) ? (v0 + v2) * 0.5f : (v1 + v3) * 0.5f;
	uint32_t nvi = result.vertices.size();
	result.vertices.push_back(mid);

	if (l0 < l1) {
		// Dissolving v0 -- v2
		auto adj_v0 = vgraph.at(q[0]);
		auto adj_v2 = vgraph.at(q[2]);

		auto adj = std::unordered_set <uint32_t> (adj_v0.begin(), adj_v0.end());
		adj.insert(adj_v2.begin(), adj_v2.end());

		// Fix vertex index for quads using v0 or v2
		for (uint32_t i = 0; i < result.quads.size(); i++) {
			auto &quad = result.quads[i];

			if (quad[0] == q[0] || quad[0] == q[2])
				quad[0] = nvi;

			if (quad[1] == q[0] || quad[1] == q[2])
				quad[1] = nvi;

			if (quad[2] == q[0] || quad[2] == q[2])
				quad[2] = nvi;

			if (quad[3] == q[0] || quad[3] == q[2])
				quad[3] = nvi;
		}
	} else {
		// Dissolving v1 -- v3
		auto adj_v1 = vgraph.at(q[1]);
		auto adj_v3 = vgraph.at(q[3]);

		auto adj = std::unordered_set <uint32_t> (adj_v1.begin(), adj_v1.end());
		adj.insert(adj_v3.begin(), adj_v3.end());

		// Fix vertex index for quads using v1 or v3
		for (uint32_t i = 0; i < result.quads.size(); i++) {
			auto &quad = result.quads[i];

			if (quad[0] == q[1] || quad[0] == q[3])
				quad[0] = nvi;

			if (quad[1] == q[1] || quad[1] == q[3])
				quad[1] = nvi;

			if (quad[2] == q[1] || quad[2] == q[3])
				quad[2] = nvi;

			if (quad[3] == q[1] || quad[3] == q[3])
				quad[3] = nvi;
		}
	}

	// TODO: remove old quads
	result.quads.erase(result.quads.begin() + quad);
	return result;
}

// Perform decimation on a quad mesh
// TODO: pass options
geometry <eQuad> decimate(const geometry <eQuad> &ref)
{
	geometry <eQuad> result = ref;

	for (int i = 0; i < 1000; i++) {
		// Progress
		printf("\rdecimate: %d / %d", i, 5000);
		fflush(stdout);

		vertex_graph vgraph(result);
		edge_graph egraph(result);
		dual_graph dgraph(egraph);
		// egraph.check();
		// egraph.check(1);

		const auto &areas = primitive_areas(result);

		std::vector <std::pair <float, uint32_t>> sorted(areas.size());
		for (uint32_t i = 0; i < areas.size(); i++)
			sorted[i] = { areas[i], i };

		std::sort(sorted.begin(), sorted.end(),
			[](const auto &a, const auto &b) {
				return a.first > b.first;
			}
		);

		printf("\n(B) %d vertices, %d quads\n", result.vertices.size(), result.quads.size());

		for (int j = 0; j < sorted.size(); j++) {
			// TODO: this would be an outright error
			if (dgraph.count(sorted[j].second) == 0)
				continue;

			if (dgraph.at(sorted[j].second).size() != 4)
				continue;

			result = diagonal_collapse(result, vgraph, sorted[j].second);
			break;
		}

		printf("(A) %d vertices, %d quads\n", result.vertices.size(), result.quads.size());
		result = compact(result);

		// TODO: clean up any doublets afterwards...
	}

	vertex_graph vgraph(result);
	edge_graph egraph(result);
	dual_graph dgraph(egraph);
	// egraph.check();
	// egraph.check(1);

	return result;
	// return compact(result);

	// vertex_graph vgraph(result);
	// edge_graph egraph(result);
	// dual_graph dgraph(egraph);

	// for (const auto &[f, adj] : dgraph)
	// 	ulog_assert(adj.size() == 4, "decimate", "quad %u is non-manifold; it has %lu neighbors\n", f, adj.size());

	// Iteratively collapse quad diagonals; heuristic is minimum area, along shortest diagonal
	// TODO: use a priority queue with removable elements...
	// then keep inserting new elements as they are created (put into a geometry_decimation_state : optimization_state)
	// const auto &areas = primitive_areas(result);

	// printf("areas: ");
	// for (int i = 0; i < areas.size(); i++)
	// 	printf("%f, ", areas[i]);
	// printf("\n");

	// Sort with an augmented array
	// std::vector <std::pair <float, uint32_t>> sorted(areas.size());
	// for (uint32_t i = 0; i < areas.size(); i++)
	// 	sorted[i] = { areas[i], i };
	//
	// std::sort(sorted.begin(), sorted.end(),
	// 	[](const auto &a, const auto &b) {
	// 		return a.first < b.first;
	// 	}
	// );
	//
	// printf("sorted areas: ");
	// for (int i = 0; i < 10; i++)
	// 	printf("%f (%d), ", sorted[i].first, sorted[i].second);
	// printf("\n");

	// TODO: stop at a maximum smallest area threshold?

	// TODO: maintain a optimzation state, with the graphs and valid states...

	// TODO: extract disjoint (non adjacent) quads to decimatewith diagonal collapse

	// TODO: only for faces with exactly four neighbors

	// printf("(B) %d vertices, %d quads\n", result.vertices.size(), result.quads.size());
	// result = diagonal_collapse(result, dgraph, sorted[0].second);
	// printf("(A) %d vertices, %d quads\n", result.vertices.size(), result.quads.size());
	//
	// // Rebuid the graphs (TODO:collect fix operations into a queue, and apply them...)
	// vgraph = vertex_graph(result);
	// egraph = edge_graph(result);
	// dgraph = dual_graph(egraph);

	return result;
}

int main(int argc, char *argv[])
{
	if (argc != 2) {
		ulog_error("quadrangulate", "Usage: %s <input>\n", argv[0]);
		return 1;
	}

	std::filesystem::path input { argv[1] };

	// geometry <eTriangle> ref = loader(input).get(0);
	geometry <eTriangle> ref = load_geometry <eTriangle> (input)[0];
	ulog_info("quadrangulate", "loaded %s, mesh has %d vertices and %d triangles\n", input.c_str(), ref.vertices.size(), ref.triangles.size());

	ref = compact(ref);
	ref = sanitize(ref);
	ulog_info("quadrangulate", "compacted %s, resulting mesh has %d vertices and %d triangles\n", input.c_str(), ref.vertices.size(), ref.triangles.size());

	geometry <eQuad> result = quadrangulate(ref);
	ulog_info("quadrangulate", "quadrangulated %s, resulting mesh has %d vertices and %d quads\n", input.c_str(), result.vertices.size(), result.quads.size());

	// TODO: after...
	// for (int i = 0; i < 3; i++)
	// 	result = laplacian_smoothing(result, 0.9f);

	result = sanitize(result);
	// result = cleanse_doublets(result);

	ulog_info("quadrangulate", "smoothed %s, resulting mesh has %d vertices and %d quads\n", input.c_str(), result.vertices.size(), result.quads.size());

	result = decimate(result);
	ulog_info("quadrangulate", "decimated %s, resulting mesh has %d vertices and %d quads\n", input.c_str(), result.vertices.size(), result.quads.size());

	write_geometry(result, (input.stem().string() + "_quad.obj"));

	// TODO: convert-geometry utility, which should automatically compact everything..
}
