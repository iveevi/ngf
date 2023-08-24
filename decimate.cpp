#include <fstream>

#include <eigen3/Eigen/Dense>

#include <igl/decimate.h>

#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/curve_network.h>
#include <polyscope/point_cloud.h>

#define MESH_LOAD_SAVE
#include "mesh.hpp"

namespace ps = polyscope;

#include <cstdarg>

void iassert(bool condition, const char *message, ...)
{
	if (!condition) {
		va_list args;
		va_start(args, message);
		vprintf(message, args);
		va_end(args);
		exit(1);
	}
}

std::tuple <Eigen::MatrixXd, Eigen::MatrixXi> translate_mesh(const Mesh &mesh)
{
	Eigen::MatrixXd V(mesh.vertices.size(), 3);
	Eigen::MatrixXi F(mesh.triangles.size(), 3);

	for (size_t i = 0; i < mesh.vertices.size(); i++) {
		V(i, 0) = mesh.vertices[i].x;
		V(i, 1) = mesh.vertices[i].y;
		V(i, 2) = mesh.vertices[i].z;
	}

	for (size_t i = 0; i < mesh.triangles.size(); i++) {
		F(i, 0) = mesh.triangles[i][1];
		F(i, 1) = mesh.triangles[i][0];
		F(i, 2) = mesh.triangles[i][2];
	}

	return { V, F };
}

Mesh translate_mesh(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F)
{
	Mesh mesh;

	mesh.vertices.resize(V.rows());
	mesh.normals.resize(V.rows());
	mesh.triangles.resize(F.rows());

	for (size_t i = 0; i < V.rows(); i++) {
		mesh.vertices[i] = { V(i, 0), V(i, 1), V(i, 2) };
		mesh.normals[i] = { 0, 0, 0 };
	}

	for (size_t i = 0; i < F.rows(); i++)
		mesh.triangles[i] = { F(i, 1), F(i, 0), F(i, 2) };

	return mesh;
}

int main(int argc, char *argv[])
{
	// Load arguments
	if (argc != 2) {
		printf("Usage: %s <filename>\n", argv[0]);
		return 1;
	}

	std::filesystem::path path = std::filesystem::weakly_canonical(argv[1]);

	// Load mesh
	Mesh mesh = deduplicate(load_mesh(path)).first;
	printf("Loaded mesh with %lu vertices and %lu triangles\n", mesh.vertices.size(), mesh.triangles.size());

	auto [Vertices, Faces] = translate_mesh(mesh);

	Eigen::MatrixXd V = Vertices;
	Eigen::MatrixXi F = Faces;
	Eigen::VectorXi a0;
	Eigen::VectorXi a1;

	size_t count = 0.01 * mesh.triangles.size();
	if (!igl::decimate(Vertices, Faces, count, V, F, a0, a1)) {
		printf("Decimation failed\n");
		return 1;
	} else {
		printf("Decimated mesh with %lu vertices and %lu triangles\n", V.rows(), F.rows());
	}

	Mesh decimated_mesh = translate_mesh(V, F);
	decimated_mesh = deduplicate(decimated_mesh).first;

	auto [vgraph, egraph, dual] = build_graphs(decimated_mesh);

	// Check the dual graph
	for (const auto &[f, adj] : dual)
		iassert(adj.size() == 3, "Triangle %u has %lu adjacent triangles\n", f, adj.size());

	// Quadrangulate the decimated mesh
	struct ordered_pair {
		uint32_t a;
		uint32_t b;

		ordered_pair(uint32_t a_, uint32_t b_) {
			a = a_;
			b = b_;

			if (a > b)
				std::swap(a, b);
		}

		bool operator==(const ordered_pair &other) const {
			return a == other.a && b == other.b;
		}

		struct hash {
			size_t operator()(const ordered_pair &p) const {
				return std::hash <uint32_t> ()(p.a) ^ std::hash <uint32_t> ()(p.b);
			}
		};
	};

	std::unordered_set <ordered_pair, ordered_pair::hash> quad_pairs;
	std::unordered_set <uint32_t> remaining;

	for (uint32_t t = 0; t < decimated_mesh.triangles.size(); t++)
		remaining.insert(t);

	while (!remaining.empty()) {
		bool found = false;

		std::unordered_set <uint32_t> to_remove;
		for (uint32_t t : remaining) {
			if (to_remove.count(t))
				continue;

			for (uint32_t tn : dual[t]) {
				if (remaining.count(tn) == 0 || to_remove.count(tn))
					continue;

				// Pick the first one arbitrarily (in the future use some metric)
				quad_pairs.insert({ t, tn });
				to_remove.insert(t);
				to_remove.insert(tn);
				found = true;
				break;
			}
		}

		for (uint32_t t : to_remove)
			remaining.erase(t);

		if (!found) {
			printf("Failed to find a pair of triangles to quadrangulate\n");
			break;
		}
	}

	printf("Left with %lu triangles\n", remaining.size());
	printf("Number of pairs: %lu\n", quad_pairs.size());
	assert(remaining.size() % 2 == 0);

	///////////////////////////////////////
	// Deal with the remaining triangles //
	///////////////////////////////////////

	// Need new connectivity information
	enum class polygon_type {
		quad,
		triangle
	};

	struct polygon {
		polygon_type type;

		uint32_t t0;
		uint32_t t1;

		std::vector <uint32_t> vertices;
	};

	std::vector <polygon> polygons;
	std::unordered_map <uint32_t, uint32_t> polygon_map;
	std::unordered_set <uint32_t> unused;

	for (uint32_t t : remaining) {
		const Triangle &tri = decimated_mesh.triangles[t];

		polygon poly;

		poly.type = polygon_type::triangle;
		poly.t0 = t;

		// any order is cyclic
		poly.vertices.push_back(tri[0]);
		poly.vertices.push_back(tri[1]);
		poly.vertices.push_back(tri[2]);

		uint32_t index = polygons.size();
		polygons.push_back(poly);
		unused.insert(index);

		polygon_map[t] = index;
	}

	auto cyclic_order = [](const Triangle &a, const Triangle &b) -> std::vector <uint32_t> {
		std::unordered_set <uint32_t> a_set = { a[0], a[1], a[2] };
		std::unordered_set <uint32_t> b_set = { b[0], b[1], b[2] };

		std::unordered_set <uint32_t> shared;
		std::unordered_set <uint32_t> unique;

		for (uint32_t v : a_set) {
			if (b_set.count(v))
				shared.insert(v);
			else
				unique.insert(v);
		}

		for (uint32_t v : b_set) {
			if (a_set.count(v))
				shared.insert(v);
			else
				unique.insert(v);
		}

		assert(shared.size() == 2);
		assert(unique.size() == 2);

		uint32_t v0 = *unique.begin();
		uint32_t v1 = *std::next(unique.begin());

		uint32_t s0 = *shared.begin();
		uint32_t s1 = *std::next(shared.begin());

		return { v0, s0, v1, s1 };
	};

	for (auto [t0, t1] : quad_pairs) {
		const Triangle &tri0 = decimated_mesh.triangles[t0];
		const Triangle &tri1 = decimated_mesh.triangles[t1];

		polygon poly;

		poly.type = polygon_type::quad;
		poly.t0 = t0;
		poly.t1 = t1;
		poly.vertices = cyclic_order(tri0, tri1);

		uint32_t index = polygons.size();
		polygons.push_back(poly);

		polygon_map[t0] = index;
		polygon_map[t1] = index;
	}

	// Adjacency information amongst the polygons
	std::unordered_map <uint32_t, std::unordered_set <uint32_t>> polygon_adjacency;

	for (auto &[e, fs] : egraph) {
		iassert(fs.size() == 2, "Edge graph should only contain edges with two adjacent faces");
		uint32_t f0 = *fs.begin();
		uint32_t f1 = *std::next(fs.begin());

		uint32_t p0 = polygon_map[f0];
		uint32_t p1 = polygon_map[f1];

		if (p0 == p1)
			continue;

		polygon_adjacency[p0].insert(p1);
		polygon_adjacency[p1].insert(p0);
	}

	// Double check adjacency information
	for (const auto &[p, adj] : polygon_adjacency)
		iassert(adj.size() >= 2, "Polygon %u has %lu adjacent polygons\n", p, adj.size());

	uint32_t triangle_count = 0;
	for (uint32_t i = 0; i < polygons.size(); i++) {
		const polygon &p0 = polygons[i];
		if (p0.type == polygon_type::triangle)
			triangle_count++;
	}

	printf("Number of triangles: %u\n", triangle_count);

	// 1. Merge leftovers in close proximity, with only only quad in between
	printf("Adjacencies of unused:\n");
	for (const auto &p : polygon_adjacency) {
		if (unused.count(p.first) == 0)
			continue;

		printf("%u: ", p.first);
		for (uint32_t i : p.second)
			printf("%u ", i);
		printf("\n");
	}

	// Find all pairs of one-adjscent polygons
	std::unordered_map <ordered_pair, uint32_t, ordered_pair::hash> one_adjacent_pairs;

	for (uint32_t r : unused) {
		iassert(polygons[r].type == polygon_type::triangle, "Remainder polygon %u is not a triangle\n", r);

		const std::unordered_set <uint32_t> &adj = polygon_adjacency[r];

		std::vector <std::pair <uint32_t, uint32_t>> adj_list;
		for (uint32_t a : adj) {
			// Although not possible, make sure a is not a triangle
			iassert(polygons[a].type == polygon_type::quad, "Polygon %u is not a quad", a);
			for (uint32_t b : polygon_adjacency[a]) {
				if (b == r)
					continue;

				if (unused.count(b) == 0)
					continue;

				iassert(polygons[b].type == polygon_type::triangle, "Connecting polygon %u is not a triangle", b);
				adj_list.push_back({ a, b });
			}
		}

		for (auto [a, b] : adj_list) {
			iassert(polygons[a].type == polygon_type::quad, "Polygon %u is not a quad", a);
			iassert(polygons[b].type == polygon_type::triangle, "Connecting polygon %u is not a triangle", b);
			one_adjacent_pairs.insert({ ordered_pair(r, b), a });
		}
	}

	printf("One-adjacent pairs:\n");
	for (const auto &p : one_adjacent_pairs)
		printf("%u -> %u -> %u\n", p.first.a, p.second, p.first.b);

	// Remove duplicate/confliting pairs
	std::unordered_set <uint32_t> taken;

	for (auto it = one_adjacent_pairs.begin(); it != one_adjacent_pairs.end(); ) {
		uint32_t a = it->first.a;
		uint32_t b = it->first.b;
		uint32_t c = it->second;

		bool remove = false;
		remove |= taken.count(a) > 0;
		remove |= taken.count(b) > 0;
		remove |= taken.count(c) > 0;

		if (remove) {
			it = one_adjacent_pairs.erase(it);
		} else {
			taken.insert(a);
			taken.insert(b);
			taken.insert(c);
			it++;
		}
	}

	// Classify the operations needed
	// true -> transpose
	// false -> split
	std::vector <std::tuple <uint32_t, uint32_t, uint32_t, bool>> operations;

	for (const auto &p : one_adjacent_pairs) {
		// If the two triangles share an edge of the
		// same triangle, then it is a split operation
		// otherwise it is transpose

		uint32_t ta = p.first.a;
		uint32_t tb = p.first.b;

		uint32_t qa = p.second;
		uint32_t qb = p.second;

		ta = polygons[ta].t0;
		tb = polygons[tb].t0;

		qa = polygons[qa].t0;
		qb = polygons[qb].t1;

		bool ta_qa = (dual[ta].count(qa) > 0);
		bool ta_qb = (dual[ta].count(qb) > 0);

		bool tb_qa = (dual[tb].count(qa) > 0);
		bool tb_qb = (dual[tb].count(qb) > 0);

		bool transpose = false;
		transpose |= ta_qa && tb_qb;
		transpose |= ta_qb && tb_qa;

		operations.push_back({ p.first.a, p.first.b, p.second, transpose });
	}

	printf("Number of distinct operations: %zu\n", operations.size());

	// Perform all transpose operations
	std::unordered_set <uint32_t> polygons_to_remove;

	uint32_t transpose_count = 0;
	for (auto it = operations.begin(); it != operations.end(); ) {
		uint32_t ta = std::get <0> (*it);
		uint32_t tb = std::get <1> (*it);
		uint32_t q = std::get <2> (*it);
		bool transpose = std::get <3> (*it);

		if (transpose) {
			// Transpose
			printf("Transpose %u -> %u -> %u\n", ta, q, tb);

			polygons_to_remove.insert(q);
			polygons_to_remove.insert(ta);
			polygons_to_remove.insert(tb);

			const polygon &pa = polygons[ta];
			const polygon &pb = polygons[tb];
			const polygon &pq = polygons[q];

			uint32_t ta = pa.t0;
			uint32_t tb = pb.t0;

			uint32_t qa = pq.t0;
			uint32_t qb = pq.t1;

			quad_pairs.erase({ qa, qb });

			bool case0 = (dual[ta].count(qa) > 0) && (dual[tb].count(qb) > 0);
			bool case1 = (dual[ta].count(qb) > 0) && (dual[tb].count(qa) > 0);

			const Triangle &tri_ta = decimated_mesh.triangles[ta];
			const Triangle &tri_tb = decimated_mesh.triangles[tb];
			const Triangle &tri_qa = decimated_mesh.triangles[qa];
			const Triangle &tri_qb = decimated_mesh.triangles[qb];

			if (case0) {
				quad_pairs.insert({ ta, qa });
				quad_pairs.insert({ tb, qb });

				polygon p0;
				p0.type = polygon_type::quad;
				p0.t0 = ta;
				p0.t1 = qa;
				p0.vertices = cyclic_order(tri_ta, tri_qa);

				polygon p1;
				p1.type = polygon_type::quad;
				p1.t0 = tb;
				p1.t1 = qb;
				p1.vertices = cyclic_order(tri_tb, tri_qb);

				polygons.push_back(p0);
				polygons.push_back(p1);
			} else if (case1) {
				quad_pairs.insert({ ta, qb });
				quad_pairs.insert({ tb, qa });

				polygon p0;
				p0.type = polygon_type::quad;
				p0.t0 = ta;
				p0.t1 = qb;
				p0.vertices = cyclic_order(tri_ta, tri_qb);

				polygon p1;
				p1.type = polygon_type::quad;
				p1.t0 = tb;
				p1.t1 = qa;
				p1.vertices = cyclic_order(tri_tb, tri_qa);

				polygons.push_back(p0);
				polygons.push_back(p1);
			} else {
				throw std::runtime_error("Invalid case");
			}

			// Remove the operation
			it = operations.erase(it);
			transpose_count++;
		} else {
			it++;
		}
	}

	// Perform all split operations
	std::unordered_set <uint32_t> triangles_to_remove;

	auto shared_vertices = [](const Triangle &ref, const Triangle &exp) -> std::tuple <uint32_t, uint32_t, uint32_t> {
		// Return shared0, shared1, unique
		std::unordered_set <uint32_t> ref_set = { ref[0], ref[1], ref[2] };
		std::unordered_set <uint32_t> exp_set = { exp[0], exp[1], exp[2] };

		std::unordered_set <uint32_t> shared;
		std::unordered_set <uint32_t> unique;

		for (uint32_t v : exp_set) {
			if (ref_set.count(v))
				shared.insert(v);
			else
				unique.insert(v);
		}

		uint32_t s0 = *shared.begin();
		uint32_t s1 = *std::next(shared.begin());

		uint32_t v0 = *unique.begin();

		iassert(shared.size() == 2, "Invalid shared size");
		iassert(unique.size() == 1, "Invalid unique size");

		return { s0, s1, v0 };
	};

	uint32_t split_count = 0;
	for (auto it = operations.begin(); it != operations.end(); ) {
		uint32_t ta = std::get <0> (*it);
		uint32_t tb = std::get <1> (*it);
		uint32_t q = std::get <2> (*it);
		bool transpose = std::get <3> (*it);

		if (!transpose) {
			// Split
			printf("Split %u -> %u -> %u\n", ta, q, tb);

			const polygon &pa = polygons[ta];
			const polygon &pb = polygons[tb];
			const polygon &pq = polygons[q];

			polygons_to_remove.insert(ta);
			polygons_to_remove.insert(tb);
			polygons_to_remove.insert(q);

			uint32_t ta = pa.t0;
			uint32_t tb = pb.t0;

			uint32_t qa = pq.t0;
			uint32_t qb = pq.t1;

			quad_pairs.erase({ qa, qb });

			triangles_to_remove.insert(ta);
			triangles_to_remove.insert(tb);
			triangles_to_remove.insert(qa);
			triangles_to_remove.insert(qb);

			// Find the central triangle
			bool acentral = (dual[qa].count(ta) > 0) && (dual[qa].count(tb) > 0);
			bool bcentral = (dual[qb].count(ta) > 0) && (dual[qb].count(tb) > 0);

			iassert(acentral || bcentral, "Invalid split operation");
			if (bcentral)
				std::swap(qa, qb);

			const Triangle &t = decimated_mesh.triangles[qa];
			glm::vec3 p0 = decimated_mesh.vertices[t[0]];
			glm::vec3 p1 = decimated_mesh.vertices[t[1]];
			glm::vec3 p2 = decimated_mesh.vertices[t[2]];
			glm::vec3 p = (p0 + p1 + p2) / 3.0f;

			uint32_t new_vertex = decimated_mesh.vertices.size();
			decimated_mesh.vertices.push_back(p);

			std::array <Triangle, 3> ref_triangles = {
				decimated_mesh.triangles[ta],
				decimated_mesh.triangles[tb],
				decimated_mesh.triangles[qb]
			};

			for (const Triangle &tri : ref_triangles) {
				auto [s0, s1, v0] = shared_vertices(t, tri);

				uint32_t new_triangle = decimated_mesh.triangles.size();
				decimated_mesh.triangles.push_back({ s0, new_vertex, v0 });
				decimated_mesh.triangles.push_back({ v0, s1, new_vertex });

				quad_pairs.insert({ new_triangle, new_triangle + 1 });

				const Triangle &ntri0 = decimated_mesh.triangles[new_triangle];
				const Triangle &ntri1 = decimated_mesh.triangles[new_triangle + 1];

				polygon poly;
				poly.type = polygon_type::quad;
				poly.t0 = new_triangle;
				poly.t1 = new_triangle + 1;
				poly.vertices = cyclic_order(ntri0, ntri1);

				polygons.push_back(poly);
			}

			// Remove the operation
			it = operations.erase(it);
			split_count++;
		} else {
			it++;
		}
	}

	printf("In total, %u transpose and %u split operations were performed\n", transpose_count, split_count);

	// Erase polygons used by the transpose operations
	{
		std::vector <uint32_t> sorted_polygons_to_remove(
			polygons_to_remove.begin(),
			polygons_to_remove.end());

		std::sort(sorted_polygons_to_remove.begin(), sorted_polygons_to_remove.end(), std::greater <uint32_t> ());

		uint32_t removed_triangle_count = 0;
		for (uint32_t p : sorted_polygons_to_remove) {
			removed_triangle_count += (polygons[p].type == polygon_type::triangle);
			polygons.erase(polygons.begin() + p);
		}

		printf("Triangular polygons removed: %u\n", removed_triangle_count);
	}

	// Delete triangles from the decimated mesh
	{
		std::vector <Triangle> new_triangles;
		std::unordered_map <uint32_t, uint32_t> remap;

		for (uint32_t i = 0; i < decimated_mesh.triangles.size(); i++) {
			if (triangles_to_remove.count(i) == 0) {
				remap[i] = new_triangles.size();
				new_triangles.push_back(decimated_mesh.triangles[i]);
			}
		}

		// Fix the polygon references
		for (polygon &poly : polygons) {
			poly.t0 = remap[poly.t0];

			if (poly.type == polygon_type::quad)
				poly.t1 = remap[poly.t1];
		}

		// Fix quad pairs
		std::unordered_set <ordered_pair, ordered_pair::hash> new_quad_pairs;
		for (const auto &[a, b] : quad_pairs)
			new_quad_pairs.insert({ remap[a], remap[b] });

		quad_pairs = new_quad_pairs;

		// Replace the decimated mesh triangles
		decimated_mesh.triangles = new_triangles;
	}

	// Rebuild adjacency information
	std::tie(vgraph, egraph, dual) = build_graphs(decimated_mesh);

	polygon_adjacency.clear();
	polygon_map.clear();

	for (uint32_t i = 0; i < polygons.size(); i++) {
		const polygon &p = polygons[i];
		polygon_map[p.t0] = i;
		if (p.type == polygon_type::quad)
			polygon_map[p.t1] = i;
	}

	for (auto &[e, fs] : egraph) {
		iassert(fs.size() == 2, "Edge graph should only contain edges with two adjacent faces");
		uint32_t f0 = *fs.begin();
		uint32_t f1 = *std::next(fs.begin());

		uint32_t p0 = polygon_map[f0];
		uint32_t p1 = polygon_map[f1];

		if (p0 == p1)
			continue;

		polygon_adjacency[p0].insert(p1);
		polygon_adjacency[p1].insert(p0);
	}

	// Collect unused triangles
	unused.clear();
	for (uint32_t i = 0; i < polygons.size(); i++) {
		const polygon &p0 = polygons[i];
		if (p0.type == polygon_type::triangle)
			unused.insert(i);
	}

	printf("Remaining leftovers: %u\n", unused.size());

	printf("Adjacent polygons: %u\n", polygon_adjacency.size());
	for (const auto &[i, adj] : polygon_adjacency)
		iassert(adj.size() >= 2, "Polygon %u has %u adjacent polygons", i, adj.size());

	// Pair up the remaining triangles by greedy proximity
	std::unordered_set <ordered_pair, ordered_pair::hash> triangle_pairs;

	auto unused_copy = unused;
	while (unused.size() > 0) {
		uint32_t best_i = 0;
		uint32_t best_j = 0;
		float best_distance = std::numeric_limits <float>::infinity();

		for (uint32_t i : unused) {
			const polygon &p0 = polygons[i];

			glm::vec3 mid0 = glm::vec3(0.0f);
			for (uint32_t vindex : p0.vertices) {
				const glm::vec3 &v = decimated_mesh.vertices[vindex];
				mid0 += v;
			}

			mid0 /= (float) p0.vertices.size();

			for (uint32_t j : unused) {
				if (i == j)
					continue;

				const polygon &p1 = polygons[j];

				glm::vec3 mid1 = glm::vec3(0.0f);
				for (uint32_t vindex : p1.vertices) {
					const glm::vec3 &v = decimated_mesh.vertices[vindex];
					mid1 += v;
				}

				mid1 /= (float) p1.vertices.size();

				float distance = glm::distance(mid0, mid1);
				if (distance < best_distance) {
					best_distance = distance;
					best_i = i;
					best_j = j;
				}
			}
		}

		triangle_pairs.insert({ best_i, best_j });
		unused.erase(best_i);
		unused.erase(best_j);

		printf("Best pair is %u %u with distance %f\n", best_i, best_j, best_distance);
	}

	printf("# of triangle pairs: %u\n", triangle_pairs.size());

	// For each pair, construct a path of quads joining them (BFS)
	std::unordered_set <uint32_t> visited;

	struct quad_path {
		uint32_t start;
		uint32_t end;

		std::vector <uint32_t> quads;
	};

	std::vector <quad_path> paths;
	for (const auto &[s, e] : triangle_pairs) {
		std::unordered_map <uint32_t, uint32_t> parent;
		std::queue <uint32_t> queue;
		queue.push(s);

		printf("Searching for path from %u to %u\n", s, e);

		std::unordered_set <uint32_t> local_visited;
		while (!queue.empty()) {
			uint32_t current = queue.front();
			queue.pop();

			if (current == e) {
				// Found a path
				printf("Found path from %u to %u\n", s, e);
				break;
			}

			// local_visited.insert(current);
			for (uint32_t neighbor : polygon_adjacency[current]) {
				if (unused_copy.count(neighbor) > 0 && neighbor != e)
					continue;

				if (visited.count(neighbor) > 0)
					continue;

				if (local_visited.count(neighbor) > 0)
					continue;

				parent[neighbor] = current;
				queue.push(neighbor);
				local_visited.insert(neighbor);
			}
		}

		if (parent.count(e) == 0) {
			printf("No path found\n");
			continue;
		}

		std::vector <uint32_t> quads;

		uint32_t current = parent[e];
		while (current != s) {
			quads.push_back(current);
			current = parent[current];
		}

		paths.push_back({ e, s, quads });
		for (uint32_t v : quads)
			visited.insert(v);
	}

	printf("Paths: %u\n", paths.size());
	for (const quad_path &p : paths) {
		printf("Path from %u to %u with %u quads\n", p.start, p.end, p.quads.size());
		for (uint32_t q : p.quads)
			printf("%u ", q);
		printf("\n");
	}

	// Perform tranpositions to resolve each path
	polygons_to_remove.clear();

	for (auto &[s, e, quads] : paths) {
		polygon *ps = &polygons[s];
		polygon *pe = &polygons[e];

		for (int32_t i = 0; i < quads.size(); i++) {
			polygon *p0 = &polygons[quads[i]];

			bool case0 = (dual[ps->t0].count(p0->t0) > 0);
			bool case1 = (dual[ps->t0].count(p0->t1) > 0);

			if (!case0 && !case1) {
				// printf("No dual edge found between %u and %u\n", ps->t0, p0->t0);
				// printf("No dual edge found between %u and %u\n", ps->t0, p0->t1);
				// printf("  need to flip the triangles!\n");

				// Flip the previous triangles and go back
				polygon *p0 = &polygons[quads[--i]];

				const Triangle &t0 = decimated_mesh.triangles[p0->t0];
				const Triangle &t1 = decimated_mesh.triangles[p0->t1];

				auto vs = cyclic_order(t0, t1);

				Triangle n0 = { vs[0], vs[1], vs[2] };
				Triangle n1 = { vs[0], vs[2], vs[3] };

				decimated_mesh.triangles[p0->t0] = n0;
				decimated_mesh.triangles[p0->t1] = n1;
				i--;

				// Rebuild the graphs
				std::tie(vgraph, egraph, dual) = build_graphs(decimated_mesh);
				continue;
			}

			quad_pairs.erase({ p0->t0, p0->t1 });
			if (case0)
				std::swap(ps->t0, p0->t1);
			else
				std::swap(ps->t0, p0->t0);

			// Repair the quads
			const Triangle &t0 = decimated_mesh.triangles[p0->t0];
			const Triangle &t1 = decimated_mesh.triangles[p0->t1];

			p0->vertices = cyclic_order(t0, t1);
			quad_pairs.insert({ p0->t0, p0->t1 });
		}

		// Last chance to flip at the end
		bool connected = dual[ps->t0].count(pe->t0) > 0;
		if (!connected) {
			// printf("No dual edge found between FINAL %u and %u\n", ps->t0, pe->t0);

			polygon *p0 = &polygons[quads.back()];

			const Triangle &t0 = decimated_mesh.triangles[p0->t0];
			const Triangle &t1 = decimated_mesh.triangles[p0->t1];

			auto vs = cyclic_order(t0, t1);

			Triangle n0 = { vs[0], vs[1], vs[2] };
			Triangle n1 = { vs[0], vs[2], vs[3] };

			decimated_mesh.triangles[p0->t0] = n0;
			decimated_mesh.triangles[p0->t1] = n1;

			std::tie(vgraph, egraph, dual) = build_graphs(decimated_mesh);

			// Perform a final flip
			bool case0 = (dual[ps->t0].count(p0->t0) > 0);
			bool case1 = (dual[ps->t0].count(p0->t1) > 0);

			quad_pairs.erase({ p0->t0, p0->t1 });
			if (case0)
				std::swap(ps->t0, p0->t1);
			else
				std::swap(ps->t0, p0->t0);

			// Repair the quads
			const Triangle &t2 = decimated_mesh.triangles[p0->t0];
			const Triangle &t3 = decimated_mesh.triangles[p0->t1];

			p0->vertices = cyclic_order(t2, t3);
			quad_pairs.insert({ p0->t0, p0->t1 });
		}

		// Clean up old and add new polygons
		iassert(dual[ps->t0].count(pe->t0) > 0, "No dual edge found between %u and %u\n", ps->t0, pe->t0);

		polygons_to_remove.insert(s);
		polygons_to_remove.insert(e);

		quad_pairs.insert({ ps->t0, pe->t0 });

		polygon new_polygon;
		new_polygon.type = polygon_type::quad;
		new_polygon.t0 = ps->t0;
		new_polygon.t1 = pe->t0;

		const Triangle &t0 = decimated_mesh.triangles[new_polygon.t0];
		const Triangle &t1 = decimated_mesh.triangles[new_polygon.t1];

		new_polygon.vertices = cyclic_order(t0, t1);

		uint32_t new_polygon_id = polygons.size();
		polygons.push_back(new_polygon);
	}

	// Erase polygons used by the transpose operations
	{
		std::vector <uint32_t> sorted_polygons_to_remove(
			polygons_to_remove.begin(),
			polygons_to_remove.end());

		std::sort(sorted_polygons_to_remove.begin(), sorted_polygons_to_remove.end(), std::greater <uint32_t> ());

		uint32_t removed_triangle_count = 0;
		for (uint32_t p : sorted_polygons_to_remove) {
			removed_triangle_count += (polygons[p].type == polygon_type::triangle);
			polygons.erase(polygons.begin() + p);
		}

		printf("Triangular polygons removed: %u\n", removed_triangle_count);
	}

	// Check that every triangle is now covered exactly by one QUAD
	std::vector <uint32_t> covered;
	covered.resize(decimated_mesh.triangles.size(), 0);

	for (const auto &p : polygons) {
		iassert(p.type == polygon_type::quad, "Polygon is not a quad\n");
		covered[p.t0]++;
		covered[p.t1]++;
	}

	for (uint32_t c : covered)
		iassert(c == 1, "Triangle is not covered exactly once\n");

	printf("%zu triangles covered exactly once by %zu quads\n", covered.size(), polygons.size());

	// Visualize
	std::vector <std::array <uint32_t, 2>> curve_points;
	for (const auto &p : polygons) {
		for (uint32_t i = 0; i < p.vertices.size(); i++) {
			uint32_t v0 = p.vertices[i];
			uint32_t v1 = p.vertices[(i + 1) % p.vertices.size()];

			curve_points.push_back({ v0, v1 });
		}
	}

	std::vector <glm::vec3> face_colors;
	face_colors.resize(decimated_mesh.triangles.size(), glm::vec3(0.0));

	glm::vec3 color_wheel[] = {
		{ 0.750, 0.250, 0.250 },
		{ 0.750, 0.500, 0.250 },
		{ 0.750, 0.750, 0.250 },
		{ 0.500, 0.750, 0.250 },
		{ 0.250, 0.750, 0.250 },
		{ 0.250, 0.750, 0.500 },
		{ 0.250, 0.750, 0.750 },
		{ 0.250, 0.500, 0.750 },
		{ 0.250, 0.250, 0.750 },
		{ 0.500, 0.250, 0.750 },
		{ 0.750, 0.250, 0.750 },
		{ 0.750, 0.250, 0.500 }
	};

	uint32_t i = 0;
	for (const auto &[t0, t1] : quad_pairs) {
		const glm::vec3 &c = color_wheel[i++ % 12];

		face_colors[t0] = c;
		face_colors[t1] = c;
	}

	// Send to polyscope to visualize
	ps::init();

	ps::registerSurfaceMesh("original", mesh.vertices, mesh.triangles);

	auto dm = ps::registerSurfaceMesh("decimated", decimated_mesh.vertices, decimated_mesh.triangles);
	dm->addFaceColorQuantity("coding", face_colors);

	auto c = ps::registerCurveNetwork("curve", decimated_mesh.vertices, curve_points);
	c->setColor({ 0.0, 0.0, 0.0 })->setRadius(0.001);

	ps::show();

	// Save to file
	std::filesystem::path opath = path.stem().string() + ".quads";
	std::ofstream ofile(opath, std::ios::binary);

	// Write the original mesh path
	std::string original_path = path.string();
	uint32_t length = original_path.size();

	ofile.write((char *) &length, sizeof(uint32_t));
	ofile.write(original_path.c_str(), length);

	// Write the mesh, then the quad indices
	uint32_t nv = decimated_mesh.vertices.size();
	uint32_t nt = decimated_mesh.triangles.size();

	ofile.write((char *) &nv, sizeof(uint32_t));
	ofile.write((char *) &nt, sizeof(uint32_t));

	ofile.write((char *) decimated_mesh.vertices.data(), nv * sizeof(glm::vec3));
	ofile.write((char *) decimated_mesh.triangles.data(), nt * sizeof(glm::uvec3));

	uint32_t np = polygons.size();
	ofile.write((char *) &np, sizeof(uint32_t));

	for (const auto &p : polygons) {
		ofile.write((char *) &p.t0, sizeof(uint32_t));
		ofile.write((char *) &p.t1, sizeof(uint32_t));
	}

	ofile.close();
}
