#include <iostream>
#include <random>

#include <omp.h>

#include <glm/glm.hpp>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include <littlevk/littlevk.hpp>

#define MESH_LOAD_SAVE
#include "closest_point.cuh"
#include "mesh.hpp"
#include "viewer.hpp"

inline bool ray_x_triangle(const Mesh &mesh, size_t tindex, const glm::vec3 &x, const glm::vec3 &d)
{
	const Triangle &tri = mesh.triangles[tindex];

	glm::vec3 v0 = mesh.vertices[tri[0]];
	glm::vec3 v1 = mesh.vertices[tri[1]];
	glm::vec3 v2 = mesh.vertices[tri[2]];

	glm::vec3 e1 = v1 - v0;
	glm::vec3 e2 = v2 - v0;
	glm::vec3 p = cross(d, e2);

	float a = dot(e1, p);
	if (std::abs(a) < 1e-6)
		return false;

	float f = 1.0 / a;
	glm::vec3 s = x - v0;
	float u = f * dot(s, p);

	if (u < 0.0 || u > 1.0)
		return false;

	glm::vec3 q = cross(s, e1);
	float v = f * dot(d, q);

	if (v < 0.0 || u + v > 1.0)
		return false;

	float t = f * dot(e2, q);
	return t > 1e-6;
}

// Acceleration structure for interior point queries
struct ipqas {
	using bin = std::vector <uint32_t>;

	Mesh ref;
	uint32_t resolution;

	std::vector <bin> bins_xy;
	std::vector <bin> bins_xz;
	std::vector <bin> bins_yz;

	glm::vec3 ext_min;
	glm::vec3 ext_max;

	ipqas(const Mesh &ref_, uint32_t resolution_)
			: ref(ref_), resolution(resolution_) {
		ext_min = glm::vec3(std::numeric_limits <float> ::max());
		ext_max = glm::vec3(std::numeric_limits <float> ::min());

		for (const glm::vec3 &vertex : ref.vertices) {
			ext_min = glm::min(ext_min, vertex);
			ext_max = glm::max(ext_max, vertex);
		}

		bins_xy.resize(resolution * resolution);
		bins_xz.resize(resolution * resolution);
		bins_yz.resize(resolution * resolution);

		for (size_t i = 0; i < ref.triangles.size(); i++) {
			const Triangle &tri = ref.triangles[i];

			// Find the bounding box of the triangle
			glm::vec3 min = glm::vec3(std::numeric_limits <float> ::max());
			glm::vec3 max = glm::vec3(std::numeric_limits <float> ::lowest());

			for (size_t j = 0; j < 3; ++j) {
				glm::vec3 v = ref.vertices[tri[j]];
				v = (v - ext_min) / (ext_max - ext_min);
				min = glm::min(min, v);
				max = glm::max(max, v);
			}

			// Find the bins that the triangle overlaps
			glm::vec3 min_bin = glm::floor(min * glm::vec3(resolution));
			glm::vec3 max_bin = glm::ceil(max * glm::vec3(resolution));

			for (size_t x = min_bin.x; x < max_bin.x; x++) {
				for (size_t y = min_bin.y; y < max_bin.y; y++)
					bins_xy[x + y * resolution].push_back(i);
			}

			for (size_t x = min_bin.x; x < max_bin.x; x++) {
				for (size_t z = min_bin.z; z < max_bin.z; z++)
					bins_xz[x + z * resolution].push_back(i);
			}

			for (size_t y = min_bin.y; y < max_bin.y; y++) {
				for (size_t z = min_bin.z; z < max_bin.z; z++)
					bins_yz[y + z * resolution].push_back(i);
			}
		}
	}

	bool query(const glm::vec3 &v) const {
		static constexpr glm::vec3 dx { 1, 0, 0 };
		static constexpr glm::vec3 dy { 0, 1, 0 };
		static constexpr glm::vec3 dz { 0, 0, 1 };

		glm::vec3 normed = (v - ext_min) / (ext_max - ext_min);
		glm::vec3 voxel = normed * glm::vec3(resolution);
		voxel = glm::clamp(voxel, glm::vec3(0), glm::vec3(resolution - 1));

		uint32_t x = voxel.x;
		uint32_t y = voxel.y;
		uint32_t z = voxel.z;

		uint32_t xy_count = 0;
		uint32_t xy_count_neg = 0;

		for (size_t tindex : bins_xy[x + y * resolution]) {
			if (ray_x_triangle(ref, tindex, v, dz))
				xy_count++;

			if (ray_x_triangle(ref, tindex, v, -dz))
				xy_count_neg++;
		}

		uint32_t xz_count = 0;
		uint32_t xz_count_neg = 0;

		for (size_t tindex : bins_xz[x + z * resolution]) {
			if (ray_x_triangle(ref, tindex, v, dy))
				xz_count++;

			if (ray_x_triangle(ref, tindex, v, -dy))
				xz_count_neg++;
		}

		uint32_t yz_count = 0;
		uint32_t yz_count_neg = 0;

		for (size_t tindex : bins_yz[y + z * resolution]) {
			if (ray_x_triangle(ref, tindex, v, dx))
				yz_count++;

			if (ray_x_triangle(ref, tindex, v, -dx))
				yz_count_neg++;
		}

		bool xy_in = (xy_count % 2 == 1) && (xy_count_neg % 2 == 1);
		bool xz_in = (xz_count % 2 == 1) && (xz_count_neg % 2 == 1);
		bool yz_in = (yz_count % 2 == 1) && (yz_count_neg % 2 == 1);

		return xy_in || xz_in || yz_in;
	}

	void dump() const {
		// TODO: microlog
		float load_xy = 0.0f;
		float load_xz = 0.0f;
		float load_yz = 0.0f;

		for (const auto &bin : bins_xy)
			load_xy += bin.size();

		for (const auto &bin : bins_xz)
			load_xz += bin.size();

		for (const auto &bin : bins_yz)
			load_yz += bin.size();

		printf("Average bin load: %f (xy), %f (xz), %f (yz)\n",
			load_xy / bins_xy.size(),
			load_xz / bins_xz.size(),
			load_yz / bins_yz.size());
	}
};

std::pair <float, glm::vec3> closest_point(const Mesh &ref, const glm::vec3 &p)
{
	std::vector <std::pair <float, glm::vec3>> distances;
	distances.resize(ref.triangles.size());

	#pragma omp parallel for
	for (size_t i = 0; i < ref.triangles.size(); ++i) {
		const Triangle &tri = ref.triangles[i];
		glm::vec3 v0 = ref.vertices[tri[0]];
		glm::vec3 v1 = ref.vertices[tri[1]];
		glm::vec3 v2 = ref.vertices[tri[2]];
		glm::vec3 closest = triangle_closest_point(v0, v1, v2, p);
		distances[i] = std::make_pair(glm::distance(p, closest), closest);
	}

	std::sort(distances.begin(), distances.end(),
		[](const auto &a, const auto &b) {
			return a.first < b.first;
		}
	);

	return distances[0];
}

int main(int argc, char *argv[])
{
	// Load arguments
	if (argc != 3) {
		printf("Usage: %s <filename> <resolution>\n", argv[0]);
		return 1;
	}

	std::filesystem::path path = std::filesystem::weakly_canonical(argv[1]);
	size_t resolution = std::atoi(argv[2]);

	// Load mesh
	Mesh mesh = load_mesh(path);
	printf("Loaded mesh with %lu vertices and %lu triangles\n", mesh.vertices.size(), mesh.triangles.size());
	printf("Resolution: %lu\n", resolution);

	// Construct the voxel mesh
	ipqas interior_query_as(mesh, resolution);
	interior_query_as.dump();

	glm::vec3 ext_min = interior_query_as.ext_min;
	glm::vec3 ext_max = interior_query_as.ext_max;

	uint32_t voxel_count = 0;
	std::vector <uint32_t> voxels;
	voxels.resize(resolution * resolution * resolution);

	#pragma omp parallel for reduction(+:voxel_count)
	for (uint32_t i = 0; i < voxels.size(); i++) {
		uint32_t x = i % resolution;
		uint32_t y = (i / resolution) % resolution;
		uint32_t z = i / (resolution * resolution);

		glm::vec3 center = glm::vec3(x + 0.5f, y + 0.5f, z + 0.5f) / (float) resolution;
		center = center * (ext_max - ext_min) + ext_min;
	
		if (interior_query_as.query(center)) {
			voxels[i] = 1;
			voxel_count++;
		}
	}

	printf("Voxel count: %u/%u\n", voxel_count, voxels.size());

	// Adding voxel elements to a mesh
	auto add_voxel = [&](Mesh &mesh, const size_t i) {
		uint32_t x = i % resolution;
		uint32_t y = (i / resolution) % resolution;
		uint32_t z = i / (resolution * resolution);

		// Load all eight corners of the voxel
		glm::vec3 corners[8];

		corners[0] = glm::vec3(x, y, z) / (float) resolution;
		corners[1] = glm::vec3(x + 1, y, z) / (float) resolution;
		corners[2] = glm::vec3(x, y + 1, z) / (float) resolution;
		corners[3] = glm::vec3(x + 1, y + 1, z) / (float) resolution;
		corners[4] = glm::vec3(x, y, z + 1) / (float) resolution;
		corners[5] = glm::vec3(x + 1, y, z + 1) / (float) resolution;
		corners[6] = glm::vec3(x, y + 1, z + 1) / (float) resolution;
		corners[7] = glm::vec3(x + 1, y + 1, z + 1) / (float) resolution;

		for (glm::vec3 &v : corners)
			v = v * (ext_max - ext_min) + ext_min;

		// Fill the data in the mesh
		uint32_t base = mesh.vertices.size();

		for (const glm::vec3 &v : corners)
			mesh.vertices.push_back(v);

		mesh.triangles.push_back({ base + 0, base + 1, base + 2 });
		mesh.triangles.push_back({ base + 1, base + 3, base + 2 });
		mesh.triangles.push_back({ base + 0, base + 2, base + 4 });
		mesh.triangles.push_back({ base + 2, base + 6, base + 4 });
		mesh.triangles.push_back({ base + 0, base + 4, base + 1 });
		mesh.triangles.push_back({ base + 1, base + 4, base + 5 });
		mesh.triangles.push_back({ base + 1, base + 5, base + 3 });
		mesh.triangles.push_back({ base + 3, base + 5, base + 7 });
		mesh.triangles.push_back({ base + 2, base + 3, base + 6 });
		mesh.triangles.push_back({ base + 3, base + 7, base + 6 });
		mesh.triangles.push_back({ base + 4, base + 6, base + 5 });
		mesh.triangles.push_back({ base + 5, base + 6, base + 7 });
	};

	// Create a mesh out of the voxels
	Mesh voxel_mesh;

	for (size_t i = 0; i < voxels.size(); i++) {
		if (voxels[i] == 0)
			continue;

		add_voxel(voxel_mesh, i);
	}

	voxel_mesh.normals.resize(voxel_mesh.vertices.size());
	voxel_mesh = deduplicate(voxel_mesh).first;

	printf("Voxel mesh: %u vertices, %u triangles\n", voxel_mesh.vertices.size(), voxel_mesh.triangles.size());

	// Adjaceny for the voxel elements
	std::unordered_map <uint32_t, std::vector<uint32_t>> voxel_adjacency;

	for (size_t i = 0; i < voxels.size(); i++) {
		if (voxels[i] == 0)
			continue;

		// Find all neighbors that are valid
		int dx[6] = { -1, 1, 0, 0, 0, 0 };
		int dy[6] = { 0, 0, -1, 1, 0, 0 };
		int dz[6] = { 0, 0, 0, 0, -1, 1 };

		int32_t x = i % resolution;
		int32_t y = (i / resolution) % resolution;
		int32_t z = i / (resolution * resolution);

		for (int j = 0; j < 6; j++) {
			int32_t nx = x + dx[j];
			int32_t ny = y + dy[j];
			int32_t nz = z + dz[j];

			if (nx < 0 || nx >= resolution || ny < 0 || ny >= resolution || nz < 0 || nz >= resolution)
				continue;

			int32_t nindex = nx + ny * resolution + nz * resolution * resolution;

			if (voxels[nindex] == 0)
				continue;

			voxel_adjacency[i].push_back(nindex);
		}
	}

	float adj_avg = 0.0f;
	for (const auto &p : voxel_adjacency)
		adj_avg += p.second.size();

	printf("Average adjacency: %f\n", adj_avg / (float) voxel_adjacency.size());

	// Remove voxels with only one neighbor
	for (auto it = voxel_adjacency.begin(); it != voxel_adjacency.end(); ) {
		if (it->second.size() <= 1) {
			printf("Removing voxel %u\n", it->first);
			it = voxel_adjacency.erase(it);
		} else
			it++;
	}

	if (voxel_adjacency.size() == 0) {
		printf("No voxels left after removing single-neighbor voxels\n");
		return -1;
	}

	// Find all the connected components and select the largest one
	std::vector <std::unordered_set <uint32_t>> components;

	std::unordered_set <uint32_t> remaining;
	for (auto [v, _] : voxel_adjacency)
		remaining.insert(v);

	while (remaining.size() > 0) {
		uint32_t seed = *remaining.begin();

		std::unordered_set <uint32_t> component;
		std::queue <uint32_t> queue;

		queue.push(seed);
		component.insert(seed);
		remaining.erase(seed);

		while (!queue.empty()) {
			uint32_t v = queue.front();
			queue.pop();

			for (uint32_t n : voxel_adjacency[v]) {
				if (remaining.count(n) == 0)
					continue;

				queue.push(n);
				component.insert(n);
				remaining.erase(n);
			}
		}

		components.push_back(component);
	}

	printf("Found %u components\n", components.size());
	for (size_t i = 0; i < components.size(); i++)
		printf("  component %u: %u voxels\n", i, components[i].size());

	std::sort(components.begin(), components.end(),
		[](const auto &a, const auto &b) {
			return a.size() > b.size();
		}
	);

	std::unordered_set <uint32_t> working_set = components[0];

	// Extract the voxel mesh
	Mesh reduced_mesh;
	for (size_t i = 0; i < voxels.size(); i++) {
		if (working_set.count(i) == 0)
			continue;

		add_voxel(reduced_mesh, i);
	}

	reduced_mesh.normals.resize(reduced_mesh.vertices.size());
	reduced_mesh = deduplicate(reduced_mesh).first;

	// Remove all the duplicate triangles
	auto triangle_eq = [](const Triangle &t1, const Triangle &t2) {
		// Any permutation of the vertices is considered equal
		bool perm1 = (t1[0] == t2[0] && t1[1] == t2[1] && t1[2] == t2[2]);
		bool perm2 = (t1[0] == t2[0] && t1[1] == t2[2] && t1[2] == t2[1]);
		bool perm3 = (t1[0] == t2[1] && t1[1] == t2[0] && t1[2] == t2[2]);
		bool perm4 = (t1[0] == t2[1] && t1[1] == t2[2] && t1[2] == t2[0]);
		bool perm5 = (t1[0] == t2[2] && t1[1] == t2[0] && t1[2] == t2[1]);
		bool perm6 = (t1[0] == t2[2] && t1[1] == t2[1] && t1[2] == t2[0]);
		return perm1 || perm2 || perm3 || perm4 || perm5 || perm6;
	};

	auto triangle_hash = [](const Triangle &t) {
		std::hash <uint32_t> hasher;
		return hasher(t[0]) ^ hasher(t[1]) ^ hasher(t[2]);
	};

	std::unordered_set <Triangle, decltype(triangle_hash), decltype(triangle_eq)> triangles(0, triangle_hash, triangle_eq);
	std::unordered_set <Triangle, decltype(triangle_hash), decltype(triangle_eq)> triangles_to_remove(0, triangle_hash, triangle_eq);

	for (const auto &t : reduced_mesh.triangles) {
		if (triangles.count(t) > 0)
			triangles_to_remove.insert(t);
		else
			triangles.insert(t);
	}

	printf("Number of triangles to remove: %u\n", triangles_to_remove.size());

	Mesh skin_mesh = reduced_mesh;
	
	std::vector <Triangle> new_triangles;
	for (const auto &t : skin_mesh.triangles) {
		if (triangles_to_remove.count(t) == 0)
			new_triangles.push_back(t);
	}

	skin_mesh.triangles = new_triangles;
	skin_mesh = deduplicate(skin_mesh).first;

	// glm::vec3 color_wheel[] = {
	// 	{ 0.910, 0.490, 0.490 },
	// 	{ 0.910, 0.700, 0.490 },
	// 	{ 0.910, 0.910, 0.490 },
	// 	{ 0.700, 0.910, 0.490 },
	// 	{ 0.490, 0.910, 0.490 },
	// 	{ 0.490, 0.910, 0.700 },
	// 	{ 0.490, 0.910, 0.910 },
	// 	{ 0.490, 0.700, 0.910 },
	// 	{ 0.490, 0.490, 0.910 },
	// 	{ 0.700, 0.490, 0.910 },
	// 	{ 0.910, 0.490, 0.910 },
	// 	{ 0.910, 0.490, 0.700 }
	// };

	// Create the acceleration structure for point queries on the target mesh
	cas_grid cas = cas_grid(mesh, 128);
	cas.precache_query(skin_mesh.vertices);
	cas.precache_device();

	printf("CAS grid has been preloaded\n");

	// Precomputation setup for subdivision
	auto [vgraph, egraph, dual] = build_graphs(skin_mesh);

	// Find all complexes (4-cycles) with edges parallel to canonical axes
	using complex = std::array <uint32_t, 4>;

	auto complex_eq = [](const complex &c1, const complex &c2) {
		// Equal if they are the same under rotation
		for (size_t i = 0; i < 4; i++) {
			bool found = true;

			for (size_t j = 0; j < 4; j++) {
				if (c1[j] != c2[(i + j) % 4]) {
					found = false;
					break;
				}
			}

			if (found)
				return true;
		}

		// Also check the reverse
		for (size_t i = 0; i < 4; i++) {
			bool found = true;

			for (size_t j = 0; j < 4; j++) {
				if (c1[j] != c2[(i - j + 4) % 4]) {
					found = false;
					break;
				}
			}

			if (found)
				return true;
		}

		// Also the same if middle elements are swapped
		// if (c1[0] == c2[0] && c1[1] == c2[2] && c1[2] == c2[1] && c1[3] == c2[3])
		// 	return true;

		printf("Distinct complex: (%u, %u, %u, %u) vs (%u, %u, %u, %u)\n", c1[0], c1[1], c1[2], c1[3], c2[0], c2[1], c2[2], c2[3]);

		return false;
	};

	auto complex_hash = [](const complex &c) {
		std::hash <uint32_t> hasher;
		return hasher(c[0]) ^ hasher(c[1]) ^ hasher(c[2]) ^ hasher(c[3]);
	};

	std::unordered_set <complex, decltype(complex_hash), decltype(complex_eq)> complexes(0, complex_hash, complex_eq);

	auto simplified_graph = vgraph;

	// Remove non-axial edges
	for (auto &[v, adj] : simplified_graph) {
		glm::vec3 vv = skin_mesh.vertices[v];

		for (auto it = adj.begin(); it != adj.end(); ) {
			glm::vec3 va = skin_mesh.vertices[*it];
			glm::vec3 e = va - vv;

			uint32_t x_zero = std::abs(e.x) < 1e-6f;
			uint32_t y_zero = std::abs(e.y) < 1e-6f;
			uint32_t z_zero = std::abs(e.z) < 1e-6f;

			if (x_zero + y_zero + z_zero < 2)
				it = adj.erase(it);
			else
				it++;
		}
	}

	for (size_t v = 0; v < skin_mesh.vertices.size(); v++) {
		using cycle = std::array <uint32_t, 5>;
		using cycle_state = std::tuple <cycle, uint32_t, uint32_t>;

		std::vector <cycle> potential;
		std::queue <cycle_state> q;
		q.push({ { 0, 0, 0, 0, 0 }, v, 0 });

		while (!q.empty()) {
			// printf("Queue size: %zu\n", q.size());
			auto [c, v, depth] = q.front();
			q.pop();

			if (depth == 5) {
				potential.push_back(c);
				continue;
			}

			c[depth] = v;
			for (uint32_t e : simplified_graph[v]) {
				// Fill the next index in the complex
				cycle_state next = { c, e, depth + 1 };
				q.push(next);
			}
		}

		// Find all cycles starting and ending with the same vertex
		for (const auto &c : potential) {
			if (c[0] != c[4])
				continue;

			// Cannot have duplicates in the middle
			bool ok = true;
			for (size_t i = 0; i < 4; i++) {
				for (size_t j = i + 1; j < 4; j++) {
					if (c[i] == c[j]) {
						ok = false;
						break;
					}
				}

				if (!ok)
					break;
			}

			if (!ok)
				continue;

			complex c2 = { c[0], c[1], c[2], c[3] };
			complexes.insert(c2);
		}
	}

	// Remove interior complexes (midpoint is inside the mesh)
	uint32_t removed = 0;
	for (auto it = complexes.begin(); it != complexes.end(); ) {
		const auto &c = *it;
		const auto &vertices = skin_mesh.vertices;
		glm::vec3 midpoint = 0.25f * (vertices[c[0]] + vertices[c[1]] + vertices[c[2]] + vertices[c[3]]);

		auto [d, v] = closest_point(skin_mesh, midpoint);
		if (d > 1e-3f) {
			printf("Removing complex with midpoint outside the mesh: %f\n", d);
			it = complexes.erase(it);
			removed++;
		} else {
			it++;
		}
	}

	printf("Number of (unique) complexes: %zu (%zu removed)\n", complexes.size(), removed);

	// Linearize these canonical complexes and record their subdivision state
	struct subdivision_state {
		uint32_t size;
		std::vector <uint32_t> vertices;

		std::vector <glm::vec3> upscale(const Mesh &ref) const {
			assert(vertices.size() == size * size);

			std::vector <glm::vec3> base;
			base.reserve(vertices.size());

			for (uint32_t v : vertices)
				base.push_back(ref.vertices[v]);

			std::vector <glm::vec3> result;

			uint32_t new_size = 2 * size;
			result.resize(new_size * new_size);

			// Bilerp each new vertex
			for (uint32_t i = 0; i < new_size; i++) {
				for (uint32_t j = 0; j < new_size; j++) {
					float u = (float) i / (new_size - 1);
					float v = (float) j / (new_size - 1);

					float lu = u * (size - 1);
					float lv = v * (size - 1);

					uint32_t u0 = std::floor(lu);
					uint32_t u1 = std::ceil(lu);

					uint32_t v0 = std::floor(lv);
					uint32_t v1 = std::ceil(lv);

					glm::vec3 p00 = base[u0 * size + v0];
					glm::vec3 p10 = base[u1 * size + v0];
					glm::vec3 p01 = base[u0 * size + v1];
					glm::vec3 p11 = base[u1 * size + v1];

					lu -= u0;
					lv -= v0;

					glm::vec3 p = p00 * (1.0f - lu) * (1.0f - lv) +
					              p10 * lu * (1.0f - lv) +
					              p01 * (1.0f - lu) * lv +
					              p11 * lu * lv;

					result[i * new_size + j] = p;
				}
			}

			return result;
		}
	};

	struct srnm_optimizer {
		Mesh ref;

		uint32_t size;

		std::vector <complex> complexes;
		std::vector <subdivision_state> subdivision_states;

		srnm_optimizer(const Mesh &ref_, const std::vector <complex> &complexes_)
				: ref(ref_), complexes(complexes_), size(2) {
			subdivision_states.resize(complexes.size());

			for (size_t i = 0; i < complexes.size(); i++) {
				complex c = complexes[i];
				subdivision_state &s = subdivision_states[i];

				s.size = 2;
				s.vertices.resize(4);

				s.vertices[0] = c[0];
				s.vertices[1] = c[1];
				s.vertices[2] = c[2];
				s.vertices[3] = c[3];
			}
		}

		srnm_optimizer(const Mesh &ref_, const std::vector <complex> &complexes_, const std::vector <subdivision_state> &subdivision_states_, uint32_t size_)
				: ref(ref_), complexes(complexes_), subdivision_states(subdivision_states_), size(size_) {}

		srnm_optimizer upscale() const {
			Mesh new_ref;

			uint32_t new_size = 2 * size;
			printf("Upscaling from %d to %d\n", size, new_size);

			std::vector <complex> new_complexes;
			std::vector <subdivision_state> new_subdivision_states;

			for (const auto &sdv : subdivision_states) {
				auto new_vertices = sdv.upscale(ref);

				subdivision_state new_sdv;
				new_sdv.size = new_size;

				// Fill the vertices
				uint32_t offset = new_ref.vertices.size();
				for (const auto &v : new_vertices) {
					new_sdv.vertices.push_back(new_ref.vertices.size());
					new_ref.vertices.push_back(v);
					new_ref.normals.push_back(glm::vec3(0.0f));
				}

				// Fill the triangles
				for (uint32_t i = 0; i < new_size - 1; i++) {
					for (uint32_t j = 0; j < new_size - 1; j++) {
						uint32_t i00 = i * new_size + j;
						uint32_t i10 = (i + 1) * new_size + j;
						uint32_t i01 = i * new_size + j + 1;
						uint32_t i11 = (i + 1) * new_size + j + 1;

						Triangle t1 { offset + i00, offset + i10, offset + i11 };
						Triangle t2 { offset + i00, offset + i01, offset + i11 };

						new_ref.triangles.push_back(t1);
						new_ref.triangles.push_back(t2);
					}
				}

				complex new_c;
				new_c[0] = new_sdv.vertices[0];
				new_c[1] = new_sdv.vertices[new_size - 1];
				new_c[2] = new_sdv.vertices[new_size * (new_size - 1)];
				new_c[3] = new_sdv.vertices[new_size * new_size - 1];

				new_complexes.push_back(new_c);
				new_subdivision_states.push_back(new_sdv);
			}

			// TODO: deduplicate vertices
			auto res = deduplicate(new_ref, 1e-6f);
			printf("Before/after: %d/%d\n", new_ref.vertices.size(), res.first.vertices.size());
			new_ref = res.first;

			auto remap = res.second;
			for (auto &c : new_complexes) {
				for (auto &v : c)
					v = remap[v];
			}

			for (auto &sdv : new_subdivision_states) {
				for (auto &v : sdv.vertices)
					v = remap[v];
			}

			return srnm_optimizer(new_ref, new_complexes, new_subdivision_states, new_size);
		}
	};

	// Create the initial optimization state
	std::vector <complex> complexes_linearized(complexes.begin(), complexes.end());

	// Reorder the complexes from cyclic to array format
	for (auto &c : complexes_linearized)
		std::swap(c[2], c[3]);

	printf("Complexes: %u\n", complexes_linearized.size());
	for (const auto &c : complexes_linearized)
		printf("  %u %u %u %u\n", c[0], c[1], c[2], c[3]);

	srnm_optimizer optimizer(skin_mesh, complexes_linearized);
	printf("Optimizer ref details: %u vertices, %u faces\n", optimizer.ref.vertices.size(), optimizer.ref.triangles.size());

	Viewer viewer;
	viewer.add("mesh", mesh, Viewer::Mode::Shaded);
	viewer.add("ref", optimizer.ref, Viewer::Mode::Wireframe);
	viewer.add("skin", skin_mesh, Viewer::Mode::Wireframe);

	viewer.ref("ref")->color = { 0.3f, 0.3f, 0.8f };
	viewer.ref("skin")->color = { 0.3f, 0.8f, 0.3f };

	// Optimization (while rendering)
	closest_point_kinfo kinfo;
	kinfo.point_count = optimizer.ref.vertices.size();

	cudaMalloc(&kinfo.points, sizeof(glm::vec3) * optimizer.ref.vertices.size());
	cudaMalloc(&kinfo.closest, sizeof(glm::vec3) * optimizer.ref.vertices.size());

	while (true) {
		GLFWwindow *window = viewer.window->handle;

		// Check window close state
		glfwPollEvents();
		if (glfwWindowShouldClose(window))
			break;

		// Other inputs
		if (glfwGetKey(window, GLFW_KEY_EQUAL) == GLFW_PRESS)
			viewer.camera.radius *= 1.1f;
		else if (glfwGetKey(window, GLFW_KEY_MINUS) == GLFW_PRESS)
			viewer.camera.radius /= 1.1f;

		if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
			viewer.camera.theta -= 0.1f;
		else if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
			viewer.camera.theta += 0.1f;

		if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
			viewer.camera.phi += 0.1f;
		else if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
			viewer.camera.phi -= 0.1f;

		viewer.camera.phi = glm::clamp(viewer.camera.phi, float(-M_PI / 2.0f), float(M_PI / 2.0f));

		// Refine on enter
		if (glfwGetKey(window, GLFW_KEY_ENTER) == GLFW_PRESS) {
			optimizer = optimizer.upscale();
			viewer.replace("ref", optimizer.ref);
			printf("Upscaled ref details: %u vertices, %u faces\n", optimizer.ref.vertices.size(), optimizer.ref.triangles.size());

			cudaFree(kinfo.points);
			cudaFree(kinfo.closest);

			kinfo.point_count = optimizer.ref.vertices.size();
			cudaMalloc(&kinfo.points, sizeof(glm::vec3) * optimizer.ref.vertices.size());
			cudaMalloc(&kinfo.closest, sizeof(glm::vec3) * optimizer.ref.vertices.size());
		}

		// Optimize the skin mesh around the target (original) mesh
		std::vector <glm::vec3> host_closest(optimizer.ref.vertices.size());

		// TODO: memcpy async while caching
		cudaMemcpy(kinfo.points, optimizer.ref.vertices.data(),
				sizeof(glm::vec3) * optimizer.ref.vertices.size(),
				cudaMemcpyHostToDevice);

		bool updated = cas.precache_query(optimizer.ref.vertices);
		if (updated)
			cas.precache_device();

		// cas.query(optimizer.ref.vertices, host_closest);
		cas.query_device(kinfo);

		cudaMemcpy(host_closest.data(), kinfo.closest,
				sizeof(glm::vec3) * optimizer.ref.vertices.size(),
				cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();

		std::vector <glm::vec3> gradients;
		gradients.reserve(optimizer.ref.vertices.size());

		for (uint32_t i = 0; i < optimizer.ref.vertices.size(); i++) {
			const glm::vec3 &v = optimizer.ref.vertices[i];
			const glm::vec3 &w = host_closest[i];
			gradients[i] = (w - v);
		}

		// TODO: adam optimizer...
		// TODO: apply gradients in cuda, using the imported vulkan buffer
		for (uint32_t i = 0; i < optimizer.ref.vertices.size(); i++)
			optimizer.ref.vertices[i] += 0.01f * gradients[i];

		viewer.refresh("ref", optimizer.ref);
		viewer.render();
	}

	viewer.destroy();

	// Save the state of the subdivison complexes
	std::filesystem::path sdv_complexes = path.stem().string() + ".sdv";

	std::ofstream fout(sdv_complexes, std::ios::binary);

	std::string source = std::filesystem::absolute(path).string();
	uint32_t source_size = source.size();

	fout.write((char *) &source_size, sizeof(uint32_t));
	fout.write(source.c_str(), source_size);

	std::unordered_map <uint32_t, uint32_t> complex_remap;
	auto add_unique_corners = [&](uint32_t c) -> uint32_t {
		if (complex_remap.count(c))
			return complex_remap[c];

		uint32_t csize = complex_remap.size();
		complex_remap[c] = csize;
		return csize;
	};

	std::vector <complex> normalized_complexes;
	for (uint32_t i = 0; i < optimizer.complexes.size(); i++) {
		auto [c0, c1, c2, c3] = optimizer.complexes[i];
		c0 = add_unique_corners(c0);
		c1 = add_unique_corners(c1);
		c2 = add_unique_corners(c2);
		c3 = add_unique_corners(c3);
		normalized_complexes.push_back({ c0, c1, c2, c3 });
	}

	printf("Normalized %u complexes\n", normalized_complexes.size());
	for (uint32_t i = 0; i < normalized_complexes.size(); i++) {
		auto [c0, c1, c2, c3] = normalized_complexes[i];
		auto [o0, o1, o2, o3] = optimizer.complexes[i];
		printf("  %u: %u %u %u %u (was %u %u %u %u)\n", i, c0, c1, c2, c3, o0, o1, o2, o3);
	}

	uint32_t corner_count = complex_remap.size();
	fout.write((char *) &corner_count, sizeof(uint32_t));

	for (auto [c, csize] : complex_remap) {
		glm::vec3 v = optimizer.ref.vertices[c];
		fout.write((char *) &v, sizeof(glm::vec3));
	}

	uint32_t complex_count = normalized_complexes.size();
	fout.write((char *) &complex_count, sizeof(uint32_t));

	for (const complex &c : normalized_complexes)
		fout.write((char *) &c, sizeof(complex));

	// Write the vertex data for all the subdivison complexes
	for (uint32_t i = 0; i < optimizer.subdivision_states.size(); i++) {
		const subdivision_state &sdv = optimizer.subdivision_states[i];

		uint32_t size = sdv.size;
		fout.write((char *) &size, sizeof(uint32_t));

		std::vector <glm::vec3> vertices;
		for (size_t vi : sdv.vertices)
			vertices.push_back(optimizer.ref.vertices[vi]);

		uint32_t vertex_count = vertices.size();
		fout.write((char *) &vertex_count, sizeof(uint32_t));
		fout.write((char *) vertices.data(), sizeof(glm::vec3) * vertices.size());
	}

	fout.close();

	printf("Saved subdivison complexes to %s\n", sdv_complexes.string().c_str());
}
