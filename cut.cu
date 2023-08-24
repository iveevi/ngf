#include <array>
#include <chrono>
#include <unordered_set>

// #include <CGAL/Polygon_mesh_processing/corefinement.h>

// #include <polyscope/polyscope.h>
// #include <polyscope/surface_mesh.h>
// #include <polyscope/point_cloud.h>

#include <indicators/block_progress_bar.hpp>

#define MESH_LOAD_SAVE
#include "mesh.hpp"
#include "casdf/casdf.hpp"
#include "microlog.h"
#include "viewer.hpp"

float chamfer(const Mesh &ref, const Mesh &source)
{
	float sum = 0;

	// Normalization constants
	float max_extent = 0.0f;
	glm::vec3 min { std::numeric_limits <float> ::max() };
	glm::vec3 max { -std::numeric_limits <float> ::max() };

	for (const glm::vec3 &v : ref.vertices) {
		min = glm::min(min, v);
		max = glm::max(max, v);
	}

	max_extent = glm::distance(min, max);

	// Compute the distance matrix
	uint32_t rv = ref.vertices.size();
	uint32_t sv = source.vertices.size();

	std::vector <float> distance_matrix(rv * sv);
	for (uint32_t i = 0; i < rv; i++) {
		#pragma omp parallel for
		for (uint32_t j = 0; j < sv; j++) {
			float d = glm::distance(ref.vertices[i], source.vertices[j]);
			distance_matrix[i * sv + j] = d/max_extent;
		}
	}

	// Get mins from each row and column
	#pragma omp parallel for reduction(+:sum)
	for (uint32_t i = 0; i < rv; i++) {
		float min = std::numeric_limits <float> ::max();
		for (uint32_t j = 0; j < sv; j++)
			min = std::min(min, distance_matrix[i * sv + j]);
		sum += min/float(rv);
	}

	#pragma omp parallel for reduction(+:sum)
	for (uint32_t i = 0; i < sv; i++) {
		float min = std::numeric_limits <float> ::max();
		for (uint32_t j = 0; j < rv; j++)
			min = std::min(min, distance_matrix[j * sv + i]);
		sum += min/float(sv);
	}

	return sum;
}

std::array <uint32_t, 4> array_order(const Triangle &a, const Triangle &b)
{
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

	return { v0, s0, s1, v1 };
}

struct quad {
	uint32_t t0;
	uint32_t t1;
	std::array <uint32_t, 4> vertices;
};

void repair(const Mesh &ref, Mesh &opt, cas_grid &cas)
{
	// Then fix the normals
	for (uint32_t i = 0; i < opt.triangles.size(); i++) {
		Triangle &t = opt.triangles[i];

		glm::vec3 p0 = opt.vertices[t[0]];
		glm::vec3 p1 = opt.vertices[t[1]];
		glm::vec3 p2 = opt.vertices[t[2]];

		glm::vec3 mid = (p0 + p1 + p2) / 3.0f;

		cas.precache_query(mid);
		auto [a, b, c, tid] = cas.query(mid);

		const Triangle &t2 = ref.triangles[tid];

		glm::vec3 v0 = ref.vertices[t2[0]];
		glm::vec3 v1 = ref.vertices[t2[1]];
		glm::vec3 v2 = ref.vertices[t2[2]];

		glm::vec3 nexp = glm::normalize(glm::cross(v1 - v0, v2 - v0));
		glm::vec3 nact = glm::normalize(glm::cross(p1 - p0, p2 - p0));

		if (glm::dot(nexp, nact) < 0.0f)
			std::swap(t[1], t[2]);
	}
}

int main(int argc, char *argv[])
{
	// Load arguments
	if (argc != 2) {
		printf("Usage: %s <filename>\n", argv[0]);
		return 1;
	}

	std::filesystem::path path = std::filesystem::weakly_canonical(argv[1]);

	// TODO: refactor everything to lower case

	// Load data
	uint32_t length = 0;
	uint32_t nv = 0;
	uint32_t nt = 0;
	uint32_t nq = 0;

	FILE *file = fopen(path.c_str(), "rb");
	ulog_assert(file, "opt", "Failed to open file: %s\n", path.c_str());

	fread(&length, sizeof(uint32_t), 1, file);

	std::string target_file;
	target_file.resize(length);

	fread(target_file.data(), sizeof(char), length, file);

	printf("Original source file: %s\n", target_file.c_str());

	Mesh target = load_mesh(target_file);
	target = deduplicate(target).first;

	fread(&nv, sizeof(uint32_t), 1, file);
	fread(&nt, sizeof(uint32_t), 1, file);

	printf("Decimated mesh: %u vertices, %u triangles\n", nv, nt);

	Mesh source;
	source.vertices.resize(nv);
	source.normals.resize(nv);
	source.triangles.resize(nt);

	fread(source.vertices.data(), sizeof(float), 3 * nv, file);
	fread(source.triangles.data(), sizeof(uint32_t), 3 * nt, file);

	fread(&nq, sizeof(uint32_t), 1, file);
	printf("Quad count: %u\n", nq);

	std::vector <quad> quads;
	quads.resize(nq);

	for (uint32_t i = 0; i < nq; ++i) {
		fread(&quads[i].t0, sizeof(uint32_t), 1, file);
		fread(&quads[i].t1, sizeof(uint32_t), 1, file);

		const Triangle &t0 = source.triangles[quads[i].t0];
		const Triangle &t1 = source.triangles[quads[i].t1];

		quads[i].vertices = array_order(t0, t1);
	}

	// TODO: check disjointness and coverage
	printf("Loaded %u quads\n", nq);
	// for (uint32_t i = 0; i < nq; i++)
	// 	printf("quads: %u & %u\n", quads[i].t0, quads[i].t1);

	// Optimizing the source mesh
	Mesh opt = source;

	// 1. Obtain softer normals by smoothing the mesh
	auto smooth = [](const Mesh &ref, float factor) {
		Mesh result = ref;

		auto [vgraph, egraph, dual] = build_graphs(result);

		#pragma omp parallel for
		for (uint32_t i = 0 ; i < vgraph.size(); i++) {
			const auto &adj = vgraph[i];

			glm::vec3 sum = glm::vec3(0.0f);
			for (uint32_t j : adj)
				sum += result.vertices[j];
			sum /= (float) adj.size();

			result.vertices[i] = result.vertices[i] * (1.0f - factor) + sum * factor;
		}

		return result;
	};

	Mesh target_smoothed = target;
	// for (int i = 0; i < 10; i++)
	// 	target_smoothed = smooth(target_smoothed, 0.9f);
	// recompute_normals(target_smoothed);

	// 2. Use sdf to optimize the source mesh
	cas_grid cas(target, 128);

	std::vector <glm::vec3> gradients(nv);

	std::vector <glm::vec3> closest(nv);
	std::vector <glm::vec3> bary(nv);
	std::vector <float> sdfs(nv);
	std::vector <uint32_t> indices(nv);
	
	closest_point_kinfo kinfo = closest_point_kinfo_alloc(opt.vertices.size());

	printf("Starting chamfer metric: %f\n", chamfer(opt, target));

	for (int i = 0; i < 1000; i++) {
		// printf("Iteration %d\n", i);
		const auto &queries = opt.vertices;

		// TODO: make a distinction between sdf and interior point...
		// cas.precache_query(queries);
		// cas.query(queries, closest, bary, sdfs, indices);
		
		if (cas.precache_query(opt.vertices))
			cas.precache_device();

		cudaMemcpy(kinfo.points, opt.vertices.data(), sizeof(glm::vec3) * kinfo.point_count, cudaMemcpyHostToDevice);

		cas.query_device(kinfo);
		cudaDeviceSynchronize();

		// cas.query(queries, closest, bary, sdfs, indices);
		cudaMemcpy(closest.data(), kinfo.closest, sizeof(glm::vec3) * kinfo.point_count, cudaMemcpyDeviceToHost);
		cudaMemcpy(bary.data(), kinfo.bary, sizeof(glm::vec3) * kinfo.point_count, cudaMemcpyDeviceToHost);
		// cudaMemcpy(sdfs.data(), kinfo.distnace, sizeof(float) * kinfo.point_count, cudaMemcpyDeviceToHost);
		cudaMemcpy(indices.data(), kinfo.triangles, sizeof(uint32_t) * kinfo.point_count, cudaMemcpyDeviceToHost);

		for (uint32_t j = 0; j < opt.vertices.size(); j++)
			gradients[j] = closest[j] - opt.vertices[j];

		// Push apart nearby points
		for (uint32_t j = 0; j < opt.vertices.size(); j++) {
			for (uint32_t k = j + 1; k < opt.vertices.size(); k++) {
				float dist = glm::distance(opt.vertices[j], opt.vertices[k]);
				if (dist < 0.01f) {
					gradients[j] += glm::normalize(opt.vertices[j] - opt.vertices[k]) * 0.1f;
					gradients[k] += glm::normalize(opt.vertices[k] - opt.vertices[j]) * 0.1f;
				}
			}
		}
		
		// Apply the gradients
		for (uint32_t j = 0; j < opt.vertices.size(); j++)
			opt.vertices[j] += gradients[j] * 0.1f;

		// repair(target, opt, quads, cas);
		// printf("Iteration %d\n", i);
	}

	printf("Final chamfer metric: %f\n", chamfer(opt, target));

	// Rediagonalize the quads using sdf
	for (auto &q : quads) {
		// Current diagonals are v0-v3 and v1-v2
		glm::vec3 v0 = opt.vertices[q.vertices[0]];
		glm::vec3 v1 = opt.vertices[q.vertices[1]];
		glm::vec3 v2 = opt.vertices[q.vertices[2]];
		glm::vec3 v3 = opt.vertices[q.vertices[3]];

		// Sample points on the diagonals
		constexpr uint32_t N = 10;

		glm::vec3 diag0[N];
		glm::vec3 diag1[N];

		for (uint32_t i = 0; i < N; i++) {
			float t = (float) i / (float) (N - 1);

			diag0[i] = v0 * (1.0f - t) + v3 * t;
			diag1[i] = v1 * (1.0f - t) + v2 * t;
		}

		std::vector <glm::vec3> all_queries;
		for (uint32_t i = 0; i < N; i++) {
			all_queries.push_back(diag0[i]);
			all_queries.push_back(diag1[i]);
		}

		cas.precache_query(all_queries);

		// Sum the sdfs
		float sdf0 = 0.0f;
		float sdf1 = 0.0f;

		for (uint32_t i = 0; i < N; i++) {
			auto [c0, b0, s0, idx0] = cas.query(diag0[i]);
			auto [c1, b1, s1, idx1] = cas.query(diag1[i]);

			sdf0 += s0;
			sdf1 += s1;
		}

		// Choose smaller sdf
		Triangle n0;
		Triangle n1;

		if (sdf0 < sdf1) {
			n0 = { q.vertices[0], q.vertices[1], q.vertices[3] };
			n1 = { q.vertices[0], q.vertices[3], q.vertices[2] };
		} else {
			n0 = { q.vertices[0], q.vertices[1], q.vertices[2] };
			n1 = { q.vertices[1], q.vertices[3], q.vertices[2] };
		}

		opt.triangles[q.t0] = n0;
		opt.triangles[q.t1] = n1;

		q.vertices = array_order(n0, n1);
	}

	repair(target, opt, cas);

	// Create cut points on the mesh for each edge
	std::unordered_set <Edge> edges;

	for (auto &q : quads) {
		edges.insert({ q.vertices[0], q.vertices[1] });
		edges.insert({ q.vertices[1], q.vertices[3] });
		edges.insert({ q.vertices[3], q.vertices[2] });
		edges.insert({ q.vertices[2], q.vertices[0] });
	}

	std::vector <glm::vec3> cut_points;

	for (const auto &[a, b] : edges) {
		glm::vec3 v0 = opt.vertices[a];
		glm::vec3 v1 = opt.vertices[b];

		constexpr uint32_t N = 100;
		std::vector <glm::vec3> samples(N);

		for (uint32_t i = 0; i < N; i++) {
			float t = (float) i / (float) (N - 1);
			samples[i] = v0 * (1.0f - t) + v1 * t;
		}

		cas.precache_query(samples);

		std::vector <glm::vec3> closest_points(N);
		std::vector <glm::vec3> closest_barys(N);
		std::vector <float> closest_sdfs(N);
		std::vector <uint32_t> closest_indices(N);

		cas.query(samples, closest_points, closest_barys, closest_sdfs, closest_indices);
		cut_points.insert(cut_points.end(), closest_points.begin(), closest_points.end());
	}

	std::vector <Mesh> cuboids;

	for (const auto &q : quads) {
		glm::vec3 v0 = opt.vertices[q.vertices[0]];
		glm::vec3 v1 = opt.vertices[q.vertices[1]];
		glm::vec3 v2 = opt.vertices[q.vertices[2]];
		glm::vec3 v3 = opt.vertices[q.vertices[3]];
		
		glm::vec3 n0 = opt.normals[q.vertices[0]];
		glm::vec3 n1 = opt.normals[q.vertices[1]];
		glm::vec3 n2 = opt.normals[q.vertices[2]];
		glm::vec3 n3 = opt.normals[q.vertices[3]];

		// Cuboid extending the quad in both normal directions
		constexpr float eps = 0.05f;

		Mesh cuboid;

		cuboid.vertices.push_back(v0 + 10.0f * n0 * eps);
		cuboid.vertices.push_back(v1 + 10.0f * n1 * eps);
		cuboid.vertices.push_back(v2 + 10.0f * n2 * eps);
		cuboid.vertices.push_back(v3 + 10.0f * n3 * eps);

		cuboid.vertices.push_back(v0 - n0 * eps);
		cuboid.vertices.push_back(v1 - n1 * eps);
		cuboid.vertices.push_back(v2 - n2 * eps);
		cuboid.vertices.push_back(v3 - n3 * eps);

		cuboid.triangles.push_back({ 0, 1, 2 });
		cuboid.triangles.push_back({ 0, 2, 3 });

		cuboid.triangles.push_back({ 4, 6, 5 });
		cuboid.triangles.push_back({ 4, 7, 6 });

		cuboid.triangles.push_back({ 0, 4, 5 });
		cuboid.triangles.push_back({ 0, 5, 1 });

		cuboid.triangles.push_back({ 1, 5, 6 });
		cuboid.triangles.push_back({ 1, 6, 2 });

		cuboid.triangles.push_back({ 2, 6, 7 });
		cuboid.triangles.push_back({ 2, 7, 3 });

		cuboid.triangles.push_back({ 3, 7, 4 });
		cuboid.triangles.push_back({ 3, 4, 0 });

		recompute_normals(cuboid);
		cuboids.push_back(cuboid);
		break;
	}

	// Visualize
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

	std::vector <glm::vec3> opt_face_colors;
	opt_face_colors.resize(opt.triangles.size(), { 0.0f, 0.0f, 0.0f });

	{
		uint32_t color_index = 0;
		for (const auto &q : quads) {
			const glm::vec3 &color = color_wheel[color_index++ % 12];

			opt_face_colors[q.t0] = color;
			opt_face_colors[q.t1] = color;
		}
	}

	Viewer viewer;

	viewer.add("source", source, Viewer::Mode::Shaded);
	viewer.add("opt", opt, Viewer::Mode::Shaded);
	viewer.add("target", target, Viewer::Mode::Shaded);

	for (uint32_t i = 0; i < cuboids.size(); i++) {
		std::string name = "cuboid_" + std::to_string(i);
		viewer.add(name, cuboids[i], Viewer::Mode::Wireframe);
		viewer.ref(name)->color = color_wheel[i % 12];
	}

	while (true) {
		glfwPollEvents();
		if (glfwWindowShouldClose(viewer.window->handle))
			break;

		viewer.render();
	}

	// namespace ps = polyscope;
	// ps::init();
	// 
	// auto sm = ps::registerSurfaceMesh("source", source.vertices, source.triangles);
	// auto om = ps::registerSurfaceMesh("opt", opt.vertices, opt.triangles);
	//
	// ps::registerSurfaceMesh("target", target.vertices, target.triangles);
	// // ps::registerSurfaceMesh("target_smoothed", target_smoothed.vertices, target_smoothed.triangles);
	// 
	// ps::registerSurfaceMesh("cuboids", cuboids.vertices, cuboids.triangles);
	//
	// // ps::registerPointCloud("cut_points", cut_points);
	//
	// om->addFaceColorQuantity("face_colors", opt_face_colors);
	//
	// ps::show();
}
