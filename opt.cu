#include <array>
#include <chrono>
#include <cstdint>
#include <unordered_set>

#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>

#include <littlevk/littlevk.hpp>

#include <indicators/block_progress_bar.hpp>

#define MESH_LOAD_SAVE
#include "mesh.hpp"
#include "casdf/casdf.hpp"
#include "microlog.h"

enum render_mode {
	flat_shading,
	wireframe,
	count
};

struct render_ref {
	littlevk::Buffer vertices;
	littlevk::Buffer indices;
	glm::vec3 color;
	render_mode mode;
};

struct cluter_refs {
	std::vector <render_ref *> refs;
	render_mode mode;
};

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

std::vector <uint32_t> array_order(const Triangle &a, const Triangle &b)
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

	// return { v0, s0, v1, s1 };
	return { v0, s0, s1, v1 };
}

struct quad {
	uint32_t size;
	std::vector <uint32_t> triangles;
	std::vector <uint32_t> vertices;
};

void save(const Mesh &opt, const std::vector <quad> &quads, const std::string target_file, const std::filesystem::path &path)
{
	std::ofstream fout(path, std::ios::binary);

	uint32_t target_file_size = target_file.size();
	fout.write((char *) &target_file_size, sizeof(uint32_t));
	fout.write((char *) target_file.data(), target_file_size);

	std::vector <std::array <uint32_t, 4>> quad_corners;
	for (const quad &q : quads) {
		quad_corners.push_back({
			q.vertices[0],
			q.vertices[q.size - 1],
			q.vertices[q.size * (q.size - 1)],
			q.vertices[q.size * q.size - 1],
		});

		glm::vec3 c0 = opt.vertices[q.vertices[0]];
		glm::vec3 c1 = opt.vertices[q.vertices[q.size - 1]];
		glm::vec3 c2 = opt.vertices[q.vertices[q.size * (q.size - 1)]];
		glm::vec3 c3 = opt.vertices[q.vertices[q.size * q.size - 1]];

		printf("quad: (%f, %f, %f) (%f, %f, %f) (%f, %f, %f) (%f, %f, %f)\n",
			c0.x, c0.y, c0.z,
			c1.x, c1.y, c1.z,
			c2.x, c2.y, c2.z,
			c3.x, c3.y, c3.z);
	}

	std::unordered_map <uint32_t, uint32_t> corner_map;

	auto add_unique_corners = [&](uint32_t c) -> uint32_t {
		if (corner_map.count(c))
			return corner_map[c];

		uint32_t csize = corner_map.size();
		corner_map[c] = csize;
		return csize;
	};
		
	std::vector <uint32_t> normalized_complexes;

	for (const std::array <uint32_t, 4> &c : quad_corners) {
		normalized_complexes.push_back(add_unique_corners(c[0]));
		normalized_complexes.push_back(add_unique_corners(c[1]));
		normalized_complexes.push_back(add_unique_corners(c[2]));
		normalized_complexes.push_back(add_unique_corners(c[3]));
	}

	printf("Normalized complexes: %lu\n", normalized_complexes.size()/12);
	for (uint32_t i = 0; i < normalized_complexes.size(); i += 4)
		printf("%u %u %u %u\n", normalized_complexes[i], normalized_complexes[i + 1], normalized_complexes[i + 2], normalized_complexes[i + 3]);

	std::vector <glm::vec3> corners;
	corners.resize(corner_map.size());

	// std::vector <uint32_t> corner_indices;
	for (const auto &p : corner_map) {
		// corners.push_back(opt.vertices[p.first]);
		corners[p.second] = opt.vertices[p.first];
		// printf("%u: (%f, %f, %f)\n", p.second, corners.back().x, corners.back().y, corners.back().z);
	}
	
	printf("Corner count: %u\n", corner_map.size());
	for (uint32_t i = 0; i < corners.size(); i++)
		printf("%u: (%f, %f, %f)\n", i, corners[i].x, corners[i].y, corners[i].z);

	uint32_t corner_count = corner_map.size();
	fout.write((char *) &corner_count, sizeof(uint32_t));
	fout.write((char *) corners.data(), corners.size() * sizeof(glm::vec3));

	uint32_t quad_count = quads.size();
	fout.write((char *) &quad_count, sizeof(uint32_t));
	fout.write((char *) normalized_complexes.data(), normalized_complexes.size() * sizeof(uint32_t));

	for (const auto &q : quads) {
		uint32_t size = q.size;
		fout.write((char *) &size, sizeof(uint32_t));

		std::vector <glm::vec3> vertices;
		for (uint32_t v : q.vertices)
			vertices.push_back(opt.vertices[v]);

		uint32_t vertex_count = vertices.size();
		fout.write((char *) &vertex_count, sizeof(uint32_t));
		fout.write((char *) vertices.data(), vertices.size() * sizeof(glm::vec3));
	}
}

std::vector <glm::vec3> upscale(const Mesh &ref, const quad &q)
{
	ULOG_ASSERT(q.vertices.size() == q.size * q.size);

	std::vector <glm::vec3> base;
	base.reserve(q.vertices.size());

	for (uint32_t v : q.vertices)
		base.push_back(ref.vertices[v]);

	std::vector <glm::vec3> result;

	uint32_t new_size = 2 * q.size;
	result.resize(new_size * new_size);

	// Bilerp each new vertex
	for (uint32_t i = 0; i < new_size; i++) {
		for (uint32_t j = 0; j < new_size; j++) {
			float u = (float) i / (new_size - 1);
			float v = (float) j / (new_size - 1);

			float lu = u * (q.size - 1);
			float lv = v * (q.size - 1);

			uint32_t u0 = std::floor(lu);
			uint32_t u1 = std::ceil(lu);

			uint32_t v0 = std::floor(lv);
			uint32_t v1 = std::ceil(lv);

			glm::vec3 p00 = base[u0 * q.size + v0];
			glm::vec3 p10 = base[u1 * q.size + v0];
			glm::vec3 p01 = base[u0 * q.size + v1];
			glm::vec3 p11 = base[u1 * q.size + v1];

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

std::pair <Mesh, std::vector <quad>> upscale(const Mesh &ref, const std::vector <quad> &quads)
{
	Mesh new_ref;
	std::vector <quad> new_quads;

	uint32_t new_size = 2 * quads[0].size;
	for (const auto &q : quads) {
		auto new_vertices = upscale(ref, q);

		quad new_quad;
		new_quad.size = new_size;

		// Fill the vertices
		uint32_t offset = new_ref.vertices.size();
		for (const auto &v : new_vertices) {
			new_quad.vertices.push_back(new_ref.vertices.size());
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
				Triangle t2 { offset + i00, offset + i11, offset + i01 };

				new_quad.triangles.push_back(new_ref.triangles.size());
				new_quad.triangles.push_back(new_ref.triangles.size() + 1);

				new_ref.triangles.push_back(t1);
				new_ref.triangles.push_back(t2);
			}
		}

		new_quads.push_back(new_quad);
	}

	auto res = deduplicate(new_ref, 1e-6f);

	for (auto &q: new_quads) {
		for (auto &v : q.vertices)
			v = res.second[v];
	}

	return { res.first, new_quads };
}

void repair(const Mesh &ref, Mesh &opt, const std::vector <quad> &quads, cas_grid &cas)
{
	// First rediagonalize the quads
	for (uint32_t j = 0; j < quads.size(); j++) {
		const auto &quad = quads[j];

		// Go by every pair, which forms a quad
		for (size_t k = 0; k < quad.triangles.size(); k += 2) {
			Triangle &t0 = opt.triangles[quad.triangles[k]];
			Triangle &t1 = opt.triangles[quad.triangles[k + 1]];

			auto vec = array_order(t0, t1);

			// Diagonals are v0 -> v3 and v1 -> v2
			glm::vec3 v0 = opt.vertices[vec[0]];
			glm::vec3 v1 = opt.vertices[vec[1]];
			glm::vec3 v2 = opt.vertices[vec[2]];
			glm::vec3 v3 = opt.vertices[vec[3]];

			glm::vec3 d0 = v3 - v0;
			glm::vec3 d1 = v2 - v1;

			Triangle n0;
			Triangle n1;

			// Choose the shorter one
			if (glm::length(d0) < glm::length(d1)) {
				n0 = { vec[0], vec[1], vec[3] };
				n1 = { vec[0], vec[3], vec[2] };
			} else {
				n0 = { vec[1], vec[0], vec[2] };
				n1 = { vec[1], vec[2], vec[3] };
			}

			opt.triangles[quad.triangles[k]] = n0;
			opt.triangles[quad.triangles[k + 1]] = n1;
		}
	}

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

void sdf_gradients(cas_grid &cas, const std::vector <glm::vec3> &points, std::vector <glm::vec3> &gradients)
{
	uint32_t n = points.size();
	ULOG_ASSERT(n == gradients.size());

	std::vector <glm::vec3> epsiloned(6 * n);
	for (uint32_t i = 0; i < n; i++) {
		epsiloned[6 * i + 0] = points[i] + glm::vec3(1e-3f, 0.0f, 0.0f);
		epsiloned[6 * i + 1] = points[i] + glm::vec3(-1e-3f, 0.0f, 0.0f);
		epsiloned[6 * i + 2] = points[i] + glm::vec3(0.0f, 1e-3f, 0.0f);
		epsiloned[6 * i + 3] = points[i] + glm::vec3(0.0f, -1e-3f, 0.0f);
		epsiloned[6 * i + 4] = points[i] + glm::vec3(0.0f, 0.0f, 1e-3f);
		epsiloned[6 * i + 5] = points[i] + glm::vec3(0.0f, 0.0f, -1e-3f);
	}

	cas.precache_query(epsiloned);

	std::vector <glm::vec3> closest(6 * n);
	std::vector <glm::vec3> bary(6 * n);
	std::vector <float> dist(6 * n);
	std::vector <uint32_t> tid(6 * n);

	cas.query(epsiloned, closest, bary, dist, tid);
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
		uint32_t qt0;
		uint32_t qt1;

		fread(&qt0, sizeof(uint32_t), 1, file);
		fread(&qt1, sizeof(uint32_t), 1, file);

		const Triangle &t0 = source.triangles[qt0];
		const Triangle &t1 = source.triangles[qt1];

		quads[i].size = 2;
		quads[i].triangles = { qt0, qt1 };
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
	using stage_result = std::pair <Mesh, std::vector <quad>>;

	std::vector <stage_result> stages;

	cas_grid cas(target, 128);

	{
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

		stages.push_back({ opt, quads });
	}

	// Upscale for the next stage
	auto new_opt = opt;
	auto new_quads = quads;

	std::vector <std::vector <float>> densities;

	// Epoch time
	auto start = std::chrono::high_resolution_clock::now();

	for (size_t it = 0; it < 3; it++) {
		// TODO: plot this...
		// TODO: preview this in real-time...
		std::tie(new_opt, new_quads) = upscale(new_opt, new_quads);
		
		std::vector <glm::vec3> closest(new_opt.vertices.size());
		std::vector <glm::vec3> bary(new_opt.vertices.size());
		std::vector <float> sdfs(new_opt.vertices.size());
		std::vector <uint32_t> indices(new_opt.vertices.size());

		std::vector <glm::vec3> gradients(new_opt.vertices.size());
		std::vector <glm::vec3> diffused(new_opt.vertices.size());

		closest_point_kinfo kinfo = closest_point_kinfo_alloc(new_opt.vertices.size());

		using namespace indicators;

		BlockProgressBar bar {
			option::BarWidth { 50 },
			option::ForegroundColor { Color::white },
			option::FontStyles { std::vector <FontStyle> { FontStyle::bold} }
		};

		double query_time = 0.0;
		double update_time = 0.0;
		double remesh_time = 0.0;
		double total_time = 0.0;

		std::chrono::high_resolution_clock::time_point t1, t2;
		std::chrono::high_resolution_clock::time_point lstart, lend;
		std::chrono::high_resolution_clock clock;

		constexpr uint32_t TARGET_SAMPLES = 1000;

		cumesh cumesh_opt = cumesh_alloc(new_opt);
		cumesh cumesh_target = cumesh_alloc(target);

		sample_result target_samples = sample_result_alloc(TARGET_SAMPLES);
		closest_point_kinfo kinfo_from_target = closest_point_kinfo_alloc(TARGET_SAMPLES);

		std::vector <glm::vec3> host_target_samples (TARGET_SAMPLES);
		std::vector <glm::vec3> from_target_closest (TARGET_SAMPLES);
		std::vector <glm::vec3> from_target_bary    (TARGET_SAMPLES);
		std::vector <float>     from_target_sdfs    (TARGET_SAMPLES);
		std::vector <uint32_t>  from_target_indices (TARGET_SAMPLES);

		// Get connectivity information
		auto [vgraph, egraph, dual] = build_graphs(new_opt);

		std::vector <float> sampling_density(target.triangles.size(), 0.0f);
		std::vector <uint32_t> sampled_triangles(TARGET_SAMPLES);
		float total_samples = 0.0f;

		bool enable_target_sampling = true;
		for (int i = 0; i < 1000; i++) {
			t1 = clock.now();
			lstart = clock.now();

			// First query from the optimized mesh to the target
			const auto &queries = new_opt.vertices;

			// TODO: make a distinction between sdf and interior point...
			if (cas.precache_query(queries))
				cas.precache_device();

			cudaMemcpy(kinfo.points, new_opt.vertices.data(), sizeof(glm::vec3) * kinfo.point_count, cudaMemcpyHostToDevice);

			cas.query_device(kinfo);
			cudaDeviceSynchronize();

			// cas.query(queries, closest, bary, sdfs, indices);
			cudaMemcpy(closest.data(), kinfo.closest, sizeof(glm::vec3) * kinfo.point_count, cudaMemcpyDeviceToHost);
			cudaMemcpy(bary.data(), kinfo.bary, sizeof(glm::vec3) * kinfo.point_count, cudaMemcpyDeviceToHost);
			// cudaMemcpy(sdfs.data(), kinfo.distnace, sizeof(float) * kinfo.point_count, cudaMemcpyDeviceToHost);
			cudaMemcpy(indices.data(), kinfo.triangles, sizeof(uint32_t) * kinfo.point_count, cudaMemcpyDeviceToHost);

			for (uint32_t j = 0; j < new_opt.vertices.size(); j++) {
				gradients[j] = closest[j] - new_opt.vertices[j];
				// if (glm::length(gradients[j]) > 0.0f)
				// 	gradients[j] = glm::normalize(gradients[j]);
			}

			// Now query from the target to the optimized mesh
			if (enable_target_sampling) {
				auto qnow = std::chrono::high_resolution_clock::now();
				float epoch = std::chrono::duration_cast <std::chrono::duration <float>> (qnow - start).count();
				cumesh_reload(cumesh_opt, new_opt);
				sample(target_samples, cumesh_target, epoch);

				cudaMemcpy(kinfo_from_target.points, target_samples.points, sizeof(glm::vec3) * TARGET_SAMPLES, cudaMemcpyDeviceToDevice);

				cudaMemcpy(sampled_triangles.data(), target_samples.indices, sizeof(uint32_t) * TARGET_SAMPLES, cudaMemcpyDeviceToHost);
				cudaMemcpy(host_target_samples.data(), target_samples.points, sizeof(glm::vec3) * TARGET_SAMPLES, cudaMemcpyDeviceToHost);

				brute_closest_point(cumesh_opt, kinfo_from_target);

				cudaMemcpy(from_target_closest.data(), kinfo_from_target.closest, sizeof(glm::vec3) * TARGET_SAMPLES, cudaMemcpyDeviceToHost);
				cudaMemcpy(from_target_bary.data(), kinfo_from_target.bary, sizeof(glm::vec3) * TARGET_SAMPLES, cudaMemcpyDeviceToHost);
				// cudaMemcpy(from_target_sdfs.data(), kinfo_from_target.distnace, sizeof(float) * TARGET_SAMPLES, cudaMemcpyDeviceToHost);
				cudaMemcpy(from_target_indices.data(), kinfo_from_target.triangles, sizeof(uint32_t) * TARGET_SAMPLES, cudaMemcpyDeviceToHost);

				for (uint32_t j = 0; j < TARGET_SAMPLES; j++) {
					glm::vec3 w = host_target_samples[j];
					glm::vec3 v = from_target_closest[j];
					glm::vec3 b = from_target_bary[j];
					uint32_t t = from_target_indices[j];

					const Triangle &tri = new_opt.triangles[t];

					glm::vec3 v0 = new_opt.vertices[tri[0]];
					glm::vec3 v1 = new_opt.vertices[tri[1]];
					glm::vec3 v2 = new_opt.vertices[tri[2]];

					glm::vec3 delta = w - v;
					// if (glm::length(delta) > 0.0f)
					// 	delta = glm::normalize(delta);

					b = glm::clamp(b, 0.0f, 1.0f);
					glm::vec3 gv0 = b.x * delta;
					glm::vec3 gv1 = b.y * delta;
					glm::vec3 gv2 = b.z * delta;

					// ulog_assert(b.x >= 0.0f && b.x <= 1.0f, "loop", "b.x = %f", b.x);
					// ulog_assert(b.y >= 0.0f && b.y <= 1.0f, "loop", "b.y = %f", b.y);
					// ulog_assert(b.z >= 0.0f && b.z <= 1.0f, "loop", "b.z = %f", b.z);

					// ULOG_ASSERT(b.x >= 0.0f && b.x <= 1.0f);
					// ULOG_ASSERT(b.y >= 0.0f && b.y <= 1.0f);
					// ULOG_ASSERT(b.z >= 0.0f && b.z <= 1.0f);

					ULOG_ASSERT(!isnan(gv0.x));
					ULOG_ASSERT(!isnan(gv0.y));
					ULOG_ASSERT(!isnan(gv0.z));

					ULOG_ASSERT(tri[0] < gradients.size());
					ULOG_ASSERT(tri[1] < gradients.size());
					ULOG_ASSERT(tri[2] < gradients.size());

					gradients[tri[0]] += gv0;
					gradients[tri[1]] += gv1;
					gradients[tri[2]] += gv2;
				}

				total_samples += TARGET_SAMPLES;

				// printf(" > ");
				for (uint32_t j = 0; j < TARGET_SAMPLES; j++) {
					// printf("%d ", sampled_triangles[j]);
					ULOG_ASSERT(sampled_triangles[j] < target.triangles.size());
					sampling_density[sampled_triangles[j]]++;
				}
				// printf("\n");
			}

			t2 = clock.now();
			query_time += std::chrono::duration_cast <std::chrono::microseconds> (t2 - t1).count() / 1000.0;

			// Apply the gradients
			t1 = clock.now();

			// FIXME: First diffuse the gradients across neighboring vertices

			// std::copy(diffused.begin(), diffused.end(), gradients.begin());
			for (uint32_t j = 0; j < new_opt.vertices.size(); j++) {
				glm::vec3 g = gradients[j];
				if (glm::any(glm::isnan(g)))
					continue;
				new_opt.vertices[j] += g * 0.1f;
			}

			t2 = clock.now();
			update_time += std::chrono::duration_cast <std::chrono::microseconds> (t2 - t1).count() / 1000.0;

			// Rediagonalize each quad sparsely
			t1 = clock.now();

			if (i > 0 && i % 10 == 0) {
				if (i > 700)
					enable_target_sampling = false;
				else
					new_opt = smooth(new_opt, 0.1f);
			}

			if (i % 100 == 0) {
				repair(target, new_opt, new_quads, cas);
				std::tie(vgraph, egraph, dual) = build_graphs(new_opt);
			}

			t2 = clock.now();
			remesh_time += std::chrono::duration_cast <std::chrono::microseconds> (t2 - t1).count() / 1000.0;

			// Calculate total time
			lend = clock.now();
			total_time += std::chrono::duration_cast <std::chrono::microseconds> (lend - lstart).count() / 1000.0;

			// Update progress bar
			bar.set_progress(100.0f * i/1000.0f);
		}

		// Show timings
		query_time /= 1000.0;
		update_time /= 1000.0;
		remesh_time /= 1000.0;
		total_time /= 1000.0;

		printf("\nAverage timings:\n");
		printf("Query time  %3.2f ms  (%3.2f%%)\n", query_time, query_time / total_time * 100.0f);
		printf("Update time %3.2f ms  (%3.2f%%)\n", update_time, update_time / total_time * 100.0f);
		printf("Remesh time %3.2f ms  (%3.2f%%)\n", remesh_time, remesh_time / total_time * 100.0f);
		printf("Total time  %3.2f ms\n", total_time);

		stages.push_back({ new_opt, new_quads });
		densities.push_back(sampling_density);
	}

	// Visualize
	constexpr glm::vec3 color_wheel[] = {
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

	namespace ps = polyscope;
	ps::init();

	ps::registerSurfaceMesh("source", source.vertices, source.triangles);

	auto tm = ps::registerSurfaceMesh("target", target.vertices, target.triangles);
	for (uint32_t i = 0; i < densities.size(); i++)
		tm->addFaceScalarQuantity("sampling_density_" + std::to_string(i), densities[i]);

	for (uint32_t i = 0; i < stages.size(); i++) {
		auto [opt, quads] = stages[i];
		std::string basename = "stage_" + std::to_string(i);
		auto m = ps::registerSurfaceMesh(basename, opt.vertices, opt.triangles);

		std::vector <glm::vec3> face_colors;

		uint32_t color_index = 0;
		for (const auto &q : quads) {
			const glm::vec3 &color = color_wheel[color_index++ % 12];

			for (uint32_t tri : q.triangles)
				face_colors.push_back(color);
		}

		m->addFaceColorQuantity("face_colors", face_colors);
	}

	ps::show();

	// Save results
	std::filesystem::path sdv_quads = path.stem().string() + ".sdv";
	save(new_opt, new_quads, target_file, sdv_quads);
}
