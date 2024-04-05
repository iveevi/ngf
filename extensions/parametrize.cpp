#include <algorithm>
#include <cmath>
#include <optional>
#include <thread>

#include <glm/gtc/random.hpp>

#include "common.hpp"
#include "util.hpp"

using VertexList = std::vector <glm::vec3>;
using FaceList = std::vector <glm::ivec3>;

struct Connectivity {
	std::vector <std::unordered_set <int32_t>> adjacency; // vertices -> vertices
	std::vector <std::unordered_set <int32_t>> neighbors; // vertices -> faces

	static Connectivity from
	(
		const std::vector <glm::vec3> &vertices,
		const std::vector <glm::ivec3> &faces
	)
	{
		Connectivity conn;

		conn.adjacency.resize(vertices.size());
		conn.neighbors.resize(vertices.size());

		for (size_t i = 0; i < faces.size(); i++) {
			const glm::ivec3 &f = faces[i];

			for (int32_t j = 0; j < 3; j++)
				conn.neighbors[f[j]].insert(i);

			for (int32_t j = 0; j < 3; j++) {
				int32_t nj = (j + 1) % 3;
				conn.adjacency[f[j]].insert(f[nj]);
				conn.adjacency[f[nj]].insert(f[j]);
			}
		}

		return conn;
	}
};

static void relax_harmonic_parametrization
(
	const Connectivity &conn,
	const std::unordered_set <int32_t> &bset,
	std::vector <glm::vec2> &uvs
)
{
	std::vector <glm::vec2> buffer(uvs.size());
	for (size_t i = 0; i < uvs.size(); i++) {
		if (bset.count(i)) {
			buffer[i] = uvs[i];
		} else {
			glm::vec2 mean(0.0f);
			for (int32_t j : conn.adjacency[i])
				mean += uvs[j];
			mean /= conn.adjacency[i].size();
			buffer[i] = mean;
		}
	}

	uvs = buffer;
}

static std::vector <glm::vec2> harmonic_disk_parametrization
(
	const VertexList &vertices,
	const Connectivity &conn,
	const std::vector <int32_t> &boundary,
	const std::unordered_set <int32_t> &bset
)
{
	std::vector <glm::vec2> uvs;
	uvs.resize(conn.adjacency.size());

	// Random initialization
	for (size_t i = 0; i < uvs.size(); i++)
		uvs[i] = glm::linearRand(glm::vec2(0.0), glm::vec2(1.0));

	// Compute the boundary length
	float length = 0.0f;
	for (size_t i = 0; i < boundary.size(); i++) {
		int32_t ni = (i + 1) % boundary.size();
		const glm::vec3 vi = vertices[boundary[i]];
		const glm::vec3 vni = vertices[boundary[ni]];
		length += glm::length(vi - vni);
	}

	int32_t c0 = 0;
	int32_t index = 0;

	float segment;

	segment = 0.0f;
	do {
		int32_t ni = (index + 1) % boundary.size();
		const glm::vec3 vi = vertices[boundary[index]];
		const glm::vec3 vni = vertices[boundary[ni]];
		segment += glm::length(vi - vni);
		index++;
	} while(segment < 0.25f * length);
	int32_t c1 = index;

	do {
		int32_t ni = (index + 1) % boundary.size();
		const glm::vec3 vi = vertices[boundary[index]];
		const glm::vec3 vni = vertices[boundary[ni]];
		segment += glm::length(vi - vni);
		index++;
	} while(segment < 0.5f * length);
	int32_t c2 = index;

	do {
		int32_t ni = (index + 1) % boundary.size();
		const glm::vec3 vi = vertices[boundary[index]];
		const glm::vec3 vni = vertices[boundary[ni]];
		segment += glm::length(vi - vni);
		index++;
	} while(segment < 0.75f * length);
	int32_t c3 = index;

	// Set boundary condition
	segment = 0.0f;
	for (int32_t i = c0; i < c1; i++) {
		int32_t ni = (i + 1) % boundary.size();
		const glm::vec3 vi = vertices[boundary[i]];
		const glm::vec3 vni = vertices[boundary[ni]];
		uvs[boundary[i]] = glm::vec2(segment/(0.25 * length), 0.0f);
		segment += glm::length(vi - vni);
	}

	segment = 0.0f;
	for (int32_t i = c1; i < c2; i++) {
		int32_t ni = (i + 1) % boundary.size();
		const glm::vec3 vi = vertices[boundary[i]];
		const glm::vec3 vni = vertices[boundary[ni]];
		uvs[boundary[i]] = glm::vec2(1.0f, segment/(0.25 * length));
		segment += glm::length(vi - vni);
	}

	segment = 0.0f;
	for (int32_t i = c2; i < c3; i++) {
		int32_t ni = (i + 1) % boundary.size();
		const glm::vec3 vi = vertices[boundary[i]];
		const glm::vec3 vni = vertices[boundary[ni]];
		uvs[boundary[i]] = glm::vec2(1.0f - segment/(0.25 * length), 1.0f);
		segment += glm::length(vi - vni);
	}

	segment = 0.0f;
	for (size_t i = c3; i < boundary.size(); i++) {
		int32_t ni = (i + 1) % boundary.size();
		const glm::vec3 vi = vertices[boundary[i]];
		const glm::vec3 vni = vertices[boundary[ni]];
		uvs[boundary[i]] = glm::vec2(0.0f, 1.0f - segment/(0.25 * length));
		segment += glm::length(vi - vni);
	}

	uvs[boundary[0]] = glm::vec2(0.0f);

	// Perform iterative relaxations
	for (int32_t i = 0; i < 100; i++)
		relax_harmonic_parametrization(conn, bset, uvs);

	return uvs;
}

static void fix_faces
(
		const VertexList              &vertices,
		FaceList                      &faces,
		const std::vector <glm::vec2> &uvs
)
{
	for (glm::ivec3 &f : faces) {
		glm::vec2 uv1 = uvs[f.x];
		glm::vec2 uv2 = uvs[f.y];
		glm::vec2 uv3 = uvs[f.z];

		float s1 = uv1.x;
		float s2 = uv2.x;
		float s3 = uv3.x;

		float t1 = uv1.y;
		float t2 = uv2.y;
		float t3 = uv3.y;

		// TODO: cross product?
		float A = ((s2 - s1) * (t3 - t1) - (s3 - s1) * (t2 - t1));
		if (A < 0)
			std::swap(f.y, f.z);
	}
}

inline float cross(const glm::vec2 &a, const glm::vec2 &b)
{
	return a.x * b.y - a.y * b.x;
}

static glm::vec3 stretch_metric
(
		const glm::vec3 &q1,
		const glm::vec3 &q2,
		const glm::vec3 &q3,
		const glm::vec2 &uv1,
		const glm::vec2 &uv2,
		const glm::vec2 &uv3
)
{
	float area = glm::length(glm::cross(q1 - q2, q1 - q3));

	float s1 = uv1.x;
	float s2 = uv2.x;
	float s3 = uv3.x;

	float t1 = uv1.y;
	float t2 = uv2.y;
	float t3 = uv3.y;

	float A = 0.5 * ((s2 - s1) * (t3 - t1) - (s3 - s1) * (t2 - t1)) + 1e-6f;
	// glm::vec2 e0 = uv1 - uv2;
	// glm::vec2 e1 = uv1 - uv3;
	// float A = 0.5f * glm::length(cross(e0, e1));
	if (A < 0)
		return { FLT_MAX, area, FLT_MAX };

	glm::vec3 Ss = (q1 * (t2 - t3) + q2 * (t3 - t1) + q3 * (t1 - t2))/(2 * A);
	glm::vec3 St = (q1 * (s3 - s2) + q2 * (s1 - s3) + q3 * (s2 - s1))/(2 * A);

	float a = glm::dot(Ss, Ss);
	float c = glm::dot(St, St);

	float stretch = glm::sqrt(0.5 * (a + c));

	if (glm::isnan(stretch)) {
                printf("   NAN STRETCH! points are (%.2f, %.2f), (%.2f, %.2f), "
                       "(%.2f, %.2f) | Ss = (%.2f, %.2f) | St = (%.2f, %.2f) | "
                       "A = %f\n",
                       s1, t1, s2, t3, s3, t3, Ss.x, Ss.y, St.x, St.y, A);
        }

	return { stretch, area, A };
}

static float vertex_neighbor_stretch
(
	const std::unordered_set <int32_t> &neighbor,
	const VertexList                   &vertices,
	const FaceList                     &faces,
	const std::vector <glm::vec2>      &uvs,
	const std::vector <int32_t>        &boundary,
	int32_t                            vertex,
	glm::vec2                          vertex_uv
)
{
	float sum = 0;
	float weights = 0;

	for (int32_t fi : neighbor) {
		const glm::ivec3 &f = faces[fi];

		glm::vec2 uv1 = uvs[f.x];
		glm::vec2 uv2 = uvs[f.y];
		glm::vec2 uv3 = uvs[f.z];

		if (f.x == vertex)
			uv1 = vertex_uv;
		else if (f.y == vertex)
		      uv2 = vertex_uv;
		else if (f.z == vertex)
		      uv3 = vertex_uv;

		glm::vec3 m = stretch_metric
		(
			vertices[f.x],
			vertices[f.y],
			vertices[f.z],
			uv1, uv2, uv3
		);

		sum += m.x * m.x * m.y;
		weights += m.y;
	}

	return glm::sqrt(sum/weights);
}

static std::tuple <glm::vec2, float> vertex_stretch_linesearch
(
	const std::unordered_set <int32_t> &neighbor,
	const VertexList                   &vertices,
	const FaceList                     &faces,
	const std::vector <glm::vec2>      &uvs,
	const std::vector <int32_t>        &boundary,
	int32_t                            vertex,
	glm::vec2                          vuv,
	glm::vec2                          delta,        // Assume normalized
	float                              a,
	float                              b,
	float                              tolerance
)
{
	constexpr float phi = 0.5f * (1 + glm::sqrt(5));

	// float area = parametrization_area(vertices, faces, uvs);
	while (std::fabs(a - b) > tolerance) {
		float c = b - (b - a)/phi;
		float d = a + (b - a)/phi;

		glm::vec2 c_uv = vuv + c * delta;
		glm::vec2 d_uv = vuv + d * delta;

		float m_c = vertex_neighbor_stretch
		(
			neighbor,
			vertices, faces, uvs, boundary,
			vertex, c_uv
		);

		float m_d = vertex_neighbor_stretch
		(
			neighbor,
			vertices, faces, uvs, boundary,
			vertex, d_uv
		);

		if (m_c < m_d)
			b = d;
		else
			a = c;
	}

	// Make sure its not infinity, otherwise return the old
	float opt = 0.5f * (b + a);

	glm::vec2 opt_uv = vuv + opt * delta;
	float m_opt = vertex_neighbor_stretch
	(
		neighbor,
		vertices, faces, uvs, boundary,
		vertex, opt_uv
	);

	return { opt_uv, m_opt };
}

static std::tuple <float, float> uv_bounds(const glm::vec2 &vuv, const glm::vec2 &delta)
{
	float one_tu = (1 - vuv.x)/delta.x;
	float one_tv = (1 - vuv.y)/delta.y;

	float zero_tu = -vuv.x/delta.x;
	float zero_tv = -vuv.y/delta.y;

	float min_u = glm::min(one_tu, zero_tu);
	float max_u = glm::max(one_tu, zero_tu);

	float min_v = glm::min(one_tv, zero_tv);
	float max_v = glm::max(one_tv, zero_tv);

	float a = glm::min(max_u, max_v);
	float b = glm::max(min_u, min_v);

	return { a, b };
}

static void geometric_stretch_optimization_iteration
(
	const Connectivity &conn,
	const VertexList &vertices,
	const FaceList &faces,
	std::vector <glm::vec2>	&uvs,
	const std::vector <int32_t> &boundary,
	const std::unordered_set <int32_t> &bset,
	float tolerance
)
{
	std::vector <int32_t> indices;
	std::vector <float> costs;

	indices.resize(vertices.size());
	costs.resize(vertices.size());

	for (size_t i = 0; i < vertices.size(); i++) {
		indices[i] = i;
		costs[i] = vertex_neighbor_stretch
		(
			conn.neighbors[i],
			vertices, faces, uvs, boundary,
			i, uvs[i]
		);
	}

	std::sort
	(
		indices.begin(),
		indices.end(),
		[&costs](int32_t a, int32_t b) {
			return costs[a] > costs[b];
		}
	);

	for (int32_t vi : indices) {
		if (bset.count(vi))
			continue;

		glm::vec2 delta = glm::circularRand(1.0f);

		auto [a, b] = uv_bounds(uvs[vi], delta);
		// float a = 0.001f;
		// float b = -0.001f;
		auto [opt_uv, new_cost] = vertex_stretch_linesearch
		(
			conn.neighbors[vi],
			vertices, faces, uvs,
			(bset.count(vi)) ? boundary : std::vector <int32_t> {},
			vi, uvs[vi], delta,
			a, b, tolerance
		);

		if (new_cost < costs[vi])
			uvs[vi] = opt_uv;
	}
}

torch::Tensor parametrize
(
	const torch::Tensor &tch_vertices,
	const torch::Tensor &tch_faces,
	const std::vector <int32_t> &boundary
)
{
	tensor_check <torch::kCPU, torch::kFloat32, 3> (tch_vertices);
	tensor_check <torch::kCPU, torch::kInt32, 3>   (tch_faces);

	// Localizing buffers
	std::vector <glm::vec3> vertices;
	std::vector <glm::ivec3> faces;

	vertices.resize(tch_vertices.size(0));
	faces.resize(tch_faces.size(0));

	const float *vertices_raw = tch_vertices.data_ptr <float> ();
	const int32_t *faces_raw = tch_faces.data_ptr <int32_t> ();

	std::memcpy(vertices.data(), vertices_raw, vertices.size() * sizeof(glm::vec3));
	std::memcpy(faces.data(), faces_raw, faces.size() * sizeof(glm::ivec3));

	// Chart triangle topology
	Connectivity conn = Connectivity::from(vertices, faces);

	// Faster boundary checking
	std::unordered_set <int32_t> bset;
	for (int32_t vi : boundary)
		bset.insert(vi);

	// Parametrize
	std::vector <glm::vec2> huvs = harmonic_disk_parametrization(vertices, conn, boundary, bset);
	// std::vector <glm::vec2> uvs = geometric_stretch_optimization(conn, vertices, faces, huvs, boundary, bset);

	return vector_to_tensor <glm::vec2, torch::kFloat32, 2> (huvs);
	// return vector_to_tensor <glm::vec2, torch::kFloat32, 2> (uvs);
}

std::vector <torch::Tensor> parametrize_parallel
(
	const std::vector <
		std::tuple <
			torch::Tensor,
			torch::Tensor,
			std::vector <int32_t>
		>
	> &patches
)
{
	int32_t threads = std::thread::hardware_concurrency();

	// Return
	std::vector <torch::Tensor> parametrizations;
	parametrizations.resize(patches.size());

	// Job counter (index)
	std::mutex jobber;

	int32_t next = 0;

	// Worker pool
	std::vector <std::thread> workers;
	for (int32_t i = 0; i < threads; i++) {
		workers.emplace_back
		(
			[&](int32_t tid) {
				while (true) {
					int32_t index = -1;
					{
						std::lock_guard lock { jobber };
						index = next++;
					}

					if (index >= (int32_t) patches.size())
						break;

					parametrizations[index] = parametrize
					(
						std::get <0> (patches[index]),
						std::get <1> (patches[index]),
						std::get <2> (patches[index])
					);
				}
			}, i
		);
	}

	// Wait to finish
	for (int32_t i = 0; i < threads; i++)
		workers[i].join();

	return parametrizations;
}
