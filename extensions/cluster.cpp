#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>
#include <unordered_map>

#include "common.hpp"

// Chartifying geometry into N clusters
static std::vector <std::unordered_set <int32_t>> cluster_once(const geometry &g, const geometry::dual_graph &dgraph, const std::vector <int32_t> &seeds, const std::string &metric)
{
	std::unordered_map <int32_t, int32_t> face_to_cluster;
	std::vector <std::unordered_set <int32_t>> clusters;
	std::vector <glm::vec3> cluster_normals;

	// Initialize the clusters
	for (int32_t s : seeds) {
		assert(s < g.triangles.size());

		face_to_cluster[s] = clusters.size();
		clusters.push_back(std::unordered_set <int32_t>{ s });
		cluster_normals.push_back(g.face_normal(s));
	}

	// Costs
	std::vector <float> costs;
	costs.resize(g.triangles.size(), FLT_MAX);

	// Comparator for the map
	auto compare = [&](const size_t &a, const size_t &b) {
		// Cost, then index
		return (costs[a] < costs[b]) || ((std::abs(costs[a] - costs[b]) < 1e-6f) && (a < b));
	};

	// Initialize the queue (set)
	std::set <int32_t, decltype(compare)> queue(compare);
	for (size_t i = 0; i < g.triangles.size(); i++) {
		if (std::find(seeds.begin(), seeds.end(), i) != seeds.end())
			costs[i] = 0.0f;

		queue.insert(i);
	}

	// Grow the charts
	while (queue.size() > 0) {
		int32_t face = *queue.begin();
		queue.erase(queue.begin());

		if (std::isinf(costs[face]))
			break;

		// clear line
		// TODO: progress bar instead...
		int32_t ci = face_to_cluster[face];
		glm::vec3 cn = cluster_normals[ci];

//		printf("\033[K\rRemaining: %zu; current is %d with %f (index = %d)", queue.size(), face, costs[face], ci);

		for (int32_t neighbor : dgraph.at(face)) {
			glm::vec3 nn = g.face_normal(neighbor);
			float dc     = glm::length(g.centroid(face) - g.centroid(neighbor));
			float dn     = 1 - glm::dot(cn, nn);
			float new_cost = costs[face] + ((metric == "flat") ? dn * dc : dc);

			if (queue.count(neighbor) && new_cost < costs[neighbor]) {
				queue.erase(neighbor);

				float size = clusters[ci].size();
				glm::vec3 new_normal = (cn * size + nn)/(size + 1.0f);

				cluster_normals[ci] = new_normal;
				costs[neighbor] = new_cost;

				if (face_to_cluster.count(neighbor)) {
					int32_t old = face_to_cluster[neighbor];
					clusters[old].erase(neighbor);
				}

				face_to_cluster[neighbor] = ci;
				clusters[ci].insert(neighbor);

				queue.insert(neighbor);
			}
		}
	}

	return clusters;
}

std::vector <std::vector <int32_t>> cluster_geometry(const geometry &g, const std::vector <int32_t> &seeds, int32_t iterations, const std::string &metric)
{
	assert(metric == "uniform" || metric == "flat");

	// Make the dual graph
	auto egraph = g.make_edge_graph();
	auto dgraph = g.make_dual_graph(egraph);

	std::vector <std::unordered_set <int32_t>> clusters;
	std::vector <int32_t> next_seeds = seeds;

	for (int32_t i = 0; i < iterations; i++) {
		clusters = cluster_once(g, dgraph, next_seeds, metric);
		if (i == iterations - 1)
			break;

		// Find the central faces for each cluster,
		// i.e. the face closest to the centroid
		std::vector <glm::vec3> centroids;
		for (const auto &c : clusters) {
			glm::vec3 centroid(0.0f);
			float wsum = 0.0f;
			for (int32_t f : c) {
				float w = 1.0f; // g.area(f);
				centroid += g.centroid(f) * w;
				wsum += w;
			}

			centroid /= wsum;
			centroids.push_back(centroid);
		}

		// For each cluster, find the closest face
		next_seeds.clear();
		for (size_t j = 0; j < clusters.size(); j++) {
			const auto &c = clusters[j];
			const auto &centroid = centroids[j];

			float min_dist = FLT_MAX;
			int32_t min_face = -1;

			for (int32_t f : c) {
				float dist = glm::length(centroid - g.centroid(f));
				if (dist < min_dist) {
					min_dist = dist;
					min_face = f;
				}
			}

			next_seeds.push_back(min_face);
		}
	}

	std::vector <std::vector <int32_t>> clusters_linear;
	for (const auto &c : clusters)
		clusters_linear.emplace_back(c.begin(), c.end());

	return clusters_linear;
}
