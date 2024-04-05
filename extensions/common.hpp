#pragma once

#include <vector>
#include <cstdint>

#include <torch/extension.h>

#include "geometry.hpp"

// Surface smoothing utilities
struct Graph {
	std::unordered_map <int32_t, std::unordered_set <int32_t>> graph;

	int32_t *dev_graph = nullptr;

	int32_t max = 0;
	int32_t max_adj = 0;

	Graph(const torch::Tensor &);
	~Graph();

	void allocate_device_graph();
	void initialize_from_triangles(const torch::Tensor &);
	void initialize_from_quadrilaterals(const torch::Tensor &);

	torch::Tensor smooth(const torch::Tensor &, float) const;
};

std::vector <std::vector <int32_t>> cluster_geometry
(const geometry &, const std::vector <int32_t> &, int32_t, const std::string &);

// Patch parametrization (multichart geometry images)
// std::tuple <torch::Tensor, torch::Tensor> parametrize
torch::Tensor parametrize
(const torch::Tensor &, const torch::Tensor &, const std::vector <int32_t> &);

std::vector <torch::Tensor> parametrize_parallel
(const std::vector <std::tuple <torch::Tensor, torch::Tensor, std::vector <int32_t>>> &);

// Triangulation utilities
// TODO: refactor
torch::Tensor triangulate_shorted(const torch::Tensor &, size_t, size_t);

// Loading a mesh
std::tuple <torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
load_mesh(const std::string &);
