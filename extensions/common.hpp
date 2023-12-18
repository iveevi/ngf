#pragma once

#include <vector>
#include <cstdint>

#include <torch/extension.h>

#include "geometry.hpp"

std::vector <std::vector <int32_t>> cluster_once(const geometry &g, const geometry::dual_graph &dgraph, const std::vector <int32_t> &seeds);

std::vector <std::vector <int32_t>> cluster_geometry(const geometry &g, const std::vector <int32_t> &seeds, int32_t iterations);
