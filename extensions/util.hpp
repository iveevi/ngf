#pragma once

#include <torch/extension.h>

template <torch::DeviceType device, torch::ScalarType type, size_t components>
inline void tensor_check(const torch::Tensor &tensor)
{
	assert(tensor.device() == device);
	assert(tensor.dtype() == type);
	assert(tensor.dim() == 2);
	assert(tensor.size(1) == components);
}

template <typename T, torch::ScalarType type, size_t components>
torch::Tensor vector_to_tensor(const std::vector <T> &data)
{
	auto options = torch::TensorOptions().dtype(type).device(torch::kCPU, 0);

	torch::Tensor tch = torch::zeros({ (long) data.size(), components }, options);

	void *raw = tch.data_ptr();
	std::memcpy(raw, data.data(), sizeof(T) * data.size());

	return tch;
}
