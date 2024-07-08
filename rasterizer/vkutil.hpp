#pragma once

#include <utility>
#include <optional>

#include <glm/glm.hpp>

#include <littlevk/littlevk.hpp>

struct DeviceRenderContext;

std::optional <std::pair <vk::CommandBuffer, littlevk::SurfaceOperation>>
new_frame(DeviceRenderContext &, size_t);

void end_frame(const vk::Queue &, const littlevk::PresentSyncronization &,
		const vk::CommandBuffer &, size_t);

void present_frame(DeviceRenderContext &, const littlevk::SurfaceOperation &, size_t);

void render_pass_begin(const DeviceRenderContext &, const vk::CommandBuffer &,
		const littlevk::SurfaceOperation &, const glm::vec4 &);

void render_pass_end(const DeviceRenderContext &, const vk::CommandBuffer &);

// Allocate and copy in one step
template <typename T>
littlevk::Image general_allocator(const vk::Device &device,
		const vk::PhysicalDeviceMemoryProperties &memory_properties,
		const vk::CommandPool &command_pool,
		const vk::Queue &graphics_queue,
		littlevk::Deallocator *dal,
		const std::vector <T> &buffer, vk::Extent2D extent,
		const vk::ImageType type = vk::ImageType::e2D,
		const vk::Format format = vk::Format::eR32G32B32A32Sfloat)
{
	vk::ImageViewType view = (type == vk::ImageType::e2D)
		? vk::ImageViewType::e2D
		: vk::ImageViewType::e1D;

	littlevk::Image texture;
	littlevk::Buffer staging;
	std::tie(texture, staging) = littlevk::linked_device_allocator
		(device, memory_properties, dal)
		.image(extent, format,
			vk::ImageUsageFlagBits::eSampled
				| vk::ImageUsageFlagBits::eTransferDst,
			vk::ImageAspectFlagBits::eColor,
			type, view)
		.buffer(buffer, vk::BufferUsageFlagBits::eTransferSrc);

	littlevk::submit_now(device, command_pool, graphics_queue,
		[&](const vk::CommandBuffer &cmd) {
			littlevk::transition(cmd, texture,
					vk::ImageLayout::eUndefined,
					vk::ImageLayout::eTransferDstOptimal);

			littlevk::copy_buffer_to_image(cmd, texture, staging,
					vk::ImageLayout::eTransferDstOptimal);

			littlevk::transition(cmd, texture,
					vk::ImageLayout::eTransferDstOptimal,
					vk::ImageLayout::eShaderReadOnlyOptimal);
		}
	);

	return texture;
};
