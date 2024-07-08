#include "vkutil.hpp"
#include "context.hpp"

std::optional <std::pair <vk::CommandBuffer, littlevk::SurfaceOperation>> new_frame(DeviceRenderContext &engine, size_t frame)
{
	// Get next image
	littlevk::SurfaceOperation op;
	op = littlevk::acquire_image(engine.device, engine.swapchain.swapchain, engine.sync[frame]);
	if (op.status == littlevk::SurfaceOperation::eResize) {
		engine.resize();
		return std::nullopt;
	}

	vk::CommandBuffer cmd = engine.command_buffers[frame];
	cmd.begin(vk::CommandBufferBeginInfo {});

	littlevk::viewport_and_scissor(cmd, littlevk::RenderArea(engine.window));

	// Record command buffer
	return std::make_pair(cmd, op);
}

void end_frame(const vk::Queue &queue, const littlevk::PresentSyncronization &sync, const vk::CommandBuffer &cmd, size_t frame)
{
	cmd.end();

	// Submit command buffer while signaling the semaphore
	vk::PipelineStageFlags wait_stage = vk::PipelineStageFlagBits::eColorAttachmentOutput;

	vk::SubmitInfo submit_info {
		1, &sync.image_available[frame],
		&wait_stage,
		1, &cmd,
		1, &sync.render_finished[frame]
	};

	queue.submit(submit_info, sync.in_flight[frame]);
}

void present_frame(DeviceRenderContext &engine, const littlevk::SurfaceOperation &op, size_t frame)
{
	// Send image to the screen
	littlevk::SurfaceOperation pop = littlevk::present_image(engine.present_queue, engine.swapchain.swapchain, engine.sync[frame], op.index);
	if (pop.status == littlevk::SurfaceOperation::eResize)
		engine.resize();
}

void render_pass_begin(const DeviceRenderContext &engine, const vk::CommandBuffer &cmd, const littlevk::SurfaceOperation &op, const glm::vec4 &color)
{
	const auto &rpbi = littlevk::default_rp_begin_info <2>
		(engine.render_pass, engine.framebuffers[op.index], engine.window)
		.clear_value(0, vk::ClearColorValue {
			std::array <float, 4> { color.x, color.y, color.z, color.w }
		});

	return cmd.beginRenderPass(rpbi, vk::SubpassContents::eInline);
}

void render_pass_end(const DeviceRenderContext &engine, const vk::CommandBuffer &cmd)
{
	return cmd.endRenderPass();
}

