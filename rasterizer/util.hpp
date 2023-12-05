#pragma once

#include <chrono>
#include <string>
#include <vector>
#include <deque>

#include <cuda_runtime.h>

struct timer {
	using clock = std::chrono::high_resolution_clock;
	using time_point = typename clock::time_point;

	clock clk;
	time_point start;
	time_point end;

	timer() = default;

	void tick() {
		start = clk.now();
	}

	double tock() {
		end = clk.now();
		return std::chrono::duration_cast <std::chrono::microseconds> (end - start).count()/1000.0;
	}
};

struct cuda_timer {
	cudaEvent_t start;
	cudaEvent_t end;

	cuda_timer() {
		cudaEventCreate(&start);
		cudaEventCreate(&end);
	}

	void tick() {
		cudaEventRecord(start);
	}

	double tock() {
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		float ms;
		cudaEventElapsedTime(&ms, start, end);
		return ms;
	}
};

// TODO: scoped & nested timers
struct timeframe {
	std::string name;
	double time;
	std::deque <timeframe> children;

	timeframe *child(std::string name) {
		children.emplace_back(timeframe { name, 0.0 });
		return &children.back();
	}
};

enum Device {
	eCPU,
	eCUDA
};

template <Device D>
struct scoped_timer {
	// TODO: selector template
	using timer_t = std::conditional_t <D == eCPU, timer, cuda_timer>;

	timer_t timer;
	timeframe *frame;

	scoped_timer(std::string name, timeframe *frame) : frame(frame) {
		timer.tick();
	}

	~scoped_timer() {
		frame->time = timer.tock();
	}
};
