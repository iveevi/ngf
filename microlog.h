#pragma once

#include <stdarg.h>
#include <stdio.h>
#include <time.h>

void ulog_error   (const char *, const char *, ...);
void ulog_warning (const char *, const char *, ...);
void ulog_info    (const char *, const char *, ...);
void ulog_assert  (bool, const char *, const char *, ...);

// TODO: time logging modes
//
// default is full time
// currently implementing delta time

static struct {
	bool timer;
} ulog_config = { .timer = false };

static struct {
	bool enabled;
	struct timespec previous;
} ulog_timer = { .enabled = false };

static inline void ulog_timer_start()
{
	if (!ulog_timer.enabled) {
		ulog_timer.enabled = true;
		clock_gettime(CLOCK_MONOTONIC, &ulog_timer.previous);
	}
}

// returns delta time in milliseconds
static inline float ulog_timer_update()
{
	if (ulog_timer.enabled) {
		struct timespec time;
		clock_gettime(CLOCK_MONOTONIC, &time);

		float delta = (time.tv_sec - ulog_timer.previous.tv_sec) * 1000.0f + (time.tv_nsec - ulog_timer.previous.tv_nsec) / 1e6f;
		ulog_timer.previous = time;

		return delta;
	}

	return 0.0f;
}

inline void ulog_error(const char *header, const char *format, ...)
{
	static const char *error = "\033[31;1m";
	static const char *reset = "\033[0m";

	ulog_timer_start();
	printf("%s[x]%s (%s) ", error, reset, header);

	va_list args;
	va_start(args, format);
	vprintf(format, args);
	va_end(args);
}

inline void ulog_warning(const char *header, const char *format, ...)
{
	static const char *warning = "\033[33;1m";
	static const char *reset = "\033[0m";

	ulog_timer_start();
	printf("%s[!]%s (%s) ", warning, reset, header);

	va_list args;
	va_start(args, format);
	vprintf(format, args);
	va_end(args);
}

inline void ulog_info(const char *header, const char *format, ...)
{
	static const char *info = "\033[34;1m";
	static const char *reset = "\033[0m";

	if (ulog_config.timer) {
		ulog_timer_start();
		printf("%s[*]%s [%+8.2f ms] (%s) ", info, reset, ulog_timer_update(), header);
	} else {
		printf("%s[*]%s (%s) ", info, reset, header);
	}

	va_list args;
	va_start(args, format);
	vprintf(format, args);
	va_end(args);
}

inline void ulog_assert(bool condition, const char *header, const char *format, ...)
{
	ulog_timer_start();
	if (!condition) {
		static const char *error = "\033[35;1m";
		static const char *reset = "\033[0m";

		printf("%s[#]%s (%s) ", error, reset, header);

		va_list args;
		va_start(args, format);
		vprintf(format, args);
		va_end(args);

		fflush(stdout);
		exit(EXIT_FAILURE);
	}
}
