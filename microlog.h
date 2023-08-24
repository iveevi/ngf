// Header file...
#pragma once

#include <stdio.h>
#include <stdarg.h>

inline void ulog_error(const char *header, const char *format, ...)
{
	static const char *error = "\033[31;1m";
	static const char *reset = "\033[0m";

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

	printf("%s[*]%s (%s) ", info, reset, header);

	va_list args;
	va_start(args, format);
	vprintf(format, args);
	va_end(args);
}

inline void ulog_assert(bool condition, const char *header, const char *format, ...)
{
	static const char *assert_fail = "\033[91;1m";
	static const char *reset = "\033[0m";

	if (condition)
		return;

	printf("%s[!]%s (%s) ", assert_fail, reset, header);

	va_list args;
	va_start(args, format);
	vprintf(format, args);
	va_end(args);

	fflush(stdout);
	abort();
}

#define ULOG_ASSERT(cond) ulog_assert(cond, __FILE__, "%s failed on line %lu.\n", #cond, __LINE__)
