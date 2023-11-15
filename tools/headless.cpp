#include <iostream>

#include <GLFW/glfw3.h>

int main()
{
	if (!glfwInit()) {
		fprintf(stderr, "GLFW failed to initialize, assuming headless server.\n");
		return 1;
	}

	fprintf(stderr, "GLFW successfully initialized, permitting glfw applications.\n");
	return 0;
}
