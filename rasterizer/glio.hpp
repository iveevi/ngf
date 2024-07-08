#pragma once

#include <GLFW/glfw3.h>

struct Transform;

struct MouseInfo {
	bool left_drag = false;
	bool voided = true;
	float last_x = 0.0f;
	float last_y = 0.0f;
} extern mouse;

void button_callback(GLFWwindow *, int, int, int);
void cursor_callback(GLFWwindow *, double, double);
void handle_key_input(GLFWwindow *, Transform &);
