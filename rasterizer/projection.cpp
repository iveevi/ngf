#include <fstream>
#include <sstream>

#include "common.hpp"

// Transform
void Transform::from(const glm::vec3 &position_, const glm::vec3 &rotation_, const glm::vec3 &scale_)
{
	position = position_;
	rotation = rotation_;
	scale = scale_;
}

glm::mat4 Transform::matrix() const
{
	glm::mat4 pmat = glm::translate(glm::mat4(1.0f), position);
	glm::mat4 rmat = glm::mat4_cast(glm::quat(glm::radians(rotation)));
	glm::mat4 smat = glm::scale(glm::mat4(1.0f), scale);
	return pmat * rmat * smat;
}

glm::vec3 Transform::right() const
{
	glm::quat q = glm::quat(rotation);
	return glm::normalize(glm::vec3(q * glm::vec4(1.0f, 0.0f, 0.0f, 0.0f)));
}

glm::vec3 Transform::up() const
{
	glm::quat q = glm::quat(rotation);
	return glm::normalize(glm::vec3(q * glm::vec4(0.0f, 1.0f, 0.0f, 0.0f)));
}

glm::vec3 Transform::forward() const
{
	glm::quat q = glm::quat(rotation);
	return glm::normalize(glm::vec3(q * glm::vec4(0.0f, 0.0f, 1.0f, 0.0f)));
}

std::tuple <glm::vec3, glm::vec3, glm::vec3> Transform::axes() const
{
	glm::quat q = glm::quat(rotation);
	return std::make_tuple(
		glm::normalize(glm::vec3(q * glm::vec4(1.0f, 0.0f, 0.0f, 0.0f))),
		glm::normalize(glm::vec3(q * glm::vec4(0.0f, 1.0f, 0.0f, 0.0f))),
		glm::normalize(glm::vec3(q * glm::vec4(0.0f, 0.0f, 1.0f, 0.0f)))
	);
}

// Camera
void Camera::from(float aspect_, float fov_, float near_, float far_)
{
	aspect = aspect_;
	fov = fov_;
	near = near_;
	far = far_;
}

glm::mat4 Camera::perspective_matrix() const
{
	return glm::perspective(
		glm::radians(fov),
		aspect, near, far
	);
}

RayFrame Camera::rayframe(const Transform &transform) const
{
	auto [right, up, forward] = transform.axes();

	// Convert FOV to radians
        float vfov = glm::radians(fov);

        float h = std::tan(vfov / 2);

        float vheight = 2 * h;
        float vwidth = vheight * aspect;

        glm::vec3 w = glm::normalize(-forward);
        glm::vec3 u = glm::normalize(cross(up, w));
        glm::vec3 v = glm::cross(w, u);

	glm::vec3 horizontal = u * vwidth;
        glm::vec3 vertical = v * vheight;

	return RayFrame {
		.origin = transform.position,
		.lower_left = transform.position - horizontal/2.0f - vertical/2.0f - w,
		.horizontal = horizontal,
		.vertical = vertical
	};
}

glm::mat4 Camera::view_matrix(const Transform &transform)
{
	auto [right, up, forward] = transform.axes();
	return glm::lookAt(
		transform.position,
		transform.position + forward,
		up
	);
}
