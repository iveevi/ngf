#include <getopt.h>

#include "geometry.hpp"
#include "microlog.h"

int main(int argc, char *argv[])
{
	// convert <input mesh> -f <format extension> <options>
	// options: -c (compact)

	std::filesystem::path input_mesh;
	std::string format_extension = ".ply";
	bool compact = false;

	int c;
	while ((c = getopt(argc, argv, "f:ch")) != -1) {
		switch (c) {
		case 'c':
			compact = true;
			break;
		case 'f':
			format_extension = optarg;
			break;
		case 'h':
			printf("Usage: %s <input mesh> -f <format extension> <options>\n", argv[0]);
		default:
			exit(1);
		}
	}

	if (optind < argc) {
		input_mesh = argv[optind];
	} else {
		printf("Usage: %s <input mesh> -f <format extension> <options>\n", argv[0]);
		exit(1);
	}

	// Only for triangle meshes
	geometry <eTriangle> geometry = load_geometry <eTriangle> (input_mesh)[0];
	ulog_info("convert", "loaded geometry with %d vertices and %d triangles\n", geometry.vertices.size(), geometry.triangles.size());

	std::filesystem::path output_mesh = input_mesh;
	output_mesh.replace_extension(format_extension);

	// TODO: compact if needed...

	write_geometry(geometry, output_mesh);

	return 0;
}
