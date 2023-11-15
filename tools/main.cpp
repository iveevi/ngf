#include <stdio.h>
#include <time.h>

#include "simplify.h"

const char *help_str = R"(
Usage: simplify <input> <output> <ratio> <agressiveness>
    input:         name of existing OBJ format mesh
    output:        name for decimated OBJ format mesh
    ratio:         for example 0.2 will decimate 80%% of triangles  (=0.5) 
    agressiveness: faster or better decimation                      (=7.0) 
)";

void help(const char *argv[])
{
	printf("%s\n", help_str);
}

int main(int argc, const char *argv[])
{
        if (argc < 3) {
		help(argv);
		return EXIT_SUCCESS;
	}
	
	Simplify::load_obj(argv[1]);
	if ((Simplify::triangles.size() < 3) || (Simplify::vertices.size() < 3))
		return EXIT_FAILURE;

	int target_count = Simplify::triangles.size() >> 1;
	if (argc > 3) {
		float reduceFraction = atof(argv[3]);
		if (reduceFraction > 1.0)
			reduceFraction = 1.0; // lossless only
		if (reduceFraction <= 0.0) {
			printf("Ratio must be BETWEEN zero and one.\n");
			return EXIT_FAILURE;
		}
		target_count = round((float) Simplify::triangles.size() * atof(argv[3]));
	}

	if (target_count < 4) {
		printf("Object will not survive such extreme decimation\n");
		return EXIT_FAILURE;
	}

	double agressiveness = 7.0;
	if (argc > 4)
		agressiveness = atof(argv[4]);

	clock_t start = clock();
	printf("Input: %zu vertices, %zu triangles (target %d)\n",
			Simplify::vertices.size(), Simplify::triangles.size(), target_count);

	long unsigned int startSize = Simplify::triangles.size();

	Simplify::simplify_mesh(target_count, agressiveness, true);
	if (Simplify::triangles.size() >= startSize) {
		printf("Unable to reduce mesh.\n");
		return EXIT_FAILURE;
	}

	Simplify::write_obj(argv[2]);
        printf("Output: %zu vertices, %zu triangles (%f reduction; %.4f sec)\n",
               Simplify::vertices.size(), Simplify::triangles.size(),
               (float)Simplify::triangles.size() / (float)startSize,
               ((float)(clock() - start)) / CLOCKS_PER_SEC);

        return EXIT_SUCCESS;
}
