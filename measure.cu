#include <filesystem>
#include <random>
#include <unordered_map>

#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <indicators/block_progress_bar.hpp>

#include "casdf/casdf.hpp"
#include "microlog.h"

std::vector <glm::vec3> normals(const geometry &g)
{
	std::vector <glm::vec3> normals(g.vertices.size(), glm::vec3(0.0f));

	for (auto &t : g.triangles) {
		glm::vec3 a = g.vertices[t.x];
		glm::vec3 b = g.vertices[t.y];
		glm::vec3 c = g.vertices[t.z];

		glm::vec3 n = glm::cross(b - a, c - a);

		normals[t.x] += n;
		normals[t.y] += n;
		normals[t.z] += n;
	}

	for (auto &n : normals)
		n = glm::normalize(n);

	return normals;
}

struct loader {
	std::vector <geometry> meshes;

	loader(const std::filesystem::path &path) {
		Assimp::Importer importer;

		// Read scene
		const aiScene *scene;
		scene = importer.ReadFile(path, aiProcess_GenNormals | aiProcess_Triangulate);

		// Check if the scene was loaded
		if ((!scene | scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) || !scene->mRootNode) {
			ulog_error("loader", "Assimp error: \"%s\"\n", importer.GetErrorString());
			return;
		}

		process_node(scene->mRootNode, scene, path.parent_path());
	}

	void process_node(aiNode *node, const aiScene *scene, const std::string &directory) {
		// Process all the node's meshes (if any)
		for (uint32_t i = 0; i < node->mNumMeshes; i++) {
			aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
			process_mesh(mesh, scene, directory);
		}

		// Recusively process all the node's children
		for (uint32_t i = 0; i < node->mNumChildren; i++)
			process_node(node->mChildren[i], scene, directory);

	}

	void process_mesh(aiMesh *, const aiScene *, const std::string &);

	const geometry &get(uint32_t i) const {
		return meshes[i];
	}
};

void loader::process_mesh(aiMesh *mesh, const aiScene *scene, const std::string &dir)
{

	std::vector <glm::vec3> vertices;
        std::vector <glm::uvec3> triangles;

	// Process all the mesh's vertices
	for (uint32_t i = 0; i < mesh->mNumVertices; i++) {
		vertices.push_back({
			mesh->mVertices[i].x,
			mesh->mVertices[i].y,
			mesh->mVertices[i].z
		});
	}

	// Process all the mesh's triangles
	// std::stack <uint32_t> buffer;
	for (uint32_t i = 0; i < mesh->mNumFaces; i++) {
		aiFace face = mesh->mFaces[i];
		ulog_assert(face.mNumIndices == 3, "process_mesh", "Only triangles are supported, got %d-sided polygon instead\n", face.mNumIndices);
		triangles.push_back({
			face.mIndices[0],
			face.mIndices[1],
			face.mIndices[2]
		});
	}

	meshes.push_back({ vertices, triangles });
}

int main(int argc, char *argv[])
{
	if (argc != 2) {
		ulog_error("measure", "Usage: %s <directory>\n", argv[0]);
		return 1;
	}

	std::filesystem::path dir { argv[1] };

	// Find the relevant files

	// Find the files complexes.bin, encodings.bin, model.bin, and points.bin
	std::filesystem::path complexes_path;
	std::filesystem::path encodings_path;
	std::filesystem::path model_path;
	std::filesystem::path points_path;

	for (auto &p : std::filesystem::directory_iterator(dir)) {
		std::string fname = p.path().filename().string();
		if (fname == "complexes.bin")
			complexes_path = p.path();
		else if (fname == "encodings.bin")
			encodings_path = p.path();
		else if (fname == "model.bin")
			model_path = p.path();
		else if (fname == "points.bin")
			points_path = p.path();
	}

	complexes_path = std::filesystem::absolute(complexes_path);
	encodings_path = std::filesystem::absolute(encodings_path);
	model_path = std::filesystem::absolute(model_path);
	points_path = std::filesystem::absolute(points_path);

	ulog_info("measure", "Complexes:      %s\n", complexes_path.c_str());
	ulog_info("measure", "Encodings:      %s\n", encodings_path.c_str());
	ulog_info("measure", "Model:          %s\n", model_path.c_str());
	ulog_info("measure", "Points:         %s\n", points_path.c_str());

	// Reference mesh is ref.*
	std::filesystem::path ref_path;
	for (auto &p : std::filesystem::directory_iterator(dir)) {
		std::string fname = p.path().filename().string();
		if (fname.find("ref") == 0) {
			ref_path = p.path();
			break;
		}
	}

	ulog_info("measure", "Reference mesh: %s\n", std::filesystem::absolute(ref_path).c_str());

	// Target meshes are mesh[X]x[X].*
	std::vector <std::pair <uint32_t, std::filesystem::path>> target_paths;

	for (auto &p : std::filesystem::directory_iterator(dir)) {
		std::string fname = p.path().filename().string();
		if (fname.find("mesh") == 0) {
			std::string res_str = fname.substr(4, fname.find('.') - 4);
			uint32_t res = std::stoi(res_str);
			ulog_info("measure", "Target mesh %2d: %s\n", res, std::filesystem::absolute(p.path()).c_str());
			target_paths.push_back({ res, p.path() });
		}
	}

	if (target_paths.size() == 0) {
		ulog_error("measure", "No target meshes found\n");
		return 1;
	}

	std::sort(target_paths.begin(), target_paths.end(),
		[](auto &a, auto &b) {
			return a.first < b.first;
		}
	);

	// Load all objects
	loader ref { ref_path };

	std::vector <std::pair <uint32_t, loader>> targets;
	for (auto &[r, p] : target_paths)
		targets.push_back({ r, p });

	geometry ref_geometry = ref.get(0);

	glm::vec3 min { std::numeric_limits <float>::max() };
	glm::vec3 max { -std::numeric_limits <float>::max() };

	for (auto &v : ref_geometry.vertices) {
		min = glm::min(min, v);
		max = glm::max(max, v);
	}

	auto normalize = [&](glm::vec3 v) {
		return (v - min) / (max - min);
	};

	std::vector <std::pair <uint32_t, geometry>> target_geometry;
	for (auto &[r, t] : targets)
		target_geometry.push_back({ r, t.get(0) });

	// Compute normals for the ref and each target
	auto ref_normals = normals(ref_geometry);

	std::unordered_map <uint32_t, std::vector <glm::vec3>> target_normals;
	for (auto &[r, t] : target_geometry)
		target_normals[r] = normals(t);

	// Generate cache grids
	constexpr uint32_t resolution = 128;

	cas_grid ref_cas { ref_geometry, resolution };

	std::vector <std::pair <uint32_t, cas_grid>> target_cas;
	for (auto &[r, t] : target_geometry)
		target_cas.push_back({ r, { t, resolution } });

	// Sample and evaluate errors for each target
	constexpr uint32_t samples = 1000000;
	constexpr uint32_t sample_batch = 10000;

	std::random_device rd;
	std::mt19937 gen { rd() };

	std::uniform_real_distribution <float> barycentric_dist { 0.0f, 1.0f };

	std::vector <glm::vec3> closest;
	std::vector <glm::vec3> bary;
	std::vector <float> distance;
	std::vector <uint32_t> triangle;

	std::vector <glm::vec3> ref_samples;
	std::vector <glm::vec3> target_samples;

	std::vector <glm::vec3> ref_bary;
	std::vector <glm::vec3> target_bary;

	std::vector <uint32_t> ref_triangles;
	std::vector <uint32_t> target_triangles;

	closest.resize(sample_batch);
	bary.resize(sample_batch);
	distance.resize(sample_batch);
	triangle.resize(sample_batch);

	ref_samples.resize(sample_batch);
	target_samples.resize(sample_batch);

	ref_bary.resize(sample_batch);
	target_bary.resize(sample_batch);

	ref_triangles.resize(sample_batch);
	target_triangles.resize(sample_batch);

	auto sample_point = [](const geometry &g, uint32_t tri, glm::vec3 bary) {
		const glm::uvec3 &t = g.triangles[tri];
		const glm::vec3 &v0 = g.vertices[t.x];
		const glm::vec3 &v1 = g.vertices[t.y];
		const glm::vec3 &v2 = g.vertices[t.z];
		return v0 * bary.x + v1 * bary.y + v2 * bary.z;
	};

	std::vector <std::tuple <uint32_t, double, double>> errors;
	for (auto &[r, tcas] : target_cas) {
		const auto &tgt_normals = target_normals[r];

		// Progress bar
		using namespace indicators;

		BlockProgressBar bar {
			option::BarWidth { 50 },
			option::ForegroundColor { Color::white },
			option::PostfixText { "Evaluating target mesh with resolution " + std::to_string(r) },
			option::FontStyles { std::vector <FontStyle> { FontStyle::bold} }
		};

		// Sampling objects
		std::uniform_int_distribution <uint32_t> ref_triangle_dist
			{ 0, ref_cas.ref.triangles.size() - 1 };

		std::uniform_int_distribution <uint32_t> target_triangle_dist
			{ 0, tcas.ref.triangles.size() - 1 };

		double dpm = 0.0f;
		double dnormal = 0.0f;

		for (uint32_t it = 0; it < samples/sample_batch; it++) {
			// Generate samples on both surfaces
			for (uint32_t i = 0; i < sample_batch; i++) {
				glm::vec3 barycentric0 {
					barycentric_dist(gen),
					barycentric_dist(gen),
					0
				};

				if (barycentric0.x + barycentric0.y > 1.0f) {
					barycentric0.x = 1.0f - barycentric0.x;
					barycentric0.y = 1.0f - barycentric0.y;
				}

				barycentric0.z = 1.0f - barycentric0.x - barycentric0.y;
				ref_bary[i] = barycentric0;

				glm::vec3 barycentric1 {
					barycentric_dist(gen),
					barycentric_dist(gen),
					0
				};

				if (barycentric1.x + barycentric1.y > 1.0f) {
					barycentric1.x = 1.0f - barycentric1.x;
					barycentric1.y = 1.0f - barycentric1.y;
				}

				barycentric1.z = 1.0f - barycentric1.x - barycentric1.y;
				target_bary[i] = barycentric1;

				// Sample reference surface
				uint32_t ref_triangle = ref_triangle_dist(gen);
				ref_samples[i] = sample_point(ref_geometry, ref_triangle, barycentric0);
				ref_triangles[i] = ref_triangle;

				// Sample target surface
				uint32_t target_triangle = target_triangle_dist(gen);
				target_samples[i] = sample_point(tcas.ref, target_triangle, barycentric1);
				target_triangles[i] = target_triangle;
			}

			// Compute closest points on either end for the error metric
			ref_cas.precache_query(target_samples);
			tcas.precache_query(ref_samples);

			// Compute target samples -> ref
			ref_cas.query(target_samples, closest, bary, distance, triangle);

			for (uint32_t i = 0; i < sample_batch; i++) {
				glm::vec3 v_tgt = normalize(target_samples[i]);
				glm::vec3 v_ref = normalize(closest[i]);
				dpm += glm::distance(v_tgt, v_ref);
			}

			#pragma omp parallel for reduction(+:dnormal)
			for (uint32_t i = 0; i < sample_batch; i++) {
				glm::vec3 n_tgt;

				{
					const glm::uvec3 &t = tcas.ref.triangles[target_triangles[i]];
					const glm::vec3 &n0 = tgt_normals[t.x];
					const glm::vec3 &n1 = tgt_normals[t.y];
					const glm::vec3 &n2 = tgt_normals[t.z];
					const glm::vec3 &b = target_bary[i];
					n_tgt = n0 * b.x + n1 * b.y + n2 * b.z;
				}

				glm::vec3 n_ref;

				{
					const glm::uvec3 &t = ref_geometry.triangles[triangle[i]];
					const glm::vec3 &n0 = ref_normals[t.x];
					const glm::vec3 &n1 = ref_normals[t.y];
					const glm::vec3 &n2 = ref_normals[t.z];
					const glm::vec3 &b = bary[i];
					n_ref = n0 * b.x + n1 * b.y + n2 * b.z;
				}

				// Compute angular difference
				if (glm::dot(n_tgt, n_ref) < 0.0f)
					n_tgt = -n_tgt;

				float dot = glm::dot(n_tgt, n_ref);
				dot = glm::clamp(dot, -1.0f, 1.0f);
				dnormal += glm::degrees(std::acos(dot));
			}

			// Compute ref samples -> target
			tcas.query(ref_samples, closest, bary, distance, triangle);

			for (uint32_t i = 0; i < sample_batch; i++) {
				glm::vec3 v_tgt = normalize(closest[i]);
				glm::vec3 v_ref = normalize(ref_samples[i]);
				dpm += glm::distance(v_tgt, v_ref);
			}

			for (uint32_t i = 0; i < sample_batch; i++) {
				glm::vec3 n_tgt;

				{
					const glm::uvec3 &t = tcas.ref.triangles[triangle[i]];
					const glm::vec3 &n0 = tgt_normals[t.x];
					const glm::vec3 &n1 = tgt_normals[t.y];
					const glm::vec3 &n2 = tgt_normals[t.z];
					const glm::vec3 &b = bary[i];
					n_tgt = n0 * b.x + n1 * b.y + n2 * b.z;
				}

				glm::vec3 n_ref;

				{
					const glm::uvec3 &t = ref_geometry.triangles[ref_triangles[i]];
					const glm::vec3 &n0 = ref_normals[t.x];
					const glm::vec3 &n1 = ref_normals[t.y];
					const glm::vec3 &n2 = ref_normals[t.z];
					const glm::vec3 &b = ref_bary[i];
					n_ref = n0 * b.x + n1 * b.y + n2 * b.z;
				}

				// Compute angular difference
				if (glm::dot(n_tgt, n_ref) < 0.0f)
					n_tgt = -n_tgt;

				float dot = glm::dot(n_tgt, n_ref);
				dot = glm::clamp(dot, -1.0f, 1.0f);
				dnormal += glm::degrees(std::acos(dot));
			}

			bar.set_progress(100.0f * float(it)/float(samples/sample_batch));
		}

		dpm /= 2 * samples;
		dnormal /= 2 * samples;
		errors.push_back({ r, dpm, dnormal });
	}

	// TODO: compare representation sizes using purely the geometric data

	// Compute file sizes
	double complexes_size = std::filesystem::file_size(complexes_path);
	double encodings_size = std::filesystem::file_size(encodings_path);
	double model_size = std::filesystem::file_size(model_path);
	double points_size = std::filesystem::file_size(points_path);

	// Compute reference geometry size
	double ref_geometry_size = 0;
	ref_geometry_size += ref_geometry.vertices.size() * sizeof(glm::vec3);
	ref_geometry_size += ref_geometry.triangles.size() * sizeof(glm::uvec3);

	// Convert to MB and show again
	complexes_size /= 1024 * 1024;
	encodings_size /= 1024 * 1024;
	model_size /= 1024 * 1024;
	points_size /= 1024 * 1024;

	double total_size = complexes_size + encodings_size + model_size + points_size;

	double prop_complexes = complexes_size / total_size;
	double prop_encodings = encodings_size / total_size;
	double prop_model = model_size / total_size;
	double prop_points = points_size / total_size;

	ref_geometry_size /= 1024 * 1024;

	printf("\r\033[2K");
	printf("\n%s\n", std::string(80, '-').c_str());
	printf("Analytics:\n");
	printf("%s\n\n", std::string(80, '-').c_str());

	printf("Reference geometry: %4.3f MB (%7d vertices, %7d triangles)\n", ref_geometry_size, ref_geometry.vertices.size(), ref_geometry.triangles.size());
	printf("Total:              %4.3f MB\n\n", total_size);

	printf("Complexes:          %4.3f MB (%3.2f%%)\n", complexes_size, prop_complexes * 100.0f);
	printf("Encodings:          %4.3f MB (%3.2f%%)\n", encodings_size, prop_encodings * 100.0f);
	printf("Model:              %4.3f MB (%3.2f%%)\n", model_size, prop_model * 100.0f);
	printf("Points:             %4.3f MB (%3.2f%%)\n\n", points_size, prop_points * 100.0f);

	printf("Compression ratio:  %4.3fx\n\n", (ref_geometry_size - total_size) / total_size);

	for (const auto &e : errors)
		printf("Resolution %2d: dpm = %2.5f, dnormal = %2.5f\n", std::get <0> (e), std::get <1> (e), std::get <2> (e));
}
