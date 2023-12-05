#include "common.hpp"

// Matrix multiplication with bias
static std::vector <float> matmul_biased(const std::vector <float> &Wm_c,
		const std::vector <float> &Xbatched,
		size_t in, size_t out)
{
	// Check sizes
	ulog_assert(Wm_c.size() == (in + 1) * out, "matmul_biased", "Wm size mismatch\n");

	ulog_assert(Xbatched.size() % in == 0,  "matmul_biased", "X size mismatch\n");

	// Prepare result
	size_t batch = Xbatched.size() / in;

	std::vector <float> Ybatched(out * batch);

	// Perform matrix multiplication and add bias
	#pragma omp parallel
	for (size_t b = 0; b < batch; b++) {
		for (size_t i = 0; i < out; i++) {
			float sum = Wm_c[i * (in + 1) + in];
			for (size_t j = 0; j < in; j++)
				sum += Wm_c[i * (in + 1) + j] * Xbatched[b * in + j];
			Ybatched[b * out + i] = sum;
		}
	}

	return Ybatched;
}

// Interpolate to obtain the vertex positions and features
static auto interpolate(uint32_t sample_rate)
{
	uint32_t vertex_count = sample_rate * sample_rate * g_sdc.complex_count;
	ulog_info("interpolate", "Evaluating %d vertices\n", vertex_count);

	std::vector <glm::vec3> lerped_P;
	std::vector <float>     lerped_F;

	lerped_P.resize(vertex_count);
	lerped_F.resize(vertex_count * g_sdc.feature_size);

	uint32_t i = 0;

	#pragma omp parallel for
	for (uint32_t c = 0; c < g_sdc.complex_count; c++) {
		const glm::ivec4 &complex = g_sdc.complexes[c];

		glm::vec3 v00 = g_sdc.vertices[complex.x];
		glm::vec3 v10 = g_sdc.vertices[complex.y];
		glm::vec3 v01 = g_sdc.vertices[complex.w];
		glm::vec3 v11 = g_sdc.vertices[complex.z];

		auto get_feature = [&](uint32_t v) {
			std::vector <float> f(g_sdc.feature_size);
			for (uint32_t i = 0; i < g_sdc.feature_size; i++)
				f[i] = g_sdc.features[v * g_sdc.feature_size + i];
			return f;
		};

		auto set_feature = [&](uint32_t v, std::vector <float> &f) {
			for (uint32_t i = 0; i < g_sdc.feature_size; i++)
				lerped_F[v * g_sdc.feature_size + i] = f[i];
		};

		std::vector <float> f00 = get_feature(complex.x);
		std::vector <float> f10 = get_feature(complex.y);
		std::vector <float> f01 = get_feature(complex.w);
		std::vector <float> f11 = get_feature(complex.z);

		for (uint32_t ix = 0; ix < sample_rate; ix++) {
			for (uint32_t iy = 0; iy < sample_rate; iy++) {
				float u = (float) ix / (sample_rate - 1);
				float v = (float) iy / (sample_rate - 1);

				{
					glm::vec3 lp00 = v00 * (1 - u) * (1 - v);
					glm::vec3 lp10 = v10 * u * (1 - v);
					glm::vec3 lp01 = v01 * (1 - u) * v;
					glm::vec3 lp11 = v11 * u * v;
					lerped_P[i] = lp00 + lp10 + lp01 + lp11;
				}

				{
					std::vector <float> f(g_sdc.feature_size);
					for (uint32_t k = 0; k < g_sdc.feature_size; k++) {
						float f00k = f00[k] * (1 - u) * (1 - v);
						float f10k = f10[k] * u * (1 - v);
						float f01k = f01[k] * (1 - u) * v;
						float f11k = f11[k] * u * v;
						f[k] = f00k + f10k + f01k + f11k;
					}

					set_feature(i, f);
				}

				i++;
			}
		}
	}

	return std::make_pair(lerped_P, lerped_F);
}

// Embedding
static auto embedding(const std::vector <glm::vec3> &lerped_P, const std::vector <float> &lerped_E, uint32_t sample_rate)
{
	uint32_t vertex_count = sample_rate * sample_rate * g_sdc.complex_count;
	uint32_t ffwd_size = g_sdc.ffwd_size();

	ulog_assert(ffwd_size == g_dnn.Hs[0], "eval", "embedded_size != g_dnn.H0 [%u != %u]\n", ffwd_size, g_dnn.Hs[0]);

	std::vector <float> embedded;
	embedded.resize(vertex_count * ffwd_size);

	#pragma omp parallel for
	for (uint32_t i = 0; i < vertex_count; i++) {
		float *vembedded = &embedded[i * ffwd_size];

		// First copy the feature
		const float *lfeature = &lerped_E[i * g_sdc.feature_size];
		for (uint32_t k = 0; k < g_sdc.feature_size; k++)
			vembedded[k] = lfeature[k];

		// Positional encoding
		const glm::vec3 &p = lerped_P[i];

		std::vector <float> pos_enc;
		pos_enc.push_back(p.x);
		pos_enc.push_back(p.y);
		pos_enc.push_back(p.z);

		for (uint32_t j = 0; j < FREQUENCIES; j++) {
			glm::vec3 sp = glm::sin(powf(2.0f, j) * p);
			glm::vec3 cp = glm::cos(powf(2.0f, j) * p);
			pos_enc.push_back(sp.x);
			pos_enc.push_back(sp.y);
			pos_enc.push_back(sp.z);
			pos_enc.push_back(cp.x);
			pos_enc.push_back(cp.y);
			pos_enc.push_back(cp.z);
		}

		ulog_assert(pos_enc.size() == g_sdc.vertex_encoding_size(), "eval", "pos_enc.size() != encoding_size [%lu != %u]\n", pos_enc.size(), g_sdc.vertex_encoding_size());

		for (uint32_t k = 0; k < pos_enc.size(); k++)
			vembedded[g_sdc.feature_size + k] = pos_enc[k];
	}

	return embedded;
}

// Evaluate the neural network
inline float leaky_relu(float x)
{
	return x > 0 ? x : 0.01f * x;
}

std::vector <glm::vec3> eval(uint32_t sample_rate)
{
	// Interpolate to obtain the vertex positions and features
	auto [lerped_P, lerped_F] = interpolate(sample_rate);

	// Construct the network input with embeddings
	uint32_t vertex_count = sample_rate * sample_rate * g_sdc.complex_count;

	// Evaluate the neural network layers
	std::vector <float> hidden = embedding(lerped_P, lerped_F, sample_rate);
	ulog_info("eval", "Constructed network input embeddings\n");

	#pragma omp parallel for
	for (uint32_t i = 0; i < g_dnn.Wm_c.size(); i++) {
		ulog_info("eval", "Evaluating layer %u\n", i);

		hidden = matmul_biased(g_dnn.Wm_c[i], hidden, g_dnn.Hs[i], g_dnn.Ws[i]);
		if (i < g_dnn.Wm_c.size() - 1) {
			for (uint32_t j = 0; j < hidden.size(); j++)
				hidden[j] = leaky_relu(hidden[j]);
		}
	}

	// Apply displacements
	glm::vec3 *displacements = (glm::vec3 *) hidden.data();

	std::vector <glm::vec3> final_P(vertex_count);
	for (uint32_t i = 0; i < vertex_count; i++)
		final_P[i] = lerped_P[i] + displacements[i];

	return final_P;
}
