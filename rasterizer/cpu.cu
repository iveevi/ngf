#include "common.hpp"

// Matrix multiplication with bias
template <float (*act)(float)>
static std::vector <float> matmul_biased(const std::vector <float> &W,
		const std::vector <float> &Xbatched,
		size_t in, size_t out)
{
	// Check sizes
	ulog_assert(W.size() == (in + 1) * out, "matmul_biased", "W size mismatch\n");
	ulog_assert(Xbatched.size() % in == 0,  "matmul_biased", "X size mismatch\n");

	// Prepare result
	size_t batch = Xbatched.size() / in;

	std::vector <float> Ybatched(out * batch);

	// Perform matrix multiplication and add bias
	#pragma omp parallel
	for (size_t b = 0; b < batch; b++) {
		const float *Xrow = &Xbatched[b * in];
		float *Yrow = &Ybatched[b * out];

		for (size_t i = 0; i < out; i++) {
			const float *Wrow = &W[i * (in + 1)];
			float sum = Wrow[in];
			for (size_t j = 0; j < in; j++)
				sum += Wrow[j] * Xrow[j];

			if constexpr (act)
				Yrow[i] = act(sum);
			else
				Yrow[i] = sum;
		}
	}

	return Ybatched;
}

// Matrix multiplication without bias or activation
static std::vector <float> matmul(const std::vector <float> &W,
		const std::vector <float> &Xbatched,
		size_t in, size_t out)
{
	// Check sizes
	ulog_assert(W.size() == (in + 1) * out, "matmul", "W size mismatch\n");
	ulog_assert(Xbatched.size() % in == 0,  "matmul", "X size mismatch\n");

	// Prepare result
	size_t batch = Xbatched.size() / in;

	std::vector <float> Ybatched(out * batch);

	// Perform matrix multiplication and add bias
	#pragma omp parallel
	for (size_t b = 0; b < batch; b++) {
		const float *Xrow = &Xbatched[b * in];
		float *Yrow = &Ybatched[b * out];

		for (size_t i = 0; i < out; i++) {
			const float *Wrow = &W[i * (in + 1)];
			float sum = 0.0f;
			for (size_t j = 0; j < in; j++)
				sum += Wrow[j] * Xrow[j];
			Yrow[i] = sum;
		}
	}

	return Ybatched;
}

// Plain GEMM
static std::vector <float> gemm(const std::vector <float> &A, const std::vector <float> &B, size_t in, size_t out)
{
	ulog_assert(A.size() == in * out, "gemm", "A size mismatch\n");
	ulog_assert(B.size() % in == 0, "gemm", "B size mismatch\n");

	size_t rows = B.size() / in;
	std::vector <float> C(out * rows);

	#pragma omp parallel
	for (size_t i = 0; i < out; i++) {
		const float *Arow = &A[i * in];
		float *Crow = &C[i * rows];

		for (size_t j = 0; j < rows; j++) {
			const float *Brow = &B[j * in];
			float sum = 0.0f;
			for (size_t k = 0; k < in; k++)
				sum += Arow[k] * Brow[k];
			Crow[j] = sum;
		}
	}

	return C;
}

// General matrix multiplication for gradient computation; assumes the last row is the bias and ignores it
static std::vector <float> gradient_gemm(const std::vector <float> &A, const std::vector <float> &B, size_t in, size_t out)
{
	// Compute A * B
	ulog_assert(A.size() == (in + 1) * out, "gradient gemm", "A size mismatch\n");
	ulog_assert(B.size() == in * 3, "gradient gemm", "B size mismatch\n");

	std::vector <float> C(out * 3);

	#pragma omp parallel
	for (size_t i = 0; i < out; i++) {
		const float *Arow = &A[i * (in + 1)];
		float *Crow = &C[i * 3];

		Crow[0] = 0.0f;
		Crow[1] = 0.0f;
		Crow[2] = 0.0f;

		for (size_t j = 0; j < in; j++) {
			const float *Brow = &B[j * 3];
			Crow[0] += Arow[j] * Brow[0];
			Crow[1] += Arow[j] * Brow[1];
			Crow[2] += Arow[j] * Brow[2];
		}
	}

	return C;
}

// Shur multiplication between two vectors
static std::vector <float> shur(const std::vector <float> &X, const std::vector <float> &Y)
{
	// Check sizes
	ulog_assert(X.size() == Y.size(), "shur", "X/Y size mismatch\n");

	// Prepare result
	size_t batch = X.size();

	std::vector <float> Z(batch);

	// Perform matrix multiplication and add bias
	#pragma omp parallel
	for (size_t b = 0; b < batch; b++)
		Z[b] = X[b] * Y[b];

	return Z;
}

std::vector <glm::vec3> eval(uint32_t sample_rate)
{
	uint32_t vertex_count = sample_rate * sample_rate * g_sdc.complex_count;
	ulog_info("eval", "Evaluating %d vertices\n", vertex_count);

	std::vector <glm::vec3> lerped_P;
	std::vector <float>     lerped_E;

	lerped_P.resize(vertex_count);
	lerped_E.resize(vertex_count * g_sdc.feature_count);

	uint32_t i = 0;
	for (uint32_t c = 0; c < g_sdc.complex_count; c++) {
		const glm::uvec4 &complex = g_sdc.complexes[c];

		glm::vec3 v00 = g_sdc.vertices[complex.x];
		glm::vec3 v10 = g_sdc.vertices[complex.y];
		glm::vec3 v01 = g_sdc.vertices[complex.w];
		glm::vec3 v11 = g_sdc.vertices[complex.z];

		auto get_feature = [&](uint32_t v) {
			std::vector <float> f(g_sdc.feature_count);
			for (uint32_t i = 0; i < g_sdc.feature_count; i++)
				f[i] = g_sdc.features[v * g_sdc.feature_count + i];
			return f;
		};

		auto set_feature = [&](uint32_t v, std::vector <float> &f) {
			for (uint32_t i = 0; i < g_sdc.feature_count; i++)
				lerped_E[v * g_sdc.feature_count + i] = f[i];
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
					glm::vec3 lp00 = v00 * u * v;
					glm::vec3 lp10 = v10 * (1.0f - u) * v;
					glm::vec3 lp01 = v01 * u * (1.0f - v);
					glm::vec3 lp11 = v11 * (1.0f - u) * (1.0f - v);
					lerped_P[i] = lp00 + lp10 + lp01 + lp11;
				}

				{
					std::vector <float> f(g_sdc.feature_count);
					for (uint32_t k = 0; k < g_sdc.feature_count; k++) {
						float f00k = f00[k] * u * v;
						float f10k = f10[k] * (1.0f - u) * v;
						float f01k = f01[k] * u * (1.0f - v);
						float f11k = f11[k] * (1.0f - u) * (1.0f - v);
						f[k] = f00k + f10k + f01k + f11k;
					}

					set_feature(i, f);
				}

				i++;
			}
		}
	}

	ulog_info("eval", "Lerped vertices and features\n");

	// Construct the network input with embeddings
	// TODO: get these from the file as well...
	constexpr uint32_t L = 8;

	uint32_t embedded_size = g_sdc.feature_count + 3 * (2 * L + 1);
	ulog_assert(embedded_size == g_dnn.H0, "eval", "embedded_size != g_dnn.H0 [%u != %u]\n", embedded_size, g_dnn.H0);

	std::vector <float> embedded;
	embedded.resize(vertex_count * embedded_size);

	for (uint32_t i = 0; i < vertex_count; i++) {
		float *vembedded = &embedded[i * embedded_size];

		// First copy the feature
		float *lfeature = &lerped_E[i * g_sdc.feature_count];
		for (uint32_t k = 0; k < g_sdc.feature_count; k++)
			vembedded[k] = lfeature[k];

		// Positional encoding
		const glm::vec3 &p = lerped_P[i];

		std::vector <float> pos_enc;
		pos_enc.push_back(p.x);
		pos_enc.push_back(p.y);
		pos_enc.push_back(p.z);

		for (uint32_t j = 0; j < L; j++) {
			glm::vec3 sp = glm::sin(powf(2.0f, j) * p);
			glm::vec3 cp = glm::cos(powf(2.0f, j) * p);
			pos_enc.push_back(sp.x);
			pos_enc.push_back(sp.y);
			pos_enc.push_back(sp.z);
			pos_enc.push_back(cp.x);
			pos_enc.push_back(cp.y);
			pos_enc.push_back(cp.z);
		}

		ulog_assert(pos_enc.size() == 3 * (2 * L + 1), "eval", "pos_enc.size() != 3 * (2 * L + 1) [%lu != %u]\n", pos_enc.size(), 3 * (2 * L + 1));

		for (uint32_t k = 0; k < pos_enc.size(); k++)
			vembedded[g_sdc.feature_count + k] = pos_enc[k];
	}

	ulog_info("eval", "Constructed network input embeddings\n");

	printf("embedding[0]:");
	for (uint32_t i = 0; i < embedded_size; i++)
		printf(" %f", embedded[i]);
	printf("\n");

	// Evaluate the first network layer
	std::vector <float> hidden;

	hidden = matmul_biased <sinf> (g_dnn.Wm0c, embedded, embedded_size, g_dnn.W0);
	ulog_assert(hidden.size() == vertex_count * g_dnn.W0, "eval", "hidden.size() != vertex_count * g_dnn.W0 [%lu != %u]\n", hidden.size(), vertex_count * g_dnn.W0);
	ulog_info("eval", "Evaluated first network layer\n");

	hidden = matmul_biased <sinf> (g_dnn.Wm1c, hidden, g_dnn.W0, g_dnn.W1);
	ulog_assert(hidden.size() == vertex_count * g_dnn.W1, "eval", "hidden.size() != vertex_count * g_dnn.W1 [%lu != %u]\n", hidden.size(), vertex_count * g_dnn.W1);
	ulog_assert(g_dnn.W2 == 3, "eval", "W2 != 3");
	ulog_info("eval", "Evaluated second network layer\n");

	hidden = matmul_biased <nullptr> (g_dnn.Wm2c, hidden, g_dnn.W1, g_dnn.W2);
	ulog_assert(hidden.size() == vertex_count * g_dnn.W2, "eval", "hidden.size() != vertex_count * g_dnn.W2 [%lu != %u]\n", hidden.size(), vertex_count * g_dnn.W2);
	ulog_info("eval", "Evaluated final network layer\n");

	// Apply displacements
	glm::vec3 *displacements = (glm::vec3 *) hidden.data();

	printf("displacements[0]: %f %f %f\n", displacements[0].x, displacements[0].y, displacements[0].z);

	std::vector <glm::vec3> final_P(vertex_count);
	for (uint32_t i = 0; i < vertex_count; i++)
		final_P[i] = lerped_P[i] + displacements[i];

	printf("final_P[0]: %f %f %f\n", final_P[0].x, final_P[0].y, final_P[0].z);

	return final_P;
}

std::vector <glm::vec3> eval_normals(uint32_t sample_rate)
{
	// Compute the embedding gradients with respect to U and V
	uint32_t vertex_count = sample_rate * sample_rate * g_sdc.complex_count;
	ulog_info("eval", "Evaluating normal vector for %d vertices\n", vertex_count);

	// Compute the embedding derivatives
	constexpr uint32_t L = 8;

	uint32_t embedded_size = g_sdc.feature_count + 3 * (2 * L + 1);

	std::vector <glm::vec3> DV_u(vertex_count);
	std::vector <glm::vec3> DV_v(vertex_count);

	std::vector <float> Dembedding_u(vertex_count * embedded_size);
	std::vector <float> Dembedding_v(vertex_count * embedded_size);

	uint32_t i = 0;
	for (uint32_t c = 0; c < g_sdc.complex_count; c++) {
		const glm::uvec4 &complex = g_sdc.complexes[c];

		glm::vec3 v00 = g_sdc.vertices[complex.x];
		glm::vec3 v10 = g_sdc.vertices[complex.y];
		glm::vec3 v01 = g_sdc.vertices[complex.w];
		glm::vec3 v11 = g_sdc.vertices[complex.z];

		auto get_feature = [&](uint32_t v) {
			std::vector <float> f(g_sdc.feature_count);
			for (uint32_t i = 0; i < g_sdc.feature_count; i++)
				f[i] = g_sdc.features[v * g_sdc.feature_count + i];
			return f;
		};

		std::vector <float> f00 = get_feature(complex.x);
		std::vector <float> f10 = get_feature(complex.y);
		std::vector <float> f01 = get_feature(complex.w);
		std::vector <float> f11 = get_feature(complex.z);

		for (uint32_t ix = 0; ix < sample_rate; ix++) {
			for (uint32_t iy = 0; iy < sample_rate; iy++) {
				float u = (float) ix / (sample_rate - 1);
				float v = (float) iy / (sample_rate - 1);

				// Feature derivatives
				float *du_base = &Dembedding_u[i * embedded_size];
				float *dv_base = &Dembedding_v[i * embedded_size];

				for (uint32_t k = 0; k < g_sdc.feature_count; k++) {
					float dfdu = v * (f00[k] - f10[k]) + (1 - v) * (f01[k] - f11[k]);
					float dfdv = u * (f00[k] - f01[k]) + (1 - u) * (f10[k] - f11[k]);

					du_base[k] = dfdu;
					dv_base[k] = dfdv;
				}

				// Vertex derivatives
				du_base += g_sdc.feature_count;
				dv_base += g_sdc.feature_count;

				glm::vec3 dPdu = v * (v00 - v10) + (1 - v) * (v01 - v11);
				glm::vec3 dPdv = u * (v00 - v01) + (1 - u) * (v10 - v11);

				du_base[0] = dPdu.x;
				du_base[1] = dPdu.y;
				du_base[2] = dPdu.z;

				dv_base[0] = dPdv.x;
				dv_base[1] = dPdv.y;
				dv_base[2] = dPdv.z;

				DV_u[i] = dPdu;
				DV_v[i] = dPdv;

				glm::vec3 P = v00 * u * v
					+ v10 * (1 - u) * v
					+ v01 * u * (1 - v)
					+ v11 * (1 - u) * (1 - v);

				// Encoded derivatives
				du_base += 3;
				dv_base += 3;

				for (uint32_t k = 0; k < L; k++) {
					float K = powf(2.0f, k);
					glm::vec3 dsin = K * glm::cos(K * P);
					glm::vec3 dcos = -K * glm::sin(K * P);

					glm::vec3 dsin_u = dsin * dPdu;
					glm::vec3 dcos_u = dcos * dPdu;

					glm::vec3 dsin_v = dsin * dPdv;
					glm::vec3 dcos_v = dcos * dPdv;

					du_base[0] = dsin_u.x;
					du_base[1] = dsin_u.y;
					du_base[2] = dsin_u.z;
					du_base[3] = dcos_u.x;
					du_base[4] = dcos_u.y;
					du_base[5] = dcos_u.z;

					dv_base[0] = dsin_v.x;
					dv_base[1] = dsin_v.y;
					dv_base[2] = dsin_v.z;
					dv_base[3] = dcos_v.x;
					dv_base[4] = dcos_v.y;
					dv_base[5] = dcos_v.z;

					du_base += 6;
					dv_base += 6;
				}

				i++;
			}
		}
	}

	ulog_info("eval", "Computed embedding derivatives\n");
	ulog_assert(embedded_size == g_dnn.H0, "eval", "embedded_size != g_dnn.H0 [%u != %u]\n", embedded_size, g_dnn.H0);

	// First layer derivatives
	auto Dphi0_u = matmul(g_dnn.Wm0c, Dembedding_u, embedded_size, g_dnn.W0);
	auto Dsin_phi0_u = matmul_biased <cosf> (g_dnn.Wm0c, Dembedding_u, embedded_size, g_dnn.W0);
	auto DPhi0_u = shur(Dphi0_u, Dsin_phi0_u);

	auto Dphi0_v = matmul(g_dnn.Wm0c, Dembedding_v, embedded_size, g_dnn.W0);
	auto Dsin_phi0_v = matmul_biased <cosf> (g_dnn.Wm0c, Dembedding_v, embedded_size, g_dnn.W0);
	auto DPhi0_v = shur(Dphi0_v, Dsin_phi0_v);

	ulog_info("eval", "Computed first layer derivatives\n");
	ulog_assert(DPhi0_u.size() % g_dnn.W0 == 0, "eval", "DPhi0_u.size() %% g_dnn.W0 != 0 [%u %% %u != 0]\n", DPhi0_u.size(), g_dnn.W0);
	ulog_assert(DPhi0_v.size() % g_dnn.W0 == 0, "eval", "DPhi0_v.size() %% g_dnn.W0 != 0 [%u %% %u != 0]\n", DPhi0_v.size(), g_dnn.W0);

	// Second layer derivatives
	auto Dphi1_u = matmul(g_dnn.Wm1c, DPhi0_u, g_dnn.W0, g_dnn.W1);
	auto Dsin_phi1_u = matmul_biased <cosf> (g_dnn.Wm1c, DPhi0_u, g_dnn.W0, g_dnn.W1);
	auto DPhi1_u = shur(Dphi1_u, Dsin_phi1_u);

	auto Dphi1_v = matmul(g_dnn.Wm1c, DPhi0_v, g_dnn.W0, g_dnn.W1);
	auto Dsin_phi1_v = matmul_biased <cosf> (g_dnn.Wm1c, DPhi0_v, g_dnn.W0, g_dnn.W1);
	auto DPhi1_v = shur(Dphi1_v, Dsin_phi1_v);

	ulog_info("eval", "Computed second layer derivatives\n");
	ulog_assert(DPhi1_u.size() % g_dnn.W1 == 0, "eval", "DPhi1_u.size() %% g_dnn.W1 != 0 [%u %% %u != 0]\n", DPhi1_u.size(), g_dnn.W1);
	ulog_assert(DPhi1_v.size() % g_dnn.W1 == 0, "eval", "DPhi1_v.size() %% g_dnn.W1 != 0 [%u %% %u != 0]\n", DPhi1_v.size(), g_dnn.W1);

	// Final layer derivatives
	auto Dmlp_u = matmul(g_dnn.Wm2c, DPhi1_u, g_dnn.W1, g_dnn.W2);
	auto Dmlp_v = matmul(g_dnn.Wm2c, DPhi1_v, g_dnn.W1, g_dnn.W2);

	ulog_info("eval", "Computed final layer derivatives\n");
	ulog_assert(Dmlp_u.size() == vertex_count * 3, "eval", "Dmlp_u.size() != vertex_count * 3 [%u != %u * 3]\n", Dmlp_u.size(), vertex_count);
	ulog_assert(Dmlp_v.size() == vertex_count * 3, "eval", "Dmlp_v.size() != vertex_count * 3 [%u != %u * 3]\n", Dmlp_v.size(), vertex_count);

	// Compute normal vector as the cross product of the two partial derivatives
	glm::vec3 *partial_u = (glm::vec3 *) Dmlp_u.data();
	glm::vec3 *partial_v = (glm::vec3 *) Dmlp_v.data();

	std::vector <glm::vec3> normals(vertex_count);
	for (unsigned int i = 0; i < vertex_count; i++) {
		partial_u[i] += DV_u[i];
		partial_v[i] += DV_v[i];

		glm::vec3 normal = glm::cross(partial_u[i], partial_v[i]);
		normals[i] = glm::normalize(normal);
	}

	printf("Computed normals: [0] = (%f, %f, %f)\n", normals[0].x, normals[0].y, normals[0].z);

	return normals;
}
