#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <omp.h>
#include <immintrin.h>

float dot_product_avx(const float *a, const float *b, int D) {
    __m256 sum_vec = _mm256_setzero_ps();
    int i = 0;
    for (; i <= D - 8; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 b_vec = _mm256_loadu_ps(b + i);
        sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
    }
    __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum_vec, 1), _mm256_castps256_ps128(sum_vec));
    __m128 sum64 = _mm_hadd_ps(sum128, sum128);
    __m128 sum32 = _mm_hadd_ps(sum64, sum64);
    float result = _mm_cvtss_f32(sum32);
    for (; i < D; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    uint32_t N, D;
    std::cin.read(reinterpret_cast<char *>(&N), sizeof(N));
    std::cin.read(reinterpret_cast<char *>(&D), sizeof(D));

    std::vector<float> data(N * D);
    std::cin.read(reinterpret_cast<char *>(data.data()), N * D * sizeof(float));

    std::vector<float> norms(N);
    #pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < N; ++i) {
        const float *vec = data.data() + i * D;
        norms[i] = std::sqrt(dot_product_avx(vec, vec, D));
    }

    std::vector<float> dot_products(N * N);
    const int B = 64;

    #pragma omp parallel for schedule(dynamic) collapse(2)
    for (uint32_t i_block = 0; i_block < N; i_block += B) {
        for (uint32_t j_block = 0; j_block < N; j_block += B) {
            uint32_t i_max = std::min(i_block + B, N);
            uint32_t j_max = std::min(j_block + B, N);
            for (uint32_t i = i_block; i < i_max; ++i) {
                for (uint32_t j = j_block; j < j_max; ++j) {
                    dot_products[i * N + j] = dot_product_avx(data.data() + i * D, data.data() + j * D, D);
                }
            }
        }
    }

    std::vector<float> final_results(N * 4);
    #pragma omp parallel for schedule(dynamic)
    for (uint32_t i = 0; i < N; ++i) {
        std::vector<float> cosine_sim(N);
        const float norm_i = norms[i];
        for (uint32_t j = 0; j < N; ++j) {
            cosine_sim[j] = dot_products[i * N + j] / (norm_i * norms[j] + 1e-12f);
        }

        std::partial_sort(cosine_sim.begin(), cosine_sim.begin() + 5, cosine_sim.end(), std::greater<float>());
        
        float* result_ptr = final_results.data() + i * 4;
        std::copy(cosine_sim.data() + 1, cosine_sim.data() + 5, result_ptr);
    }

    std::cout.write(reinterpret_cast<char *>(final_results.data()), N * 4 * sizeof(float));

    return 0;
}