#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <omp.h>
#include <immintrin.h>

float dot_product_avx512_aligned(const float *a, const float *b, int D) {
    __m512 sum_vec = _mm512_setzero_ps();
    int i = 0;
    for (; i <= D - 16; i += 16) {
        __m512 a_vec = _mm512_load_ps(a + i);
        __m512 b_vec = _mm512_load_ps(b + i);
        sum_vec = _mm512_fmadd_ps(a_vec, b_vec, sum_vec);
    }
    return _mm512_reduce_add_ps(sum_vec);
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    uint32_t N, D;
    std::cin.read(reinterpret_cast<char *>(&N), sizeof(N));
    std::cin.read(reinterpret_cast<char *>(&D), sizeof(D));

    float *data = (float *)_mm_malloc((size_t)N * D * sizeof(float), 64);
    std::cin.read(reinterpret_cast<char *>(data), (long)N * D * sizeof(float));

    std::vector<float> inv_norms(N);
    #pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < N; ++i) {
        float norm_sq = dot_product_avx512_aligned(data + (long)i * D, data + (long)i * D, D);
        inv_norms[i] = 1.0f / (std::sqrt(norm_sq) + 1e-12f);
    }

    std::vector<float> dot_products(N * N);
    const int B = 64;

    #pragma omp parallel for schedule(dynamic)
    for (int i_block = 0; i_block < N; i_block += B) {
        for (int j_block = i_block; j_block < N; j_block += B) {
            int i_max = std::min(i_block + B, (int)N);
            int j_max = std::min(j_block + B, (int)N);

            if (i_block == j_block) {
                for (int i = i_block; i < i_max; ++i) {
                    for (int j = i; j < j_max; ++j) {
                        float dp = dot_product_avx512_aligned(data + (long)i * D, data + (long)j * D, D);
                        dot_products[(long)i * N + j] = dp;
                        dot_products[(long)j * N + i] = dp;
                    }
                }
            } else {
                for (int i = i_block; i < i_max; ++i) {
                    int j = j_block;
                    for (; j <= j_max - 2; j += 2) {
                        const float* vec_i = data + (long)i * D;
                        
                        const float* vec_j0 = data + (long)j * D;
                        float dp0 = dot_product_avx512_aligned(vec_i, vec_j0, D);
                        dot_products[(long)i * N + j] = dp0;
                        dot_products[(long)j * N + i] = dp0;

                        const float* vec_j1 = data + (long)(j + 1) * D;
                        float dp1 = dot_product_avx512_aligned(vec_i, vec_j1, D);
                        dot_products[(long)i * N + (j + 1)] = dp1;
                        dot_products[(long)(j + 1) * N + i] = dp1;
                    }
                    if (j < j_max) {
                    if (j < j_max) {
                         float dp = dot_product_avx512_aligned(data + (long)i * D, data + (long)j * D, D);
                         dot_products[(long)i * N + j] = dp;
                         dot_products[(long)j * N + i] = dp;
                    }
                }
            }
        }
    }

    std::vector<float> final_results(N * 4);
    #pragma omp parallel
    {
        std::vector<float> cosine_sim_thread_buffer(N);
        #pragma omp for schedule(static)
        for (uint32_t i = 0; i < N; ++i) {
            const float inv_norm_i = inv_norms[i];
            for (uint32_t j = 0; j < N; ++j) {
                cosine_sim_thread_buffer[j] = dot_products[(long)i * N + j] * inv_norm_i * inv_norms[j];
            }

            std::partial_sort(cosine_sim_thread_buffer.begin(),
                              cosine_sim_thread_buffer.begin() + 5,
                              cosine_sim_thread_buffer.end(),
                              std::greater<float>());
            
            float* result_ptr = final_results.data() + (long)i * 4;
            std::copy(cosine_sim_thread_buffer.data() + 1, cosine_sim_thread_buffer.data() + 5, result_ptr);
        }
    }

    std::cout.write(reinterpret_cast<char *>(final_results.data()), (long)N * 4 * sizeof(float));
    
    _mm_free(data);

    return 0;
}