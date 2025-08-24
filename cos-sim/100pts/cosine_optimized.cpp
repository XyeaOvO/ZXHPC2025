#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <omp.h>
#include <immintrin.h>

float dot_product_avx512_aligned(const float *a, const float *b, int D) {
    __m512 sum_vec = _mm512_setzero_ps();
    for (int i = 0; i <= D - 16; i += 16) {
        __m512 a_vec = _mm512_load_ps(a + i);
        __m512 b_vec = _mm512_load_ps(b + i);
        sum_vec = _mm512_fmadd_ps(a_vec, b_vec, sum_vec);
    }
    return _mm512_reduce_add_ps(sum_vec);
}

inline void micro_kernel_4x4(int i_start, int j_start, int D, int N, const float* data, float* dot_products) {
    const float* i_ptr0 = data + (long)(i_start + 0) * D;
    const float* i_ptr1 = data + (long)(i_start + 1) * D;
    const float* i_ptr2 = data + (long)(i_start + 2) * D;
    const float* i_ptr3 = data + (long)(i_start + 3) * D;

    const float* j_ptr0 = data + (long)(j_start + 0) * D;
    const float* j_ptr1 = data + (long)(j_start + 1) * D;
    const float* j_ptr2 = data + (long)(j_start + 2) * D;
    const float* j_ptr3 = data + (long)(j_start + 3) * D;

    __m512 acc00 = _mm512_setzero_ps(), acc01 = _mm512_setzero_ps(), acc02 = _mm512_setzero_ps(), acc03 = _mm512_setzero_ps();
    __m512 acc10 = _mm512_setzero_ps(), acc11 = _mm512_setzero_ps(), acc12 = _mm512_setzero_ps(), acc13 = _mm512_setzero_ps();
    __m512 acc20 = _mm512_setzero_ps(), acc21 = _mm512_setzero_ps(), acc22 = _mm512_setzero_ps(), acc23 = _mm512_setzero_ps();
    __m512 acc30 = _mm512_setzero_ps(), acc31 = _mm512_setzero_ps(), acc32 = _mm512_setzero_ps(), acc33 = _mm512_setzero_ps();

    for (int k = 0; k < D; k += 16) {
        __m512 v_i0 = _mm512_load_ps(i_ptr0 + k);
        __m512 v_i1 = _mm512_load_ps(i_ptr1 + k);
        __m512 v_i2 = _mm512_load_ps(i_ptr2 + k);
        __m512 v_i3 = _mm512_load_ps(i_ptr3 + k);

        __m512 v_j0 = _mm512_load_ps(j_ptr0 + k);
        acc00 = _mm512_fmadd_ps(v_i0, v_j0, acc00);
        acc10 = _mm512_fmadd_ps(v_i1, v_j0, acc10);
        acc20 = _mm512_fmadd_ps(v_i2, v_j0, acc20);
        acc30 = _mm512_fmadd_ps(v_i3, v_j0, acc30);

        __m512 v_j1 = _mm512_load_ps(j_ptr1 + k);
        acc01 = _mm512_fmadd_ps(v_i0, v_j1, acc01);
        acc11 = _mm512_fmadd_ps(v_i1, v_j1, acc11);
        acc21 = _mm512_fmadd_ps(v_i2, v_j1, acc21);
        acc31 = _mm512_fmadd_ps(v_i3, v_j1, acc31);

        __m512 v_j2 = _mm512_load_ps(j_ptr2 + k);
        acc02 = _mm512_fmadd_ps(v_i0, v_j2, acc02);
        acc12 = _mm512_fmadd_ps(v_i1, v_j2, acc12);
        acc22 = _mm512_fmadd_ps(v_i2, v_j2, acc22);
        acc32 = _mm512_fmadd_ps(v_i3, v_j2, acc32);
        
        __m512 v_j3 = _mm512_load_ps(j_ptr3 + k);
        acc03 = _mm512_fmadd_ps(v_i0, v_j3, acc03);
        acc13 = _mm512_fmadd_ps(v_i1, v_j3, acc13);
        acc23 = _mm512_fmadd_ps(v_i2, v_j3, acc23);
        acc33 = _mm512_fmadd_ps(v_i3, v_j3, acc33);
    }
    
    float dps[4][4];
    dps[0][0]=_mm512_reduce_add_ps(acc00); dps[0][1]=_mm512_reduce_add_ps(acc01); dps[0][2]=_mm512_reduce_add_ps(acc02); dps[0][3]=_mm512_reduce_add_ps(acc03);
    dps[1][0]=_mm512_reduce_add_ps(acc10); dps[1][1]=_mm512_reduce_add_ps(acc11); dps[1][2]=_mm512_reduce_add_ps(acc12); dps[1][3]=_mm512_reduce_add_ps(acc13);
    dps[2][0]=_mm512_reduce_add_ps(acc20); dps[2][1]=_mm512_reduce_add_ps(acc21); dps[2][2]=_mm512_reduce_add_ps(acc22); dps[2][3]=_mm512_reduce_add_ps(acc23);
    dps[3][0]=_mm512_reduce_add_ps(acc30); dps[3][1]=_mm512_reduce_add_ps(acc31); dps[3][2]=_mm512_reduce_add_ps(acc32); dps[3][3]=_mm512_reduce_add_ps(acc33);

    for(int i=0; i<4; ++i) {
        for(int j=0; j<4; ++j) {
            dot_products[(long)(i_start + i) * N + (j_start + j)] = dps[i][j];
            dot_products[(long)(j_start + j) * N + (i_start + i)] = dps[i][j];
        }
    }
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
    const int IR = 4;
    const int JR = 4;

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
                int i = i_block;
                for (; i <= i_max - IR; i += IR) {
                    int j = j_block;
                    for (; j <= j_max - JR; j += JR) {
                        micro_kernel_4x4(i, j, D, N, data, dot_products.data());
                    }
                    for (; j < j_max; ++j) {
                        for(int ii=0; ii<IR; ++ii) {
                           float dp = dot_product_avx512_aligned(data + (long)(i + ii) * D, data + (long)j * D, D);
                           dot_products[(long)(i + ii) * N + j] = dp;
                           dot_products[(long)j * N + (i + ii)] = dp;
                        }
                    }
                }
                for (; i < i_max; ++i) {
                    for (int j = j_block; j < j_max; ++j) {
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