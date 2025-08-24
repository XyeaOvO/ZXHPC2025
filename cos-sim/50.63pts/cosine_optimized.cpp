#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <omp.h>

float dot_product(const float *a, const float *b, int D) {
    float result = 0.0f;
    for (int i = 0; i < D; ++i) {
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
    for (uint32_t i = 0; i < N; ++i) {
        float sum_sq = 0.0f;
        const float *vec = data.data() + i * D;
        for (uint32_t j = 0; j < D; ++j) {
            sum_sq += vec[j] * vec[j];
        }
        norms[i] = std::sqrt(sum_sq);
    }
    std::vector<float> final_results(N * 4);

    #pragma omp parallel for schedule(dynamic)
    for (uint32_t i = 0; i < N; ++i) {
        std::vector<float> cosine_sim(N);
        const float *vec_i = data.data() + i * D;

        for (uint32_t j = 0; j < N; ++j) {
            const float *vec_j = data.data() + j * D;
            float dot_prod = dot_product(vec_i, vec_j, D);
            cosine_sim[j] = dot_prod / (norms[i] * norms[j] + 1e-12f);
        }

        std::partial_sort(cosine_sim.begin(), cosine_sim.begin() + 5, cosine_sim.end(), std::greater<float>());
        
        float* result_ptr = final_results.data() + i * 4;
        std::copy(cosine_sim.data() + 1, cosine_sim.data() + 5, result_ptr);
    }

    std::cout.write(reinterpret_cast<char *>(final_results.data()), N * 4 * sizeof(float));

    return 0;
}