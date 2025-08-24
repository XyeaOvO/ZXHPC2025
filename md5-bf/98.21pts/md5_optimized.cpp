#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <atomic>
#include <omp.h>
#include <climits>
#include <cstring>

#include <immintrin.h>


#define SIMD_WIDTH 8

#define F(x, y, z) _mm256_or_si256(_mm256_and_si256(x, y), _mm256_andnot_si256(x, z))
#define G(x, y, z) _mm256_or_si256(_mm256_and_si256(x, z), _mm256_andnot_si256(z, y))
#define H(x, y, z) _mm256_xor_si256(x, _mm256_xor_si256(y, z))
#define I(x, y, z) _mm256_xor_si256(y, _mm256_or_si256(x, _mm256_xor_si256(z, _mm256_set1_epi32(0xFFFFFFFF))))

#define ROTATE_LEFT(x, n) _mm256_or_si256(_mm256_slli_epi32(x, n), _mm256_srli_epi32(x, 32 - n))

#define FF(a, b, c, d, x, s, ac) { \
    (a) = _mm256_add_epi32((a), F((b), (c), (d))); \
    (a) = _mm256_add_epi32((a), (x)); \
    (a) = _mm256_add_epi32((a), _mm256_set1_epi32(ac)); \
    (a) = ROTATE_LEFT((a), (s)); \
    (a) = _mm256_add_epi32((a), (b)); \
}
#define GG(a, b, c, d, x, s, ac) { \
    (a) = _mm256_add_epi32((a), G((b), (c), (d))); \
    (a) = _mm256_add_epi32((a), (x)); \
    (a) = _mm256_add_epi32((a), _mm256_set1_epi32(ac)); \
    (a) = ROTATE_LEFT((a), (s)); \
    (a) = _mm256_add_epi32((a), (b)); \
}
#define HH(a, b, c, d, x, s, ac) { \
    (a) = _mm256_add_epi32((a), H((b), (c), (d))); \
    (a) = _mm256_add_epi32((a), (x)); \
    (a) = _mm256_add_epi32((a), _mm256_set1_epi32(ac)); \
    (a) = ROTATE_LEFT((a), (s)); \
    (a) = _mm256_add_epi32((a), (b)); \
}
#define II(a, b, c, d, x, s, ac) { \
    (a) = _mm256_add_epi32((a), I((b), (c), (d))); \
    (a) = _mm256_add_epi32((a), (x)); \
    (a) = _mm256_add_epi32((a), _mm256_set1_epi32(ac)); \
    (a) = ROTATE_LEFT((a), (s)); \
    (a) = _mm256_add_epi32((a), (b)); \
}

void md5_8x_48_byte(const uint64_t inputs[SIMD_WIDTH][6], uint32_t digests[SIMD_WIDTH][4]) {
    __m256i a, b, c, d;
    __m256i x[16];

    uint32_t* input_ptr = (uint32_t*)inputs;
    for (int j = 0; j < 12; ++j) {
        x[j] = _mm256_set_epi32(
            input_ptr[7 * 12 + j], input_ptr[6 * 12 + j], input_ptr[5 * 12 + j], input_ptr[4 * 12 + j],
            input_ptr[3 * 12 + j], input_ptr[2 * 12 + j], input_ptr[1 * 12 + j], input_ptr[0 * 12 + j]
        );
    }

    x[12] = _mm256_set1_epi32(0x80);
    x[13] = _mm256_setzero_si256();
    x[14] = _mm256_set1_epi32(384);
    x[15] = _mm256_setzero_si256();

    a = _mm256_set1_epi32(0x67452301); b = _mm256_set1_epi32(0xefcdab89);
    c = _mm256_set1_epi32(0x98badcfe); d = _mm256_set1_epi32(0x10325476);
    __m256i A = a, B = b, C = c, D = d;

    FF(a, b, c, d, x[ 0], 7,  0xd76aa478); FF(d, a, b, c, x[ 1], 12, 0xe8c7b756); FF(c, d, a, b, x[ 2], 17, 0x242070db); FF(b, c, d, a, x[ 3], 22, 0xc1bdceee);
    FF(a, b, c, d, x[ 4], 7,  0xf57c0faf); FF(d, a, b, c, x[ 5], 12, 0x4787c62a); FF(c, d, a, b, x[ 6], 17, 0xa8304613); FF(b, c, d, a, x[ 7], 22, 0xfd469501);
    FF(a, b, c, d, x[ 8], 7,  0x698098d8); FF(d, a, b, c, x[ 9], 12, 0x8b44f7af); FF(c, d, a, b, x[10], 17, 0xffff5bb1); FF(b, c, d, a, x[11], 22, 0x895cd7be);
    FF(a, b, c, d, x[12], 7,  0x6b901122); FF(d, a, b, c, x[13], 12, 0xfd987193); FF(c, d, a, b, x[14], 17, 0xa679438e); FF(b, c, d, a, x[15], 22, 0x49b40821);
    GG(a, b, c, d, x[ 1], 5,  0xf61e2562); GG(d, a, b, c, x[ 6], 9,  0xc040b340); GG(c, d, a, b, x[11], 14, 0x265e5a51); GG(b, c, d, a, x[ 0], 20, 0xe9b6c7aa);
    GG(a, b, c, d, x[ 5], 5,  0xd62f105d); GG(d, a, b, c, x[10], 9,  0x02441453); GG(c, d, a, b, x[15], 14, 0xd8a1e681); GG(b, c, d, a, x[ 4], 20, 0xe7d3fbc8);
    GG(a, b, c, d, x[ 9], 5,  0x21e1cde6); GG(d, a, b, c, x[14], 9,  0xc33707d6); GG(c, d, a, b, x[ 3], 14, 0xf4d50d87); GG(b, c, d, a, x[ 8], 20, 0x455a14ed);
    GG(a, b, c, d, x[13], 5,  0xa9e3e905); GG(d, a, b, c, x[ 2], 9,  0xfcefa3f8); GG(c, d, a, b, x[ 7], 14, 0x676f02d9); GG(b, c, d, a, x[12], 20, 0x8d2a4c8a);
    HH(a, b, c, d, x[ 5], 4,  0xfffa3942); HH(d, a, b, c, x[ 8], 11, 0x8771f681); HH(c, d, a, b, x[11], 16, 0x6d9d6122); HH(b, c, d, a, x[14], 23, 0xfde5380c);
    HH(a, b, c, d, x[ 1], 4,  0xa4beea44); HH(d, a, b, c, x[ 4], 11, 0x4bdecfa9); HH(c, d, a, b, x[ 7], 16, 0xf6bb4b60); HH(b, c, d, a, x[10], 23, 0xbebfbc70);
    HH(a, b, c, d, x[13], 4,  0x289b7ec6); HH(d, a, b, c, x[ 0], 11, 0xeaa127fa); HH(c, d, a, b, x[ 3], 16, 0xd4ef3085); HH(b, c, d, a, x[ 6], 23, 0x04881d05);
    HH(a, b, c, d, x[ 9], 4,  0xd9d4d039); HH(d, a, b, c, x[12], 11, 0xe6db99e5); HH(c, d, a, b, x[15], 16, 0x1fa27cf8); HH(b, c, d, a, x[ 2], 23, 0xc4ac5665);
    II(a, b, c, d, x[ 0], 6,  0xf4292244); II(d, a, b, c, x[ 7], 10, 0x432aff97); II(c, d, a, b, x[14], 15, 0xab9423a7); II(b, c, d, a, x[ 5], 21, 0xfc93a039);
    II(a, b, c, d, x[12], 6,  0x655b59c3); II(d, a, b, c, x[ 3], 10, 0x8f0ccc92); II(c, d, a, b, x[10], 15, 0xffeff47d); II(b, c, d, a, x[ 1], 21, 0x85845dd1);
    II(a, b, c, d, x[ 8], 6,  0x6fa87e4f); II(d, a, b, c, x[15], 10, 0xfe2ce6e0); II(c, d, a, b, x[ 6], 15, 0xa3014314); II(b, c, d, a, x[13], 21, 0x4e0811a1);
    II(a, b, c, d, x[ 4], 6,  0xf7537e82); II(d, a, b, c, x[11], 10, 0xbd3af235); II(c, d, a, b, x[ 2], 15, 0x2ad7d2bb); II(b, c, d, a, x[ 9], 21, 0xeb86d391);
    
    A = _mm256_add_epi32(A, a); B = _mm256_add_epi32(B, b);
    C = _mm256_add_epi32(C, c); D = _mm256_add_epi32(D, d);

    alignas(32) uint32_t temp_A[8], temp_B[8], temp_C[8], temp_D[8];
    _mm256_store_si256((__m256i*)temp_A, A);
    _mm256_store_si256((__m256i*)temp_B, B);
    _mm256_store_si256((__m256i*)temp_C, C);
    _mm256_store_si256((__m256i*)temp_D, D);

    for (int i = 0; i < SIMD_WIDTH; i++) {
        digests[i][0] = temp_A[i];
        digests[i][1] = temp_B[i];
        digests[i][2] = temp_C[i];
        digests[i][3] = temp_D[i];
    }
}

class RndGen {
public:
    RndGen(uint64_t s0, uint64_t s1, uint64_t s2) { s_[0] = s0; s_[1] = s1; s_[2] = s2; }
    RndGen() = delete;
    void generate(uint64_t out[6]) {
        out[0] = xorshift64(s_[0]); out[1] = 0;
        out[2] = xorshift64(s_[1]); out[3] = 0;
        out[4] = xorshift64(s_[2]); out[5] = 0;
    }
private:
    uint64_t xorshift64(uint64_t &state) {
        uint64_t x = state;
        x ^= (x << 13); x ^= (x >> 7); x ^= (x << 17);
        return state = x;
    }
    uint64_t s_[3];
};

uint64_t hex_to_u64(const std::string& hex) {
    uint64_t res; std::stringstream ss; ss << std::hex << hex; ss >> res; return res;
}

void hex_to_bytes(const std::string& hex, unsigned char* bytes) {
    for (unsigned int i = 0; i < hex.length() / 2; i++) {
        bytes[i] = (unsigned char)strtol(hex.substr(i * 2, 2).c_str(), NULL, 16);
    }
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    std::string s0_hex, s1_hex, s2_hex;
    std::cin >> s0_hex >> s1_hex >> s2_hex;
    uint64_t s0 = hex_to_u64(s0_hex), s1 = hex_to_u64(s1_hex), s2 = hex_to_u64(s2_hex);

    std::string target_hash_hex;
    std::cin >> target_hash_hex;
    alignas(32) unsigned char target_hash_bytes[16];
    hex_to_bytes(target_hash_hex, target_hash_bytes);

    std::atomic<uint64_t> min_found_n(ULLONG_MAX);

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        
        RndGen generator(s0, s1, s2);
        
        uint64_t dummy_out[6];
        for (int i = 0; i < thread_id * SIMD_WIDTH; ++i) {
            generator.generate(dummy_out);
        }

        uint64_t current_n_base = (uint64_t)thread_id * SIMD_WIDTH + 1;
        const int skip_amount = (num_threads - 1) * SIMD_WIDTH;

        alignas(32) uint64_t input_buffer[SIMD_WIDTH][6];
        alignas(32) uint32_t hash_output[SIMD_WIDTH][4];

        while (current_n_base < min_found_n.load(std::memory_order_relaxed)) {
            for (int i = 0; i < SIMD_WIDTH; ++i) {
                generator.generate(input_buffer[i]);
            }

            md5_8x_48_byte(input_buffer, hash_output);

            for (int i = 0; i < SIMD_WIDTH; ++i) {
                if (memcmp(hash_output[i], target_hash_bytes, 16) == 0) {
                    uint64_t found_n = current_n_base + i;
                    uint64_t prev_min = min_found_n.load(std::memory_order_relaxed);
                    while (found_n < prev_min) {
                        if (min_found_n.compare_exchange_weak(prev_min, found_n)) break;
                    }
                }
            }

            for (int i = 0; i < skip_amount; ++i) {
                generator.generate(dummy_out);
            }
            current_n_base += (uint64_t)num_threads * SIMD_WIDTH;
        }
    }

    std::cout << min_found_n.load() << std::endl;

    return 0;
}