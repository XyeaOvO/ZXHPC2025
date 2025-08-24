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
#include <array>

#include <immintrin.h>

#define SIMD_WIDTH 16
#define F(x, y, z) _mm512_or_si512(_mm512_and_si512(x, y), _mm512_andnot_si512(x, z))
#define G(x, y, z) _mm512_or_si512(_mm512_and_si512(x, z), _mm512_andnot_si512(z, y))
#define H(x, y, z) _mm512_xor_si512(x, _mm512_xor_si512(y, z))
#define I(x, y, z) _mm512_xor_si512(y, _mm512_or_si512(x, _mm512_xor_si512(z, _mm512_set1_epi32(0xFFFFFFFF))))
#define ROTATE_LEFT(x, n) _mm512_or_si512(_mm512_slli_epi32(x, n), _mm512_srli_epi32(x, 32 - n))
#define FF(a, b, c, d, x, s, ac) { (a) = _mm512_add_epi32((a), F((b), (c), (d))); (a) = _mm512_add_epi32((a), (x)); (a) = _mm512_add_epi32((a), _mm512_set1_epi32(ac)); (a) = ROTATE_LEFT((a), (s)); (a) = _mm512_add_epi32((a), (b)); }
#define GG(a, b, c, d, x, s, ac) { (a) = _mm512_add_epi32((a), G((b), (c), (d))); (a) = _mm512_add_epi32((a), (x)); (a) = _mm512_add_epi32((a), _mm512_set1_epi32(ac)); (a) = ROTATE_LEFT((a), (s)); (a) = _mm512_add_epi32((a), (b)); }
#define HH(a, b, c, d, x, s, ac) { (a) = _mm512_add_epi32((a), H((b), (c), (d))); (a) = _mm512_add_epi32((a), (x)); (a) = _mm512_add_epi32((a), _mm512_set1_epi32(ac)); (a) = ROTATE_LEFT((a), (s)); (a) = _mm512_add_epi32((a), (b)); }
#define II(a, b, c, d, x, s, ac) { (a) = _mm512_add_epi32((a), I((b), (c), (d))); (a) = _mm512_add_epi32((a), (x)); (a) = _mm512_add_epi32((a), _mm512_set1_epi32(ac)); (a) = ROTATE_LEFT((a), (s)); (a) = _mm512_add_epi32((a), (b)); }

void md5_16x_48_byte(const uint64_t inputs[SIMD_WIDTH][6], uint32_t digests[SIMD_WIDTH][4]) {
    __m512i a, b, c, d;
    __m512i x[16];
    uint32_t* input_ptr = (uint32_t*)inputs;
    for (int j = 0; j < 12; ++j) {
        x[j] = _mm512_set_epi32(
            input_ptr[15*12+j], input_ptr[14*12+j], input_ptr[13*12+j], input_ptr[12*12+j],
            input_ptr[11*12+j], input_ptr[10*12+j], input_ptr[9*12+j],  input_ptr[8*12+j],
            input_ptr[7*12+j],  input_ptr[6*12+j],  input_ptr[5*12+j],  input_ptr[4*12+j],
            input_ptr[3*12+j],  input_ptr[2*12+j],  input_ptr[1*12+j],  input_ptr[0*12+j]
        );
    }
    x[12] = _mm512_set1_epi32(0x80); x[13] = _mm512_setzero_si512();
    x[14] = _mm512_set1_epi32(384);  x[15] = _mm512_setzero_si512();
    a = _mm512_set1_epi32(0x67452301); b = _mm512_set1_epi32(0xefcdab89);
    c = _mm512_set1_epi32(0x98badcfe); d = _mm512_set1_epi32(0x10325476);
    __m512i A=a, B=b, C=c, D=d;
    FF(a,b,c,d,x[0],7,0xd76aa478); FF(d,a,b,c,x[1],12,0xe8c7b756); FF(c,d,a,b,x[2],17,0x242070db); FF(b,c,d,a,x[3],22,0xc1bdceee);
    FF(a,b,c,d,x[4],7,0xf57c0faf); FF(d,a,b,c,x[5],12,0x4787c62a); FF(c,d,a,b,x[6],17,0xa8304613); FF(b,c,d,a,x[7],22,0xfd469501);
    FF(a,b,c,d,x[8],7,0x698098d8); FF(d,a,b,c,x[9],12,0x8b44f7af); FF(c,d,a,b,x[10],17,0xffff5bb1); FF(b,c,d,a,x[11],22,0x895cd7be);
    FF(a,b,c,d,x[12],7,0x6b901122); FF(d,a,b,c,x[13],12,0xfd987193); FF(c,d,a,b,x[14],17,0xa679438e); FF(b,c,d,a,x[15],22,0x49b40821);
    GG(a,b,c,d,x[1],5,0xf61e2562); GG(d,a,b,c,x[6],9,0xc040b340); GG(c,d,a,b,x[11],14,0x265e5a51); GG(b,c,d,a,x[0],20,0xe9b6c7aa);
    GG(a,b,c,d,x[5],5,0xd62f105d); GG(d,a,b,c,x[10],9,0x02441453); GG(c,d,a,b,x[15],14,0xd8a1e681); GG(b,c,d,a,x[4],20,0xe7d3fbc8);
    GG(a,b,c,d,x[9],5,0x21e1cde6); GG(d,a,b,c,x[14],9,0xc33707d6); GG(c,d,a,b,x[3],14,0xf4d50d87); GG(b,c,d,a,x[8],20,0x455a14ed);
    GG(a,b,c,d,x[13],5,0xa9e3e905); GG(d,a,b,c,x[2],9,0xfcefa3f8); GG(c,d,a,b,x[7],14,0x676f02d9); GG(b,c,d,a,x[12],20,0x8d2a4c8a);
    HH(a,b,c,d,x[5],4,0xfffa3942); HH(d,a,b,c,x[8],11,0x8771f681); HH(c,d,a,b,x[11],16,0x6d9d6122); HH(b,c,d,a,x[14],23,0xfde5380c);
    HH(a,b,c,d,x[1],4,0xa4beea44); HH(d,a,b,c,x[4],11,0x4bdecfa9); HH(c,d,a,b,x[7],16,0xf6bb4b60); HH(b,c,d,a,x[10],23,0xbebfbc70);
    HH(a,b,c,d,x[13],4,0x289b7ec6); HH(d,a,b,c,x[0],11,0xeaa127fa); HH(c,d,a,b,x[3],16,0xd4ef3085); HH(b,c,d,a,x[6],23,0x04881d05);
    HH(a,b,c,d,x[9],4,0xd9d4d039); HH(d,a,b,c,x[12],11,0xe6db99e5); HH(c,d,a,b,x[15],16,0x1fa27cf8); HH(b,c,d,a,x[2],23,0xc4ac5665);
    II(a,b,c,d,x[0],6,0xf4292244); II(d,a,b,c,x[7],10,0x432aff97); II(c,d,a,b,x[14],15,0xab9423a7); II(b,c,d,a,x[5],21,0xfc93a039);
    II(a,b,c,d,x[12],6,0x655b59c3); II(d,a,b,c,x[3],10,0x8f0ccc92); II(c,d,a,b,x[10],15,0xffeff47d); II(b,c,d,a,x[1],21,0x85845dd1);
    II(a,b,c,d,x[8],6,0x6fa87e4f); II(d,a,b,c,x[15],10,0xfe2ce6e0); II(c,d,a,b,x[6],15,0xa3014314); II(b,c,d,a,x[13],21,0x4e0811a1);
    II(a,b,c,d,x[4],6,0xf7537e82); II(d,a,b,c,x[11],10,0xbd3af235); II(c,d,a,b,x[2],15,0x2ad7d2bb); II(b,c,d,a,x[9],21,0xeb86d391);
    A=_mm512_add_epi32(A,a); B=_mm512_add_epi32(B,b); C=_mm512_add_epi32(C,c); D=_mm512_add_epi32(D,d);
    alignas(64) uint32_t temp_A[16], temp_B[16], temp_C[16], temp_D[16];
    _mm512_store_si512((__m512i*)temp_A,A); _mm512_store_si512((__m512i*)temp_B,B);
    _mm512_store_si512((__m512i*)temp_C,C); _mm512_store_si512((__m512i*)temp_D,D);
    for (int i=0; i<SIMD_WIDTH; i++) {
        digests[i][0]=temp_A[i]; digests[i][1]=temp_B[i];
        digests[i][2]=temp_C[i]; digests[i][3]=temp_D[i];
    }
}

namespace XorshiftJump {
    using matrix = std::array<uint64_t, 64>;
    uint64_t transform(const matrix& mat, uint64_t state) {
        uint64_t new_state = 0;
        for (int i=0; i<64; ++i) if ((state>>i)&1) new_state ^= mat[i];
        return new_state;
    }
    matrix multiply(const matrix& a, const matrix& b) {
        matrix result{};
        for (int i=0; i<64; ++i) result[i] = transform(b, a[i]);
        return result;
    }
    matrix power(matrix base, uint64_t exp) {
        matrix result{};
        for (int i=0; i<64; ++i) result[i] = 1ULL << i;
        while (exp > 0) {
            if (exp % 2 == 1) result = multiply(result, base);
            base = multiply(base, base);
            exp /= 2;
        }
        return result;
    }
    matrix get_xorshift_matrix() {
        matrix mat{};
        for (int i=0; i<64; ++i) {
            uint64_t basis = 1ULL << i;
            basis ^= basis << 13; basis ^= basis >> 7; basis ^= basis << 17;
            mat[i] = basis;
        }
        return mat;
    }
}

class RndGen {
public:
    RndGen(uint64_t s0, uint64_t s1, uint64_t s2) { s_[0]=s0; s_[1]=s1; s_[2]=s2; }
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
    for (unsigned int i=0; i<hex.length()/2; i++) {
        bytes[i] = (unsigned char)strtol(hex.substr(i*2,2).c_str(), NULL, 16);
    }
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    for (int i = 0; i < 5; ++i) {
        std::string s0_hex, s1_hex, s2_hex;
        std::cin >> s0_hex >> s1_hex >> s2_hex;
        uint64_t s0 = hex_to_u64(s0_hex), s1 = hex_to_u64(s1_hex), s2 = hex_to_u64(s2_hex);

        std::string target_hash_hex;
        std::cin >> target_hash_hex;
        alignas(32) unsigned char target_hash_bytes[16];
        hex_to_bytes(target_hash_hex, target_hash_bytes);

        std::atomic<uint64_t> min_found_n(ULLONG_MAX);

        XorshiftJump::matrix single_step_mat = XorshiftJump::get_xorshift_matrix();
        XorshiftJump::matrix thread_jump_mat;
        XorshiftJump::matrix stride_jump_mat;

        #pragma omp parallel
        {
            #pragma omp single
            {
                int num_threads = omp_get_num_threads();
                thread_jump_mat = XorshiftJump::power(single_step_mat, SIMD_WIDTH);
                stride_jump_mat = XorshiftJump::power(thread_jump_mat, num_threads);
            }

            int thread_id = omp_get_thread_num();
            
            XorshiftJump::matrix start_jump_mat = XorshiftJump::power(thread_jump_mat, thread_id);
            
            uint64_t s0_block_start = XorshiftJump::transform(start_jump_mat, s0);
            uint64_t s1_block_start = XorshiftJump::transform(start_jump_mat, s1);
            uint64_t s2_block_start = XorshiftJump::transform(start_jump_mat, s2);

            uint64_t current_n_base = (uint64_t)thread_id * SIMD_WIDTH + 1;

            alignas(64) uint64_t input_buffer[SIMD_WIDTH][6];
            alignas(64) uint32_t hash_output[SIMD_WIDTH][4];

            while (current_n_base < min_found_n.load(std::memory_order_relaxed)) {
                RndGen generator(s0_block_start, s1_block_start, s2_block_start);
                for (int j = 0; j < SIMD_WIDTH; ++j) {
                    generator.generate(input_buffer[j]);
                }

                md5_16x_48_byte(input_buffer, hash_output);

                for (int j = 0; j < SIMD_WIDTH; ++j) {
                    if (memcmp(hash_output[j], target_hash_bytes, 16) == 0) {
                        uint64_t found_n = current_n_base + j;
                        uint64_t prev_min = min_found_n.load(std::memory_order_relaxed);
                        while (found_n < prev_min) {
                            if (min_found_n.compare_exchange_weak(prev_min, found_n)) break;
                        }
                    }
                }
                
                s0_block_start = XorshiftJump::transform(stride_jump_mat, s0_block_start);
                s1_block_start = XorshiftJump::transform(stride_jump_mat, s1_block_start);
                s2_block_start = XorshiftJump::transform(stride_jump_mat, s2_block_start);
                current_n_base += (uint64_t)omp_get_num_threads() * SIMD_WIDTH;
            }
        }

        std::cout << min_found_n.load() << std::endl;
    }

    return 0;
}