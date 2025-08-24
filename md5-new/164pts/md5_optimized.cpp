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
#define FF(a, b, c, d, x, s, k) { (a) = _mm512_add_epi32((a), F((b), (c), (d))); (a) = _mm512_add_epi32((a), (x)); (a) = _mm512_add_epi32((a), (k)); (a) = ROTATE_LEFT((a), (s)); (a) = _mm512_add_epi32((a), (b)); }
#define GG(a, b, c, d, x, s, k) { (a) = _mm512_add_epi32((a), G((b), (c), (d))); (a) = _mm512_add_epi32((a), (x)); (a) = _mm512_add_epi32((a), (k)); (a) = ROTATE_LEFT((a), (s)); (a) = _mm512_add_epi32((a), (b)); }
#define HH(a, b, c, d, x, s, k) { (a) = _mm512_add_epi32((a), H((b), (c), (d))); (a) = _mm512_add_epi32((a), (x)); (a) = _mm512_add_epi32((a), (k)); (a) = ROTATE_LEFT((a), (s)); (a) = _mm512_add_epi32((a), (b)); }
#define II(a, b, c, d, x, s, k) { (a) = _mm512_add_epi32((a), I((b), (c), (d))); (a) = _mm512_add_epi32((a), (x)); (a) = _mm512_add_epi32((a), (k)); (a) = ROTATE_LEFT((a), (s)); (a) = _mm512_add_epi32((a), (b)); }

struct MD5_Constants {
    __m512i A_init, B_init, C_init, D_init;
    __m512i X12, X13, X14, X15;
    __m512i target_A, target_B, target_C, target_D;
    __m512i K[64];
};

uint16_t md5_16x_48_byte_compare(const uint32_t soa_inputs[12][SIMD_WIDTH], const MD5_Constants& consts) {
    __m512i a = consts.A_init, b = consts.B_init, c = consts.C_init, d = consts.D_init;
    __m512i A=a, B=b, C=c, D=d;
    const __m512i x0  = _mm512_load_si512((const __m512i*)soa_inputs[0]); const __m512i x1  = _mm512_load_si512((const __m512i*)soa_inputs[1]);
    const __m512i x2  = _mm512_load_si512((const __m512i*)soa_inputs[2]); const __m512i x3  = _mm512_load_si512((const __m512i*)soa_inputs[3]);
    const __m512i x4  = _mm512_load_si512((const __m512i*)soa_inputs[4]); const __m512i x5  = _mm512_load_si512((const __m512i*)soa_inputs[5]);
    const __m512i x6  = _mm512_load_si512((const __m512i*)soa_inputs[6]); const __m512i x7  = _mm512_load_si512((const __m512i*)soa_inputs[7]);
    const __m512i x8  = _mm512_load_si512((const __m512i*)soa_inputs[8]); const __m512i x9  = _mm512_load_si512((const __m512i*)soa_inputs[9]);
    const __m512i x10 = _mm512_load_si512((const __m512i*)soa_inputs[10]); const __m512i x11 = _mm512_load_si512((const __m512i*)soa_inputs[11]);
    FF(a,b,c,d,x0, 7,consts.K[0]);  FF(d,a,b,c,x1, 12,consts.K[1]);  FF(c,d,a,b,x2, 17,consts.K[2]);  FF(b,c,d,a,x3, 22,consts.K[3]);
    FF(a,b,c,d,x4, 7,consts.K[4]);  FF(d,a,b,c,x5, 12,consts.K[5]);  FF(c,d,a,b,x6, 17,consts.K[6]);  FF(b,c,d,a,x7, 22,consts.K[7]);
    FF(a,b,c,d,x8, 7,consts.K[8]);  FF(d,a,b,c,x9, 12,consts.K[9]);  FF(c,d,a,b,x10,17,consts.K[10]); FF(b,c,d,a,x11,22,consts.K[11]);
    FF(a,b,c,d,consts.X12,7,consts.K[12]); FF(d,a,b,c,consts.X13,12,consts.K[13]); FF(c,d,a,b,consts.X14,17,consts.K[14]); FF(b,c,d,a,consts.X15,22,consts.K[15]);
    GG(a,b,c,d,x1, 5,consts.K[16]); GG(d,a,b,c,x6, 9,consts.K[17]); GG(c,d,a,b,x11,14,consts.K[18]); GG(b,c,d,a,x0, 20,consts.K[19]);
    GG(a,b,c,d,x5, 5,consts.K[20]); GG(d,a,b,c,x10,9,consts.K[21]); GG(c,d,a,b,consts.X15,14,consts.K[22]); GG(b,c,d,a,x4, 20,consts.K[23]);
    GG(a,b,c,d,x9, 5,consts.K[24]); GG(d,a,b,c,consts.X14,9,consts.K[25]); GG(c,d,a,b,x3, 14,consts.K[26]); GG(b,c,d,a,x8, 20,consts.K[27]);
    GG(a,b,c,d,consts.X13,5,consts.K[28]); GG(d,a,b,c,x2, 9,consts.K[29]); GG(c,d,a,b,x7, 14,consts.K[30]); GG(b,c,d,a,consts.X12,20,consts.K[31]);
    HH(a,b,c,d,x5, 4,consts.K[32]); HH(d,a,b,c,x8, 11,consts.K[33]); HH(c,d,a,b,x11,16,consts.K[34]); HH(b,c,d,a,consts.X14,23,consts.K[35]);
    HH(a,b,c,d,x1, 4,consts.K[36]); HH(d,a,b,c,x4, 11,consts.K[37]); HH(c,d,a,b,x7, 16,consts.K[38]); HH(b,c,d,a,x10,23,consts.K[39]);
    HH(a,b,c,d,consts.X13,4,consts.K[40]); HH(d,a,b,c,x0, 11,consts.K[41]); HH(c,d,a,b,x3, 16,consts.K[42]); HH(b,c,d,a,x6, 23,consts.K[43]);
    HH(a,b,c,d,x9, 4,consts.K[44]); HH(d,a,b,c,consts.X12,11,consts.K[45]); HH(c,d,a,b,consts.X15,16,consts.K[46]); HH(b,c,d,a,x2, 23,consts.K[47]);
    II(a,b,c,d,x0, 6,consts.K[48]); II(d,a,b,c,x7, 10,consts.K[49]); II(c,d,a,b,consts.X14,15,consts.K[50]); II(b,c,d,a,x5, 21,consts.K[51]);
    II(a,b,c,d,consts.X12,6,consts.K[52]); II(d,a,b,c,x3, 10,consts.K[53]); II(c,d,a,b,x10,15,consts.K[54]); II(b,c,d,a,x1, 21,consts.K[55]);
    II(a,b,c,d,x8, 6,consts.K[56]); II(d,a,b,c,consts.X15,10,consts.K[57]); II(c,d,a,b,x6, 15,consts.K[58]); II(b,c,d,a,consts.X13,21,consts.K[59]);
    II(a,b,c,d,x4, 6,consts.K[60]); II(d,a,b,c,x11,10,consts.K[61]); II(c,d,a,b,x2, 15,consts.K[62]); II(b,c,d,a,x9, 21,consts.K[63]);
    A = _mm512_add_epi32(A, a); B = _mm512_add_epi32(B, b); C = _mm512_add_epi32(C, c); D = _mm512_add_epi32(D, d);
    const __mmask16 mask_A = _mm512_cmpeq_epi32_mask(A, consts.target_A); const __mmask16 mask_B = _mm512_cmpeq_epi32_mask(B, consts.target_B);
    const __mmask16 mask_C = _mm512_cmpeq_epi32_mask(C, consts.target_C); const __mmask16 mask_D = _mm512_cmpeq_epi32_mask(D, consts.target_D);
    return mask_A & mask_B & mask_C & mask_D;
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
alignas(64) std::array<std::array<uint64_t, 256>, 8> g_jump_luts;
void precompute_jump_luts(const XorshiftJump::matrix& jump_matrix) {
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 256; ++j) {
            g_jump_luts[i][j] = XorshiftJump::transform(jump_matrix, (uint64_t)j << (i * 8));
        }
    }
}
inline uint64_t transform_lut(uint64_t state) {
    return g_jump_luts[0][(state >> 0) & 0xFF] ^ g_jump_luts[1][(state >> 8) & 0xFF] ^ g_jump_luts[2][(state >> 16) & 0xFF] ^
           g_jump_luts[3][(state >> 24) & 0xFF] ^ g_jump_luts[4][(state >> 32) & 0xFF] ^ g_jump_luts[5][(state >> 40) & 0xFF] ^
           g_jump_luts[6][(state >> 48) & 0xFF] ^ g_jump_luts[7][(state >> 56) & 0xFF];
}
uint64_t hex_to_u64(const std::string& hex) {
    uint64_t res; std::stringstream ss; ss << std::hex << hex; ss >> res; return res;
}
void hex_to_bytes(const std::string& hex, unsigned char* bytes) {
    for (unsigned int i=0; i<hex.length()/2; i++) {
        bytes[i] = (unsigned char)strtol(hex.substr(i*2,2).c_str(), NULL, 16);
    }
}
inline void xorshift64(uint64_t& state) {
    uint64_t x = state;
    x ^= (x << 13); x ^= (x >> 7); x ^= (x << 17);
    state = x;
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    MD5_Constants constants;
    constants.A_init = _mm512_set1_epi32(0x67452301);
    constants.B_init = _mm512_set1_epi32(0xefcdab89);
    constants.C_init = _mm512_set1_epi32(0x98badcfe);
    constants.D_init = _mm512_set1_epi32(0x10325476);
    constants.X12 = _mm512_set1_epi32(0x80);
    constants.X13 = _mm512_setzero_si512();
    constants.X14 = _mm512_set1_epi32(384);
    constants.X15 = _mm512_setzero_si512();
    
    static const uint32_t K_scalar[64] = {
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
        0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
        0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
        0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
        0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
        0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
        0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391
    };
    for(int j=0; j<64; ++j) { constants.K[j] = _mm512_set1_epi32(K_scalar[j]); }


    for (int i = 0; i < 5; ++i) {
        std::string s0_hex, s1_hex, s2_hex;
        std::cin >> s0_hex >> s1_hex >> s2_hex;
        uint64_t s0 = hex_to_u64(s0_hex), s1 = hex_to_u64(s1_hex), s2 = hex_to_u64(s2_hex);

        std::string target_hash_hex;
        std::cin >> target_hash_hex;
        alignas(32) unsigned char target_hash_bytes[16];
        hex_to_bytes(target_hash_hex, target_hash_bytes);
        uint32_t target_hash_u32[4];
        memcpy(target_hash_u32, target_hash_bytes, 16);

        constants.target_A = _mm512_set1_epi32(target_hash_u32[0]);
        constants.target_B = _mm512_set1_epi32(target_hash_u32[1]);
        constants.target_C = _mm512_set1_epi32(target_hash_u32[2]);
        constants.target_D = _mm512_set1_epi32(target_hash_u32[3]);

        std::atomic<uint64_t> min_found_n(ULLONG_MAX);
        XorshiftJump::matrix single_step_mat = XorshiftJump::get_xorshift_matrix();
        XorshiftJump::matrix stride_jump_mat;

        #pragma omp parallel
        {
            #pragma omp single
            {
                int num_threads = omp_get_num_threads();
                stride_jump_mat = XorshiftJump::power(single_step_mat, (uint64_t)num_threads * SIMD_WIDTH);
                precompute_jump_luts(stride_jump_mat);
            }

            int thread_id = omp_get_thread_num();
            XorshiftJump::matrix start_jump_mat = XorshiftJump::power(single_step_mat, (uint64_t)thread_id * SIMD_WIDTH);
            uint64_t s0_block_start = XorshiftJump::transform(start_jump_mat, s0);
            uint64_t s1_block_start = XorshiftJump::transform(start_jump_mat, s1);
            uint64_t s2_block_start = XorshiftJump::transform(start_jump_mat, s2);
            
            uint64_t current_n_base = (uint64_t)thread_id * SIMD_WIDTH + 1;
            alignas(64) uint32_t soa_input_buffer[12][SIMD_WIDTH];
            memset(soa_input_buffer, 0, sizeof(soa_input_buffer));

            while (current_n_base < min_found_n.load(std::memory_order_relaxed)) {
                uint64_t ts0 = s0_block_start, ts1 = s1_block_start, ts2 = s2_block_start;
                for (int j = 0; j < SIMD_WIDTH; ++j) {
                    xorshift64(ts0); soa_input_buffer[0][j] = (uint32_t)ts0; soa_input_buffer[1][j] = (uint32_t)(ts0 >> 32);
                    xorshift64(ts1); soa_input_buffer[4][j] = (uint32_t)ts1; soa_input_buffer[5][j] = (uint32_t)(ts1 >> 32);
                    xorshift64(ts2); soa_input_buffer[8][j] = (uint32_t)ts2; soa_input_buffer[9][j] = (uint32_t)(ts2 >> 32);
                }
                
                uint16_t match_mask = md5_16x_48_byte_compare(soa_input_buffer, constants);
                if (match_mask != 0) {
                    int first_match_idx = __builtin_ctz(match_mask);
                    uint64_t found_n = current_n_base + first_match_idx;
                    uint64_t prev_min = min_found_n.load(std::memory_order_relaxed);
                    while (found_n < prev_min) {
                        if (min_found_n.compare_exchange_weak(prev_min, found_n)) break;
                    }
                }
                
                s0_block_start = transform_lut(s0_block_start);
                s1_block_start = transform_lut(s1_block_start);
                s2_block_start = transform_lut(s2_block_start);
                current_n_base += (uint64_t)omp_get_num_threads() * SIMD_WIDTH;
            }
        }
        std::cout << min_found_n.load() << std::endl;
    }
    return 0;
}