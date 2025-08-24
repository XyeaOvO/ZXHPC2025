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
    return g_jump_luts[0][(state >> 0) & 0xFF] ^ g_jump_luts[1][(state >> 8) & 0xFF] ^
           g_jump_luts[2][(state >> 16) & 0xFF] ^ g_jump_luts[3][(state >> 24) & 0xFF] ^
           g_jump_luts[4][(state >> 32) & 0xFF] ^ g_jump_luts[5][(state >> 40) & 0xFF] ^
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

        MD5_Constants constants;
        constants.A_init = _mm512_set1_epi32(0x67452301); constants.B_init = _mm512_set1_epi32(0xefcdab89);
        constants.C_init = _mm512_set1_epi32(0x98badcfe); constants.D_init = _mm512_set1_epi32(0x10325476);
        constants.X12 = _mm512_set1_epi32(0x80); constants.X13 = _mm512_setzero_si512();
        constants.X14 = _mm512_set1_epi32(384); constants.X15 = _mm512_setzero_si512();
        constants.target_A = _mm512_set1_epi32(target_hash_u32[0]); constants.target_B = _mm512_set1_epi32(target_hash_u32[1]);
        constants.target_C = _mm512_set1_epi32(target_hash_u32[2]); constants.target_D = _mm512_set1_epi32(target_hash_u32[3]);
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

        std::atomic<uint64_t> min_found_n(ULLONG_MAX);
        XorshiftJump::matrix single_step_mat = XorshiftJump::get_xorshift_matrix();
        XorshiftJump::matrix stride_jump_mat;

        #pragma omp parallel
        {
            #pragma omp single
            {
                int num_threads = omp_get_num_threads();
                stride_jump_mat = XorshiftJump::power(single_step_mat, (uint64_t)num_threads * SIMD_WIDTH * 2);
                precompute_jump_luts(stride_jump_mat);
            }

            int thread_id = omp_get_thread_num();
            XorshiftJump::matrix start_jump_mat = XorshiftJump::power(single_step_mat, (uint64_t)thread_id * SIMD_WIDTH * 2);
            uint64_t s0_block_start = XorshiftJump::transform(start_jump_mat, s0);
            uint64_t s1_block_start = XorshiftJump::transform(start_jump_mat, s1);
            uint64_t s2_block_start = XorshiftJump::transform(start_jump_mat, s2);
            
            uint64_t current_n_base = (uint64_t)thread_id * SIMD_WIDTH * 2 + 1;
            alignas(64) uint32_t soa_input_buffer0[12][SIMD_WIDTH], soa_input_buffer1[12][SIMD_WIDTH];
            memset(soa_input_buffer0, 0, sizeof(soa_input_buffer0));
            memset(soa_input_buffer1, 0, sizeof(soa_input_buffer1));

            while (current_n_base < min_found_n.load(std::memory_order_relaxed)) {
                uint64_t ts0 = s0_block_start, ts1 = s1_block_start, ts2 = s2_block_start;
                for (int j = 0; j < SIMD_WIDTH; ++j) {
                    xorshift64(ts0); soa_input_buffer0[0][j] = (uint32_t)ts0; soa_input_buffer0[1][j] = (uint32_t)(ts0 >> 32);
                    xorshift64(ts1); soa_input_buffer0[4][j] = (uint32_t)ts1; soa_input_buffer0[5][j] = (uint32_t)(ts1 >> 32);
                    xorshift64(ts2); soa_input_buffer0[8][j] = (uint32_t)ts2; soa_input_buffer0[9][j] = (uint32_t)(ts2 >> 32);
                }
                for (int j = 0; j < SIMD_WIDTH; ++j) {
                    xorshift64(ts0); soa_input_buffer1[0][j] = (uint32_t)ts0; soa_input_buffer1[1][j] = (uint32_t)(ts0 >> 32);
                    xorshift64(ts1); soa_input_buffer1[4][j] = (uint32_t)ts1; soa_input_buffer1[5][j] = (uint32_t)(ts1 >> 32);
                    xorshift64(ts2); soa_input_buffer1[8][j] = (uint32_t)ts2; soa_input_buffer1[9][j] = (uint32_t)(ts2 >> 32);
                }
                
                __m512i a0=constants.A_init, b0=constants.B_init, c0=constants.C_init, d0=constants.D_init;
                __m512i a1=constants.A_init, b1=constants.B_init, c1=constants.C_init, d1=constants.D_init;
                __m512i A0=a0, B0=b0, C0=c0, D0=d0;
                __m512i A1=a1, B1=b1, C1=c1, D1=d1;

                #define LOAD0(i) _mm512_load_si512((const __m512i*)soa_input_buffer0[i])
                #define LOAD1(i) _mm512_load_si512((const __m512i*)soa_input_buffer1[i])
                
                FF(a0,b0,c0,d0,LOAD0(0), 7,constants.K[0]);   FF(a1,b1,c1,d1,LOAD1(0), 7,constants.K[0]);
                FF(d0,a0,b0,c0,LOAD0(1), 12,constants.K[1]);  FF(d1,a1,b1,c1,LOAD1(1), 12,constants.K[1]);
                FF(c0,d0,a0,b0,LOAD0(2), 17,constants.K[2]);  FF(c1,d1,a1,b1,LOAD1(2), 17,constants.K[2]);
                FF(b0,c0,d0,a0,LOAD0(3), 22,constants.K[3]);  FF(b1,c1,d1,a1,LOAD1(3), 22,constants.K[3]);
                FF(a0,b0,c0,d0,LOAD0(4), 7,constants.K[4]);   FF(a1,b1,c1,d1,LOAD1(4), 7,constants.K[4]);
                FF(d0,a0,b0,c0,LOAD0(5), 12,constants.K[5]);  FF(d1,a1,b1,c1,LOAD1(5), 12,constants.K[5]);
                FF(c0,d0,a0,b0,LOAD0(6), 17,constants.K[6]);  FF(c1,d1,a1,b1,LOAD1(6), 17,constants.K[6]);
                FF(b0,c0,d0,a0,LOAD0(7), 22,constants.K[7]);  FF(b1,c1,d1,a1,LOAD1(7), 22,constants.K[7]);
                FF(a0,b0,c0,d0,LOAD0(8), 7,constants.K[8]);   FF(a1,b1,c1,d1,LOAD1(8), 7,constants.K[8]);
                FF(d0,a0,b0,c0,LOAD0(9), 12,constants.K[9]);  FF(d1,a1,b1,c1,LOAD1(9), 12,constants.K[9]);
                FF(c0,d0,a0,b0,LOAD0(10),17,constants.K[10]); FF(c1,d1,a1,b1,LOAD1(10),17,constants.K[10]);
                FF(b0,c0,d0,a0,LOAD0(11),22,constants.K[11]); FF(b1,c1,d1,a1,LOAD1(11),22,constants.K[11]);
                FF(a0,b0,c0,d0,constants.X12,7,constants.K[12]); FF(a1,b1,c1,d1,constants.X12,7,constants.K[12]);
                FF(d0,a0,b0,c0,constants.X13,12,constants.K[13]);FF(d1,a1,b1,c1,constants.X13,12,constants.K[13]);
                FF(c0,d0,a0,b0,constants.X14,17,constants.K[14]);FF(c1,d1,a1,b1,constants.X14,17,constants.K[14]);
                FF(b0,c0,d0,a0,constants.X15,22,constants.K[15]);FF(b1,c1,d1,a1,constants.X15,22,constants.K[15]);
                GG(a0,b0,c0,d0,LOAD0(1), 5,constants.K[16]);  GG(a1,b1,c1,d1,LOAD1(1), 5,constants.K[16]);
                GG(d0,a0,b0,c0,LOAD0(6), 9,constants.K[17]);  GG(d1,a1,b1,c1,LOAD1(6), 9,constants.K[17]);
                GG(c0,d0,a0,b0,LOAD0(11),14,constants.K[18]); GG(c1,d1,a1,b1,LOAD1(11),14,constants.K[18]);
                GG(b0,c0,d0,a0,LOAD0(0), 20,constants.K[19]);  GG(b1,c1,d1,a1,LOAD1(0), 20,constants.K[19]);
                GG(a0,b0,c0,d0,LOAD0(5), 5,constants.K[20]);  GG(a1,b1,c1,d1,LOAD1(5), 5,constants.K[20]);
                GG(d0,a0,b0,c0,LOAD0(10),9,constants.K[21]);  GG(d1,a1,b1,c1,LOAD1(10),9,constants.K[21]);
                GG(c0,d0,a0,b0,constants.X15,14,constants.K[22]);GG(c1,d1,a1,b1,constants.X15,14,constants.K[22]);
                GG(b0,c0,d0,a0,LOAD0(4), 20,constants.K[23]);  GG(b1,c1,d1,a1,LOAD1(4), 20,constants.K[23]);
                GG(a0,b0,c0,d0,LOAD0(9), 5,constants.K[24]);  GG(a1,b1,c1,d1,LOAD1(9), 5,constants.K[24]);
                GG(d0,a0,b0,c0,constants.X14,9,constants.K[25]); GG(d1,a1,b1,c1,constants.X14,9,constants.K[25]);
                GG(c0,d0,a0,b0,LOAD0(3), 14,constants.K[26]);  GG(c1,d1,a1,b1,LOAD1(3), 14,constants.K[26]);
                GG(b0,c0,d0,a0,LOAD0(8), 20,constants.K[27]);  GG(b1,c1,d1,a1,LOAD1(8), 20,constants.K[27]);
                GG(a0,b0,c0,d0,constants.X13,5,constants.K[28]); GG(a1,b1,c1,d1,constants.X13,5,constants.K[28]);
                GG(d0,a0,b0,c0,LOAD0(2), 9,constants.K[29]);  GG(d1,a1,b1,c1,LOAD1(2), 9,constants.K[29]);
                GG(c0,d0,a0,b0,LOAD0(7), 14,constants.K[30]);  GG(c1,d1,a1,b1,LOAD1(7), 14,constants.K[30]);
                GG(b0,c0,d0,a0,constants.X12,20,constants.K[31]);GG(b1,c1,d1,a1,constants.X12,20,constants.K[31]);
                HH(a0,b0,c0,d0,LOAD0(5), 4,constants.K[32]);  HH(a1,b1,c1,d1,LOAD1(5), 4,constants.K[32]);
                HH(d0,a0,b0,c0,LOAD0(8), 11,constants.K[33]);  HH(d1,a1,b1,c1,LOAD1(8), 11,constants.K[33]);
                HH(c0,d0,a0,b0,LOAD0(11),16,constants.K[34]);  HH(c1,d1,a1,b1,LOAD1(11),16,constants.K[34]);
                HH(b0,c0,d0,a0,constants.X14,23,constants.K[35]);HH(b1,c1,d1,a1,constants.X14,23,constants.K[35]);
                HH(a0,b0,c0,d0,LOAD0(1), 4,constants.K[36]);  HH(a1,b1,c1,d1,LOAD1(1), 4,constants.K[36]);
                HH(d0,a0,b0,c0,LOAD0(4), 11,constants.K[37]);  HH(d1,a1,b1,c1,LOAD1(4), 11,constants.K[37]);
                HH(c0,d0,a0,b0,LOAD0(7), 16,constants.K[38]);  HH(c1,d1,a1,b1,LOAD1(7), 16,constants.K[38]);
                HH(b0,c0,d0,a0,LOAD0(10),23,constants.K[39]);  HH(b1,c1,d1,a1,LOAD1(10),23,constants.K[39]);
                HH(a0,b0,c0,d0,constants.X13,4,constants.K[40]); HH(a1,b1,c1,d1,constants.X13,4,constants.K[40]);
                HH(d0,a0,b0,c0,LOAD0(0), 11,constants.K[41]);  HH(d1,a1,b1,c1,LOAD1(0), 11,constants.K[41]);
                HH(c0,d0,a0,b0,LOAD0(3), 16,constants.K[42]);  HH(c1,d1,a1,b1,LOAD1(3), 16,constants.K[42]);
                HH(b0,c0,d0,a0,LOAD0(6), 23,constants.K[43]);  HH(b1,c1,d1,a1,LOAD1(6), 23,constants.K[43]);
                HH(a0,b0,c0,d0,LOAD0(9), 4,constants.K[44]);  HH(a1,b1,c1,d1,LOAD1(9), 4,constants.K[44]);
                HH(d0,a0,b0,c0,constants.X12,11,constants.K[45]);HH(d1,a1,b1,c1,constants.X12,11,constants.K[45]);
                HH(c0,d0,a0,b0,constants.X15,16,constants.K[46]);HH(c1,d1,a1,b1,constants.X15,16,constants.K[46]);
                HH(b0,c0,d0,a0,LOAD0(2), 23,constants.K[47]);  HH(b1,c1,d1,a1,LOAD1(2), 23,constants.K[47]);
                II(a0,b0,c0,d0,LOAD0(0), 6,constants.K[48]);  II(a1,b1,c1,d1,LOAD1(0), 6,constants.K[48]);
                II(d0,a0,b0,c0,LOAD0(7), 10,constants.K[49]);  II(d1,a1,b1,c1,LOAD1(7), 10,constants.K[49]);
                II(c0,d0,a0,b0,constants.X14,15,constants.K[50]);II(c1,d1,a1,b1,constants.X14,15,constants.K[50]);
                II(b0,c0,d0,a0,LOAD0(5), 21,constants.K[51]);  II(b1,c1,d1,a1,LOAD1(5), 21,constants.K[51]);
                II(a0,b0,c0,d0,constants.X12,6,constants.K[52]); II(a1,b1,c1,d1,constants.X12,6,constants.K[52]);
                II(d0,a0,b0,c0,LOAD0(3), 10,constants.K[53]);  II(d1,a1,b1,c1,LOAD1(3), 10,constants.K[53]);
                II(c0,d0,a0,b0,LOAD0(10),15,constants.K[54]);  II(c1,d1,a1,b1,LOAD1(10),15,constants.K[54]);
                II(b0,c0,d0,a0,LOAD0(1), 21,constants.K[55]);  II(b1,c1,d1,a1,LOAD1(1), 21,constants.K[55]);
                II(a0,b0,c0,d0,LOAD0(8), 6,constants.K[56]);  II(a1,b1,c1,d1,LOAD1(8), 6,constants.K[56]);
                II(d0,a0,b0,c0,constants.X15,10,constants.K[57]);II(d1,a1,b1,c1,constants.X15,10,constants.K[57]);
                II(c0,d0,a0,b0,LOAD0(6), 15,constants.K[58]);  II(c1,d1,a1,b1,LOAD1(6), 15,constants.K[58]);
                II(b0,c0,d0,a0,constants.X13,21,constants.K[59]);II(b1,c1,d1,a1,constants.X13,21,constants.K[59]);
                II(a0,b0,c0,d0,LOAD0(4), 6,constants.K[60]);  II(a1,b1,c1,d1,LOAD1(4), 6,constants.K[60]);
                II(d0,a0,b0,c0,LOAD0(11),10,constants.K[61]);  II(d1,a1,b1,c1,LOAD1(11),10,constants.K[61]);
                II(c0,d0,a0,b0,LOAD0(2), 15,constants.K[62]);  II(c1,d1,a1,b1,LOAD1(2), 15,constants.K[62]);
                II(b0,c0,d0,a0,LOAD0(9), 21,constants.K[63]);  II(b1,c1,d1,a1,LOAD1(9), 21,constants.K[63]);
                #undef LOAD0
                #undef LOAD1

                A0=_mm512_add_epi32(A0,a0); B0=_mm512_add_epi32(B0,b0); C0=_mm512_add_epi32(C0,c0); D0=_mm512_add_epi32(D0,d0);
                A1=_mm512_add_epi32(A1,a1); B1=_mm512_add_epi32(B1,b1); C1=_mm512_add_epi32(C1,c1); D1=_mm512_add_epi32(D1,d1);

                uint16_t match_mask = _mm512_cmpeq_epi32_mask(A0, constants.target_A) & _mm512_cmpeq_epi32_mask(B0, constants.target_B) & _mm512_cmpeq_epi32_mask(C0, constants.target_C) & _mm512_cmpeq_epi32_mask(D0, constants.target_D);
                if (match_mask != 0) {
                    int first_match_idx = __builtin_ctz(match_mask);
                    uint64_t found_n = current_n_base + first_match_idx;
                    uint64_t prev_min = min_found_n.load(std::memory_order_relaxed);
                    while (found_n < prev_min) { if (min_found_n.compare_exchange_weak(prev_min, found_n)) break; }
                }

                match_mask = _mm512_cmpeq_epi32_mask(A1, constants.target_A) & _mm512_cmpeq_epi32_mask(B1, constants.target_B) & _mm512_cmpeq_epi32_mask(C1, constants.target_C) & _mm512_cmpeq_epi32_mask(D1, constants.target_D);
                if (match_mask != 0) {
                    int first_match_idx = __builtin_ctz(match_mask);
                    uint64_t found_n = current_n_base + SIMD_WIDTH + first_match_idx;
                    uint64_t prev_min = min_found_n.load(std::memory_order_relaxed);
                    while (found_n < prev_min) { if (min_found_n.compare_exchange_weak(prev_min, found_n)) break; }
                }
                
                s0_block_start = transform_lut(s0_block_start);
                s1_block_start = transform_lut(s1_block_start);
                s2_block_start = transform_lut(s2_block_start);
                
                current_n_base += (uint64_t)omp_get_num_threads() * SIMD_WIDTH * 2;
            }
        }
        std::cout << min_found_n.load() << std::endl;
    }
    return 0;
}