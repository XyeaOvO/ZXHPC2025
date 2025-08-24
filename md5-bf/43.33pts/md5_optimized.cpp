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

// Constants for MD5Transform routine.
#define S11 7
#define S12 12
#define S13 17
#define S14 22
#define S21 5
#define S22 9
#define S23 14
#define S24 20
#define S31 4
#define S32 11
#define S33 16
#define S34 23
#define S41 6
#define S42 10
#define S43 15
#define S44 21

class MD5 {
public:
    MD5() { init(); }
    void update(const unsigned char *buf, size_t length);
    void final(unsigned char digest[16]);
private:
    void init();
    void transform(const uint8_t block[64]);
    static void encode(uint8_t *output, const uint32_t *input, size_t len);
    inline uint32_t F(uint32_t x, uint32_t y, uint32_t z) { return (x&y) | (~x&z); }
    inline uint32_t G(uint32_t x, uint32_t y, uint32_t z) { return (x&z) | (y&~z); }
    inline uint32_t H(uint32_t x, uint32_t y, uint32_t z) { return x^y^z; }
    inline uint32_t I(uint32_t x, uint32_t y, uint32_t z) { return y ^ (x | ~z); }
    inline uint32_t rotate_left(uint32_t x, int n) { return (x << n) | (x >> (32-n)); }
    inline void FF(uint32_t &a, uint32_t b, uint32_t c, uint32_t d, uint32_t x, uint32_t s, uint32_t ac) { a = rotate_left(a+ F(b,c,d) + x + ac, s) + b; }
    inline void GG(uint32_t &a, uint32_t b, uint32_t c, uint32_t d, uint32_t x, uint32_t s, uint32_t ac) { a = rotate_left(a + G(b,c,d) + x + ac, s) + b; }
    inline void HH(uint32_t &a, uint32_t b, uint32_t c, uint32_t d, uint32_t x, uint32_t s, uint32_t ac) { a = rotate_left(a + H(b,c,d) + x + ac, s) + b; }
    inline void II(uint32_t &a, uint32_t b, uint32_t c, uint32_t d, uint32_t x, uint32_t s, uint32_t ac) { a = rotate_left(a + I(b,c,d) + x + ac, s) + b; }
    uint32_t state[4];
    uint32_t count[2];
    uint8_t buffer[64];
};

void MD5::init() {
    count[0] = count[1] = 0;
    state[0] = 0x67452301;
    state[1] = 0xefcdab89;
    state[2] = 0x98badcfe;
    state[3] = 0x10325476;
}

void MD5::update(const unsigned char *input, size_t inputLen) {
    size_t i, index, partLen;
    index = (unsigned int)((count[0] >> 3) & 0x3F);
    if ((count[0] += ((uint32_t)inputLen << 3)) < ((uint32_t)inputLen << 3)) count[1]++;
    count[1] += ((uint32_t)inputLen >> 29);
    partLen = 64 - index;
    if (inputLen >= partLen) {
        memcpy(&buffer[index], input, partLen);
        transform(buffer);
        for (i = partLen; i + 63 < inputLen; i += 64) transform(&input[i]);
        index = 0;
    } else i = 0;
    memcpy(&buffer[index], &input[i], inputLen - i);
}

void MD5::final(unsigned char digest[16]) {
    static unsigned char PADDING[64] = {
      0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };
    unsigned char bits[8];
    encode(bits, count, 8);
    size_t index = (unsigned int)((count[0] >> 3) & 0x3f);
    size_t padLen = (index < 56) ? (56 - index) : (120 - index);
    update(PADDING, padLen);
    update(bits, 8);
    encode(digest, state, 16);
    memset(this, 0, sizeof(*this));
    init();
}

void MD5::transform(const uint8_t block[64]) {
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3], x[16];
    std::memcpy(x, block, 64);
    FF(a, b, c, d, x[ 0], S11, 0xd76aa478); FF(d, a, b, c, x[ 1], S12, 0xe8c7b756); FF(c, d, a, b, x[ 2], S13, 0x242070db); FF(b, c, d, a, x[ 3], S14, 0xc1bdceee);
    FF(a, b, c, d, x[ 4], S11, 0xf57c0faf); FF(d, a, b, c, x[ 5], S12, 0x4787c62a); FF(c, d, a, b, x[ 6], S13, 0xa8304613); FF(b, c, d, a, x[ 7], S14, 0xfd469501);
    FF(a, b, c, d, x[ 8], S11, 0x698098d8); FF(d, a, b, c, x[ 9], S12, 0x8b44f7af); FF(c, d, a, b, x[10], S13, 0xffff5bb1); FF(b, c, d, a, x[11], S14, 0x895cd7be);
    FF(a, b, c, d, x[12], S11, 0x6b901122); FF(d, a, b, c, x[13], S12, 0xfd987193); FF(c, d, a, b, x[14], S13, 0xa679438e); FF(b, c, d, a, x[15], S14, 0x49b40821);
    GG(a, b, c, d, x[ 1], S21, 0xf61e2562); GG(d, a, b, c, x[ 6], S22, 0xc040b340); GG(c, d, a, b, x[11], S23, 0x265e5a51); GG(b, c, d, a, x[ 0], S24, 0xe9b6c7aa);
    GG(a, b, c, d, x[ 5], S21, 0xd62f105d); GG(d, a, b, c, x[10], S22, 0x02441453); GG(c, d, a, b, x[15], S23, 0xd8a1e681); GG(b, c, d, a, x[ 4], S24, 0xe7d3fbc8);
    GG(a, b, c, d, x[ 9], S21, 0x21e1cde6); GG(d, a, b, c, x[14], S22, 0xc33707d6); GG(c, d, a, b, x[ 3], S23, 0xf4d50d87); GG(b, c, d, a, x[ 8], S24, 0x455a14ed);
    GG(a, b, c, d, x[13], S21, 0xa9e3e905); GG(d, a, b, c, x[ 2], S22, 0xfcefa3f8); GG(c, d, a, b, x[ 7], S23, 0x676f02d9); GG(b, c, d, a, x[12], S24, 0x8d2a4c8a);
    HH(a, b, c, d, x[ 5], S31, 0xfffa3942); HH(d, a, b, c, x[ 8], S32, 0x8771f681); HH(c, d, a, b, x[11], S33, 0x6d9d6122); HH(b, c, d, a, x[14], S34, 0xfde5380c);
    HH(a, b, c, d, x[ 1], S31, 0xa4beea44); HH(d, a, b, c, x[ 4], S32, 0x4bdecfa9); HH(c, d, a, b, x[ 7], S33, 0xf6bb4b60); HH(b, c, d, a, x[10], S34, 0xbebfbc70);
    HH(a, b, c, d, x[13], S31, 0x289b7ec6); HH(d, a, b, c, x[ 0], S32, 0xeaa127fa); HH(c, d, a, b, x[ 3], S33, 0xd4ef3085); HH(b, c, d, a, x[ 6], S34, 0x04881d05);
    HH(a, b, c, d, x[ 9], S31, 0xd9d4d039); HH(d, a, b, c, x[12], S32, 0xe6db99e5); HH(c, d, a, b, x[15], S33, 0x1fa27cf8); HH(b, c, d, a, x[ 2], S34, 0xc4ac5665);
    II(a, b, c, d, x[ 0], S41, 0xf4292244); II(d, a, b, c, x[ 7], S42, 0x432aff97); II(c, d, a, b, x[14], S43, 0xab9423a7); II(b, c, d, a, x[ 5], S44, 0xfc93a039);
    II(a, b, c, d, x[12], S41, 0x655b59c3); II(d, a, b, c, x[ 3], S42, 0x8f0ccc92); II(c, d, a, b, x[10], S43, 0xffeff47d); II(b, c, d, a, x[ 1], S44, 0x85845dd1);
    II(a, b, c, d, x[ 8], S41, 0x6fa87e4f); II(d, a, b, c, x[15], S42, 0xfe2ce6e0); II(c, d, a, b, x[ 6], S43, 0xa3014314); II(b, c, d, a, x[13], S44, 0x4e0811a1);
    II(a, b, c, d, x[ 4], S41, 0xf7537e82); II(d, a, b, c, x[11], S42, 0xbd3af235); II(c, d, a, b, x[ 2], S43, 0x2ad7d2bb); II(b, c, d, a, x[ 9], S44, 0xeb86d391);
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    memset(x, 0, sizeof x);
}

void MD5::encode(uint8_t *output, const uint32_t *input, size_t len) {
    for (size_t i = 0, j = 0; j < len; i++, j += 4) {
        output[j] = (uint8_t)(input[i] & 0xff);
        output[j+1] = (uint8_t)((input[i] >> 8) & 0xff);
        output[j+2] = (uint8_t)((input[i] >> 16) & 0xff);
        output[j+3] = (uint8_t)((input[i] >> 24) & 0xff);
    }
}


class RndGen
{
public:
    RndGen(uint64_t s0, uint64_t s1, uint64_t s2) {
        s_[0] = s0; s_[1] = s1; s_[2] = s2;
    }
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
    uint64_t res;
    std::stringstream ss;
    ss << std::hex << hex;
    ss >> res;
    return res;
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
    uint64_t s0 = hex_to_u64(s0_hex);
    uint64_t s1 = hex_to_u64(s1_hex);
    uint64_t s2 = hex_to_u64(s2_hex);

    std::string target_hash_hex;
    std::cin >> target_hash_hex;
    unsigned char target_hash[16];
    hex_to_bytes(target_hash_hex, target_hash);

    std::atomic<uint64_t> min_found_n(ULLONG_MAX);

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        RndGen generator(s0, s1, s2);
        
        uint64_t dummy_out[6];
        for (int i = 0; i < thread_id; ++i) {
            generator.generate(dummy_out);
        }

        uint64_t current_n = thread_id + 1;
        unsigned char hash_output[16];
        uint64_t input_buffer[6];
        MD5 md5;

        while (current_n < min_found_n.load(std::memory_order_relaxed)) {
            generator.generate(input_buffer);
            
            md5.update(reinterpret_cast<unsigned char*>(input_buffer), sizeof(input_buffer));
            md5.final(hash_output);

            if (memcmp(hash_output, target_hash, 16) == 0) {
                uint64_t prev_min = min_found_n.load();
                while(current_n < prev_min) {
                    if (min_found_n.compare_exchange_weak(prev_min, current_n)) break;
                }
                break;
            }

            for (int i = 0; i < num_threads - 1; ++i) {
                generator.generate(dummy_out);
            }
            current_n += num_threads;
        }
    }

    std::cout << min_found_n.load() << std::endl;

    return 0;
}