#include <iostream>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <utility>

// --- 开始：整数转换与哈希优化 ---

// 快速将 string_view 解析为无符号整数
inline uint64_t sv_to_u64(std::string_view sv) {
    uint64_t res = 0;
    for (char c : sv) {
        res = res * 10 + (c - '0');
    }
    return res;
}

// 快速将 string_view 解析为 uint32_t 类型的IPv4地址
inline uint32_t parse_ipv4(std::string_view sv) {
    uint32_t ip = 0;
    uint32_t part = 0;
    for (char c : sv) {
        if (c == '.') {
            ip = (ip << 8) | part;
            part = 0;
        } else {
            part = part * 10 + (c - '0');
        }
    }
    ip = (ip << 8) | part;
    return ip;
}

// TCP五元组中的四元组作为Key
struct TCPKey {
    uint32_t src_ip;
    uint32_t dst_ip;
    uint16_t src_port;
    uint16_t dst_port;

    bool operator==(const TCPKey& other) const {
        return src_ip == other.src_ip && dst_ip == other.dst_ip &&
               src_port == other.src_port && dst_port == other.dst_port;
    }
};

// 为 TCPKey 提供自定义哈希函数
namespace std {
template <>
struct hash<TCPKey> {
    size_t operator()(const TCPKey& k) const {
        // 使用一个简单的组合哈希函数
        size_t h1 = hash<uint32_t>{}(k.src_ip);
        size_t h2 = hash<uint32_t>{}(k.dst_ip);
        size_t h3 = hash<uint16_t>{}(k.src_port);
        size_t h4 = hash<uint16_t>{}(k.dst_port);
        // 将多个哈希值组合起来
        size_t seed = h1;
        seed ^= h2 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= h3 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= h4 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};
}

// 将 uint32_t 类型的IPv4地址格式化为字符串
std::string format_ipv4(uint32_t ip) {
    std::string ip_str;
    ip_str += std::to_string((ip >> 24) & 0xFF);
    ip_str += '.';
    ip_str += std::to_string((ip >> 16) & 0xFF);
    ip_str += '.';
    ip_str += std::to_string((ip >> 8) & 0xFF);
    ip_str += '.';
    ip_str += std::to_string(ip & 0xFF);
    return ip_str;
}

// --- 结束：整数转换与哈希优化 ---


inline void skip_spaces(std::string_view s, size_t &i) {
    while (i < s.size() && s[i] == ' ') ++i;
}

inline std::string_view next_token_sv(std::string_view s, size_t &i) {
    skip_spaces(s, i);
    if (i >= s.size()) return {};
    size_t j = i;
    while (j < s.size() && s[j] != ' ') ++j;
    std::string_view tok = s.substr(i, j - i);
    i = j;
    return tok;
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    // 使用 TCPKey 作为键，char 作为状态
    std::unordered_map<TCPKey, char> flow_map;
    flow_map.reserve(1 << 22);

    // 使用 uint32_t 作为IP的键
    std::unordered_map<uint32_t, long long> dnstunnel_count;
    dnstunnel_count.reserve(1 << 20);

    std::string line;
    line.reserve(256);

    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        
        std::string_view line_sv(line);
        size_t i = 0;

        (void) next_token_sv(line_sv, i); // timestamp

        std::string_view proto = next_token_sv(line_sv, i);
        if (proto.empty()) continue;

        uint32_t src_ip_int = parse_ipv4(next_token_sv(line_sv, i));
        uint32_t dst_ip_int = parse_ipv4(next_token_sv(line_sv, i));
        uint16_t src_port_int = sv_to_u64(next_token_sv(line_sv, i));
        uint16_t dst_port_int = sv_to_u64(next_token_sv(line_sv, i));

        if (proto.front() == 'T') { // TCP
            std::string_view flags = next_token_sv(line_sv, i);
            if (flags.empty()) continue;
            
            TCPKey key = {src_ip_int, dst_ip_int, src_port_int, dst_port_int};
            
            auto it = flow_map.find(key);
            if (it == flow_map.end()) {
                if (flags == "SYN") {
                    flow_map.emplace(key, 1);
                } else {
                    flow_map.emplace(key, 2);
                }
            } else {
                if (it->second == 1) {
                    it->second = 2;
                }
            }
        } else { // DNS
            (void) next_token_sv(line_sv, i); // domain length
            std::string_view domain = next_token_sv(line_sv, i);
            if (domain.empty()) continue;

            size_t pos = domain.find('.');
            size_t prefix_len = (pos == std::string_view::npos) ? domain.size() : pos;
            
            if (prefix_len >= 30) {
                dnstunnel_count[src_ip_int] += prefix_len;
            }
        }
    }

    // 汇总 portscan
    std::unordered_map<uint32_t, long long> portscan_count;
    portscan_count.reserve(1 << 20);
    for (const auto &[key, state] : flow_map) {
        if (state == 1) {
            portscan_count[key.src_ip] += 1;
        }
    }

    // 输出
    std::vector<std::pair<std::string, long long>> portvec;
    portvec.reserve(portscan_count.size());
    for (const auto &[ip_int, count] : portscan_count) {
        portvec.emplace_back(format_ipv4(ip_int), count);
    }
    std::sort(portvec.begin(), portvec.end());
    for (const auto &p : portvec) {
        std::cout << p.first << " portscan " << p.second << '\n';
    }

    std::vector<std::pair<std::string, long long>> dnsvec;
    dnsvec.reserve(dnstunnel_count.size());
    for (const auto &[ip_int, count] : dnstunnel_count) {
        dnsvec.emplace_back(format_ipv4(ip_int), count);
    }
    std::sort(dnsvec.begin(), dnsvec.end());
    for (const auto &p : dnsvec) {
        std::cout << p.first << " tunnelling " << p.second << '\n';
    }

    return 0;
}