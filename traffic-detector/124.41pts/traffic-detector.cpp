#include <iostream>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <thread>
#include <cstdio>
#include <cstring>
#include <omp.h>
#include <numeric>

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

// --- Data structures and parsers (unchanged) ---
#include <cstdint>
inline uint64_t sv_to_u64(std::string_view sv) {
    uint64_t res = 0;
    for (char c : sv) {
        if (c >= '0' && c <= '9') res = res * 10 + (c - '0');
    }
    return res;
}

inline uint32_t parse_ipv4(std::string_view sv) {
    uint32_t ip = 0;
    uint32_t part = 0;
    for (char c : sv) {
        if (c == '.') {
            ip = (ip << 8) | part;
            part = 0;
        } else if (c >= '0' && c <= '9') {
            part = part * 10 + (c - '0');
        }
    }
    ip = (ip << 8) | part;
    return ip;
}

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

namespace std {
template <> struct hash<TCPKey> {
    size_t operator()(const TCPKey& k) const {
        uint64_t h1 = (static_cast<uint64_t>(k.src_ip) << 32) | k.dst_ip;
        uint64_t h2 = (static_cast<uint64_t>(k.src_port) << 16) | k.dst_port;
        return std::hash<uint64_t>{}(h1 ^ (h2 << 1));
    }
};
}

struct ThreadData {
    std::unordered_map<TCPKey, char> flow_map;
    std::unordered_map<uint32_t, long long> dnstunnel_count;
};

struct ParsedInfo {
    bool is_tcp;
    uint32_t src_ip, dst_ip;
    uint16_t src_port, dst_port;
    std::string_view last_token; 
};

inline bool fast_parse_line(std::string_view line, ParsedInfo& out) {
    const char* p = line.data();
    const char* const end = p + line.size();
    
    p = (const char*)memchr(p, ' ', end - p); if (!p) return false; ++p;
    
    const char* proto_end = (const char*)memchr(p, ' ', end - p); if (!proto_end) return false;
    out.is_tcp = (*p == 'T');
    p = proto_end + 1;

    const char* src_ip_end = (const char*)memchr(p, ' ', end - p); if (!src_ip_end) return false;
    out.src_ip = parse_ipv4({p, (size_t)(src_ip_end - p)});
    p = src_ip_end + 1;
    
    const char* dst_ip_end = (const char*)memchr(p, ' ', end - p); if (!dst_ip_end) return false;
    out.dst_ip = parse_ipv4({p, (size_t)(dst_ip_end - p)});
    p = dst_ip_end + 1;

    const char* src_port_end = (const char*)memchr(p, ' ', end - p); if (!src_port_end) return false;
    out.src_port = sv_to_u64({p, (size_t)(src_port_end - p)});
    p = src_port_end + 1;

    const char* dst_port_end = (const char*)memchr(p, ' ', end - p); if (!dst_port_end) return false;
    out.dst_port = sv_to_u64({p, (size_t)(dst_port_end - p)});
    p = dst_port_end + 1;

    if (out.is_tcp) {
        const char* flags_end = (const char*)memchr(p, ' ', end - p); if (!flags_end) return false;
        out.last_token = {p, (size_t)(flags_end - p)};
    } else {
        p = (const char*)memchr(p, ' ', end - p); if (!p) return false; ++p;
        if (p >= end) return false;
        out.last_token = {p, (size_t)(end - p)};
    }
    return true;
}

// The worker function is now simpler, as it only processes its own data
void worker_func(const std::vector<std::string_view>& assigned_lines, ThreadData& data) {
    ParsedInfo info;
    for (const auto& line_sv : assigned_lines) {
        if (fast_parse_line(line_sv, info)) {
            if (info.is_tcp) {
                TCPKey key = {info.src_ip, info.dst_ip, info.src_port, info.dst_port};
                auto it = data.flow_map.find(key);
                if (it == data.flow_map.end()) {
                    data.flow_map.emplace(key, (info.last_token == "SYN") ? 1 : 2);
                } else if (it->second == 1 && info.last_token != "SYN") {
                    it->second = 2;
                }
            } else { // DNS
                std::string_view domain = info.last_token;
                size_t dot_pos = domain.find('.');
                size_t prefix_len = (dot_pos == std::string_view::npos) ? domain.size() : dot_pos;
                if (prefix_len >= 30) {
                    data.dnstunnel_count[info.src_ip] += prefix_len;
                }
            }
        }
    }
}


int main() {
    std::ios_base::sync_with_stdio(false);
    
    int fd = STDIN_FILENO;
    struct stat sb;
    fstat(fd, &sb);
    size_t file_size = sb.st_size;
    if (file_size == 0) return 0;
    const char* buffer = (const char*)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    
    const int num_threads = std::thread::hardware_concurrency();
    std::vector<ThreadData> thread_data(num_threads);
    std::vector<std::vector<std::string_view>> thread_lines(num_threads);

    std::hash<TCPKey> tcp_hasher;
    std::hash<uint32_t> ip_hasher;

    // --- PHASE 1: Parallel Read and Local Partitioning ---
    std::vector<std::vector<std::vector<std::string_view>>> all_local_buckets(num_threads);

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        all_local_buckets[tid].resize(num_threads);

        size_t chunk_size = file_size / num_threads;
        size_t start_pos = tid * chunk_size;
        size_t end_pos = (tid == num_threads - 1) ? file_size : (tid + 1) * chunk_size;

        if (tid > 0 && start_pos > 0) {
            while (start_pos < file_size && buffer[start_pos - 1] != '\n') {
                start_pos++;
            }
        }
        if (tid < num_threads - 1) {
            while (end_pos < file_size && buffer[end_pos - 1] != '\n') {
                end_pos++;
            }
        }
        
        const char* ptr = buffer + start_pos;
        const char* const chunk_end = buffer + end_pos;
        
        ParsedInfo info; // Reuse this struct
        while (ptr < chunk_end) {
            const char* line_end = (const char*)memchr(ptr, '\n', chunk_end - ptr);
            if (!line_end) line_end = chunk_end;

            std::string_view line_sv(ptr, line_end - ptr);
            
            if (fast_parse_line(line_sv, info)) {
                if (info.is_tcp) {
                    TCPKey key = {info.src_ip, info.dst_ip, info.src_port, info.dst_port};
                    all_local_buckets[tid][tcp_hasher(key) % num_threads].push_back(line_sv);
                } else {
                    all_local_buckets[tid][ip_hasher(info.src_ip) % num_threads].push_back(line_sv);
                }
            }
            ptr = line_end + 1;
        }
    }

    // --- PHASE 2: Parallel Consolidation ---
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < num_threads; ++i) {
        size_t total_size = 0;
        for (int j = 0; j < num_threads; ++j) {
            total_size += all_local_buckets[j][i].size();
        }
        thread_lines[i].reserve(total_size);
        for (int j = 0; j < num_threads; ++j) {
            thread_lines[i].insert(thread_lines[i].end(), all_local_buckets[j][i].begin(), all_local_buckets[j][i].end());
            all_local_buckets[j][i].clear(); // Free memory
        }
    }

    // --- PHASE 3: Parallel Lock-Free Processing ---
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < num_threads; ++i) {
        worker_func(thread_lines[i], thread_data[i]);
    }


    // --- AGGREGATION (Simple and fast) ---
    std::unordered_map<uint32_t, long long> dnstunnel_count;
    std::unordered_map<uint32_t, long long> portscan_count;

    for(const auto& data : thread_data) {
        for(const auto& [ip, count] : data.dnstunnel_count) {
            dnstunnel_count[ip] += count;
        }
        for (const auto& [key, state] : data.flow_map) {
            if (state == 1) {
                portscan_count[key.src_ip]++;
            }
        }
    }

    // --- Output (unchanged) ---
    auto format_and_print = [](const std::string& type, std::unordered_map<uint32_t, long long>& counts) {
        std::vector<std::pair<uint32_t, long long>> sorted_counts;
        sorted_counts.reserve(counts.size());
        for (const auto& pair : counts) sorted_counts.push_back(pair);

        std::sort(sorted_counts.begin(), sorted_counts.end(), [](const auto& a, const auto& b) {
            char buf_a[16], buf_b[16];
            sprintf(buf_a, "%u.%u.%u.%u", (a.first >> 24) & 0xFF, (a.first >> 16) & 0xFF, (a.first >> 8) & 0xFF, a.first & 0xFF);
            sprintf(buf_b, "%u.%u.%u.%u", (b.first >> 24) & 0xFF, (b.first >> 16) & 0xFF, (b.first >> 8) & 0xFF, b.first & 0xFF);
            return strcmp(buf_a, buf_b) < 0;
        });
        
        std::string out_buffer;
        out_buffer.reserve(sorted_counts.size() * 50);
        char line_buf[128];
        for (const auto& [ip_int, count] : sorted_counts) {
             int len = sprintf(line_buf, "%u.%u.%u.%u %s %lld\n",
                (ip_int >> 24) & 0xFF, (ip_int >> 16) & 0xFF, (ip_int >> 8) & 0xFF, ip_int & 0xFF,
                type.c_str(), count);
            out_buffer.append(line_buf, len);
        }
        fwrite(out_buffer.data(), 1, out_buffer.size(), stdout);
    };

    format_and_print("portscan", portscan_count);
    format_and_print("tunnelling", dnstunnel_count);

    munmap((void*)buffer, file_size);
    close(fd);

    return 0;
}