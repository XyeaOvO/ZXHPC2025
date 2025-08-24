#include <iostream>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <thread> // For hardware_concurrency
#include <cstdio>
#include <cstring>
#include <omp.h> // For OpenMP

// For mmap
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

// --- Data structures and parsers (unchanged) ---
#include <cstdint>
// ... (sv_to_u64, parse_ipv4, TCPKey, and its hash struct are the same as before)
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
        uint64_t h = (static_cast<uint64_t>(k.src_ip) << 32) | k.dst_ip;
        h ^= (static_cast<uint64_t>(k.src_port) << 16) | k.dst_port;
        return std::hash<uint64_t>{}(h);
    }
};
}

// Helper to split string_view, same as before
inline std::string_view next_token_sv(std::string_view& s, char delim = ' ') {
    size_t pos = s.find(delim);
    if (pos == std::string_view::npos) {
        std::string_view tok = s;
        s.remove_prefix(s.size());
        return tok;
    }
    std::string_view tok = s.substr(0, pos);
    s.remove_prefix(pos + 1);
    return tok;
}
// --- End of unchanged section ---

// Thread-local data remains the same
struct ThreadData {
    std::unordered_map<TCPKey, char> flow_map;
    std::unordered_map<uint32_t, long long> dnstunnel_count;
};

void worker_func(const std::vector<std::string_view>& assigned_lines, ThreadData& data) {
    for (const auto& line_sv_ref : assigned_lines) {
        std::string_view line_sv = line_sv_ref; // Make a copy to modify
        if(line_sv.empty()) continue;

        next_token_sv(line_sv); // timestamp
        std::string_view proto = next_token_sv(line_sv);
        if (proto.empty()) continue;

        uint32_t src_ip_int = parse_ipv4(next_token_sv(line_sv));
        uint32_t dst_ip_int = parse_ipv4(next_token_sv(line_sv));
        uint16_t src_port_int = sv_to_u64(next_token_sv(line_sv));
        uint16_t dst_port_int = sv_to_u64(next_token_sv(line_sv));

        if (proto.front() == 'T') {
            std::string_view flags = next_token_sv(line_sv);
            if (flags.empty()) continue;
            TCPKey key = {src_ip_int, dst_ip_int, src_port_int, dst_port_int};
            auto it = data.flow_map.find(key);
            if (it == data.flow_map.end()) {
                data.flow_map.emplace(key, (flags == "SYN") ? 1 : 2);
            } else if (it->second == 1) {
                it->second = 2;
            }
        } else {
            next_token_sv(line_sv); // domain_len
            std::string_view domain = next_token_sv(line_sv);
            if (domain.empty()) continue;
            size_t dot_pos = domain.find('.');
            size_t prefix_len = (dot_pos == std::string_view::npos) ? domain.size() : dot_pos;
            if (prefix_len >= 30) {
                data.dnstunnel_count[src_ip_int] += prefix_len;
            }
        }
    }
}


int main() {
    std::ios_base::sync_with_stdio(false);
    
    // 1. Use mmap to read stdin
    int fd = STDIN_FILENO;
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        perror("fstat");
        return 1;
    }
    size_t file_size = sb.st_size;
    const char* buffer = (const char*)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (buffer == MAP_FAILED) {
        perror("mmap");
        return 1;
    }
    
    const int num_threads = std::thread::hardware_concurrency();
    std::vector<ThreadData> thread_data(num_threads);
    std::vector<std::vector<std::string_view>> thread_lines(num_threads);

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        
        // --- Step 1: Parallel line splitting and distribution ---
        size_t chunk_size = file_size / num_threads;
        size_t start = tid * chunk_size;
        size_t end = (tid == num_threads - 1) ? file_size : (tid + 1) * chunk_size;

        // Adjust start to the beginning of a line
        if (tid > 0) {
            while (start < file_size && buffer[start-1] != '\n') {
                start++;
            }
        }
        // Adjust end to the end of a line (optional but good practice)
        if (tid < num_threads - 1) {
            while (end < file_size && buffer[end-1] != '\n') {
                end++;
            }
        }

        std::vector<std::vector<std::string_view>> local_buckets(num_threads);

        const char* ptr = buffer + start;
        const char* chunk_end = buffer + end;
        while (ptr < chunk_end) {
            const char* line_end = (const char*)memchr(ptr, '\n', chunk_end - ptr);
            if (!line_end) line_end = chunk_end;

            std::string_view line_sv(ptr, line_end - ptr);
            
            // Pre-parse to get src_ip
            size_t first_space = line_sv.find(' ');
            if (first_space != std::string_view::npos) {
                 size_t second_space = line_sv.find(' ', first_space + 1);
                 if (second_space != std::string_view::npos) {
                    size_t third_space = line_sv.find(' ', second_space + 1);
                    if (third_space != std::string_view::npos) {
                        std::string_view src_ip_sv = line_sv.substr(second_space + 1, third_space - (second_space + 1));
                        if (!src_ip_sv.empty()) {
                            uint32_t src_ip_int = parse_ipv4(src_ip_sv);
                            local_buckets[src_ip_int % num_threads].push_back(line_sv);
                        }
                    }
                 }
            }
            ptr = line_end + 1;
        }

        // Merge local buckets into global task lists
        #pragma omp critical
        {
            for(int i = 0; i < num_threads; ++i) {
                if(!local_buckets[i].empty()) {
                    thread_lines[i].insert(thread_lines[i].end(), local_buckets[i].begin(), local_buckets[i].end());
                }
            }
        }

        // Wait for all threads to finish distribution
        #pragma omp barrier

        // --- Step 2: Parallel processing ---
        worker_func(thread_lines[tid], thread_data[tid]);
    }


    // 4. Aggregate results (serial, as it's fast enough)
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

    // 5. Output results (Optimized: sort ints, then format)
    auto format_and_print = [](const std::string& type, std::unordered_map<uint32_t, long long>& counts) {
        std::vector<std::pair<uint32_t, long long>> sorted_counts;
        sorted_counts.reserve(counts.size());
        for (const auto& pair : counts) {
            sorted_counts.push_back(pair);
        }

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

    // Cleanup mmap
    munmap((void*)buffer, file_size);
    close(fd);

    return 0;
}