#include <iostream>
#include <string>
#include <string_view>
#include <vector>
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

// --- Custom Hash Table (Linear Probing) ---
template<typename Key, typename Value>
class FastMap {
private:
    struct Entry {
        Key key;
        Value value;
        bool occupied = false;
    };
    std::vector<Entry> table;
    size_t table_size;
    size_t num_elements = 0;
    std::hash<Key> hasher;

public:
    FastMap(size_t initial_capacity = 1024) {
        table_size = initial_capacity;
        table.resize(table_size);
    }

    Value& operator[](const Key& key) {
        if (num_elements * 2 > table_size) {
            rehash();
        }
        size_t index = hasher(key) & (table_size - 1);
        while (table[index].occupied && !(table[index].key == key)) {
            index = (index + 1) & (table_size - 1);
        }
        if (!table[index].occupied) {
            table[index].key = key;
            table[index].occupied = true;
            num_elements++;
        }
        return table[index].value;
    }

    Entry* begin() { return table.data(); }
    Entry* end() { return table.data() + table_size; }

private:
    void rehash() {
        std::vector<Entry> old_table = std::move(table);
        table_size *= 2;
        table.assign(table_size, Entry());
        num_elements = 0;
        for (const auto& entry : old_table) {
            if (entry.occupied) {
                (*this)[entry.key] = entry.value;
            }
        }
    }
};

// --- Data structures and parsers ---
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
        // Fibonacci hashing multiplier for better distribution
        return (h1 ^ (h2 << 1)) * 11400714819323198485ull;
    }
};
}

// --- OPTIMIZATION: Ultimate Parser ---
struct ParsedInfo {
    bool is_tcp;
    uint32_t src_ip, dst_ip;
    uint16_t src_port, dst_port;
    std::string_view last_token;
};

inline bool ultimate_parser(std::string_view line, ParsedInfo& out) {
    const char* p = line.data();
    const char* const end = p + line.size();
    uint32_t temp_ip = 0, part = 0;
    uint64_t temp_port = 0;

    // Skip timestamp
    while (p < end && *p != ' ') p++;
    if (p == end) return false;
    p++;

    // Protocol
    out.is_tcp = (*p == 'T');
    while (p < end && *p != ' ') p++;
    if (p == end) return false;
    p++;

    // Src IP
    for (int i = 0; i < 4; ++i) {
        part = 0;
        while (p < end && *p >= '0' && *p <= '9') part = part * 10 + (*p++ - '0');
        temp_ip = (temp_ip << 8) | part;
        if (p < end && *p == '.') p++;
    }
    out.src_ip = temp_ip;
    if (p == end) return false;
    p++;

    // Dst IP
    temp_ip = 0;
    for (int i = 0; i < 4; ++i) {
        part = 0;
        while (p < end && *p >= '0' && *p <= '9') part = part * 10 + (*p++ - '0');
        temp_ip = (temp_ip << 8) | part;
        if (p < end && *p == '.') p++;
    }
    out.dst_ip = temp_ip;
    if (p == end) return false;
    p++;

    // Src Port
    while (p < end && *p >= '0' && *p <= '9') temp_port = temp_port * 10 + (*p++ - '0');
    out.src_port = temp_port;
    if (p == end) return false;
    p++;
    
    // Dst Port
    temp_port = 0;
    while (p < end && *p >= '0' && *p <= '9') temp_port = temp_port * 10 + (*p++ - '0');
    out.dst_port = temp_port;
    if (p == end) return false;
    p++;

    if (out.is_tcp) {
        const char* token_start = p;
        while (p < end && *p != ' ') p++;
        out.last_token = {token_start, (size_t)(p - token_start)};
    } else {
        while (p < end && *p != ' ') p++;
        if (p == end) return false;
        p++;
        out.last_token = {p, (size_t)(end - p)};
    }
    return true;
}

struct ThreadData {
    FastMap<TCPKey, char> flow_map;
    FastMap<uint32_t, long long> dnstunnel_count;
};

void worker_func(const std::vector<std::string_view>& assigned_lines, ThreadData& data) {
    ParsedInfo info;
    for (const auto& line_sv : assigned_lines) {
        if (ultimate_parser(line_sv, info)) {
            if (info.is_tcp) {
                TCPKey key = {info.src_ip, info.dst_ip, info.src_port, info.dst_port};
                char& state = data.flow_map[key];
                if (state == 0) { // New entry
                    state = (info.last_token == "SYN") ? 1 : 2;
                } else if (state == 1 && info.last_token != "SYN") {
                    state = 2;
                }
            } else {
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

    // Using std::hash for convenience, could be faster with custom integer-only hash
    std::hash<TCPKey> tcp_hasher;
    std::hash<uint32_t> ip_hasher;

    // --- PASS 1: Parallel Count (using minimal parsing for key) ---
    // The ultimate_parser is fast enough to be used directly here.
    std::vector<std::vector<size_t>> counts(num_threads, std::vector<size_t>(num_threads, 0));
    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        size_t chunk_size = file_size / num_threads;
        size_t start_pos = tid * chunk_size;
        size_t end_pos = (tid == num_threads - 1) ? file_size : (tid + 1) * chunk_size;
        if (tid > 0 && start_pos > 0) while (start_pos < file_size && buffer[start_pos - 1] != '\n') start_pos++;
        if (tid < num_threads - 1 && end_pos < file_size) while (end_pos < file_size && buffer[end_pos - 1] != '\n') end_pos++;
        const char* ptr = buffer + start_pos;
        const char* const chunk_end = buffer + end_pos;
        
        ParsedInfo info;
        while (ptr < chunk_end) {
            const char* line_end = (const char*)memchr(ptr, '\n', chunk_end - ptr);
            if (!line_end) line_end = chunk_end;
            if (ultimate_parser({ptr, (size_t)(line_end - ptr)}, info)) {
                if (info.is_tcp) {
                    TCPKey key = {info.src_ip, info.dst_ip, info.src_port, info.dst_port};
                    counts[tid][tcp_hasher(key) % num_threads]++;
                } else {
                    counts[tid][ip_hasher(info.src_ip) % num_threads]++;
                }
            }
            ptr = line_end + 1;
        }
    }

    // --- Inter-Pass: Calculate Offsets and Allocate ---
    std::vector<std::vector<size_t>> write_offsets(num_threads, std::vector<size_t>(num_threads));
    for (int dest_tid = 0; dest_tid < num_threads; ++dest_tid) {
        size_t current_offset = 0;
        for (int src_tid = 0; src_tid < num_threads; ++src_tid) {
            write_offsets[src_tid][dest_tid] = current_offset;
            current_offset += counts[src_tid][dest_tid];
        }
        thread_lines[dest_tid].resize(current_offset);
    }
    
    // --- PASS 2: Parallel Scatter ---
    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        size_t chunk_size = file_size / num_threads;
        size_t start_pos = tid * chunk_size;
        size_t end_pos = (tid == num_threads - 1) ? file_size : (tid + 1) * chunk_size;
        if (tid > 0 && start_pos > 0) while (start_pos < file_size && buffer[start_pos - 1] != '\n') start_pos++;
        if (tid < num_threads - 1 && end_pos < file_size) while (end_pos < file_size && buffer[end_pos - 1] != '\n') end_pos++;
        const char* ptr = buffer + start_pos;
        const char* const chunk_end = buffer + end_pos;
        
        ParsedInfo info;
        while (ptr < chunk_end) {
            const char* line_end = (const char*)memchr(ptr, '\n', chunk_end - ptr);
            if (!line_end) line_end = chunk_end;
            std::string_view line_sv(ptr, line_end - ptr);
            if (ultimate_parser(line_sv, info)) {
                 if (info.is_tcp) {
                    TCPKey key = {info.src_ip, info.dst_ip, info.src_port, info.dst_port};
                    size_t dest_tid = tcp_hasher(key) % num_threads;
                    thread_lines[dest_tid][write_offsets[tid][dest_tid]++] = line_sv;
                } else {
                    size_t dest_tid = ip_hasher(info.src_ip) % num_threads;
                    thread_lines[dest_tid][write_offsets[tid][dest_tid]++] = line_sv;
                }
            }
            ptr = line_end + 1;
        }
    }

    // --- PHASE 3: Parallel Lock-Free Processing with FastMap ---
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < num_threads; ++i) {
        worker_func(thread_lines[i], thread_data[i]);
    }

    // --- AGGREGATION ---
    std::unordered_map<uint32_t, long long> dnstunnel_final_count;
    std::unordered_map<uint32_t, long long> portscan_final_count;
    for(auto& data : thread_data) {
        for(auto& entry : data.dnstunnel_count) if(entry.occupied) dnstunnel_final_count[entry.key] += entry.value;
        for(auto& entry : data.flow_map) if(entry.occupied && entry.value == 1) portscan_final_count[entry.key.src_ip]++;
    }

    // --- Output with Optimized Sorting ---
    auto format_and_print = [](const std::string& type, std::unordered_map<uint32_t, long long>& counts) {
        if (counts.empty()) return;
        std::vector<std::pair<std::string, long long>> sorted_counts;
        sorted_counts.reserve(counts.size());
        char ip_buf[16];
        for (const auto& [ip_int, count] : counts) {
            sprintf(ip_buf, "%u.%u.%u.%u", (ip_int >> 24) & 0xFF, (ip_int >> 16) & 0xFF, (ip_int >> 8) & 0xFF, ip_int & 0xFF);
            sorted_counts.emplace_back(ip_buf, count);
        }
        std::sort(sorted_counts.begin(), sorted_counts.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        std::string out_buffer;
        out_buffer.reserve(sorted_counts.size() * 60);
        char line_buf[128];
        for (const auto& [ip_str, count] : sorted_counts) {
             int len = sprintf(line_buf, "%s %s %lld\n", ip_str.c_str(), type.c_str(), count);
            out_buffer.append(line_buf, len);
        }
        fwrite(out_buffer.data(), 1, out_buffer.size(), stdout);
    };

    format_and_print("portscan", portscan_final_count);
    format_and_print("tunnelling", dnstunnel_final_count);

    munmap((void*)buffer, file_size);
    close(fd);

    return 0;
}