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
#include <unordered_map>

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

// --- Custom Hash Table (Linear Probing) ---
// This custom map is generally faster than std::unordered_map due to simpler implementation.
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
        // Ensure capacity is a power of two for fast modulo
        table_size = 1;
        while(table_size < initial_capacity) table_size <<= 1;
        table.resize(table_size);
    }

    // FIX 2: Add size() method
    size_t size() const {
        return num_elements;
    }

    // FIX 2: Add empty() method
    bool empty() const {
        return num_elements == 0;
    }

    Value& operator[](const Key& key) {
        size_t index = hasher(key) & (table_size - 1);
        while (table[index].occupied && !(table[index].key == key)) {
            index = (index + 1) & (table_size - 1);
        }

        if (table[index].occupied) { // Key found
            return table[index].value;
        } else { // New key, needs insertion
            // OPTIMIZATION: Check for rehash only on insertion, not on every access.
            if (num_elements * 2 > table_size) {
                rehash();
                // After rehash, find the new index for the key.
                index = hasher(key) & (table_size - 1);
                while (table[index].occupied && !(table[index].key == key)) {
                    index = (index + 1) & (table_size - 1);
                }
            }
            table[index].key = key;
            table[index].occupied = true;
            // Default-initialize the value. Assumes Value is default-constructible.
            table[index].value = Value{}; 
            num_elements++;
            return table[index].value;
        }
    }

    // Custom iterator support to allow range-based for loops
    struct Iterator {
        Entry* ptr;
        Entry* end;
        void operator++() { 
            ptr++;
            while(ptr < end && !ptr->occupied) ptr++;
        }
        bool operator!=(const Iterator& other) const { return ptr != other.ptr; }
        Entry& operator*() { return *ptr; }
    };
    Iterator begin() { 
        Entry* p = table.data();
        Entry* e = table.data() + table_size;
        while(p < e && !p->occupied) p++;
        return {p, e};
    }
    Iterator end() { return {table.data() + table_size, table.data() + table_size}; }


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

// OPTIMIZATION 1 (MAJOR): Define a compact struct to hold parsed data.
// This struct will be passed to worker threads, eliminating the need for re-parsing.
struct ParsedData {
    bool is_tcp;
    uint32_t src_ip;
    union {
        struct { // TCP specific info
            uint32_t dst_ip;
            uint16_t src_port;
            uint16_t dst_port;
            bool is_syn;
        } tcp;
        struct { // DNS specific info
            uint16_t prefix_len;
        } dns;
    };
};


// OPTIMIZATION 1: Modify parser to perform a full parse and populate the ParsedData struct.
inline bool full_parser(std::string_view line, ParsedData& out) {
    const char* p = line.data();
    const char* const end = p + line.size();
    uint32_t temp_ip = 0, part = 0;
    uint16_t temp_port = 0;

    while (p < end && *p != ' ') p++; if (p == end) return false; p++;
    out.is_tcp = (*p == 'T');
    while (p < end && *p != ' ') p++; if (p == end) return false; p++;

    for (int i = 0; i < 4; ++i) { part = 0; while (p < end && *p >= '0' && *p <= '9') part = part * 10 + (*p++ - '0'); temp_ip = (temp_ip << 8) | part; if (p < end && *p == '.') p++; }
    out.src_ip = temp_ip;
    if (p == end) return false; p++;

    temp_ip = 0;
    for (int i = 0; i < 4; ++i) { part = 0; while (p < end && *p >= '0' && *p <= '9') part = part * 10 + (*p++ - '0'); temp_ip = (temp_ip << 8) | part; if (p < end && *p == '.') p++; }
    uint32_t dst_ip_val = temp_ip;
    if (p == end) return false; p++;
    
    temp_port = 0; while (p < end && *p >= '0' && *p <= '9') temp_port = temp_port * 10 + (*p++ - '0');
    uint16_t src_port_val = temp_port;
    if (p == end) return false; p++;
    
    temp_port = 0; while (p < end && *p >= '0' && *p <= '9') temp_port = temp_port * 10 + (*p++ - '0');
    uint16_t dst_port_val = temp_port;
    if (p == end) return false; p++;

    if (out.is_tcp) {
        out.tcp.dst_ip = dst_ip_val;
        out.tcp.src_port = src_port_val;
        out.tcp.dst_port = dst_port_val;
        out.tcp.is_syn = (p[0] == 'S' && p[1] == 'Y' && p[2] == 'N' && (p[3] == ' ' || p+3 == end));
    } else {
        while (p < end && *p != ' ') p++; if (p == end) return false; p++;
        const char* domain_start = p;
        const char* dot = (const char*)memchr(domain_start, '.', end - domain_start);
        size_t prefix_len = (dot == nullptr) ? (end - domain_start) : (dot - domain_start);
        out.dns.prefix_len = prefix_len;
    }
    return true;
}

struct ThreadData {
    FastMap<TCPKey, char> flow_map; // TCP flow state: 0=new, 1=SYN-only, 2=Established
    FastMap<uint32_t, long long> dnstunnel_count;
};

// OPTIMIZATION 1: Worker function now processes a vector of ParsedData structs. No parsing needed here.
void worker_func(const std::vector<ParsedData>& assigned_data, ThreadData& data) {
    for (const auto& p_data : assigned_data) {
        if (p_data.is_tcp) {
            TCPKey key = {p_data.src_ip, p_data.tcp.dst_ip, p_data.tcp.src_port, p_data.tcp.dst_port};
            char& state = data.flow_map[key];
            if (state == 0) { // New flow
                state = p_data.tcp.is_syn ? 1 : 2;
            } else if (state == 1 && !p_data.tcp.is_syn) { // SYN-only flow sees another packet
                state = 2;
            }
        } else { // is DNS
            if (p_data.dns.prefix_len >= 30) {
                data.dnstunnel_count[p_data.src_ip] += p_data.dns.prefix_len;
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
    // OPTIMIZATION 1: This vector will hold parsed data structs, not string_views.
    std::vector<std::vector<ParsedData>> thread_parsed_data(num_threads);

    std::hash<TCPKey> tcp_hasher;
    std::hash<uint32_t> ip_hasher;

    // --- PASS 1: Parallel Count (Parse once) ---
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
        
        ParsedData p_data;
        while (ptr < chunk_end) {
            const char* line_end = (const char*)memchr(ptr, '\n', chunk_end - ptr);
            if (!line_end) line_end = chunk_end;
            if (full_parser({ptr, (size_t)(line_end - ptr)}, p_data)) {
                if (p_data.is_tcp) {
                    TCPKey key = {p_data.src_ip, p_data.tcp.dst_ip, p_data.tcp.src_port, p_data.tcp.dst_port};
                    counts[tid][tcp_hasher(key) % num_threads]++;
                } else {
                    counts[tid][ip_hasher(p_data.src_ip) % num_threads]++;
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
        thread_parsed_data[dest_tid].resize(current_offset);
    }
    
    // --- PASS 2: Parallel Scatter (Scatter parsed structs) ---
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
        
        ParsedData p_data;
        while (ptr < chunk_end) {
            const char* line_end = (const char*)memchr(ptr, '\n', chunk_end - ptr);
            if (!line_end) line_end = chunk_end;
            if (full_parser({ptr, (size_t)(line_end - ptr)}, p_data)) {
                 if (p_data.is_tcp) {
                    TCPKey key = {p_data.src_ip, p_data.tcp.dst_ip, p_data.tcp.src_port, p_data.tcp.dst_port};
                    size_t dest_tid = tcp_hasher(key) % num_threads;
                    thread_parsed_data[dest_tid][write_offsets[tid][dest_tid]++] = p_data;
                } else {
                    size_t dest_tid = ip_hasher(p_data.src_ip) % num_threads;
                    thread_parsed_data[dest_tid][write_offsets[tid][dest_tid]++] = p_data;
                }
            }
            ptr = line_end + 1;
        }
    }

    // --- PHASE 3: Parallel Lock-Free Processing (No parsing here) ---
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < num_threads; ++i) {
        worker_func(thread_parsed_data[i], thread_data[i]);
    }

    // --- AGGREGATION ---
    // OPTIMIZATION: Use FastMap for final aggregation as well.
    FastMap<uint32_t, long long> dnstunnel_final_count(1 << 16);
    FastMap<uint32_t, long long> portscan_final_count(1 << 16);
    
    for(auto& data : thread_data) {
        for(auto& entry : data.dnstunnel_count) dnstunnel_final_count[entry.key] += entry.value;
        for(auto& entry : data.flow_map) if(entry.value == 1) portscan_final_count[entry.key.src_ip]++;
    }

    // --- Output with Optimized Sorting ---
    auto format_and_print = [](const std::string& type, auto& counts) {
        if (counts.empty()) return;
        std::vector<std::pair<std::string, long long>> sorted_counts;
        sorted_counts.reserve(counts.size());
        char ip_buf[16];
        for (const auto& entry : counts) {
            uint32_t ip_int = entry.key;
            long long count = entry.value;
            sprintf(ip_buf, "%u.%u.%u.%u", (ip_int >> 24) & 0xFF, (ip_int >> 16) & 0xFF, (ip_int >> 8) & 0xFF, ip_int & 0xFF);
            sorted_counts.emplace_back(ip_buf, count);
        }
        std::sort(sorted_counts.begin(), sorted_counts.end()); // Default pair sort is lexicographical on first element
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