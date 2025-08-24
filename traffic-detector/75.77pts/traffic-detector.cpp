#include <iostream>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <utility>

/*
  高性能解析：基于行缓冲 + string_view 分词
  目标：消除分词过程中的字符串复制与临时对象
*/

// Flow 状态：0 = 未见, 1 = 仅见一次且为 SYN, 2 = 多包或非 SYN (不可计为 portscan)
struct FlowInfo {
    char state; // 0/1/2
    // 源IP现在由外部汇总时提供，不再存储在每个 FlowInfo 中，以节省内存
};

// 跳过前导空格
inline void skip_spaces(std::string_view s, size_t &i) {
    while (i < s.size() && s[i] == ' ') ++i;
}

// 读取下一个 token，返回 string_view，几乎零开销
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

    // TCP流的键是五元组字符串，值是状态
    std::unordered_map<std::string, char> flow_map;
    flow_map.reserve(1 << 22);

    // DNS和Portscan的统计都直接使用 string 作为 key
    std::unordered_map<std::string, long long> dnstunnel_count;
    dnstunnel_count.reserve(1 << 20);

    std::string line;
    line.reserve(256); // 为行缓冲预分配一些空间

    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        
        // 使用 string_view 进行解析
        std::string_view line_sv(line);
        size_t i = 0;

        (void) next_token_sv(line_sv, i); // timestamp (skip)

        std::string_view proto = next_token_sv(line_sv, i);
        if (proto.empty()) continue;

        std::string_view src_ip = next_token_sv(line_sv, i);
        if (src_ip.empty()) continue;
        std::string_view dst_ip = next_token_sv(line_sv, i);
        if (dst_ip.empty()) continue;
        std::string_view src_port = next_token_sv(line_sv, i);
        if (src_port.empty()) continue;
        std::string_view dst_port = next_token_sv(line_sv, i);
        if (dst_port.empty()) continue;

        if (proto.front() == 'T') { // TCP
            std::string_view flags = next_token_sv(line_sv, i);
            if (flags.empty()) continue;
            
            // 构造五元组 key
            std::string key;
            key.reserve(src_ip.size() + dst_ip.size() + src_port.size() + dst_port.size() + 3);
            key.append(src_ip);
            key.push_back('|');
            key.append(dst_ip);
            key.push_back('|');
            key.append(src_port);
            key.push_back('|');
            key.append(dst_port);
            
            auto it = flow_map.find(key);
            if (it == flow_map.end()) {
                // 首次见到
                if (flags == "SYN") {
                    // 仅保存状态，源IP在最后汇总时通过解析key得到
                    flow_map.emplace(std::move(key), 1);
                } else {
                    flow_map.emplace(std::move(key), 2);
                }
            } else {
                // 已存在，若之前是 1 则变为 2（多包）
                if (it->second == 1) {
                    it->second = 2;
                }
            }
        } else { // DNS
            (void) next_token_sv(line_sv, i); // domain length (skip)
            std::string_view domain = next_token_sv(line_sv, i);
            if (domain.empty()) continue;

            size_t pos = domain.find('.');
            size_t prefix_len = (pos == std::string_view::npos) ? domain.size() : pos;
            
            if (prefix_len >= 30) {
                // 此处需要创建 string 对象作为 map 的 key
                dnstunnel_count[std::string(src_ip)] += prefix_len;
            }
        }
    }

    // 汇总 portscan
    std::unordered_map<std::string, long long> portscan_count;
    portscan_count.reserve(1 << 20);
    for (const auto &[key, state] : flow_map) {
        if (state == 1) {
            // 从 key 中解析出 src_ip
            size_t pos = key.find('|');
            portscan_count[key.substr(0, pos)] += 1;
        }
    }

    // 输出
    std::vector<std::pair<std::string, long long>> portvec;
    portvec.reserve(portscan_count.size());
    for (const auto &kv : portscan_count) portvec.emplace_back(kv.first, kv.second);
    std::sort(portvec.begin(), portvec.end());
    for (const auto &p : portvec) {
        std::cout << p.first << " portscan " << p.second << '\n';
    }

    std::vector<std::pair<std::string, long long>> dnsvec;
    dnsvec.reserve(dnstunnel_count.size());
    for (const auto &kv : dnstunnel_count) dnsvec.emplace_back(kv.first, kv.second);
    std::sort(dnsvec.begin(), dnsvec.end());
    for (const auto &p : dnsvec) {
        std::cout << p.first << " tunnelling " << p.second << '\n';
    }

    return 0;
}