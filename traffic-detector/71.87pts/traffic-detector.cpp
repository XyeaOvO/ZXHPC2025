#include <bits/stdc++.h>
using namespace std;


struct FlowInfo {
    char state;
    string src_ip;
    FlowInfo(): state(0) {}
    FlowInfo(char s, string&& ip): state(s), src_ip(std::move(ip)) {}
};

inline void skip_spaces(const string &s, size_t &i) {
    while (i < s.size() && s[i] == ' ') ++i;
}

// 读取下一个 token（以空格分隔），返回 token（不包括空格），并把 i 移到下一 token 开始位置
inline string next_token(const string &s, size_t &i) {
    skip_spaces(s, i);
    if (i >= s.size()) return string();
    size_t j = i;
    while (j < s.size() && s[j] != ' ') ++j;
    string tok = s.substr(i, j - i);
    i = j;
    return tok;
}

// 读取下一个 token 但不分配（返回开始指针与长度），更高效 -- 这里偶尔会用到
inline pair<size_t,size_t> next_token_pos(const string &s, size_t &i) {
    skip_spaces(s, i);
    if (i >= s.size()) return {string::npos, 0};
    size_t j = i;
    while (j < s.size() && s[j] != ' ') ++j;
    size_t start = i;
    size_t len = j - i;
    i = j;
    return {start, len};
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // 预分配大小（可调整），避免频繁重哈希
    unordered_map<string, FlowInfo> flow_map;
    flow_map.reserve(1 << 20); // 如果输入很大可以适当增加

    unordered_map<string, long long> dnstunnel_count;
    dnstunnel_count.reserve(1 << 20);

    string line;
    // 逐行读取
    while (std::getline(cin, line)) {
        if (line.empty()) continue;
        size_t i = 0;

        // timestamp (skip)
        (void) next_token(line, i);

        // protocol
        string proto = next_token(line, i);
        if (proto.empty()) continue;

        // src_ip, dst_ip
        string src_ip = next_token(line, i);
        if (src_ip.empty()) continue;
        string dst_ip = next_token(line, i);
        if (dst_ip.empty()) continue;

        // src_port, dst_port
        string src_port = next_token(line, i);
        if (src_port.empty()) continue;
        string dst_port = next_token(line, i);
        if (dst_port.empty()) continue;

        if (proto[0] == 'T') { // TCP
            string flags = next_token(line, i); // flags like "SYN"
            if (flags.empty()) continue;
            // data_len (we don't need data)
            string data_len = next_token(line, i);
            // data may exist but we ignore

            // 构造五元组 key（尽量少的分配）
            // key = src_ip + '|' + dst_ip + '|' + src_port + '|' + dst_port
            string key;
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
                    flow_map.emplace(std::move(key), FlowInfo(1, std::move(src_ip)));
                } else {
                    flow_map.emplace(std::move(key), FlowInfo(2, std::move(src_ip)));
                }
            } else {
                // 已存在，若之前是 0/1 则变为 2（多包或非唯一SYN）
                if (it->second.state == 1 || it->second.state == 0) {
                    it->second.state = 2;
                }
                // 已经为 2 则保持
            }
        } else { // assume DNS (题目仅有 TCP/DNS)
            // next token is domain length (int), then domain string (可能含点)
            // 读取域名长度 token（我们不强依赖它，但需跳过）
            string dom_len_tok = next_token(line, i);
            // domain：剩余第一个 token（domain 中不会含空格），直接取 next_token
            string domain = next_token(line, i);
            if (domain.empty()) continue;

            // 找 domain 的第一个 '.'
            size_t pos = domain.find('.');
            size_t prefix_len = (pos == string::npos) ? domain.size() : pos;
            if (prefix_len >= 30) {
                // 累加 prefix 长度
                dnstunnel_count[src_ip] += (long long)prefix_len;
            }
        }
    }

    // 汇总 portscan：遍历 flow_map，统计每个 src_ip 的计数
    unordered_map<string, long long> portscan_count;
    portscan_count.reserve(1 << 20);
    for (const auto &kv : flow_map) {
        const FlowInfo &fi = kv.second;
        if (fi.state == 1) {
            // 只有 state==1 的五元组计作一次 portscan，fi.src_ip 保存源 IP
            portscan_count[fi.src_ip] += 1;
        }
    }

    // 输出要求：先所有 portscan（IP 按字典序升），再 tunnelling（同样排序）
    vector<pair<string, long long>> portvec;
    portvec.reserve(portscan_count.size());
    for (auto &kv : portscan_count) portvec.emplace_back(kv.first, kv.second);
    sort(portvec.begin(), portvec.end(), [](const auto &a, const auto &b){
        return a.first < b.first;
    });
    for (auto &p : portvec) {
        cout << p.first << " portscan " << p.second << '\n';
    }

    vector<pair<string, long long>> dnsvec;
    dnsvec.reserve(dnstunnel_count.size());
    for (auto &kv : dnstunnel_count) dnsvec.emplace_back(kv.first, kv.second);
    sort(dnsvec.begin(), dnsvec.end(), [](const auto &a, const auto &b){
        return a.first < b.first;
    });
    for (auto &p : dnsvec) {
        cout << p.first << " tunnelling " << p.second << '\n';
    }

    return 0;
}
