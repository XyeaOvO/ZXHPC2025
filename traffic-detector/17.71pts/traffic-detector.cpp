#include <iostream>
#include <sstream>
#include <map>
#include <vector>
#include <string>

struct Packet {
    double timestamp;
    std::string protocol;
    std::string src_ip, dst_ip;
    int src_port = -1, dst_port = -1;
    std::string flags;
    int data_len = 0;
    std::string data;
};

Packet parse_line(const std::string& line) {
    Packet pkt;
    std::istringstream iss(line);
    iss >> pkt.timestamp >> pkt.protocol >> pkt.src_ip >> pkt.dst_ip;
    if (pkt.protocol == "TCP" || pkt.protocol == "DNS") {
        iss >> pkt.src_port >> pkt.dst_port;
        if (pkt.protocol == "TCP") {
            iss >> pkt.flags;
        }
        iss >> pkt.data_len;
        if (iss.peek() == ' ' || iss.peek() == '\t') iss.get();
        std::getline(iss, pkt.data);
        if (!pkt.data.empty() && pkt.data[0] == ' ') pkt.data.erase(0, 1);
    }
    return pkt;
}

std::string get_dns_prefix(const std::string& domain) {
    size_t dot = domain.find('.');
    if (dot != std::string::npos) return domain.substr(0, dot);
    return "";
}

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    struct FiveTuple {
        std::string src_ip, dst_ip;
        int src_port, dst_port;
    };

    auto tuple_str = [](const FiveTuple& t) {
        return t.src_ip + "|" + t.dst_ip + "|" + std::to_string(t.src_port) + "|" + std::to_string(t.dst_port);
    };

    std::map<std::string, std::vector<Packet>> syn_flows;

    std::map<std::string, int> dnstunnel_count;

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        Packet pkt = parse_line(line);

        if (pkt.protocol == "TCP") {
            FiveTuple key{pkt.src_ip, pkt.dst_ip, pkt.src_port, pkt.dst_port};
            syn_flows[tuple_str(key)].push_back(pkt);
        }
        else if (pkt.protocol == "DNS" && !pkt.data.empty()) {
            std::string prefix = get_dns_prefix(pkt.data);
            if (prefix.length() >= 30) {
                dnstunnel_count[pkt.src_ip]+=prefix.length();
            }
        }
    }

    std::map<std::string, int> portscan_ip_count;
    for (const auto& kv : syn_flows) {
        const std::vector<Packet>& pkts = kv.second;
        if (pkts.size() == 1 && pkts[0].flags == "SYN") {
            portscan_ip_count[pkts[0].src_ip]++;
        }
    }

    for (const auto& kv : portscan_ip_count) {
        std::cout << kv.first << " portscan " << kv.second << std::endl;
    }
    for (const auto& kv : dnstunnel_count) {
        std::cout << kv.first << " tunnelling " << kv.second << std::endl;
    }

    return 0;
}