#!/bin/bash
#SBATCH --job-name=cosine_sim      # 任务名称
#SBATCH --nodes=1                  # 申请 1 个节点
#SBATCH --ntasks=1                 # 1 个任务
#SBATCH --cpus-per-task=40         # 申请40个核心
#SBATCH --time=00:10:00            # 时间限制 10 分钟
#SBATCH --partition=c003t          # 提交到 c003t 分区
#SBATCH --output=cosine_%j.out     # 标准输出文件
#SBATCH --error=cosine_%j.err      # 标准错误文件

module load gcc/10.1.0

echo "Compiling the optimized code..."

g++ -std=c++11 -O3 -march=native -fopenmp cosine_optimized.cpp -o solution -mavx512f

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Compilation successful. Running the scorer..."

# 请将下面的 URL 和 Token 替换为题目页面提供的真实值！
PROBLEM_URL="https://hpci.chouhsing.org/problems/cos-sim/" # 假设的URL，请替换
TOKEN="7c699a11-8c28-5dc7-b27d-67def56181af" # 你的Token，请替换

zxscorer "$PROBLEM_URL" -t "$TOKEN" -- ./solution

