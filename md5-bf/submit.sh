#!/bin/bash
#SBATCH --job-name=md5_bf          # 任务名称
#SBATCH --nodes=1                  # 申请 1 个节点
#SBATCH --ntasks=1                 # 1 个任务
#SBATCH --cpus-per-task=40         # 申请40个核心！
#SBATCH --time=00:10:00            # 时间限制 10 分钟
#SBATCH --partition=c003t          # 提交到 c003t 分区
#SBATCH --output=md5_bf_%j.out     # 标准输出文件
#SBATCH --error=md5_bf_%j.err      # 标准错误文件

# 加载支持现代C++和OpenMP的GCC编译器
echo "Loading GCC 10.1.0 module..."
module load gcc/10.1.0

echo "Compiling the MD5 brute-force solver..."

g++ -std=c++17 -O3 -march=native -fopenmp md5_optimized.cpp -o solution -mavx512f

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Compilation successful. Running the scorer..."

# 请将下面的 URL 和 Token 替换为题目页面提供的真实值！
PROBLEM_URL="https://hpci.chouhsing.org/problems/md5-new/"
TOKEN="7c699a11-8c28-5dc7-b27d-67def56181af" 

# 运行评分器
# 远程运行
zxscorer "$PROBLEM_URL" -- ./solution