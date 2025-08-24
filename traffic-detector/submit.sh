#!/bin/bash
#SBATCH --job-name=traffic_detector_optimized
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=00:10:00
#SBATCH --partition=hpc003t
#SBATCH --output=traffic_detector_%j.out
#SBATCH --error=traffic_detector_%j.err

echo "Loading GCC 10.1.0 module..."
module load gcc/10.1.0

echo "Compiling the optimized traffic detector..."

g++ -std=c++17 -Ofast -march=native -flto -pipe -fopenmp traffic-detector.cpp -o solution

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Compilation successful. Running the scorer..."

PROBLEM_URL="https://hpci.chouhsing.org/problems/traffic-detector/"
TOKEN="7c699a11-8c28-5dc7-b27d-67def56181af" 

# 运行评分器
# --token="$TOKEN"
/home/share/chouhsing/bin/zxscorer "$PROBLEM_URL" --token="$TOKEN" -- ./solution