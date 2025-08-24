#!/bin/bash
#SBATCH --job-name=a_plus_b        # 任务名称
#SBATCH --nodes=1                  # 申请 1 个节点
#SBATCH --ntasks=1                 # 1 个任务
#SBATCH --cpus-per-task=1          # 每个任务使用1个核心
#SBATCH --time=00:05:00            # 时间限制 5 分钟
#SBATCH --partition=hpc003t          # 提交到 c003t 分区
#SBATCH --output=aplusb_%j.out     # 标准输出文件（%j 会被替换为任务ID）
#SBATCH --error=aplusb_%j.err      # 标准错误文件

# 运行评分器
zxscorer "https://hpci.chouhsing.org/problems/a-plus-b/" --token="7c699a11-8c28-5dc7-b27d-67def56181af" -- python3 aplusb.py
