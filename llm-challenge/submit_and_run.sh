#!/bin/bash

#SBATCH -J llm_challenge_job
#SBATCH -p c003t
#SBATCH -N 1
#SBATCH --cpus-per-task=40
#SBATCH --mem=40G
#SBATCH --exclusive 
#SBATCH --time=00:30:00
#SBATCH -o ./log/job_%j.out
#SBATCH -e ./log/job_%j.err

# --- 1. 设置环境 ---
echo "Setting up environment..."
source ~/llm-challenge/.venv/bin/activate
module load gcc/8.4.0

# --- 2. 定义服务器参数 ---
LLAMA_CPP_DIR="/home/u2023212641/llm-challenge/llama.cpp"
SERVER_EXE="${LLAMA_CPP_DIR}/build/bin/llama-server"
MODEL="${LLAMA_CPP_DIR}/models/microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"

SERVER_HOST="127.0.0.1"
SERVER_PORT="8080"
# --- 关键改动: 大幅提高并发度 ---
CONCURRENT_REQUESTS=5
CONTEXT_SIZE=4096
# CONTEXT_SIZE=3840
NUM_THREADS=${SLURM_CPUS_PER_TASK:-40}

SERVER_CMD=(\
    "$SERVER_EXE" \
    -m "$MODEL" \
    --host "$SERVER_HOST" \
    --port "$SERVER_PORT" \
    -t "$NUM_THREADS" \
    -tb "$NUM_THREADS" \
    -c "$CONTEXT_SIZE" \
    -b 4096 \
    -ctk q8_0 \
    -ctv q8_0 \
    --parallel "$CONCURRENT_REQUESTS" \
    # --kv-unified \
    --numa distribute \
    --timeout 600 \
    --mlock \
    --flash-attn \
)

# --- 3. 启动服务器 ---
echo "Starting llama.cpp server in the background..."
echo "Server command: ${SERVER_CMD[@]}"
# 直接后台运行，日志会进入 SLURM 的输出/错误文件
"${SERVER_CMD[@]}" &
SERVER_PID=$!
echo "Server started with PID: $SERVER_PID"

trap 'echo "Cleaning up server..."; kill -TERM $SERVER_PID; wait $SERVER_PID 2>/dev/null' EXIT

# --- 4. 等待服务器就绪 ---
echo "Waiting for server to become ready..."
MAX_WAIT=10
SECONDS=0
while true; do
    HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://${SERVER_HOST}:${SERVER_PORT}/health)
    if [ "$HTTP_STATUS" -eq 200 ]; then
        echo "Server is ready!"
        break
    fi
    if [ $SECONDS -ge $MAX_WAIT ]; then
        echo "Server failed to start within $MAX_WAIT seconds. Exiting."
        exit 1
    fi
    sleep 2
    echo -n "."
done

TOKEN="7c699a11-8c28-5dc7-b27d-67def56181af" 
# --- 5. 运行评分程序 ---
echo "Running zxscorer with solver_client.py..."
zxscorer "https://hpci.chouhsing.org/problems/llm-challenge/" \
    --token="$TOKEN" \
        -- python ~/llm-challenge/solver_client.py

echo "Scoring finished."