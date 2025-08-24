#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===== 1. 路径与参数 (客户端配置) =====
# 假设 llama.cpp 服务器正在此地址和端口上运行
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8080
# 并发请求数，应与服务器启动时的 --parallel 参数保持一致，以获得最佳性能
CONCURRENT_REQUESTS = 5

# ===== GBNF 和推理参数 =====
# GBNF (Guided Backus-Naur Form) 语法
# 这个语法强制模型只能生成 "A", "B", "C", "D" 四个字符中的一个。
# 这是确保模型输出格式正确的关键。
GBNF_GRAMMAR = r'''
root ::= ("A" | "B" | "C" | "D")
'''

# 推理参数
# 这些参数会随每个请求发送给服务器
INFERENCE_PARAMS = {
    "n_predict": 1,         # 只需要预测1个 token (即一个字母)
    "temperature": 1,       # 温度设置为1，允许模型有一定的随机性
    "top_p": 0.95,          # top-p 采样
    "stop": ["<|end|>"],    # 停止符 (虽然 n_predict=1 使其作用不大)
    "grammar": GBNF_GRAMMAR,# 使用上面定义的 GBNF 语法来约束输出
}

# ===== 2. Prompt 模板 =====
# 一个清晰的、指示性的 Prompt，引导模型执行单项选择任务。
# 即使有 GBNF 语法，一个好的 Prompt 也能提升模型理解任务的准确性。
PROMPT_TEMPLATE = """<|user>
The following is a multiple-choice question with options A, B, C, D. Select the correct answer and respond with ONLY the letter (A, B, C, or D). No ANY explanation.

Question:
{q}
<|end|>
<|assistant>
"""

# ===== 3. 并行处理函数 =====
def solve_question(question_tuple):
    """
    处理单个问题：构建 prompt，发送请求到服务器，并返回结果。
    """
    index, q = question_tuple
    try:
        # 如果问题行为空，直接返回默认答案，避免发送无效请求
        if not q.strip():
            return index, "B"
            
        # 使用模板格式化 Prompt
        prompt = PROMPT_TEMPLATE.format(q=q.strip())

        # 准备请求体
        data = {"prompt": prompt, **INFERENCE_PARAMS}
        headers = {"Connection": "close"}

        # 向正在运行的 llama.cpp 服务器发送 POST 请求
        response = requests.post(
            f"http://{SERVER_HOST}:{SERVER_PORT}/completion",
            json=data,
            timeout=600,  # 10分钟超时
            headers=headers
        )
        # 如果服务器返回错误状态码 (如 4xx, 5xx)，则抛出异常
        response.raise_for_status()
        
        # 从 JSON 响应中获取模型生成的内容
        full_output = response.json()['content']

        answer = full_output.strip().upper()

        if answer not in {"A", "B", "C", "D"}:
            print(f"WARN on question {index}: Unexpected output '{answer}'. Defaulting to 'B'.", file=sys.stderr)
            return index, "B"

        return index, answer

    except Exception as e:
        # 捕获所有可能的异常 (网络问题、超时、JSON 解析错误等)
        print(f"ERROR on question {index}: {e}", file=sys.stderr)
        # 出错时返回一个默认答案 "B"，确保程序不会中断
        return index, "B"

# ===== 4. 读取输入并并行执行 =====
def main():
    """
    主函数：从标准输入读取所有问题，使用线程池并行处理，并按原始顺序打印结果。
    """
    # 从 stdin 读取全部内容，并按双换行符分割成问题列表
    questions = sys.stdin.read().strip().split("\n\n")
    results = {}  # 使用字典来存储结果，键为原始索引，值为答案
    
    # 为每个非空问题创建一个带索引的元组
    questions_with_indices = list(enumerate(q for q in questions if q.strip()))

    # 使用线程池来并行发送请求
    with ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as executor:
        print(f"Submitting {len(questions_with_indices)} questions to the running server...", file=sys.stderr)

        # 提交所有任务到线程池
        future_to_question = {
            executor.submit(solve_question, q_tuple): q_tuple for q_tuple in questions_with_indices
        }

        try:
            from tqdm import tqdm
            # as_completed 会在任务完成时立即返回 future 对象，实现乱序完成
            progress_iterator = tqdm(as_completed(future_to_question), total=len(questions_with_indices), file=sys.stderr, desc="Processing questions")
        except ImportError:
            progress_iterator = as_completed(future_to_question)

        # 遍历已完成的任务
        for future in progress_iterator:
            index, answer = future.result()
            results[index] = answer

    # ===== 5. 按顺序输出结果 =====
    # 确保即使某些问题处理失败，也能为每个原始问题输出一个答案
    for i in range(len(questions)):
        print(results.get(i, "B"))

if __name__ == "__main__":
    main()