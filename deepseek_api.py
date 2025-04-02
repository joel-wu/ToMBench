import json
import math
import random

import numpy
import argparse
from prompts import *
from tqdm import tqdm
import os
import torch
import json
import random
import os
import asyncio
from together import Together
from argparse import ArgumentParser

# 导入必要的库
from prompts import *
from tqdm import tqdm

def format_prompt_4(d, args):
    if args.language == 'zh':
        cA = d['选项A'].replace("A. ", "")
        cB = d['选项B'].replace("B. ", "")
        cC = d['选项C'].replace("C. ", "")
        cD = d['选项D'].replace("D. ", "")
        choices = [cA, cB, cC, cD]
        random.shuffle(choices)
        prompt = UserEvaluatePrompt4Choices_zh.format(story=d['故事'], question=d['问题'], choice_a=choices[0],
                                                      choice_b=choices[1], choice_c=choices[2], choice_d=choices[3])
        map = {"A": "", "B": "", "C": "", "D": ""}

        if choices[0] == cA:
            map['A'] = 'A'
        elif choices[0] == cB:
            map['A'] = 'B'
        elif choices[0] == cC:
            map['A'] = 'C'
        elif choices[0] == cD:
            map['A'] = 'D'

        if choices[1] == cA:
            map['B'] = 'A'
        elif choices[1] == cB:
            map['B'] = 'B'
        elif choices[1] == cC:
            map['B'] = 'C'
        elif choices[1] == cD:
            map['B'] = 'D'

        if choices[2] == cA:
            map['C'] = 'A'
        elif choices[2] == cB:
            map['C'] = 'B'
        elif choices[2] == cC:
            map['C'] = 'C'
        elif choices[2] == cD:
            map['C'] = 'D'

        if choices[3] == cA:
            map['D'] = 'A'
        elif choices[3] == cB:
            map['D'] = 'B'
        elif choices[3] == cC:
            map['D'] = 'C'
        elif choices[3] == cD:
            map['D'] = 'D'
    else:
        cA = d['OPTION-A'].replace("A. ", "")
        cB = d['OPTION-B'].replace("B. ", "")
        cC = d['OPTION-C'].replace("C. ", "")
        cD = d['OPTION-D'].replace("D. ", "")
        choices = [cA, cB, cC, cD]
        random.shuffle(choices)
        prompt = UserEvaluatePrompt4Choices_en.format(story=d['STORY'], question=d['QUESTION'], choice_a=choices[0],
                                                      choice_b=choices[1], choice_c=choices[2], choice_d=choices[3])
        map = {"A": "", "B": "", "C": "", "D": ""}

        if choices[0] == cA:
            map['A'] = 'A'
        elif choices[0] == cB:
            map['A'] = 'B'
        elif choices[0] == cC:
            map['A'] = 'C'
        elif choices[0] == cD:
            map['A'] = 'D'

        if choices[1] == cA:
            map['B'] = 'A'
        elif choices[1] == cB:
            map['B'] = 'B'
        elif choices[1] == cC:
            map['B'] = 'C'
        elif choices[1] == cD:
            map['B'] = 'D'

        if choices[2] == cA:
            map['C'] = 'A'
        elif choices[2] == cB:
            map['C'] = 'B'
        elif choices[2] == cC:
            map['C'] = 'C'
        elif choices[2] == cD:
            map['C'] = 'D'

        if choices[3] == cA:
            map['D'] = 'A'
        elif choices[3] == cB:
            map['D'] = 'B'
        elif choices[3] == cC:
            map['D'] = 'C'
        elif choices[3] == cD:
            map['D'] = 'D'
    return map, prompt


def format_prompt_2(d, args):
    if args.language == 'zh':
        cA = d['选项A'].replace("A. ", "")
        cB = d['选项B'].replace("B. ", "")
        choices = [cA, cB]
        random.shuffle(choices)
        prompt = UserEvaluatePrompt2Choices_zh.format(story=d['故事'], question=d['问题'], choice_a=choices[0],
                                                      choice_b=choices[1])
        map = {"A": "", "B": "", "C": "", "D": ""}
        if choices[0] == cA:
            map['A'] = 'A'
        elif choices[0] == cB:
            map['A'] = 'B'

        if choices[1] == cA:
            map['B'] = 'A'
        elif choices[1] == cB:
            map['B'] = 'B'
    else:
        cA = d['OPTION-A'].replace("A. ", "")
        cB = d['OPTION-B'].replace("B. ", "")
        choices = [cA, cB]
        random.shuffle(choices)
        prompt = UserEvaluatePrompt2Choices_en.format(story=d['STORY'], question=d['QUESTION'], choice_a=choices[0],
                                                      choice_b=choices[1])
        map = {"A": "", "B": "", "C": "", "D": ""}
        if choices[0] == cA:
            map['A'] = 'A'
        elif choices[0] == cB:
            map['A'] = 'B'

        if choices[1] == cA:
            map['B'] = 'A'
        elif choices[1] == cB:
            map['B'] = 'B'

    return map, prompt

# 设定一个异步API客户端类，用来并发地调用Together API
class AsyncAPIClient:
    def __init__(self, api_key: str, max_concurrency: int = 5):
        self.client = Together(api_key=api_key)
        self.semaphore = asyncio.Semaphore(max_concurrency)

    # 异步请求API
    async def create_completion(self, prompt: str, model: str, temperature: float, top_p: float):
        async with self.semaphore:
            try:
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1024,
                    temperature=temperature,
                    top_p=top_p
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"API调用失败: {str(e)}")
                return None

# 处理每个任务的函数
async def process_task(file, task, try_times, data, output_dir, model_name, args, client):
    prompts = []
    meta = []

    # 生成所有需要的prompt
    for j in range(try_times):
        for i, d in enumerate(data):
            if isinstance(d['选项C'], float) and numpy.isnan(d['选项C']):
                maps, prompt = format_prompt_2(d, args)
            else:
                maps, prompt = format_prompt_4(d, args)

            # 处理系统提示，中文或英文
            if args.language == "zh":
                system_prompt = SystemEvaluatePrompt_zh_cot if args.cot else SystemEvaluatePrompt_zh
            else:
                system_prompt = SystemEvaluatePrompt_en_cot if args.cot else SystemEvaluatePrompt_en

            full_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>\n"
            prompts.append(full_prompt)
            meta.append({"idx": i, "number": j, "map": maps, "data": d})

    # 逐个调用 API
    results = []
    with tqdm(total=len(prompts), desc=f"Processing {task}") as pbar:
        for idx, prompt in enumerate(prompts):
            output = await client.create_completion(prompt, args.model_name, args.temperature, args.top_p)
            results.append((meta[idx], output))
            pbar.update(1)  # 更新进度

    # 保存结果到文件
    output_path = os.path.join(output_dir, f"{task}_{model_name.split('/')[-1]}_results.jsonl")
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding='utf-8') as f:
        for meta_item, output in results:
            if output:
                record = {
                    "idx": meta_item["idx"],
                    "number": meta_item["number"],
                    "answer": meta_item["data"].get("答案\nANSWER", ""),
                    "map": meta_item["map"],
                    "data": meta_item["data"],
                    "output": output
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

# 主函数，处理所有任务
async def main(args):
    # 获取API密钥并初始化客户端
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("需要设置环境变量TOGETHER_API_KEY")

    client = AsyncAPIClient(api_key=api_key)

    # 加载数据
    files = [f for f in os.listdir("./data") if f.endswith(".jsonl") and not f.startswith(".")]
    print(files)
    if args.task:
        files = [f for f in files if f.startswith(args.task)]

    # 处理每个文件
    for file in files:
        task_name = os.path.splitext(file)[0]
        output_path = os.path.join(args.output_dir, f"{task_name}_{args.model_name.split('/')[-1]}_results.jsonl")

        # 如果结果文件已经存在，则跳过
        if os.path.exists(output_path):
            print(f"[Skip] {output_path} already exists. Skipping this task.")
            continue

        # 读取数据
        with open(f"data/{file}", "r", encoding='utf-8') as f:
            data = [json.loads(line) for line in f.readlines()]

        # 逐个处理任务
        await process_task(file, task_name, args.try_times, data, args.output_dir, args.model_name, args, client)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, default="")
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-70B")
    parser.add_argument("--language", type=str, default="eg")
    parser.add_argument("--try_times", type=int, default=5)
    parser.add_argument("--cot", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="/home/ubuntu/ToMBench/DeepSeek-R1-Llama70B-eg")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)

    args = parser.parse_args()

    # Call the main function
    asyncio.run(main(args))