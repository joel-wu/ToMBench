import json
import random
import numpy as np
import argparse
import os
from tqdm import tqdm
from vllm import LLM, SamplingParams

from prompts import *


def format_prompt(d, args):
    """根据语言和选项数量构造 prompt 和选项映射"""
    is_2choice = isinstance(d.get("选项C", None), float) and np.isnan(d["选项C"])
    if args.language == "zh":
        if is_2choice:
            choices = [d['选项A'].replace("A. ", ""), d['选项B'].replace("B. ", "")]
            random.shuffle(choices)
            prompt = UserEvaluatePrompt2Choices_zh.format(
                story=d['故事'], question=d['问题'],
                choice_a=choices[0], choice_b=choices[1]
            )
        else:
            choices = [d['选项A'].replace("A. ", ""), d['选项B'].replace("B. ", ""),
                       d['选项C'].replace("C. ", ""), d['选项D'].replace("D. ", "")]
            random.shuffle(choices)
            prompt = UserEvaluatePrompt4Choices_zh.format(
                story=d['故事'], question=d['问题'],
                choice_a=choices[0], choice_b=choices[1],
                choice_c=choices[2], choice_d=choices[3]
            )
    else:
        if is_2choice:
            choices = [d['OPTION-A'].replace("A. ", ""), d['OPTION-B'].replace("B. ", "")]
            random.shuffle(choices)
            prompt = UserEvaluatePrompt2Choices_en.format(
                story=d['STORY'], question=d['QUESTION'],
                choice_a=choices[0], choice_b=choices[1]
            )
        else:
            choices = [d['OPTION-A'].replace("A. ", ""), d['OPTION-B'].replace("B. ", ""),
                       d['OPTION-C'].replace("C. ", ""), d['OPTION-D'].replace("D. ", "")]
            random.shuffle(choices)
            prompt = UserEvaluatePrompt4Choices_en.format(
                story=d['STORY'], question=d['QUESTION'],
                choice_a=choices[0], choice_b=choices[1],
                choice_c=choices[2], choice_d=choices[3]
            )

    # 构造 answer 映射（例如 map["A"] = "C"）
    label_map = {}
    for idx, choice in zip("ABCD", choices):
        if args.language == "zh":
            for label in "ABCD":
                if choice == d.get(f"选项{label}"):
                    label_map[idx] = label
        else:
            for label in "ABCD":
                if choice == d.get(f"OPTION-{label}"):
                    label_map[idx] = label
    return prompt, label_map


def build_all_prompts(data, args):
    all_prompts = []
    meta_info = []
    for j in range(args.try_times):
        for i, d in enumerate(data):
            prompt_text, label_map = format_prompt(d, args)
            if args.language == "zh":
                system_prompt = SystemEvaluatePrompt_zh_cot if args.cot else SystemEvaluatePrompt_zh
            else:
                system_prompt = SystemEvaluatePrompt_en_cot if args.cot else SystemEvaluatePrompt_en
            full_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{prompt_text}\n<|assistant|>\n"
            all_prompts.append(full_prompt)
            meta_info.append({
                "idx": i, "number": j, "map": label_map, "data": d
            })
    return all_prompts, meta_info


def run(args):
    # 初始化 vLLM 引擎
    llm = LLM(
        model=args.model_path,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )

    # 加载任务数据
    files = [args.task] if args.task else os.listdir("data")
    for file in files:
        task = file.split(".")[0]
        with open(os.path.join("data", file), "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f.readlines()]

        print(f"Processing {task} with {len(data)} examples * {args.try_times} = {len(data) * args.try_times} prompts")

        all_prompts, meta_info = build_all_prompts(data, args)

        # 批量推理
        batch_size = args.batch_size
        output_path = os.path.join(args.output_dir, f"{task}_{args.model_name}_results.jsonl")
        os.makedirs(args.output_dir, exist_ok=True)

        with open(output_path, "a+", encoding="utf-8") as fout:
            for i in tqdm(range(0, len(all_prompts), batch_size), desc="Running vLLM"):
                batch_prompts = all_prompts[i: i + batch_size]
                batch_meta = meta_info[i: i + batch_size]
                outputs = llm.generate(batch_prompts, sampling_params)
                for meta, out in zip(batch_meta, outputs):
                    result = {
                        "idx": meta["idx"],
                        "number": meta["number"],
                        "answer": meta["data"]["答案\nANSWER"],
                        "map": meta["map"],
                        "data": meta["data"],
                        "output": out.outputs[0].text.strip()
                    }
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="")
    parser.add_argument("--model_name", type=str, default="model")
    parser.add_argument("--language", type=str, default="zh")
    parser.add_argument("--try_times", type=int, default=5)
    parser.add_argument("--cot", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    random.seed(args.seed)
    run(args)
