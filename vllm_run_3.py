import json
import math
import random

import numpy
import argparse
from prompts import *
from tqdm import tqdm
import os
import torch
from vllm import LLM, SamplingParams


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--language", type=str, default="zh")
    parser.add_argument("--try_times", type=int, default=5)
    parser.add_argument("--cot", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=600)
    parser.add_argument("--model_name2", type=str, default="")
    args = parser.parse_args()

    random.seed(args.seed)

    from transformers import AutoTokenizer
    import shutil

    # # 检查 tokenizer 文件是否存在
    # tokenizer_path = os.path.join(args.model_path, "tokenizer_config.json")
    # print(tokenizer_path)
    # if not os.path.exists(tokenizer_path):
    #     print("[Info] tokenizer not found in fake model dir, fallback to HF tokenizer")
    #
    #     # 用 model_name 作为 HF 模型名加载
    #     hf_tokenizer = AutoTokenizer.from_pretrained(args.model_name2, trust_remote_code=True)
    #
    #     # 保存 tokenizer 到 fake 模型目录
    #     hf_tokenizer.save_pretrained(args.model_path)
    #     print("[Info] tokenizer saved to:", args.model_path)


    llm = LLM(
        model=args.model_path,
        max_model_len=args.max_model_len,
        trust_remote_code=True
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )

    model_name = args.model_name.split("/")[-1]
    files = [f for f in os.listdir("./data") if f.endswith(".jsonl") and not f.startswith(".")]
    print(files)
    if args.task != "":
        files = [args.task]

    for file in files:

        task = file.split(".")[0]
        output_path = os.path.join(args.output_dir, f"{task}_{model_name}_results.jsonl")

        # 如果结果文件已经存在，则跳过这个 task
        if os.path.exists(output_path):
            print(f"[Skip] {output_path} already exists. Skipping this task.")
            continue

        with open(f"data/{file}", "r", encoding='utf-8') as f:
            data = [json.loads(line) for line in f.readlines()]

        prompts = []
        meta = []

        for j in range(args.try_times):
            for i, d in enumerate(data):
                if isinstance(d['选项C'], float) and numpy.isnan(d['选项C']):
                    maps, prompt = format_prompt_2(d, args)
                else:
                    maps, prompt = format_prompt_4(d, args)

                if args.language == "zh":
                    system_prompt = SystemEvaluatePrompt_zh_cot if args.cot else SystemEvaluatePrompt_zh
                else:
                    system_prompt = SystemEvaluatePrompt_en_cot if args.cot else SystemEvaluatePrompt_en

                full_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>\n"
                prompts.append(full_prompt)
                meta.append({"idx": i, "number": j, "map": maps, "data": d})

        outputs = llm.generate(prompts, sampling_params)

        output_path = os.path.join(args.output_dir, f"{task}_{model_name}_results.jsonl")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(output_path, "a+", encoding='utf-8') as f:
            for meta_i, output in zip(meta, outputs):
                result = {
                    "idx": meta_i["idx"],
                    "number": meta_i["number"],
                    "answer": meta_i["data"]["答案\nANSWER"],
                    "map": meta_i["map"],
                    "data": meta_i["data"],
                    "output": output.outputs[0].text.strip()
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")