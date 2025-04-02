import os
import re
import gc
import math
import logging
import torch
import pandas as pd
import multiprocessing
import numpy as np
import math
from datasets import load_dataset
from tqdm import tqdm
import random

# vLLM & Transformers
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer

# 从旧脚本可见，这里包含所有任务定义 tsks1(S1), tsks2(S2), tsks3(S3)
from tasks import tsks1, tsks2, tsks3



###############################################################################
# 1. 构造 prompt 的辅助函数
###############################################################################

def build_s1_txt(tsk, typ="txt", reverse=False):
    """
    针对旧脚本里 S1 相关逻辑，把 txt 等拼接方式封装在这里。
    返回: final_txt, q1, q2, expected_tokens (list of 2), result_key (F/T)
    """
    # 参照旧代码中的 if-else
    if typ in ["txt_open"]:
        if tsk["NUMBER"] == 1:
            txt = tsk[
                      "txt"] + " XNAM connects to the CX and explores its contents. XNAM reads the label on the CX."
        elif tsk["NUMBER"] == 17:
            txt = tsk[
                      "txt"] + " XNAM puts one in a CD player and listens to the songs. XNAM can can clearly hear that it is full of S2 music."
        else:
            txt = tsk["txt"] + " XNAM opens the CX and looks inside. XNAM reads the label."
    else:
        if typ in ["txt_correctlabel"]:
            if tsk["NUMBER"] == 1:
                txt = tsk[
                          typ] + " XNAM does not connect to the CX and does not explore its contents. XNAM reads the label on the CX. "
            elif tsk["NUMBER"] == 17:
                txt = tsk[
                          typ] + " XNAM does not open the box and does not listen to any of the CDs. XNAM reads the label."
            else:
                txt = tsk[typ] + " XNAM does not open the CX and does not look inside. XNAM reads the label."
        if typ in ["txt_informed", "txt"]:
            if tsk["NUMBER"] == 1:
                txt = tsk[
                          "txt"] + " XNAM does not connect to the CX and does not explore its contents. XNAM reads the label on the CX. "
            elif tsk["NUMBER"] == 17:
                txt = tsk[
                          "txt"] + " XNAM does not open the box and does not listen to any of the CDs. XNAM reads the label."
            else:
                txt = tsk["txt"] + " XNAM does not open the CX and does not look inside. XNAM reads the label."
        if typ in ["txt_informed"]:
            txt = txt + " A cousin calls XNAM and tells OBJ_PRO that the CX has S1 in it, and that XPRO should ignore the label that says \"S2\". XNAM believes OBJ_PRO cousin."

    txt = txt.replace("S1", tsk["o1"]).replace("S2", tsk["o2"]).replace("CX", tsk["c"]).replace("XNAM", tsk[
        "xnam"]).replace("XPRO", tsk["xpro"]).replace("OBJ_PRO", tsk["obj_pro"]).replace("POS_PRO", tsk[
        "pos_pro"]).replace(". h", ". H").replace(". s", ". S")

    q1 = tsk["xpro"].capitalize() + " opens the " + tsk["c"] + " and looks inside. " + tsk[
        "xpro"].capitalize() + " can clearly see that it is full of"

    q2 = tsk["xnam"] + " calls a friend to tell them that " + tsk["xpro"] + " has just found a " + tsk[
        "c"] + " full of"

    if tsk["NUMBER"] == 17:
        q1 = tsk["xpro"].capitalize() + " puts one in a CD player and listens to the songs. " + tsk[
            "xpro"].capitalize() + " can clearly hear that it is full of"

    if tsk["NUMBER"] == 14:
        q2 = q2.replace("found", "received")
    if tsk["NUMBER"] == 12:
        q2 = q2.replace("found", "received")

    # 如果 reverse, 交换 S1/S2
    # 这里旧代码做法是 txt = txt.replace(o1,####).replace(o2,o1).replace(####,o2)
    if reverse:
        txt = txt.replace(tsk["o1"], "####").replace(tsk["o2"], tsk["o1"]).replace("####", tsk["o2"])

    # 计算 expected
    #   先默认 expected = [o1, o2]
    #   然后看 typ 是否在 ["txt_transparent","txt_correctlabel","txt_informed","txt_open","txt_present"] => [o1,o1]
    #   typ in ["txt_notransfer"] => [o2,o2] ...
    #   reverse 再交换
    expected = [tsk["o1"], tsk["o2"]]
    if typ in ["txt_transparent", "txt_correctlabel", "txt_informed", "txt_open", "txt_present"]:
        expected = [tsk["o1"], tsk["o1"]]
    elif typ in ["txt_notransfer"]:
        expected = [tsk["o2"], tsk["o2"]]
    elif typ in ["txt_stayed"]:
        expected = [tsk["o1"], tsk["o1"]]

    if reverse:
        # swap
        expected = [w.replace(tsk["o1"], "####").replace(tsk["o2"], tsk["o1"]).replace("####", tsk["o2"]) for w in
                    expected]

    # 最后只取第一个词
    expected_tokens = [w.split()[0] for w in expected]

    # 还需要区分 (F)/(T) 对应的 result_key
    #   在旧脚本里, (F) == reverse=False, (T) == reverse=True.
    #   例如 "false belief (F)", "false belief (T)"
    #   但因为同一个 typ 也会映射到不一样的 result_key
    #   旧脚本是:
    #     typ="txt" -> "false belief (F/T)"
    #     typ="txt_correctlabel" -> "correct label (F/T)"
    #     typ="txt_informed" -> "informed protagonist (F/T)"
    #     typ="txt_open" -> "open container (F/T)"
    #   但是 "txt_notransfer" 在S1好像没用？这是S2?
    #
    # 简单做一个映射:
    map_typ_to_key = {
        "txt": "false belief",
        "txt_correctlabel": "correct label",
        "txt_informed": "informed protagonist",
        "txt_open": "open container",
        # 还有一些并不在S1中用
    }
    # 如果 typ 不在 map_typ_to_key，就用 "false belief"
    base_key = map_typ_to_key.get(typ, "false belief")
    suffix = "(T)" if reverse else "(F)"
    result_key = f"{base_key} {suffix}"

    return txt, q1, q2, expected_tokens, result_key


def build_s2_txt(tsk, typ="txt", reverse=False):
    """
    参照旧代码 S2 的逻辑。
    - txt = tsk[typ]
    - if typ=="txt_informed": txt = tsk["txt"] + " " + tsk["txt_informed"]
    - q1 = tsk["q1"], q2 = tsk["q2"]
    - reverse => 交换 o1,o2
    - expected 逻辑同旧代码
    """
    # 基础 txt
    if typ == "txt_informed":
        txt = tsk["txt"] + " " + tsk["txt_informed"]
    else:
        txt = tsk.get(typ, tsk["txt"])  # 若 typ 不存在就用 tsk["txt"]

    # 构造 q1,q2
    q1 = tsk["q1"]
    q2 = tsk["q2"]

    if reverse:
        # 交换 S1/S2
        txt = txt.replace(tsk["o1"], "####").replace(tsk["o2"], tsk["o1"]).replace("####", tsk["o2"])

    # expected 参照S1的逻辑:
    expected = [tsk["o1"], tsk["o2"]]
    if typ in ["txt_notransfer"]:
        expected = [tsk["o2"], tsk["o2"]]
    elif typ in ["txt_transparent", "txt_correctlabel", "txt_informed", "txt_open", "txt_present"]:
        expected = [tsk["o1"], tsk["o1"]]
    elif typ in ["txt_stayed"]:
        expected = [tsk["o1"], tsk["o1"]]

    if reverse:
        expected = [w.replace(tsk["o1"], "####").replace(tsk["o2"], tsk["o1"]).replace("####", tsk["o2"]) for w in
                    expected]

    expected_tokens = [w.split()[0] for w in expected]

    # result_key 映射
    map_typ_to_key = {
        "txt": "false belief",
        "txt_notransfer": "no transfer",
        "txt_informed": "informed protagonist",
        "txt_present": "present protagonist",
        # ...
    }
    base_key = map_typ_to_key.get(typ, "false belief")
    suffix = "(T)" if reverse else "(F)"
    result_key = f"{base_key} {suffix}"

    return txt, q1, q2, expected_tokens, result_key


def build_s3_txt(tsk, typ="txt", reverse=False):
    """
    旧代码 S3:
      - txt = tsk[typ]
      - if typ=="txt_reverse": txt=tsk["txt_reverse"]
      - 生成 prompt: "Read the following story: {txt} Correct Answer:"
      - expected = ["No"] / ["Yes"] if reverse
    """
    if typ == "txt_reverse":
        txt = tsk["txt_reverse"]
    else:
        txt = tsk[typ]

    if reverse:
        # S3 里 reverse => expected=Yes
        expected = ["Yes"]
    else:
        expected = ["No"]

    expected_tokens = [w.split()[0] for w in expected]

    # result_key: "Irony" / "Irony reversed"
    if reverse:
        result_key = "Irony reversed"
    else:
        result_key = "Irony"

    # prompt 只有一个
    # 旧脚本: prob_chat("Read the following story: " + txt + " Correct Answer:")
    # q1 = None (we treat it as 1 prompt)
    return txt, expected_tokens, result_key


###############################################################################
# 2. 建立批量 Prompt 并行生成
###############################################################################

def make_prompts_s1(tsks):
    """
    针对 "unexpected" (S1), 构造所有 Prompt:
      typ in [txt, txt_correctlabel, txt_informed, txt_open]
      reverse in [False, True]
      对 20 个 tsks1
      每个任务 => 两个子Prompt(q1,q2)
    返回:
      all_prompts: List[str]
      meta_info: 与 all_prompts 对应的 [ (task_idx, result_key, expected_token, ...), ... ]
    """
    types = ["txt", "txt_correctlabel", "txt_informed", "txt_open"]
    all_prompts = []
    meta_info = []

    for i, tsk in enumerate(tsks):
        for typ in types:
            for rev in [False, True]:
                # 构造文本
                final_txt, q1, q2, exps, rkey = build_s1_txt(tsk, typ, reverse=rev)
                # print(final_txt)
                # print(exps)
                # print(rkey)
                # prompt1
                p1 = f"Complete the following story: {final_txt} {q1}"
                # prompt2
                p2 = f"Complete the following story: {final_txt} {q2}"
                # print(p1)
                # print(p2)
                # exps = [exp_q1, exp_q2]
                # rkey 同一个
                # 记录
                all_prompts.append(p1)
                meta_info.append({"task_idx": i, "result_key": rkey, "expected": exps[0]})
                all_prompts.append(p2)
                meta_info.append({"task_idx": i, "result_key": rkey, "expected": exps[1]})

    # print(all_prompts)
    # print(meta_info)

    return all_prompts, meta_info


def make_prompts_s2(tsks):
    """
    针对 "transfer" (S2)
      typ in ["txt", "txt_notransfer", "txt_informed", "txt_present"]
      reverse in [False, True]
      对 20 个 tsks2
      每个任务 => q1,q2
    """
    types = ["txt", "txt_notransfer", "txt_informed", "txt_present"]
    all_prompts = []
    meta_info = []

    for i, tsk in enumerate(tsks):
        for typ in types:
            for rev in [False, True]:
                final_txt, q1, q2, exps, rkey = build_s2_txt(tsk, typ, reverse=rev)
                # print(final_txt)
                # print(q1)
                # print(q2)
                # print(exps)
                # print(rkey)
                p1 = f"Complete the following story: {final_txt} {q1}"
                p2 = f"Complete the following story: {final_txt} {q2}"
                # print(p1)
                # print(p2)
                # exps[0], exps[1]
                all_prompts.append(p1)
                meta_info.append({"task_idx": i, "result_key": rkey, "expected": exps[0]})
                all_prompts.append(p2)
                meta_info.append({"task_idx": i, "result_key": rkey, "expected": exps[1]})

    return all_prompts, meta_info


def make_prompts_s3(tsks):
    """
    针对 "irony" (S3)
      typ in ["txt", "txt_reverse"]
      reverse in [False, True]
      对 12 个 tsks3
      每个任务 => 1 prompt
    """
    # 旧脚本里: results["Irony"] / results["Irony reversed"]
    # 其实只是 typ="txt" + reverse=False => "Irony"
    #         typ="txt_reverse" + reverse=True => "Irony reversed"
    # 但旧脚本: for i in range(12):
    #   results["Irony"].append( process_tsk(tsks3[i],..., typ="txt") )
    #   results["Irony reversed"].append( process_tsk(tsks3[i],..., typ="txt_reverse") )
    #   => 2 calls per i
    #
    # 所以:  typ in ["txt","txt_reverse"], rev in [False,True] 其实4种？
    # 但旧代码 *实际* 用法:
    #   "Irony"          => typ="txt", reverse=False
    #   "Irony reversed" => typ="txt_reverse", reverse=True
    # 我们也只需这2种:
    all_prompts = []
    meta_info = []
    # tasks3 length=12
    for i, tsk in enumerate(tsks):
        # 1) Irony
        typ = "txt"
        rev = False
        txt, exps, rkey = build_s3_txt(tsk, typ=typ, reverse=rev)
        # print(txt)
        # print(exps)
        # print(rkey)
        p = f"Read the following story: {txt} Correct Answer:"
        # print(p)
        all_prompts.append(p)
        meta_info.append({"task_idx": i, "result_key": rkey, "expected": exps[0]})

        # 2) Irony reversed
        typ = "txt_reverse"
        rev = True
        txt2, exps2, rkey2 = build_s3_txt(tsk, typ=typ, reverse=rev)
        # print(txt2)
        # print(exps2)
        # print(rkey2)
        p2 = f"Read the following story: {txt2} Correct Answer:"
        # print(p2)
        all_prompts.append(p2)
        meta_info.append({"task_idx": i, "result_key": rkey2, "expected": exps2[0]})

    return all_prompts, meta_info


def generate_in_batches(prompts, llm, batch_size=32):
    """
    通用的vLLM批量推理:
      - prompts: List[str]
      - 返回: List[str]，与prompts等长
    """
    results = []
    sampling_params = SamplingParams(
        max_tokens=5,  # 旧脚本的 generate(...,max_new_tokens=5)
        temperature=1.0,
        top_p=1.0
    )
    n = len(prompts)
    n_batches = math.ceil(n / batch_size)
    start = 0
    for b in range(n_batches):
        end = min(start + batch_size, n)
        batch_prompts = prompts[start:end]
        batch_outputs = llm.generate(batch_prompts, sampling_params)
        # each out => out.outputs[0].text
        for out in batch_outputs:
            txt = out.outputs[0].text
            results.append(txt)
            # print(txt)
        start = end
    return results


###############################################################################
# 3. evaluate_* 函数：构造 prompt -> 生成 -> 对比 -> 写入 results
###############################################################################

def evaluate_unexpected(tsks, llm, batch_size=32):
    """
    S1: "unexpected"
    返回: {
      "false belief (F)": [... 20 个值 ...],
      "false belief (T)": [...],
      "correct label (F)": [...],
      "correct label (T)": [...],
      "informed protagonist (F)": [...],
      "informed protagonist (T)": [...],
      "open container (F)": [...],
      "open container (T)": [...],
    }
    每项是 20 长度。每个任务2个子问题( prompt1, prompt2 ) => 取最小值(只要有一次出错就算0)
    旧脚本是一次 process_tsk 就返回 "1"/"0". 这里要自己聚合。
    """
    # 构造
    prompts, meta_info = make_prompts_s1(tsks)
    # 批量推理
    outs = generate_in_batches(prompts, llm, batch_size=batch_size)
    # results dict
    results = {
        "false belief (F)": [1] * 20,
        "false belief (T)": [1] * 20,
        "correct label (F)": [1] * 20,
        "correct label (T)": [1] * 20,
        "informed protagonist (F)": [1] * 20,
        "informed protagonist (T)": [1] * 20,
        "open container (F)": [1] * 20,
        "open container (T)": [1] * 20,
    }

    # meta_info[i] => {task_idx, result_key, expected}
    for i, text in enumerate(outs):
        info = meta_info[i]
        t_idx = info["task_idx"]
        rkey = info["result_key"]
        expected = info["expected"]  # 只取第一个词
        # 取输出第一个词
        tokens = text.strip().split()
        if tokens:
            first_word = re.sub(r'[.,\'";]', '', tokens[0])
        else:
            first_word = ''

        correct = (first_word == expected)
        if not correct:
            # 原逻辑: 只要一个 prompt错 => 这整条task记0
            # 但现写法: results[rkey][t_idx] = min(现有值,0)
            results[rkey][t_idx] = 0

    return results


def evaluate_transfer(tsks, llm, batch_size=32):
    """
    S2: "transfer"
    返回: {
       "false belief (F)": [... 20 ...],
       "false belief (T)": [...],
       "no transfer (F)": [...],
       "no transfer (T)": [...],
       "informed protagonist (F)": [...],
       "informed protagonist (T)": [...],
       "present protagonist (F)": [...],
       "present protagonist (T)": [...],
    }
    """
    prompts, meta_info = make_prompts_s2(tsks)
    outs = generate_in_batches(prompts, llm, batch_size=batch_size)

    results = {
        "false belief (F)": [1] * 20,
        "false belief (T)": [1] * 20,
        "no transfer (F)": [1] * 20,
        "no transfer (T)": [1] * 20,
        "informed protagonist (F)": [1] * 20,
        "informed protagonist (T)": [1] * 20,
        "present protagonist (F)": [1] * 20,
        "present protagonist (T)": [1] * 20,
    }

    for i, text in enumerate(outs):
        info = meta_info[i]
        t_idx = info["task_idx"]
        rkey = info["result_key"]
        expected = info["expected"]

        tokens = text.strip().split()
        first_word = re.sub(r'[.,\'";]', '', tokens[0]) if tokens else ''
        if first_word != expected:
            results[rkey][t_idx] = 0

    return results


def evaluate_irony(tsks, llm, batch_size=32):
    """
    S3: "irony"
    返回: {
       "Irony": [... 12 ...],
       "Irony reversed": [... 12 ...],
    }
    每个 task 做2次 (Irony, Irony reversed) => 2 prompts
    """
    prompts, meta_info = make_prompts_s3(tsks)
    outs = generate_in_batches(prompts, llm, batch_size=batch_size)

    results = {
        "Irony": [1] * 12,
        "Irony reversed": [1] * 12,
    }

    for i, text in enumerate(outs):
        info = meta_info[i]
        t_idx = info["task_idx"]
        rkey = info["result_key"]
        expected = info["expected"]

        tokens = text.strip().split()
        first_word = re.sub(r'[.,\'";]', '', tokens[0]) if tokens else ''
        if first_word != expected:
            results[rkey][t_idx] = 0

    return results

def process_model(model_name, model_path):
    batch_size = 32


    llm = LLM(
        model=model_path,
        max_model_len=8192,
        # quantization=None,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    for rep in range(1, 6):
        for t in ["unexpected", "transfer", "irony"]:
            output_path = f"/home/ubuntu/ToMBench/kosinski2/{model_name}/{rep}/0.0/{t}/results.csv"
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            if os.path.isfile(output_path):
                print("exist!")
                continue

            if t == "unexpected":
                res = evaluate_unexpected(tsks1, llm, batch_size=batch_size)
            elif t == "transfer":
                res = evaluate_transfer(tsks2, llm, batch_size=batch_size)
            else:
                res = evaluate_irony(tsks3, llm, batch_size=batch_size)

            # 存CSV
            df = pd.DataFrame(res)
            # 统计总和
            k = sum(sum(vals) for vals in res.values())
            print(k)
            print("------")

            df.to_csv(output_path, index=True)
            print(f"结果已保存到: {output_path}")


###############################################################################
# 5. main()
###############################################################################

def main():
    configurations = [
        # {"model_path": "meta-llama/Llama-3.1-8B-Instruct", "model_name": "llama3.1-8B-Instruct"},
        # {"model_path": "Qwen/Qwen2.5-7B-Instruct", "model_name": "qwen2.5-7B-Instruct"},
        # {"model_path": "/home/ubuntu/comp-trust/compression/llm-awq/fake_cache/llama3.1-8B-instruct-w8-g128-awq", "model_name": "llama3.1-8B-Instruct-awq-w8"},
        # {"model_path": "/home/ubuntu/comp-trust/compression/llm-awq/fake_cache/llama3.1-8B-instruct-w4-g128-awq", "model_name": "llama3.1-8B-Instruct-awq-w4"},
        # {"model_path": "/home/ubuntu/comp-trust/compression/llm-awq/fake_cache/llama3.1-8B-instruct-w3-g128-awq", "model_name": "llama3.1-8B-Instruct-awq-w3"},
        # {"model_path": "/home/ubuntu/comp-trust/compression/llm-awq/fake_cache/qwen2.5-7B-instruct-w3-g128-awq", "model_name": "qwen2.5-7B-Instruct-awq-w3"},
        # {"model_path": "/home/ubuntu/comp-trust/compression/llm-awq/fake_cache/qwen2.5-7B-instruct-w4-g128-awq","model_name": "qwen2.5-7B-Instruct-awq-w4"},
        # {"model_path": "/home/ubuntu/comp-trust/compression/llm-awq/fake_cache/qwen2.5-7B-instruct-w8-g128-awq","model_name": "qwen2.5-7B-Instruct-awq-w8"},
        # {"model_path": "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16", "model_name": "llama3.1-8B-Instruct-gptq-w8"},
        # {"model_path": "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4", "model_name": "llama3.1-8B-Instruct-gptq-w4"},
        # {"model_path": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4", "model_name": "qwen2.5-7B-Instruct-gptq-w4"},
        # {"model_path": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8", "model_name": "qwen2.5-7B-Instruct-gptq-w8"},
        # {"model_path": "/home/ubuntu/comp-trust/compression/wanda/output/Llama-3.1-8B-Instruct-sparsegpt-model", "model_name": "llama3.1-8B-Instruct-sparsegpt"},
        # {"model_path": "/home/ubuntu/comp-trust/compression/wanda/output/Llama-3.1-8B-Instruct-wanda-model", "model_name": "llama3.1-8B-Instruct-wanda"},
        # {"model_path": "/home/ubuntu/comp-trust/compression/wanda/output/Qwen2.5-7B-Instruct-sparsegpt-model", "model_name": "qwen2.5-7B-Instruct-sparsegpt"},
        {"model_path": "/home/ubuntu/comp-trust/compression/wanda/output/Qwen2.5-7B-Instruct-wanda-model", "model_name": "qwen2.5-7B-Instruct-wanda"},
    ]


    for config in configurations:
        model_path = config["model_path"]
        model_name = config["model_name"]
        p2 = multiprocessing.Process(target=process_model, args=(model_name, model_path))
        p2.start()
        p2.join()


if __name__ == "__main__":
    main()
