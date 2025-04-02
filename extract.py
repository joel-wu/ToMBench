import pandas as pd
import os

configurations = [
        {"model_path": "meta-llama/Llama-3.1-8B-Instruct", "model_name": "llama3.1-8B-Instruct"},
        {"model_path": "Qwen/Qwen2.5-7B-Instruct", "model_name": "qwen2.5-7B-Instruct"},
        {"model_path": "/home/ubuntu/comp-trust/compression/llm-awq/fake_cache/llama3.1-8B-instruct-w8-g128-awq", "model_name": "llama3.1-8B-Instruct-awq-w8"},
        {"model_path": "/home/ubuntu/comp-trust/compression/llm-awq/fake_cache/llama3.1-8B-instruct-w4-g128-awq", "model_name": "llama3.1-8B-Instruct-awq-w4"},
        {"model_path": "/home/ubuntu/comp-trust/compression/llm-awq/fake_cache/llama3.1-8B-instruct-w3-g128-awq", "model_name": "llama3.1-8B-Instruct-awq-w3"},
        {"model_path": "/home/ubuntu/comp-trust/compression/llm-awq/fake_cache/qwen2.5-7B-instruct-w3-g128-awq", "model_name": "qwen2.5-7B-Instruct-awq-w3"},
        {"model_path": "/home/ubuntu/comp-trust/compression/llm-awq/fake_cache/qwen2.5-7B-instruct-w4-g128-awq","model_name": "qwen2.5-7B-Instruct-awq-w4"},
        {"model_path": "/home/ubuntu/comp-trust/compression/llm-awq/fake_cache/qwen2.5-7B-instruct-w8-g128-awq","model_name": "qwen2.5-7B-Instruct-awq-w8"},
        {"model_path": "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16", "model_name": "llama3.1-8B-Instruct-gptq-w8"},
        {"model_path": "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4", "model_name": "llama3.1-8B-Instruct-gptq-w4"},
        {"model_path": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4", "model_name": "qwen2.5-7B-Instruct-gptq-w4"},
        {"model_path": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8", "model_name": "qwen2.5-7B-Instruct-gptq-w8"},
        {"model_path": "/home/ubuntu/comp-trust/compression/wanda/output/Llama-3.1-8B-Instruct-sparsegpt-model", "model_name": "llama3.1-8B-Instruct-sparsegpt"},
        {"model_path": "/home/ubuntu/comp-trust/compression/wanda/output/Llama-3.1-8B-Instruct-wanda-model", "model_name": "llama3.1-8B-Instruct-wanda"},
        {"model_path": "/home/ubuntu/comp-trust/compression/wanda/output/Qwen2.5-7B-Instruct-sparsegpt-model", "model_name": "qwen2.5-7B-Instruct-sparsegpt"},
        {"model_path": "/home/ubuntu/comp-trust/compression/wanda/output/Qwen2.5-7B-Instruct-wanda-model", "model_name": "qwen2.5-7B-Instruct-wanda"},]

for config in configurations:
    model_path = config["model_path"]
    model_name = config["model_name"]


    # 基础目录
    base_dir = f"/home/ubuntu/ToMBench/kosinski2/{model_name}/{{}}/"
    out = sorted(
        [
            name for name in os.listdir(base_dir.format(1))
            if os.path.isdir(os.path.join(base_dir.format(1), name)) and
               # 仅保留能被转换为浮点数的名称
               (lambda x: all(c.isdigit() or c in '.eE-+' for c in x))(name)
        ],
        key=lambda x: float(x)
    )

    # 初始化字典以存储每次运行的总和结果和百分比
    r = {key: {t: [] for t in ["unexpected", "transfer", "irony"]} for key in out}
    percentages = {key: {t: [] for t in ["unexpected", "transfer", "irony"]} for key in out}
    avg_percent = {key: [] for key in out}  # For storing the avg percentage of unexpected and transfer

    output_final = f"/home/ubuntu/ToMBench/kosinski2/results_{model_name}.csv"

    for i in range(1, 6):  # 遍历5次运行
        for m in out:
            print(m)
            for t in ["unexpected", "transfer", "irony"]:
                # 保存结果到指定目录
                output_path = f"{base_dir.format(i)}{m}/{t}/results.csv"

                # 检查文件是否存在
                if not os.path.isfile(output_path):
                    print(f"文件 {output_path} 不存在，跳过该文件。")
                    continue

                # 读取文件内容
                df = pd.read_csv(output_path, index_col=0)

                # 将所有数值求和
                k = df.sum().sum()

                # 计算百分比
                total_elements = df.size
                #print(total_elements)
                percentage = (k / total_elements) * 100

                # 打印调试信息
                print(f"文件 {output_path} 的总和: {k}, 百分比: {percentage}%")

                # 将结果存储在对应的键中
                r[m][t].append(k)  # 将总和结果添加
                percentages[m][t].append(percentage)  # 将百分比结果添加

            # 计算 unexpected 和 transfer 百分数的平均值
            avg_percentage = (percentages[m]["unexpected"][-1] + percentages[m]["transfer"][-1]) / 2
            avg_percent[m].append(avg_percentage)

    # 计算平均值、方差和百分比
    r_avg = {key: {t: sum(values)/len(values) for t, values in val.items()} for key, val in r.items()}
    r_var = {key: {t: pd.Series(values).var() for t, values in val.items()} for key, val in r.items()}
    percent_avg = {key: {t: sum(values)/len(values) for t, values in val.items()} for key, val in percentages.items()}
    avg_percent_final = {key: sum(values)/len(values) for key, values in avg_percent.items()}

    # 将总和平均值、方差、百分比转换为 DataFrame
    df_avg = pd.DataFrame(r_avg).T
    df_var = pd.DataFrame(r_var).T
    df_percent = pd.DataFrame(percent_avg).T
    df_avg_percent = pd.DataFrame(avg_percent_final, index=["Avg(Unexpected & Transfer)"]).T

    # 合并所有统计结果 (总和平均值，方差，百分比)
    df_final = pd.concat([df_avg, df_var, df_percent, df_avg_percent], keys=['Average', 'Variance', 'Percentage', 'Avg(Unexpected & Transfer)'])

    # 分离索引处理，数值部分保持浮点数
    numeric_part = [(float(m), t) if m not in ['Average', 'Variance', 'Percentage', 'Avg(Unexpected & Transfer)'] else (m, t) for m, t in df_final.index]

    # 将数值部分转换为浮点数并排序
    numeric_only_index = [(m, t) for m, t in numeric_part if isinstance(m, float)]
    non_numeric_only_index = [(m, t) for m, t in numeric_part if isinstance(m, str)]

    # 对数值部分排序
    numeric_only_index.sort(key=lambda x: (x[0], x[1]))

    # 重新创建MultiIndex
    sorted_index = numeric_only_index + non_numeric_only_index
    df_final.index = pd.MultiIndex.from_tuples(sorted_index)

    # 保存结果到 CSV 文件
    df_final.to_csv(output_final, index=True)
    print(f"结果已保存到: {output_final}")
