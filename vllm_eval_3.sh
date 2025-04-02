#CUDA_VISIBLE_DEVICES=0 python3 vllm_run_3.py \
#    --task "" \
#    --model_name "Qwen2.5-7B-Instruct" \
#    --model_name2 "Qwen/Qwen2.5-7B-Instruct" \
#    --language "zh" \
#    --cot True \
#    --try_times 5 \
#    --model_path "/home/ubuntu/comp-trust/compression/llm-awq/fake_cache/qwen2.5-7B-instruct-w3-g128-awq" \
#    --output_dir "/home/ubuntu/ToMBench/qwen2.5-7B-Instruct-w3-g128-awq-zh" \
#    --batch_size 64
#CUDA_VISIBLE_DEVICES=2 python3 vllm_run_3.py \
#    --task "" \
#    --model_name "llama3.1-8B-Instruct" \
#    --model_name2 "meta-llama/Llama-3.1-8B-Instruct" \
#    --language "zh" \
#    --cot True \
#    --try_times 5 \
#    --model_path "/home/ubuntu/comp-trust/compression/llm-awq/fake_cache/llama3.1-8B-instruct-w8-g128-awq" \
#    --output_dir "/home/ubuntu/ToMBench/llama3.1-8B-Instruct-w8-g128-awq-zh" \
#    --batch_size 64

#CUDA_VISIBLE_DEVICES=0 python3 vllm_run_3.py \
#    --task "" \
#    --model_name "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16" \
#    --model_name2 "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16" \
#    --language "zh" \
#    --cot True \
#    --try_times 5 \
#    --model_path "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16" \
#    --output_dir "/home/ubuntu/ToMBench/llama3.1-8B-Instruct-w8-gptq-zh" \
#    --batch_size 64

#CUDA_VISIBLE_DEVICES=3 python3 vllm_run_3.py \
#    --task "" \
#    --model_name "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4" \
#    --model_name2 "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4" \
#    --language "eg" \
#    --cot True \
#    --try_times 5 \
#    --model_path "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4" \
#    --output_dir "/home/ubuntu/ToMBench/qwen2.5-32B-Instruct-w4-gptq-eg" \
#    --batch_size 64

CUDA_VISIBLE_DEVICES=1 python3 vllm_run_3.py \
    --task "" \
    --model_name "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
    --model_name2 "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
    --language "eg" \
    --cot True \
    --try_times 5 \
    --model_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
    --output_dir "/home/ubuntu/ToMBench/DeepSeek-R1-Distill-Qwen-1.5B-eg" \
    --batch_size 64




#CUDA_VISIBLE_DEVICES=1 python3 vllm_run_3.py \
#    --task "" \
#    --model_name "Qwen2.5-72B-Instruct-awq" \
#    --model_name2 "Qwen/Qwen2.5-72B-Instruct-AWQ" \
#    --language "eg" \
#    --cot True \
#    --try_times 5 \
#    --model_path "Qwen/Qwen2.5-72B-Instruct-AWQ" \
#    --output_dir "/home/ubuntu/ToMBench/qwen2.5-7B-Instruct-awq-eg" \
#    --batch_size 64
    #CUDA_VISIBLE_DEVICES=2 python3 gptq.py --pretrained_model_dir Qwen/Qwen2.5-7B-Instruct --quantized_model_dir /home/ubuntu/comp-trust/compression/gptq/fake_cache/qwen2.5-7B-instruct-w3-gptq --bits 3 --save_and_reload --desc_act --seed 0 --num_samples 128 --calibration-template llama-2

#CUDA_VISIBLE_DEVICES=3 python3 gptq.py --pretrained_model_dir meta-llama/Llama-3.1-8B-Instruct --quantized_model_dir /home/ubuntu/comp-trust/compression/gptq/fake_cache/llama3.1-8B-instruct-w4-gptq --bits 4 --save_and_reload --desc_act --seed 0 --num_samples 128 --calibration-template llama-2

