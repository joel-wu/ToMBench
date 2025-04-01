python3 -m vllm.entrypoints.openai.api_server \
    --model  /home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B/snapshots/97e1e76335b7017d8f67c08a19d103c0504298c9\
    --dtype auto \
    --api-key 123456\
    --trust-remote-code \
    --port 5050 \
    --max-model-len 8192