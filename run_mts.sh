#!/bin/bash

# モデル名のリスト
models=(
    "meta-llama/Llama-2-7b-chat-hf"
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.2"
)

# 各モデルを使用してvanilla.pyを実行
for model_name in "${models[@]}"; do
    echo "Running mts.py with model: $model_name"
    python mts.py --model_name "$model_name" --dataset "TOEFL11"
done