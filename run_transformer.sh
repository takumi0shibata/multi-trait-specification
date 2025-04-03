#!/bin/bash

# モデル名のリスト
models=(
    # "bert-base-uncased"
    "FacebookAI/roberta-base"
)

for model_name in "${models[@]}"; do
    for prompt in {1..8}; do
        echo "Running transformer.py with model: $model_name, prompt: $prompt"
        python prompt-specific.py --model "$model_name" --prompt "$prompt"
    done
done