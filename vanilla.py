import os
import re
import torch
import transformers
from tqdm import tqdm
import json
import numpy as np
import polars as pl
from sklearn.metrics import cohen_kappa_score
from typing import Optional, Literal
from utils import load_asap_dataset, load_toefl_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

def get_score_range(dataset_name, prompt_id):
    """ASAPデータセットのスコア範囲を取得."""
    score_ranges = {
        "ASAP": {
            1: (2, 12),
            2: (1, 6),
            3: (0, 3),
            4: (0, 3),
            5: (0, 4),
            6: (0, 4),
            7: (0, 30),
            8: (0, 60),
        }
    }
    return score_ranges[dataset_name][prompt_id]


def vanilla_scoring(essay, prompt, model, tokenizer, prompt_id):
    """Vanilla Promptingによるエッセイ採点."""

    # System Promptの設定
    min_score, max_score = get_score_range('ASAP', prompt_id)
    system_prompt = f"""As an English teacher, your primary responsibility is to evaluate the writing quality of essays written by middle school students. During the assessment process, you will be provided with a prompt and an essay. First, you should provide comprehensive and conrete feedback that is closely linked to the content of the essay. It is essential to avoid offering generic remarks that could be applied to any piece of writing. To create a compelling evaluation for both the student and fellow experts, you should reference specific content of the essay to substantiate your assessment. Next, your evaluation should culminate in assigning an overall score to the student’s essay, measured on a scale from {min_score} to {max_score}, where higher score should reflect a higher level of writing quality. It’s crucial to tailor your evaluation criteria to be well-suited for middle school level writing, taking into account the developmental stage and capabilities of these students."""
    # User Promptの設定
    user_prompt = f"""[Prompt]
    {prompt}
    (end of [Prompt])
    [Note]
    I have made an effort to remove personally identifying information from the essays using the Named Entity Recognizer (NER). The relevant entities are identified in the text and then replaced with a string such as "{{PERSON}}", "{{ORGANIZATION}}", "{{LOCATION}}", "{{DATE}}", "{{TIME}}", "{{MONEY}}", "{{PERCENT}}”, “{{CAPS}}” (any capitalized word) and “{{NUM}}” (any digits). Please do not penalize the essay because of the anonymizations.
    (end of [Note])
    [Essay]
    {essay}
    (end of [Essay])
    Strictly follow the format below to give your answer. Other formats are NOT allowed. Evaluation: <evaluation>insert evaluation here</evaluation>
    Score: <score>insert score ({min_score} to {max_score}) here</score>
    """

    # メッセージの準備
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # チャットテンプレートの適用
    chat = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(chat, return_tensors="pt").to(model.device)

    # 文章生成
    gen_config = GenerationConfig(
        max_new_tokens=512,
        temperature=0.1,
        # repetition_penalty=1.1,
        do_sample=True
    )
    with torch.no_grad():
        output_tokens = model.generate(**inputs, generation_config=gen_config, return_dict_in_generate=True, output_scores=True)
    response = tokenizer.decode(output_tokens.sequences[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    # スコアの抽出
    try:
        # 数値を抽出するための正規表現パターン
        score_pattern = r'\d+'
        match = re.search(score_pattern, response)
        if match:
            score = int(match.group())
        else:
            raise ValueError("数値が見つかりませんでした")
    except (ValueError, IndexError) as e:
        print(f"Error extracting score: {e}")
        print(f"Raw response: {response}")  # デバッグ用
        score = -1 # エラー時はとりあえず-1を代入

    return response, score


def main(model_name: str):
    df = load_asap_dataset('datasets/ASAP', stratify=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=True
    )

    outputs = []
    for essay_set, essay_id, essay, score in tqdm(df.iter_rows(), total=len(df)):
        with open(f"llm_prompts/ASAP/info/prompt{essay_set}.md", "r") as f:
            prompt = f.read()
        response, score = vanilla_scoring(essay, prompt, model, tokenizer, essay_set)
        outputs.append({
            "essay_set": essay_set,
            "essay_id": essay_id,
            "response": response,
            "score": score,
        })

    # Save outputs
    os.makedirs("outputs/vanilla", exist_ok=True)
    pl.DataFrame(outputs).write_csv(f"outputs/vanilla/{model_name.split('/')[1]}.csv")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()
    main(args.model_name)
