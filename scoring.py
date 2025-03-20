import os
import re
import torch
import transformers
from tqdm import tqdm
import json
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from typing import Optional, Literal
from utils import load_asap_dataset, load_toefl_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


def mts_scoring(essay, prompt, scoring_criteria, model, tokenizer):
    """MTS (Multi-Trait Specialization) に基づくエッセイ採点."""

    # Define system prompt template
    system_prompt_template = """You are a member of the English essay writing test evaluation committee. Four teachers will be provided with a [Prompt] and an [Essay] written by a student in response to the [Prompt]. Each teacher will score the essays based on different dimensions of writing quality. Your specific responsibility is to score the essays in terms of "{trait}". {trait_desc} Focus on the content of the [Essay] and the [Scoring Rubric] to determine the score."""

    # Define initial user prompt template
    user_prompt_template = """
    [Prompt]
    {prompt}
    (end of [Prompt])
    [Essay]
    {essay}
    (end of [Essay])
    Q. List the quotations from the [Essay] that are relevant to "{trait}" and evaluate whether each quotation is well-written or not.
    """

    # Define scoring user prompt template
    scoring_prompt_template = """
    [Scoring Rubric]
    **{trait}**:
    {criteria}
    (end of [Scoring Rubric])
    Q. Based on the [Scoring Rubric] and the quotations you found, how would you rate the "{trait}" of this essay? Assign a score from 0 to 10, strictly following the [Output Format] below.
    [Output Format]
    Score: <score>insert ONLY the numeric score (from 0 to 10) here</score>
    (End of [Output Format])
    """
    
    gen_config = GenerationConfig(
        max_new_tokens=512,
        temperature=0.1,
        do_sample=True
    )
    trait_scores = []
    for info in scoring_criteria:
        # Create initial messages
        messages = [
            {"role": "system", "content": system_prompt_template.format(trait=info['name'], trait_desc=info['description'])},
            {"role": "user", "content": user_prompt_template.format(prompt=prompt, essay=essay, trait=info['name'])}
        ]

        chat = tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = tokenizer(chat, return_tensors="pt").to(model.device)

        gen_config = GenerationConfig(
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True
        )
        with torch.no_grad():
            output_tokens = model.generate(**inputs, generation_config=gen_config)

        response_1 = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

        # Add scoring prompt to messages
        messages.append({"role": "assistant", "content": response_1})
        messages.append({
            "role": "user", 
            "content": scoring_prompt_template.format(
                trait=info['name'],
                criteria=info['scoring_criteria']
            )
        })

        # Generate response for scoring
        chat = tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = tokenizer(chat, return_tensors="pt").to(model.device)

        gen_config = GenerationConfig(
            max_new_tokens=64,
            temperature=0.1,
            do_sample=True
        )
        with torch.no_grad():
            output_tokens = model.generate(**inputs, generation_config=gen_config, return_dict_in_generate=True, output_scores=True)

        response_2 = tokenizer.decode(output_tokens.sequences[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

        # Extract score
        try:
            # 数値を抽出するための正規表現パターン
            score_pattern = r'\d+'
            match = re.search(score_pattern, response_2)
            if match:
                score = int(match.group())
                trait_scores.append(score)
            else:
                raise ValueError("数値が見つかりませんでした")
        except (ValueError, IndexError) as e:
            print(f"Error extracting score for trait {info['name']}: {e}")
            print(f"Raw response: {response_2}")  # デバッグ用
            trait_scores.append(-1) # エラー時はとりあえず-1を代入
            continue

    return trait_scores


def main():
    # Load dataset
    df = load_asap_dataset('datasets/ASAP', stratify=True)

    # Load scoring criteria
    with open('outputs/multi-trait-decomposition/asap_rubrics_gpt-4o.json') as f:
        all_scoring_criteria = json.load(f)

    # Initialize model and tokenizer
    # model_name = "meta-llama/Llama-2-7b-chat-hf"
    # model_name = "meta-llama/Llama-3.1-8B-Instruct"
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    # model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        use_auth_token=True
    )

    # Process essays
    outputs = []
    for essay_set, essay_id, essay, score in tqdm(df.iter_rows(), total=len(df)):
        with open(f"llm_prompts/ASAP/info/prompt{essay_set}.md", "r") as f:
            prompt = f.read()
        scoring_criteria = all_scoring_criteria[f'prompt{essay_set}']['dimensions']
        trait_scores = mts_scoring(essay, prompt, scoring_criteria, model, tokenizer)
        print(f'essay_set: {essay_set}, scores: {trait_scores}')
        outputs.append(trait_scores)

    # CSVファイルとして保存
    df_scores = pd.DataFrame(outputs)
    df_scores.to_csv('outputs/trait_scores_llama3_3B.csv', index=False)
    return outputs


if __name__ == "__main__":
    main() 