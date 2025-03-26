import os
import re
import argparse
import torch
from tqdm import tqdm
import polars as pl
from typing import Optional, Literal
from utils import load_asap_dataset, load_toefl_dataset, get_score_range
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


def vanilla_scoring(essay, prompt, model, tokenizer, prompt_id, dataset: Literal["ASAP", "TOEFL11"]):
    """Vanilla Promptingによるエッセイ採点."""

    # System Promptの設定
    min_score, max_score = get_score_range(dataset, prompt_id)
    # Set prompt
    if dataset == "ASAP":
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
        Strictly follow the format below to give your answer. Other formats are NOT allowed.
        Evaluation: <evaluation>insert evaluation here</evaluation>
        Score: <score>insert score ({min_score} to {max_score}) here</score>
        """
    elif dataset == "TOEFL11":
        system_prompt = f"""As an English teacher, your primary responsibility is to evaluate the writing quality of essays written by second language learners on an English exam. During the assessment process, you will be provided with a prompt and an essay. First, you should provide comprehensive and conrete feedback that is closely linked to the content of the essay. It is essential to avoid offering generic remarks that could be applied to any piece of writing. To create a compelling evaluation for both the student and fellow experts, you should reference specific content of the essay to substantiate your assessment. Next, your evaluation should culminate in assigning an overall score to the student’s essay, on a three level scale of "low", "medium" and "high". It’s crucial to tailor your evaluation criteria to be well-suited for second language learners, taking into account their expected abilities."""
        # User Promptの設定
        user_prompt = f"""[Prompt]
        {prompt}
        (end of [Prompt])

        [Essay]
        {essay}
        (end of [Essay])

        Strictly follow the format below to give your answer. Other formats are NOT allowed.
        Evaluation: <evaluation>insert evaluation here</evaluation>
        Score: <score>insert score (choose one of "low", "medium", and "high") here</score>
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

    if dataset == "ASAP":
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
    elif dataset == "TOEFL11":
        # スコアの抽出
        try:
            # スコアを抽出するための正規表現パターン
            score_pattern = r'(?i)(low|medium|high)'
            match = re.search(score_pattern, response)
            if match:
                score = match.group()
            else:
                score = None
                raise ValueError("スコアが見つかりませんでした")
        except (ValueError, IndexError) as e:
            print(f"Error extracting score: {e}")
            print(f"Raw response: {response}")

    return response, score


def main(dataset: Literal["ASAP", "TOEFL11"], model_name: str):
    if dataset == 'ASAP':
        df = load_asap_dataset('datasets/ASAP', stratify=True)
    elif dataset == 'TOEFL11':
        df = load_toefl_dataset('datasets/TOEFL11')
    df = df.select(['essay_set', 'essay_id', 'essay', 'score'])
    
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
        if dataset == "ASAP":
            with open(f"llm_prompts/ASAP/info/prompt{essay_set}.md", "r") as f:
                prompt = f.read()
        elif dataset == "TOEFL11":
            with open(f"llm_prompts/TOEFL11/info/prompt{essay_set}.md", "r") as f:
                prompt = f.read()
        
        response, score = vanilla_scoring(essay, prompt, model, tokenizer, essay_set, dataset)
        print(f'essay_set: {essay_set}, essay_id: {essay_id}, score: {score}')
        outputs.append({
            "essay_set": essay_set,
            "essay_id": essay_id,
            "response": response,
            "score": score,
        })

    # Save outputs
    os.makedirs("outputs/vanilla", exist_ok=True)
    pl.DataFrame(outputs).write_csv(f"outputs/vanilla/{dataset}_{model_name.split('/')[1]}.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MTS scoring with specified model and dataset.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Model name to use.")
    parser.add_argument("--dataset", type=str, default="ASAP", help="Dataset to use (ASAP or TOEFL11).")
    args = parser.parse_args()

    main(args.dataset, args.model_name)
