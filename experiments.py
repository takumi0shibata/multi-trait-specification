import os
import torch
import transformers
from tqdm import tqdm
import json
import numpy as np
from sklearn.metrics import cohen_kappa_score
from typing import Optional, Literal
from dotenv import load_dotenv

load_dotenv()

# データセット読み込み関数 (仮の実装)
def load_dataset(dataset_name, prompt_id):
    """
    データセットを読み込む関数 (仮の実装).

    Args:
        dataset_name (str): "ASAP" or "TOEFL11".
        prompt_id (int): プロンプトID (ASAP: 1-8, TOEFL11: 1-8).

    Returns:
        list: 各要素が辞書形式のデータのリスト. 辞書は以下のキーを持つ:
            - "essay": エッセイ本文 (str)
            - "score": 正解スコア (int or str). ASAPの場合はint, TOEFL11の場合は "low", "medium", "high" のいずれか.
    """
    # ダミーデータ (必要に応じて、ご自身のデータ読み込み関数に置き換えてください)
    if dataset_name == "ASAP":
        dummy_data = [
            {"essay": "This is a sample essay for ASAP prompt 1.", "score": 8},
            {"essay": "Another example essay for ASAP prompt 1.", "score": 6},
        ] * 50 # 100個のデータ
    elif dataset_name == "TOEFL11":
        dummy_data = [
            {"essay": "This is a sample essay for TOEFL11.", "score": "high"},
            {"essay": "Another example essay for TOEFL11.", "score": "medium"},
        ] * 50

    return dummy_data


def mts_scoring(essay, prompt, scoring_criteria, model_id, traits):
    """MTS (Multi-Trait Specialization) に基づくエッセイ採点."""

    # Define system prompt template
    system_prompt_template = f"""You are a member of the English essay writing test evaluation committee. Four teachers will be provided with a [Prompt] and an [Essay] written by a student in response to the [Prompt]. Each teacher will score the essays based on different dimensions of writing quality. Your specific responsibility is to score the essays in terms of "{trait}". {trait_desc} Focus on the content of the [Essay] and the [Scoring Rubric] to determine the score."""

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

    trait_scores = []
    for trait, trait_desc in traits.items():
        # Create initial messages
        messages = [
            {"role": "system", "content": system_prompt_template.format(trait=trait, trait_desc=trait_desc)},
            {"role": "user", "content": user_prompt_template.format(prompt=prompt, essay=essay, trait=trait)}
        ]

        pipline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto"
        )

        response_1 = pipline(messages, max_new_tokens=512, temperature=0.1, repetition_penalty=1.1)[0]["generated_text"][-1]['content']

        # Add scoring prompt to messages
        messages.append({"role": "assistant", "content": response_1})
        messages.append({
            "role": "user", 
            "content": scoring_prompt_template.format(
                trait=trait,
                criteria=scoring_criteria[trait]
            )
        })

        # Generate response for scoring
        response_2 = pipline(messages, max_new_tokens=64, temperature=0.1, repetition_penalty=1.1)[0]["generated_text"][-1]['content']

        # Extract score
        try:
            # Find first number in response
            trait_score = 0
            for char in response_2:
                if char.isdigit():
                    trait_score = int(char)
                    break
            trait_scores.append(trait_score)
        except (ValueError, IndexError) as e:
            print(f"Error extracting score for trait {trait}: {e}")
            print(f"Raw response: {response_2}")  # デバッグ用
            trait_scores.append(0) # エラー時はとりあえず0を代入
            continue

    return trait_scores


def vanilla_scoring(essay, prompt, model, tokenizer, dataset_name, prompt_id):
    """Vanilla Promptingによるエッセイ採点."""

    # System Prompt (dataset_nameとprompt_idによって変わる部分に注意)
    if dataset_name == "ASAP":
        min_score, max_score = get_score_range(dataset_name, prompt_id) # スコアレンジを取得する関数（後述）
        system_prompt = f"""As an English teacher, your primary responsibility is to evaluate the writing quality of essays written by middle school students. During the assessment process, you will be provided with a prompt and an essay. First, you should provide comprehensive and conrete feedback that is closely linked to the content of the essay. It is essential to avoid offering generic remarks that could be applied to any piece of writing. To create a compelling evaluation for both the student and fellow experts, you should reference specific content of the essay to substantiate your assessment. Next, your evaluation should culminate in assigning an overall score to the student's essay, measured on a scale from {min_score} to {max_score}, where higher score should reflect a higher level of writing quality. It's crucial to tailor your evaluation criteria to be well-suited for middle school level writing, taking into account the developmental stage and capabilities of these students."""
    elif dataset_name == "TOEFL11":
        system_prompt = f"""As an English teacher, your primary responsibility is to evaluate the writing quality of essays written by second language learners on an English exam. During the assessment process, you will be provided with a prompt and an essay. First, you should provide comprehensive and conrete feedback that is closely linked to the content of the essay. It is essential to avoid offering generic remarks that could be applied to any piece of writing.  To create a compelling evaluation for both the student and fellow experts, you should reference specific content of the essay to substantiate your assessment. Next, your evaluation should culminate in assigning an overall score to the student's essay, on a three level scale of "low", "medium" and "high". It's crucial to tailor your evaluation criteria to be well-suited for second language learners, taking into account their expected abilities."""

    # User Prompt
    user_prompt = f"[Prompt]\n{prompt}\n(end of [Prompt])\n[Essay]\n{essay}\n(end of [Essay])\nStrictly follow the format below to give your answer. Other formats are NOT allowed. Evaluation: <evaluation>insert evaluation here</evaluation>"
    if dataset_name == "ASAP":
        user_prompt += f" Score: <score>insert score ({min_score} to {max_score}) here</score>"
    else:  # TOEFL11
        user_prompt += ' Score: <score>insert score (choose one of "low", "medium", and "high") here</score>'

    inputs = tokenizer(system_prompt + "\n" + user_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1, repetition_penalty=1.1,do_sample=True) # TODO: 調整
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # スコア抽出
    try:
        score_start = response.rfind("<score>") + len("<score>")
        score_end = response.rfind("</score>")
        score_str = response[score_start:score_end].strip()

        if dataset_name == "ASAP":
            score = int(score_str)
        elif dataset_name == "TOEFL11":
            score = score_str # 文字列のまま
    except (ValueError, IndexError) as e:
        print(f"Error extracting score: {e}")
        print(f"Raw response: {response}")
        if dataset_name == "ASAP":
            score = 0
        else:
            score = "low"

    return score


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


def scale_to_target_range(trait_scores, dataset_name, prompt_id):
    """trait_scoresを[0,10]からデータセットのターゲット範囲にスケーリング."""

    # 1. Outlier Clipping (Q1, Q3, IQR)
    q1 = np.percentile(trait_scores, 25)
    q3 = np.percentile(trait_scores, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    clipped_scores = np.clip(trait_scores, lower_bound, upper_bound)

    # 2. 平均
    avg_score = np.mean(clipped_scores)

    # 3. Min-Max Scaling
    if dataset_name == "ASAP":
        min_score, max_score = get_score_range(dataset_name, prompt_id)
        scaled_score = int(round(((avg_score - 0) / (10 - 0)) * (max_score - min_score) + min_score)) # [0, 10] -> [min, max]
    elif dataset_name == "TOEFL11":
        if avg_score <= 3.33:
            scaled_score = "low"
        elif avg_score <= 6.66:
            scaled_score = "medium"
        else:
            scaled_score = "high"
    return scaled_score


def convert_to_numerical_score(score):
    """TOEFL11の文字列スコアを数値に変換."""
    mapping = {"low": 1, "medium": 2, "high": 3}
    return mapping[score]


def main(model_name, dataset_name, prompt_id, method="MTS"):
    # モデルとトークナイザーの準備
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # データセット読み込み
    dataset = load_dataset(dataset_name, prompt_id)

    # ASAP prompt (仮)
    prompt = "Write an essay about the impact of technology on our daily lives."

    # MTSの場合、採点基準を生成
    if method == "MTS":
        # ASAP, TOEFL11 rubric guidelines (仮)
        if dataset_name == 'ASAP':
            rubric_guidelines = "The essay should be scored based on organization, use of supporting details, clarity of the thesis, and language use."
        else:
            rubric_guidelines = """Task Response
            This dimension evaluates how well the prompt is understood, addressed, and developed within the response.
            0-2:
            - Barely relevant or unrelated content to the given prompt.
            - Lack of identifiable position or comprehension of the question.
            - Minimal or no development of ideas; content may be tangential or copied.
            3-4:
            - Partially addresses the prompt but lacks depth or coherence.
            - Discernible position, but unclear or lacking in support.
            - Ideas are difficult to identify or irrelevant with some repetition.
            5-6:
            - Addresses main parts of the prompt but incompletely or with limited development.
            - Presents a position with unclear or repetitive development.
            - Some relevant ideas but insufficiently developed or supported.
            7-8:
            - Adequately addresses the prompt with clear and developed points.
            - Presents a coherent position with well-extended and supported ideas.
            - Some tendencies toward over-generalization or lapses in content, but mostly on point.
            9-10:
            - Fully and deeply explores the prompt with a clear, well-developed position.
            - Extensively supported ideas relevant to the prompt.
            - Extremely rare lapses in content or support; demonstrates exceptional depth and insight.

            Coherence and Cohesion
            This criterion assesses how well ideas are logically organized and connected within a written response.
            0-2:
            - Lack of coherence; response is off-topic or lacking in relevant message.
            - Minimal evidence of organizational control or logical progression.
            - Virtually absent or ineffective use of cohesive devices and paragraphing.
            3-4:
            - Ideas are discernible but arranged incoherently or lack clear progression.
            - Unclear relationships between ideas, limited use of basic cohesive devices.
            - Minimal or unclear referencing, inadequate paragraphing if attempted.
            5-6:
            - Some underlying coherence but lacks full logical organization.
            - Relationships between ideas are somewhat clear but not consistently linked.
            - Limited use of cohesive devices, with inaccuracies or overuse, and occasional repetition.
            - Inconsistent or inadequate paragraphing.
            7-8:
            - Generally organized with a clear overall progression of ideas.
            - Cohesive devices used well with occasional minor lapses.
            - Effective paragraphing supporting coherence, though some issues in sequencing or clarity within paragraphs.
            9-10:
            - Effortless follow-through of ideas with superb coherence.
            - Seamless and effective use of cohesive devices with minimal to no lapses.
            - Skilful paragraphing enhancing overall coherence and logical progression.

            Lexical Resource
            This dimension evaluates the range, precision, and appropriateness of vocabulary used within a written response.
            0-2:
            · Minimal to no resource evident; extremely limited vocabulary or reliance on memorized phrases.
            - Lack of control in word formation, spelling, and recognition of vocabulary.
            - Communication severely impeded due to the absence of lexical range.
            3-4:
            - Inadequate or limited resource; vocabulary may be basic or unrelated to the task.
            - Possible dependence on input material or memorized language.
            - Errors in word choice, formation, or spelling impede meaning.
            5-6:
            - Adequate but restricted resource for the task.
            - Limited variety and precision in vocabulary, causing simplifications and repetitions.
            - Noticeable errors in spelling/word formation, with some impact on clarity.
            7-8:
            - Sufficient resource allowing flexibility and precision in expression.
            - Ability to use less common or idiomatic items, despite occasional inaccuracies.
            - Some errors in spelling/word formation with minimal impact on communication.
            9-10:
            - Full flexibility and precise use of a wide range of vocabulary.
            - Very natural and sophisticated control of lexical features with rare minor errors.
            - Skilful use of uncommon or idiomatic items, enhancing overall expression.

            Grammatical Range and Accuracy
            This dimension assesses the breadth of grammatical structures used and the precision in applying them within written communication.
            0-2:
            - Absence or extremely limited evidence of coherent sentence structures.
            - Lack of control in grammar, minimal to no use of sentence forms.
            - Language largely incomprehensible or irrelevant to the task.
            3-4:
            - Attempts at sentence forms but predominantly error-laden.
            - Inadequate range of structures with frequent grammatical errors.
            - Limited coherence due to significant errors impacting meaning.
            5-6:
            - Limited variety in structures; attempts at complexity with faults.
            - Some accurate structures but with noticeable errors and repetitions.
            - Clear attempts at complexity but lacking precision and fluency.
            7-8:
            - Adequate variety with some flexibility in using complex structures.
            - Generally well-controlled grammar but occasional errors.
            - Clear attempts at complexity and flexibility in sentence structures.
            9-10:
            - Extensive range with full flexibility and precision in structures.
            - Virtually error-free grammar and punctuation.
            - Exceptional command with rare minor errors, showcasing nuanced and sophisticated language use.
            """
        generated_criteria = generate_scoring_criteria(prompt, rubric_guidelines, model_name)
        print(f"Generated Scoring Criteria:\n{generated_criteria}")

        # 各traitとその説明 (仮)  ChatGPTの出力から抽出する必要あり
        traits = {
            "Task Response": "This dimension evaluates how well the prompt is understood, addressed, and developed within the response." ,
            "Coherence and Cohesion": "This criterion assesses how well ideas are logically organized and connected within a written response.",
            "Lexical Resource": "This dimension evaluates the range, precision, and appropriateness of vocabulary used within a written response.",
            "Grammatical Range and Accuracy": "This dimension assesses the breadth of grammatical structures used and the precision in applying them within written communication."
        }

        # traitごとの採点基準 (仮)
        scoring_criteria = {trait: "" for trait in traits} # 空の辞書で初期化
        for trait in traits.keys():
            start_idx = generated_criteria.find(f"{trait}")
            if start_idx == -1:
                start_idx = generated_criteria.find(f"'{trait}'")
            end_idx = generated_criteria.find("0-2",start_idx)
            start_idx = generated_criteria.find(":",end_idx) + 1
            try:
                next_trait = list(traits.keys())[list(traits.keys()).index(trait) + 1]
                end_idx = generated_criteria.find(next_trait)
                if end_idx == -1:
                    end_idx = generated_criteria.find("'",end_idx)
            except:
                end_idx = len(generated_criteria)
            scoring_criteria[trait] = generated_criteria[start_idx:end_idx]

    predicted_scores = []
    true_scores = []

    for sample in tqdm(dataset, desc=f"Processing essays with {method}"):
        essay = sample["essay"]
        true_score = sample["score"]

        if method == "MTS":
            trait_scores = mts_scoring(essay, prompt, scoring_criteria, model, tokenizer, traits)
            predicted_score = scale_to_target_range(trait_scores, dataset_name, prompt_id)
        elif method == "Vanilla":
            predicted_score = vanilla_scoring(essay, prompt, model, tokenizer, dataset_name, prompt_id)
        else:
            raise ValueError("Invalid method. Choose 'MTS' or 'Vanilla'.")

        predicted_scores.append(predicted_score)
        true_scores.append(true_score)

    # QWK計算
    if dataset_name == "ASAP":
        qwk = cohen_kappa_score(true_scores, predicted_scores, weights="quadratic")
    elif dataset_name == "TOEFL11":
        true_scores_num = [convert_to_numerical_score(s) for s in true_scores]
        predicted_scores_num = [convert_to_numerical_score(s) for s in predicted_scores]
        qwk = cohen_kappa_score(true_scores_num, predicted_scores_num, weights="quadratic")

    print(f"QWK ({method}): {qwk:.4f}")
    return qwk


if __name__ == "__main__":
    model_names = [
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-2-13b-chat-hf",
        "mistralai/Mistral-7B-Instruct-v0.2"
    ]
    dataset_names = ["ASAP", "TOEFL11"]
    prompt_ids = {
        "ASAP": range(1, 9),
        "TOEFL11": range(1, 9)
    }
    methods = ["MTS", "Vanilla"]

    results = {}

    for dataset_name in dataset_names:
        results[dataset_name] = {}
        for model_name in model_names:
            results[dataset_name][model_name] = {}
            for method in methods:
                results[dataset_name][model_name][method] = {}
                for prompt_id in prompt_ids[dataset_name]:
                    print(f"Evaluating {model_name} on {dataset_name} (prompt {prompt_id}) with {method}...")
                    qwk = main(model_name, dataset_name, prompt_id, method)
                    results[dataset_name][model_name][method][prompt_id] = qwk

    # 結果表示 (promptごとのQWK, 平均QWK)
    for dataset_name in dataset_names:
        print(f"--- {dataset_name} ---")
        for model_name in model_names:
            print(f"  Model: {model_name}")
            for method in methods:
                print(f"    Method: {method}")
                qwk_values = list(results[dataset_name][model_name][method].values())
                avg_qwk = np.mean(qwk_values)
                print(f"      Prompt-wise QWK: {qwk_values}")
                print(f"      Average QWK: {avg_qwk:.4f}")