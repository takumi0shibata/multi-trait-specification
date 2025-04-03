import argparse
import numpy as np
import torch
import json
import polars as pl
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction
from sklearn.metrics import mean_squared_error, mean_absolute_error, cohen_kappa_score
from torch.utils.data import Dataset as TorchDataset # Rename to avoid conflict
from utils import load_asap_dataset, load_toefl_dataset, get_score_range

def main(args):
    TASK = "ASAP"

    PROMPT = args.prompt

    df_test = load_asap_dataset('datasets/ASAP', stratify=True)
    df_train = load_asap_dataset('datasets/ASAP', stratify=False).filter(~pl.col("essay_id").is_in(df_test['essay_id']))

    df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=12)

    df_train = df_train.filter(pl.col("essay_set") == PROMPT)
    df_val = df_val.filter(pl.col("essay_set") == PROMPT)
    df_test = df_test.filter(pl.col("essay_set") == PROMPT)

    min_score, max_score = get_score_range(TASK, PROMPT)
    df_train = df_train.with_columns(
        ((pl.col("score") - min_score) / (max_score - min_score)).alias("normalized_score")
    )
    df_val = df_val.with_columns(
        ((pl.col("score") - min_score) / (max_score - min_score)).alias("normalized_score")
    )
    df_test = df_test.with_columns(
        ((pl.col("score") - min_score) / (max_score - min_score)).alias("normalized_score")
    )

    # --- Configuration ---
    # Specify the pre-trained model name. Can be changed to "bert-base-uncased", "FacebookAI/roberta-base", "microsoft/deberta-v3-large", etc.
    model_name = args.model
    num_train_epochs = 10 # Number of training epochs (adjust as needed)
    batch_size = 32 # Batch size per device (adjust based on GPU memory)
    max_length = 512 # Max sequence length for tokenizer

    # --- 3. Load Tokenizer and Model ---
    # Load the tokenizer associated with the chosen pre-trained model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load the pre-trained model for sequence classification.
    # Set num_labels=1 for regression tasks.
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

    # --- 4. Tokenize Data ---
    # Tokenize the texts using the loaded tokenizer
    train_encodings = tokenizer(df_train['essay'].to_list(), truncation=True, padding="max_length", max_length=max_length)
    dev_encodings = tokenizer(df_val['essay'].to_list(), truncation=True, padding="max_length", max_length=max_length)
    test_encodings = tokenizer(df_test['essay'].to_list(), truncation=True, padding="max_length", max_length=max_length)

    # --- 5. Create Custom PyTorch Dataset ---
    class EssayDatasetTmp(TorchDataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            # Retrieve tokenized inputs for the given index
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            # Add the corresponding label, converting it to a tensor
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float) # Ensure label is float tensor
            return item

        def __len__(self):
            # Return the total number of samples
            return len(self.labels)

    # Instantiate the custom dataset for training and evaluation sets
    train_dataset = EssayDatasetTmp(train_encodings, df_train['normalized_score'])
    dev_dataset = EssayDatasetTmp(dev_encodings, df_val['normalized_score'])
    test_dataset = EssayDatasetTmp(test_encodings, df_test['normalized_score'])

    # --- 6. Define Training Arguments ---
    # Configure the training process using TrainingArguments (remains the same)
    training_args = TrainingArguments(
        output_dir='./outputs/prompt-specific',
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=0.1, # Number of steps for learning rate warmup
        weight_decay=2e-5, # Strength of weight decay regularization
        optim="adamw_torch", # Use the AdamW optimizer
        logging_strategy="steps",  # Log metrics at the end of each epoch
        logging_steps=10, # Log every 10 steps
        eval_strategy="epoch",     # Evaluate at the end of each epoch
        save_strategy="epoch",
        load_best_model_at_end=False, # Load the best model found during training at the end
        metric_for_best_model="eval_qwk", # Use Mean Squared Error to determine the best model
        greater_is_better=True, # Lower MSE is better
        report_to="none", # Disable external reporting integrations like WandB/TensorBoard for simplicity
        fp16=torch.cuda.is_available(), # Use mixed precision training if a GPU is available
    )

    # --- 7. Define Compute Metrics Function ---
    # Define a function to compute metrics during evaluation (MSE, MAE, and QWK for regression)
    def prepare_compute_metrics(minscore, maxscore):
        def compute_metrics(eval_pred: EvalPrediction):
            predictions, labels = eval_pred

            # タプルの場合 (logits,) をアンパックする
            if isinstance(predictions, tuple):
                predictions = predictions[0]

            if len(predictions.shape) > 1:
                predictions = predictions.squeeze(-1)

            rmse = np.sqrt(mean_squared_error(labels, predictions))
            mae = mean_absolute_error(labels, predictions)

            predictions = predictions * (maxscore - minscore) + minscore
            labels = labels * (maxscore - minscore) + minscore

            qwk = cohen_kappa_score(np.round(predictions), np.round(labels), weights="quadratic",
                                    labels=[i for i in range(minscore, maxscore + 1)])
            lwk = cohen_kappa_score(np.round(predictions), np.round(labels), weights="linear",
                                    labels=[i for i in range(minscore, maxscore + 1)])

            corr = np.corrcoef(predictions, labels)[0, 1]

            return {"rmse": rmse, "mae": mae, "qwk": qwk, "lwk": lwk, "corr": corr}
        return compute_metrics

    # --- 8. Instantiate Trainer ---
    # Initialize the Trainer with the model, arguments, custom datasets, tokenizer, and metrics function
    # Note: The tokenizer is still passed for potential use cases like saving, but not strictly needed for data loading now.
    min_score, max_score = get_score_range(TASK, PROMPT)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        compute_metrics=prepare_compute_metrics(min_score, max_score),
    )

    # --- 9. Train the Model ---
    print(f"Starting fine-tuning for {model_name}...")
    trainer.train()
    print("Fine-tuning completed.")

    # --- (Optional) Evaluate the Best Model ---
    print("Evaluating the best model on the validation set...")
    eval_results = trainer.evaluate(eval_dataset=test_dataset)
    print("Evaluation results:", eval_results)

    # Save Metrics with json
    print("Saving metrics...")
    with open(f"outputs/prompt-specific/{args.model.split('/')[1]}_prompt{PROMPT}.json", "w") as metrics_file:
        json.dump(eval_results, metrics_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a pre-trained model on a specific prompt")
    parser.add_argument("--prompt", type=int, required=True, help="Prompt number to fine-tune on")
    parser.add_argument("--model", type=str, required=True, help="Pre-trained model name to fine-tune", choices=["bert-base-uncased", "FacebookAI/roberta-base", "microsoft/deberta-v3-large"])
    args = parser.parse_args()
    main(args)