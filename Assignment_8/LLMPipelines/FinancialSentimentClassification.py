import sys
import re
import numpy as np
import pandas as pd
import os
import torch
from tqdm import tqdm
from datasets import Dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from huggingface_hub import login

import os
from dotenv import load_dotenv



# ---------------- GLOBAL CONFIG ----------------

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
compute_dtype = torch.float16
max_seq_length = 2048     # ------ reduced from 2048 to 512


# ---------------- HELPERS ----------------

def get_num_layers(model):
    numbers = set()
    for name, _ in model.named_parameters():
        for number in re.findall(r'\d+', name):
            numbers.add(int(number))
    return max(numbers) if numbers else 0


def predict(X_test, model, tokenizer):
    y_pred = []

    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        max_new_tokens=3,
        do_sample=False,
        temperature=1.0,
    )

    for i in tqdm(range(len(X_test))):
        prompt = X_test.iloc[i]["text"]
        result = pipe(prompt, pad_token_id=pipe.tokenizer.eos_token_id)

        generated_text = result[0]['generated_text']
        answer = generated_text.split("The correct option is")[-1].strip().lower()

        if "positive" in answer:
            y_pred.append("positive")
        elif "negative" in answer:
            y_pred.append("negative")
        elif "neutral" in answer:
            y_pred.append("neutral")
        else:
            y_pred.append("none")

    return y_pred


def evaluate(y_true, y_pred):
    mapping = {'positive': 2, 'neutral': 1, 'none': 1, 'negative': 0}

    def map_func(x): return mapping.get(x, 1)

    y_true_mapped = np.vectorize(map_func)(y_true)
    y_pred_mapped = np.vectorize(map_func)(y_pred)

    accuracy = accuracy_score(y_true=y_true_mapped, y_pred=y_pred_mapped)
    print(f'Accuracy: {accuracy:.3f}')

    unique_labels = sorted(list(set(y_true_mapped)))
    for label in unique_labels:
        idx = [i for i in range(len(y_true_mapped)) if y_true_mapped[i] == label]
        class_accuracy = accuracy_score(
            [y_true_mapped[i] for i in idx],
            [y_pred_mapped[i] for i in idx]
        )
        print(f'Accuracy for label {label}: {class_accuracy:.3f}')

    print("\nClassification Report:")
    print(classification_report(y_true_mapped, y_pred_mapped, zero_division=0))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true_mapped, y_pred_mapped, labels=[0, 1, 2]))


def generate_prompt(dp):
    return f"""The sentiment of the following phrase: '{dp["text"]}' is

        Positive
        Negative
        Neutral
        Cannot be determined

        Solution: The correct option is {dp["sentiment"]}""".strip()


def generate_test_prompt(dp):
    return f"""The sentiment of the following phrase: '{dp["text"]}' is

        Positive
        Negative
        Neutral
        Cannot be determined

        Solution: The correct option is""".strip()


# ---------------- MAIN ----------------

def main():
    filename = "./data/FinancialSentimentDataset/all-data.csv"

    load_dotenv()  # Load from .env file
    token = os.getenv("HF_TOKEN")
    login(token=token)

    # --- Data Preparation ---
    df = pd.read_csv(filename,
                     names=["sentiment", "text"],
                     encoding="utf-8", encoding_errors="replace")

    # Stratified Train/Test Split
    X_train = list()
    X_test = list()
    for sentiment in ["positive", "neutral", "negative"]:
        train, test  = train_test_split(df[df.sentiment==sentiment],
                                        train_size=300,
                                        test_size=300,
                                        random_state=42)
        X_train.append(train)
        X_test.append(test)

    X_train = pd.concat(X_train).sample(frac=1, random_state=10)
    X_test = pd.concat(X_test)
   
    # Separate Evaluation data
    all_indices = set(df.index)
    train_test_indices = set(X_train.index) | set(X_test.index)
    eval_idx = list(all_indices - train_test_indices)
   
    X_eval = df[df.index.isin(eval_idx)]
    X_eval = (X_eval
              .groupby('sentiment', group_keys=False)
              .apply(lambda x: x.sample(n=50, random_state=10, replace=True)))
             
    X_train = X_train.reset_index(drop=True)

    # Apply Prompt Templates
    X_train = pd.DataFrame(X_train.apply(generate_prompt, axis=1), columns=["text"])
    X_eval = pd.DataFrame(X_eval.apply(generate_prompt, axis=1), columns=["text"])

    y_true = X_test.sentiment
    X_test = pd.DataFrame(X_test.apply(generate_test_prompt, axis=1), columns=["text"])
   
    print("--- Example Training Prompt ---")
    print(X_train.iloc[0]["text"])
   
    # Convert to Hugging Face Datasets
    train_data = Dataset.from_pandas(X_train)
    eval_data = Dataset.from_pandas(X_eval)


    # ---------- MODEL LOADING (4-bit QLoRA) ----------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token = token
    )

    # Prepare model for k-bit training (REQUIRED for QLoRA!)
    model = prepare_model_for_kbit_training(model)

    model.config.use_cache = False
    model.gradient_checkpointing_enable()    # ----- Added

    tokenizer = AutoTokenizer.from_pretrained(model_name, token = token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = max_seq_length

    # ---------- INITIAL EVALUATION ----------
    print("\n--- Initial Evaluation (Base Model) ---")
    y_pred = predict(X_test, model, tokenizer)
    evaluate(y_true, y_pred)

    #Clear GPU memory before training
    import gc
    gc.collect()
    torch.cuda.empty_cache()           # ----- Added

    # ---------- LoRA CONFIG ----------
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules="all-linear",
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_arguments = TrainingArguments(
        output_dir="logs",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,    # ------- Changed from 8 to 16
        optim="paged_adamw_32bit",          # ------- Changed from 32bit to 8bit
        save_steps=0,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
        eval_strategy="epoch",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_arguments,
    )

    # ---------- TRAIN ----------
    print("\n--- Starting Fine-Tuning ---")
    trainer.train()

    # Save LoRA adapters
    trainer.model.save_pretrained("trained-model")

    # Also save tokenizer for completeness
    tokenizer.save_pretrained("trained-model")           # ------- Added

    # ===== FREE GPU MEMORY BEFORE RELOADING =====
    del trainer
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache() 
    print("GPU memory cleared.")                         # ------- Added
    # ============================================

    # ---------- RELOAD FOR INFERENCE ----------
    print("\n--- Loading Fine-Tuned Model for Evaluation ---")

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token = token
    )

    model = PeftModel.from_pretrained(base_model, "trained-model")
    model = model.merge_and_unload()

    # Final dtype fix
    model.lm_head = model.lm_head.to(torch.float16)

    # ---------- FINAL EVALUATION ----------
    print("\n--- Final Evaluation (Fine-Tuned Model) ---")
    y_pred = predict(X_test, model, tokenizer)
    evaluate(y_true, y_pred)

    # Save predictions
    evaluation = pd.DataFrame({
        "text": X_test["text"],
        "y_true": y_true,
        "y_pred": y_pred,
    })
    evaluation.to_csv("test_predictions.csv", index=False)

    print("DONE.")

    return 0


if __name__ == "__main__":
    sys.exit(int(main() or 0))
