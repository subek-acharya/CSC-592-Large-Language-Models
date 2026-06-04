import sys
from datasets import load_dataset, Dataset, DatasetDict
from dataclasses import dataclass, field
from typing import Optional
import torch
from peft import LoraConfig
from tqdm import tqdm
import pandas as pd
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, AutoTokenizer, pipeline
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login


model_name = "meta-llama/Llama-2-7b-hf"
# model_name = "meta-llama/Llama-3.1-8B-Instruct"
# model_name = "microsoft/Phi-3-mini-4k-instruct"
dataset_text_field = "text"
log_with =  'wandb'
learning_rate = 1.41e-5
batch_size =32
seq_length = 128
gradient_accumulation_steps = 2
load_in_8bit = False
load_in_4bit = True
use_peft = True
num_train_epochs = 1
peft_lora_r = 64
peft_lora_alpha = 16
logging_steps = 1
output_dir = "checkpoints"

def main():
    tqdm.pandas()
    # replace with your token
    load_dotenv()  # Load from .env file
    token = os.getenv("HF_TOKEN")

    login(token=token) 
    snli = load_dataset("stanfordnlp/snli")

    # get train, validation, and test splits
    train_data = snli["train"]
    validation_data = snli["validation"]
    test_data = snli["test"]
    
    id_to_label = {0:'entailment', 1:'neutral', 2:'contradiction'}

    question_template = "### Human: Classify the relationship between the following two sentences as one of the following: entailment, neutral, contradiction. "

    train_instructions = [f'{question_template}\npremise: {x}\nhypothesis: {y}\n\n### Assistant: {id_to_label[z]}' for x,y,z in zip(train_data['premise'],train_data['hypothesis'],train_data['label']) if z!= -1]
    print(train_instructions[0])
    
    validation_instructions = [f'{question_template}\npremise: {x}\nhypothesis: {y}\n\n### Assistant: {id_to_label[z]}' for x,y,z in zip(validation_data['premise'],validation_data['hypothesis'],validation_data['label']) if z != -1]

    ds_train = Dataset.from_dict({"text": train_instructions})
    ds_validation = Dataset.from_dict({"text": validation_instructions})
    instructions_ds_dict = DatasetDict({"train": ds_train, "eval": ds_validation})

    print(instructions_ds_dict['train']['text'][0])

    

    if load_in_8bit and load_in_4bit:
        raise ValueError("Select either 8 bits or 4 bits ...")
    elif load_in_8bit or load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit)
        device_map = {"": 0}
        torch_dtype = torch.bfloat16
    else:
        device_map = None
        quantization_config = None
        torch_dtype = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch_dtype, token=token)

    dataset = instructions_ds_dict

    # training_args = TrainingArguments(
    #     output_dir=output_dir,
    #     per_device_train_batch_size=batch_size,
    #     gradient_accumulation_steps= gradient_accumulation_steps,
    #     learning_rate= learning_rate,
    #     logging_steps= logging_steps,
    #     num_train_epochs=num_train_epochs,
    # )

    if use_peft:
        peft_config = LoraConfig(
            r=peft_lora_r,
            lora_alpha=peft_lora_alpha,
            target_modules="all-linear",
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = None

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['eval'],
        peft_config=peft_config,
        args=SFTConfig(output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps= gradient_accumulation_steps,
        learning_rate= learning_rate,
        logging_steps= logging_steps,
        num_train_epochs=num_train_epochs,
        max_length= seq_length,
        dataset_text_field= dataset_text_field
        )
    )

    trainer.train()

    trainer.save_model(output_dir)
    
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name,token=token)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map={'':0},
    )
    query = instructions_ds_dict['eval']['text'][1].split('### Assistant: ')[0] + '### Assistant:'
    queries = [instructions_ds_dict['eval']['text'][i].split('### Assistant: ')[0] + '### Assistant:' for i in range(len(instructions_ds_dict['eval']))]
    sequences = pipe(
        queries,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=3,
        early_stopping=True,
        # do_sample=True,
    )
    
    results = []

    for seq in sequences:
      result = seq[0]['generated_text'].split('### Assistant:')[1]
      results.append(result)

    labels = []

    for label in instructions_ds_dict['eval']['text']:
      result = label.split('### Assistant:')[1]
      labels.append(result)

    print("Accuracy: ", (len([1 for x, y in zip(results, labels) if y in x]) / len(labels)))
    
if __name__ == "__main__":
    sys.exit(int(main() or 0))
