import sys
from datasets import load_dataset, Dataset
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from tqdm import tqdm

model_name = "microsoft/Phi-3-mini-4k-instruct"
checkpoint_dir = "checkpoints"
load_dotenv()  # Load from .env file
token = os.getenv("HF_TOKEN")

def main():
    snli = load_dataset("stanfordnlp/snli")
    validation_data = snli["validation"]
    
    id_to_label = {0:'entailment', 1:'neutral', 2:'contradiction'}
    question_template = "### Human: Classify the relationship between the following two sentences as one of the following: entailment, neutral, contradiction. "
    
    validation_instructions = [
        f'{question_template}\npremise: {x}\nhypothesis: {y}\n\n### Assistant: {id_to_label[z]}' 
        for x,y,z in zip(validation_data['premise'], validation_data['hypothesis'], validation_data['label']) 
        if z != -1
    ]
    
    ds_validation = Dataset.from_dict({"text": validation_instructions})
    
    print("Loading trained model...")
    model = AutoPeftModelForCausalLM.from_pretrained(
        checkpoint_dir,
        dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation='eager',  # KEY: Use eager attention
    )
    model = model.merge_and_unload()
    model.eval()
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    queries = [
        ds_validation['text'][i].split('### Assistant: ')[0] + '### Assistant:' 
        for i in range(len(ds_validation))
    ]
    
    print(f"Running inference on {len(queries)} validation samples...\n")
    
    results = []
    
    # Generate predictions directly without pipeline
    for i, query in enumerate(tqdm(queries, desc="Inference")):
        try:
            inputs = tokenizer(query, return_tensors='pt', padding=True).to('cuda:0')
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=3,
                    do_sample=False,  # Greedy decoding
                    temperature=None,
                    top_p=None,
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the assistant's response
            if '### Assistant:' in generated_text:
                result = generated_text.split('### Assistant:')[-1].strip()
            else:
                result = ""
            
            results.append(result)
            
        except Exception as e:
            print(f"\n⚠️  Error at sample {i}: {str(e)}")
            results.append("")
            continue
    
    # Extract ground truth labels
    labels = [label.split('### Assistant:')[1].strip() for label in ds_validation['text']]
    
    # Calculate accuracy
    correct = sum(1 for x, y in zip(results, labels) if y.strip() in x)
    accuracy = correct / len(labels)
    
    print(f"\n{'='*50}")
    print(f"✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Correct: {correct}/{len(labels)}")
    print(f"{'='*50}")
    
    print("\n=== Sample Predictions ===")
    for i in range(min(10, len(results))):
        match = "✓" if labels[i].strip() in results[i] else "✗"
        print(f"{match} Predicted: '{results[i]:<15}' | Expected: '{labels[i]}'")

if __name__ == "__main__":
    sys.exit(int(main() or 0))