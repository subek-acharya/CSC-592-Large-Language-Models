import sys

from langchain_community.llms import HuggingFacePipeline
import math

import transformers
import torch
import warnings
warnings.filterwarnings('ignore')
from transformers import AutoTokenizer, AutoModelForCausalLM

def summary_generator(text, pipeline):
    llm_pipeline = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature':0})
    summary_prompt = text
    return llm_pipeline.invoke(summary_prompt)

def main():
    load_dotenv()  # Load from .env file
    token = os.getenv("HF_TOKEN")

    #model_name = "meta-llama/Llama-2-7b-chat-hf"
    model_name = "meta-llama/Meta-Llama-3.1-8B"
    # model_name = "microsoft/Phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name,token=token)
    model = AutoModelForCausalLM.from_pretrained(model_name,token=token).cuda()

    pipeline=transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device=0,  # "auto"  >=0 to run on GPU, -1 for CPU
    max_length=500,
    truncation=True,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
    )
    context="""
    summarize the following text between <para> and <endpara>, produce output starting with <summary> and ending with <endsummary>: 
    <para>The purpose of the DataSet object is to contain the training and test data. If the total training data is 
small, we can read all the image pixels and store all images in multi-dimensional tensors in the dataset. 
However, if the number of images is large and images happen to be relatively high resolution, then one 
option is to store the image filenames (with full path info) in the dataset. We can also apply a transform to the 
images. Transform usually resizes the image and changes the 0-255 pixel scale to either 0 to 1, or -1 to 1.<endpara>
    """
    print(context)
    print(summary_generator(context, pipeline))

if __name__ == "__main__":
    sys.exit(int(main() or 0))
