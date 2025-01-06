import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
from datasets import load_dataset
from datasets import Dataset
import bitsandbytes as bnb
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, PeftModel, get_peft_model
import transformers
import os
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import yaml
from transformers import BitsAndBytesConfig
from huggingface_hub import login
# from tensorboardX import SummaryWriter



def model_quantization(load_in_4bit=True, 
                        bnb_4bit_quant_type="nf4", 
                        bnb_4bit_compute_dtype=torch.bfloat16):
    """
    Create a BitsAndBytesConfig object for model quantization.

    Args:
        load_in_4bit (bool, optional): Whether to load the model weights in 4-bit format. Default is True.
        bnb_4bit_quant_type (str, optional): The quantization type for 4-bit quantization. Default is "nf4".
        bnb_4bit_compute_dtype (torch.dtype, optional): The data type for computations during inference. Default is torch.bfloat16.

    Returns:
        BitsAndBytesConfig: Configuration object for model quantization.
    """
    bnb_config= BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype
    )
    return bnb_config





device = "cpu"# "cuda" # or "cpu"
model_path = "ibm-granite/granite-34b-code-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
# drop device_map if running on CPU
# model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=model_quantization(), device_map=device)
model.eval()
# change input text as desired
chat = [
    { "role": "user", "content": "Write a code to find the maximum value in a list of numbers." },
]
chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
# tokenize the text
input_tokens = tokenizer(chat, return_tensors="pt")
# transfer tokenized inputs to the device
for i in input_tokens:
    input_tokens[i] = input_tokens[i].to(device)
# generate output tokens
output = model.generate(**input_tokens, max_new_tokens=100)
# decode output tokens into text
output = tokenizer.batch_decode(output)
# loop over the batch to print, in this example the batch size is 1
for i in output:
    print(i)
