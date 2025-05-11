# from transformers import LlamaTokenizer, LlamaForCausalLM

# # tokenizer = LlamaTokenizer.from_pretrained('ChanceFocus/finma-7b-full')
# tokenizer = LlamaTokenizer.from_pretrained('TheFinAI/finma-7b-full')
# # model = LlamaForCausalLM.from_pretrained('ChanceFocus/finma-7b-full', device_map='auto')
# model = LlamaForCausalLM.from_pretrained('TheFinAI/finma-7b-full', device_map='auto')



from transformers import LlamaTokenizer, LlamaForCausalLM
import os

# Define the local directory where you want to save the model and tokenizer
local_dir = "/media/ntlpt19/5250315B5031474F/temp_llm_download"

# Ensure the directory exists
os.makedirs(local_dir, exist_ok=True)

# Load the tokenizer from Hugging Face and save it locally
tokenizer = LlamaTokenizer.from_pretrained('TheFinAI/finma-7b-full')
tokenizer.save_pretrained(local_dir)

# Load the model from Hugging Face and save it locally
model = LlamaForCausalLM.from_pretrained('TheFinAI/finma-7b-full', device_map='auto')
model.save_pretrained(local_dir)

print(f"Model and tokenizer downloaded and saved to {local_dir}")



exit('OKKKKKKKKK')


from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

model_name = "tiiuae/falcon-7b-instruct"
custom_path = "/media/ntlpt19/5250315B5031474F/temp_llm_download"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=custom_path)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True, 
    device_map="auto",
    cache_dir=custom_path
)

text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

print("Model downloaded to:", custom_path)
