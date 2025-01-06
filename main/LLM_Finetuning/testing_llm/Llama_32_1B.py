import torch
from transformers import pipeline

model_id = "meta-llama/Llama-3.2-1B-Instruct"
model_id = "/home/ntlpt19/.cache/huggingface/hub/models--MBZUAI--MobiLlama-05B"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]
outputs = pipe(
    messages,
    max_new_tokens=256,
)
print('>>>>>>>>>>>>>')
print('>>>>>>>>>>>>>')
print('>>>>>>>>>>>>>')
print(outputs[0]["generated_text"][-1])
