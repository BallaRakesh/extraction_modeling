import torch
from transformers import pipeline
import transformers
import torch
from huggingface_hub import login
# login("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")
login("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")
# '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'
#  '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'
model_id = "meta-llama/Meta-Llama-3.1-3B-Instruct"
hf_token = "https://llama3-2-lightweight.llamameta.net/*?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoib3ZyYzdjdWRlMHc5amVhcGhzZ2N5em04IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvbGxhbWEzLTItbGlnaHR3ZWlnaHQubGxhbWFtZXRhLm5ldFwvKiIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcyNzUwMzUwNH19fV19&Signature=t9GQKVn3ES8HG5SxS4oN5UU-Yrdy2Hef0Qm4JTzSjmpujt8B%7ExDGuRANlCWaG2zrTO68rMmjvyZUZgVRtkpqnaBMqNtUNmA%7EMqISgEZON8JjnGk%7E0C2pYj0yi-b8DQiUVS6tVOacPnJOPeg5BEh4z1JKsLW1DvCIEH0WRxR30Fhlu1up%7EEW2wunhXEqRRWWpWf03fYybp%7ECT1H3oAV8LszV9gaRKhyN7KcNz-4Xv-07igDhRyp6gR3NX2XaPyGC5ejRxrM-w5AX0vMwW7bP4JJbHWu-MK4MCEfuTyZ5CJKkhw5sDjvDSzWX4VaP0vrw0iqX3PBuFU302lWmJuGQstw__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=851232713869796"  # Your Hugging Face access token

import torch
from transformers import pipeline

# Correct model identifier from Hugging Face
model_id = "meta-llama/Meta-Llama-3.1-3B-Instruct"

# Set up the pipeline using the correct token
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",  # Automatically chooses the best device (e.g., CUDA if available)
    use_auth_token=hf_token  # Use the correct Hugging Face token here
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

# Generate response
outputs = pipe(
    messages,
    max_new_tokens=256,
)

# Print the generated text
print(outputs[0]["generated_text"])

exit('OK')
model_id = "meta-llama/Llama-3.2-3B-Instruct"
# model_id = "Llama3.2-3B-Instruct"
hf_token = "https://llama3-2-lightweight.llamameta.net/*?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoib3ZyYzdjdWRlMHc5amVhcGhzZ2N5em04IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvbGxhbWEzLTItbGlnaHR3ZWlnaHQubGxhbWFtZXRhLm5ldFwvKiIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcyNzUwMzUwNH19fV19&Signature=t9GQKVn3ES8HG5SxS4oN5UU-Yrdy2Hef0Qm4JTzSjmpujt8B%7ExDGuRANlCWaG2zrTO68rMmjvyZUZgVRtkpqnaBMqNtUNmA%7EMqISgEZON8JjnGk%7E0C2pYj0yi-b8DQiUVS6tVOacPnJOPeg5BEh4z1JKsLW1DvCIEH0WRxR30Fhlu1up%7EEW2wunhXEqRRWWpWf03fYybp%7ECT1H3oAV8LszV9gaRKhyN7KcNz-4Xv-07igDhRyp6gR3NX2XaPyGC5ejRxrM-w5AX0vMwW7bP4JJbHWu-MK4MCEfuTyZ5CJKkhw5sDjvDSzWX4VaP0vrw0iqX3PBuFU302lWmJuGQstw__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=851232713869796"  # Your Hugging Face access token

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_auth_token=hf_token  # Pass your token here
)
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]
outputs = pipe(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])

