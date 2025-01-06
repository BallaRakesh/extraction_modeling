import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Path to your quantized model directory
model_path = "/datadrive/rakesh/quantize_models/Qwen"

# Load the model configuration (if config.json is available)
try:
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
except OSError as e:
    print(f"Error loading model: {e}")
    # If model fails to load, handle it here
    exit()

# Define input text
input_text = "Once upon a time,"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# Run inference
with torch.no_grad():  # Disable gradient calculation for inference
    outputs = model.generate(input_ids, max_length=50)

# Decode and print the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
