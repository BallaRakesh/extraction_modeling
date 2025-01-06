import torch
from transformers import AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-0.5B"
model = AutoModelForCausalLM.from_pretrained(model_name)

# Quantize only linear layers in the model
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        model._modules[name] = torch.quantization.quantize_dynamic(
            module, {torch.nn.Linear}, dtype=torch.qint8
        )

# Save the model
model.save_pretrained("qwen_quantized_cpu")
