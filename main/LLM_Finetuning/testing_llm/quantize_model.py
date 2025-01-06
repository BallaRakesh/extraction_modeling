import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# Define a helper function for model quantization
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
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype
    )
    return bnb_config

# Define different quantization configurations
quantization_variants = [
    {"bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16},
    {"bnb_4bit_quant_type": "fp4", "bnb_4bit_compute_dtype": torch.float16},
    {"bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.float32},
    {"bnb_4bit_quant_type": "fp4", "bnb_4bit_compute_dtype": torch.bfloat16}
]

# Load and save each quantized model
model_name = "Qwen/Qwen2.5-0.5B"
path_saving = '/home/ntlpt19/LLM_training/quantize_models'
for idx, config in enumerate(quantization_variants):
    quant_config = model_quantization(
        bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=config["bnb_4bit_compute_dtype"]
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config)
    
    # Define a filename for saving each variant
    save_path = f"{path_saving}/{model_name}_quantized_variant_{idx}.bin"
    model.save_pretrained(save_path)
    print(f"Saved quantized model variant {idx} to {save_path}")
