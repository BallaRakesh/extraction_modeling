from transformers import BitsAndBytesConfig

from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import prepare_model_for_kbit_training
# from peft import LoraConfig, PeftModel, get_peft_model
import torch


model_name = "Qwen/Qwen2.5-0.5B"

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


def save_model(output_dir:str, model: object, tokenizer: object):
    # Save the tokenizer and model to the specified directory
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    print(f"Model saved sucessfully at {output_dir}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=model_quantization())
model_path = '/datadrive/rakesh/quantize_models'
save_model(model_path, model, tokenizer)