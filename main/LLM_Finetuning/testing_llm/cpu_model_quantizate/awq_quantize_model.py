from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import torch
torch.set_num_threads(8)  # Adjust based on CPU

def quantize_model(config):
    model_path = config["model_path"]
    quant_path = config["quant_path"]
    safetensors_flag = eval(config["safetensors"])
    quant_config = { "zero_point": eval(config["zero_point"]), "q_group_size": int(config["q_group_size"]), "w_bit": int(config["w_bit"]), "version": config["version"] }

    # Load model
    model = AutoAWQForCausalLM.from_pretrained(model_path, safetensors=safetensors_flag, **{"low_cpu_mem_usage": True, "use_cache": False})
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Quantize
    model.quantize(tokenizer, quant_config=quant_config)

    # Save quantized model
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)

    print(f'Model is quantized and saved at "{quant_path}"')


if __name__ == "__main__":
    import configparser
    config = configparser.ConfigParser()
    config.read("config.ini")
    pat_config = config["QUANTIZATION_PTA"]
    quantize_model(pat_config)
