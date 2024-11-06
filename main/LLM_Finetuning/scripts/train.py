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
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter

from huggingface_hub import login
login("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")

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


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_model(config):
    model_path = config['paths']['model_path']
    model_name = config['model']['model_name']
    print(f"model path: {model_path}")
    print(f"model name: {model_name}")
    if not os.listdir(model_path):
        print("Folder is empty, download and load the model!")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=model_quantization())
        save_model(model_path, model, tokenizer)
    else:
        print("Folder is not empty, directly load the language model")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
    return model, tokenizer

def load_json_datasets(train_file_path, test_file_path, tokenizer):
    train_data = load_dataset('json', data_files=train_file_path, split="train")
    test_data = load_dataset('json', data_files=test_file_path, split="train")
    
    train_data = train_data.map(lambda samples: tokenizer(samples["text"]), batched=True)
    test_data = test_data.map(lambda samples: tokenizer(samples["text"]), batched=True)

    return {'train': train_data, 'test': test_data}

def enable_gradient_checkpointing_and_prepare_model(model):
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    return model

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit  # Assuming 4-bit is the default
    lora_module_names = set()

    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')  # Remove lm_head if present
    
    return list(lora_module_names)

def configure_peft_model(model, modules, config):
    lora_config = LoraConfig(
        r=config['peft']['r'],
        lora_alpha=config['peft']['lora_alpha'],
        target_modules=modules,
        lora_dropout=config['peft']['lora_dropout'],
        bias=config['peft']['bias'],
        task_type=config['peft']['task_type']
    )

    return get_peft_model(model, lora_config), lora_config

from transformers import TrainerCallback, TrainingArguments

class VisualizationCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.steps = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if 'loss' in logs:
            self.train_losses.append(logs['loss'])
            self.steps.append(state.global_step)
        if 'eval_loss' in logs:
            self.eval_losses.append(logs['eval_loss'])

    def plot_losses(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.steps, self.train_losses, label='Training Loss')
        if self.eval_losses:
            eval_steps = [step for step in self.steps if step % args.eval_steps == 0]
            plt.plot(eval_steps, self.eval_losses, label='Evaluation Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training and Evaluation Loss')
        plt.legend()
        plt.savefig('loss_plot.png')
        plt.close()


def _train(config):
    model_path = config['paths']['model_path']
    root_path = config['paths']['root_path']
    train_path = os.path.join(root_path, config['training']['train_path'])
    test_path = os.path.join(root_path, config['training']['test_path'])
    output_dir = config['paths']['output_dir']
    model, tokenizer = load_model(config)
    dataset = load_json_datasets(train_path, test_path, tokenizer)
    train_data = dataset["train"]
    test_data = dataset["test"]
    model = enable_gradient_checkpointing_and_prepare_model(model)
    modules = find_all_linear_names(model)
    model, lora_config = configure_peft_model(model, modules, config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable: {trainable} | total: {total} | Percentage: {trainable/total*100:.4f}%")
    tokenizer.pad_token = tokenizer.eos_token
    torch.cuda.empty_cache()

    visualization_callback = VisualizationCallback()
    
    class TensorBoardCallback(TrainerCallback):
        def __init__(self, writer):
            self.writer = writer
        
        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.is_world_process_zero:
                for k, v in logs.items():
                    if isinstance(v, (int, float)):
                        self.writer.add_scalar(k, v, state.global_step)
                        print(f"Logging {k} at step {state.global_step}: {v}")

    writer = SummaryWriter(log_dir='./logs')
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=test_data,
        dataset_text_field="text",
        peft_config=lora_config,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=config['training']['per_device_train_batch_size'],
            gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
            warmup_steps = float(config['training']['warmup_steps']),
            max_steps = int(config['training']['max_steps']),
            learning_rate = float(config['training']['learning_rate']),
            logging_steps = int(config['training']['logging_steps']),
            save_steps = int(config['training']['save_steps']),
            output_dir=output_dir,
            optim=config['training']['optim'],
            save_strategy=config['training']['save_strategy'],
            evaluation_strategy=config['training']['evaluation_strategy'],
            eval_steps=100,
            logging_dir='./logs', 
            report_to=["tensorboard"] 

        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[TensorBoardCallback(writer)]
        # callbacks=[visualization_callback] 

    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    return trainer

def load_base_model(config):
    base_model_path = config['paths']['base_model_path']
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map={"": 0},
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, add_eos_token=True)
    return base_model, tokenizer

def merge_models(base_model, new_model):
    merged_model = PeftModel.from_pretrained(base_model, new_model)
    merged_model = merged_model.merge_and_unload()
    return merged_model

def save_merged_model(merged_model, tokenizer, output_dir):
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

if __name__ == "__main__":
    config = load_config('../config/config.yaml')
    
    trainer = _train(config)
    print("Training begins...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    
    trainer.train()
    new_model = config['model']['new_model_name']
    trainer.model.save_pretrained(new_model)
    
    base_model, tokenizer = load_base_model(config)
    merged_model = merge_models(base_model, new_model)
    merged_model_dir = config['paths']['merged_model_path']

    save_merged_model(merged_model, tokenizer, merged_model_dir)
