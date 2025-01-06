from awq import AutoAWQForCausalLM
from awq.utils.utils import get_best_device
from transformers import AutoTokenizer, TextStreamer
from datetime import datetime
import torch

def load_model(MODEL, offload_folder: str = None):
    print("inside function")
    if get_best_device() == "cpu":
        model = AutoAWQForCausalLM.from_quantized(MODEL, use_qbits=True, fuse_layers=False,
                                                 offload_folder=offload_folder)
    else:
        model = AutoAWQForCausalLM.from_quantized(MODEL, fuse_layers=True)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    
    # Ensure EOS token is properly set
    if tokenizer.eos_token_id is None:
        # Set default EOS token ID (you might need to adjust this based on your model)
        tokenizer.eos_token = '</s>'
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    
    # Configure model's generation config
    if model.config.eos_token_id is None:
        model.config.eos_token_id = tokenizer.eos_token_id
    
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    return model, tokenizer, streamer

def chat_terminator(inp, tokenizer):
    chat = [
        {"role": "user", "content": inp},
    ]

    # Create a list of valid terminators (remove None values)
    terminators = [
        token_id for token_id in [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ] if token_id is not None
    ]
    
    # Ensure we have at least one terminator
    if not terminators:
        terminators = [tokenizer.eos_token_id]  # Use the default EOS token
    
    tokens = tokenizer.apply_chat_template(
        chat,
        return_tensors="pt"
    )
    tokens = tokens.to(get_best_device())
    return tokens, terminators

def generate_response(MODEL):
    # try:
    model, tokenizer, streamer = load_model(MODEL)
    print("Type 'exit' to end conversation")
    chat_count = 0

    while True:
        print("\n\n\n")
        question = input('Enter your question: ')
        if question.lower().strip() == 'exit':
            exit('Exiting chat. Thanks for conversation')
        else:
            start_time = datetime.now()
            question_template = question
            tokens, terminators = chat_terminator(question_template, tokenizer)
            
            generation_kwargs = {
                "streamer": streamer,
                "max_new_tokens": 64,
                "eos_token_id": terminators[0],  # Use the first valid terminator
                "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            }
            
            generation_output = model.generate(
                tokens,
                **generation_kwargs
            )
            
            generation_text = tokenizer.decode(generation_output[0], skip_special_tokens=True)
            print('ans form model', generation_text)
            end_time = datetime.now()
            print("total_execution time is :", end_time - start_time)

    # except Exception as e:
    #     print(f"An error occurred: {str(e)}")
    #     print(f"Model device: {next(model.parameters()).device}")
    #     print(f"Input tokens device: {tokens.device}")
    #     print(f"Tokenizer vocab size: {len(tokenizer)}")
    #     print(f"EOS token ID: {tokenizer.eos_token_id}")
    #     raise

if __name__ == "__main__":
    model_name = "/home/ntlpt19/LLM_training/quantize_models"
    generate_response(model_name)