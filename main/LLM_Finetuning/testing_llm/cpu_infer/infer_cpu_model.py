from awq import AutoAWQForCausalLM
from awq.utils.utils import get_best_device
from transformers import AutoTokenizer, TextStreamer
from datetime import datetime

def load_model(MODEL, offload_folder: str = None):
    print("inside function")
    # inp = str(user_query) + str(ocr_input)
    if get_best_device() == "cpu":
        model = AutoAWQForCausalLM.from_quantized(MODEL, use_qbits=True, fuse_layers=False,
                                                  offload_folder=offload_folder)
    else:
        model = AutoAWQForCausalLM.from_quantized(MODEL, fuse_layers=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    return model, tokenizer, streamer


def chat_terminator(inp, tokenizer):
    chat = [
        {"role": "system", "content": "You are a concise assistant that helps answer questions."},
        {"role": "user", "content": inp},
    ]

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    tokens = tokenizer.apply_chat_template(
        chat,
        return_tensors="pt"
    )
    tokens = tokens.to(get_best_device())
    return tokens, terminators


def generate_response(MODEL):
    # Get the llm prompt.
    model, tokenizer, streamer = load_model(MODEL)
    print("Type 'exit' to end conversation")
    # counter to draw and save updated kg-graph after every chat.
    chat_count = 0

    while True:
        print("\n\n\n")
        question = input('Enter your question: ')
        if question.lower().strip() == 'exit':
            exit(f'Exiting chat. Thanks for conversation')
        else:
            start_time = datetime.now()
            question_template = question
            tokens, terminators = chat_terminator(question_template, tokenizer)
            generation_output = model.generate(
                tokens,
                streamer=streamer,
                max_new_tokens=64,
                eos_token_id=terminators
            )
            print(generation_output)
            exit('OK')
            generation_output = tokenizer.decode(generation_output[0], skip_special_tokens=True)
            end_time = datetime.now()

            print("total_execution time is :", end_time - start_time)


if __name__ == "__main__":
    # model_name = "mistralai/Mistral-7B-Instruct-v0.2"  # Replace with the correct model name if different
    # model_name = "01-ai/Yi-1.5-9B"  # Replace with the correct model name if different
    # model_name = "Qwen/Qwen2-7B"

    # model_name = "casperhansen/llama-3-8b-instruct-awq"
    # model_name = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
    model_name = "/home/ntlpt19/LLM_training/quantize_models"

    generate_response(model_name)
