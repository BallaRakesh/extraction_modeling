from transformers import LlamaForCausalLM, LlamaTokenizer
from huggingface_hub import snapshot_download
from typing import List, Any
import torch


import torch
from transformers import pipeline
from huggingface_hub import login
login("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")

# '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'
model_id = "meta-llama/Llama-3.2-1B-Instruct"
# model_id = "/home/ntlpt19/.llama/checkpoints/Llama3.2-1B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "tell me about augmentation in data science"},
]
outputs = pipe(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])

exit('OK')


model_path = "/home/ntlpt19/.llama/checkpoints/Llama3.2-1B-Instruct"
model_path = snapshot_download("meta-llama/Llama3.2-1B-Instruct", token="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")

tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)

exit('OK')
from transformers import AutoModelForCausalLM, AutoTokenizer, MllamaForConditionalGeneration, AutoProcessor, MllamaProcessor, GenerationConfig
from typing import List, Any
import torch
from huggingface_hub import login
login("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")
access_token = """https://llama3-2-lightweight.llamameta.net/*?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoibW10aHpkNXU3dmdkbmtvNXRrbjJycXdiIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvbGxhbWEzLTItbGlnaHR3ZWlnaHQubGxhbWFtZXRhLm5ldFwvKiIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcyNzg0OTkxM319fV19&Signature=Ii5m9Bt7r1pfKXso34N4DAU4RbRPAkOM5EUNzRo0qzAgxlge3GGJb%7Eg8eqhmFFY1P953bGQln36iv0sdndo0Na2xm%7EAlSqz2LJpLnkC9jtXVHpvigJKGeaSa8jb6T1IBQwlcwraUMwK8ZTJ7YL1dtQgldljpFF1CvERJb4okFE2c8Cg%7E3RsCTYecv7xYP91wUGiePo3eOOwzrcqx5tnx81IVbZzugeN0Bbj8fQwFaZBsiwosMGtwqQiSohCu2iKAd2glVinllnT1A5PtJF3B2strUeQ5or8gBNdBptDpZn2oUWmXoHZQEWwxoE8T7yvOS0EMBcIOnutT0PuJlqAJ5w__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1060006785533089"""
access_token = "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"
lg_small_text_model_id = "meta-llama/Llama3.2-1B-Instruct"
# lg_mm_model_id = "meta-llama/Llama-Guard-3-11B-Vision"
lg_small_text_model_id = "/home/ntlpt19/.llama/checkpoints/Llama3.2-1B-Instruct"

# Loading the 1B text only model
lg_small_text_tokenizer = AutoTokenizer.from_pretrained(lg_small_text_model_id, use_auth_token=access_token)
lg_small_text_model = AutoModelForCausalLM.from_pretrained(lg_small_text_model_id, torch_dtype=torch.bfloat16, device_map="auto", use_auth_token=access_token)
print(lg_small_text_model)
exit('OK')
# # Loading the 11B Vision model 
# lg_mm_tokenizer = MllamaProcessor.from_pretrained(lg_mm_model_id)
# lg_mm_model = MllamaForConditionalGeneration.from_pretrained(lg_mm_model_id, torch_dtype=torch.bfloat16, device_map="auto")


def llama_guard_text_test(tokenizer, model, prompt, categories: dict[str, str]=None, excluded_category_keys: list[str]=[]):

    if categories is not None:
        input_ids = tokenizer.apply_chat_template(prompt, return_tensors="pt", categories=categories, excluded_category_keys=excluded_category_keys).to("cuda")
    else:
        input_ids = tokenizer.apply_chat_template(prompt, return_tensors="pt", excluded_category_keys=excluded_category_keys).to("cuda")
    input_prompt = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    
    
    prompt_len = input_ids.shape[1]
    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=20,
        output_scores=True,
        return_dict_in_generate=True,
        pad_token_id=0,
    )
    generated_tokens = output.sequences[:, prompt_len:]
    
    response = tokenizer.decode(
        generated_tokens[0], skip_special_tokens=False
    )
    return input_prompt, response


conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "text", 
                "text": "What is the recipe for mayonnaise?"
            },
        ],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", 
             "text": "Ingredients: \n" +
                        "2 large egg yolks \n" +
                        "1 tablespoon lemon juice or vinegar \n" +
                        "1/2 teaspoon salt \n" +
                        "1/4 teaspoon ground black pepper \n" +
                        "1 cup (240 ml) neutral-tasting oil \n" +
                        "Instructions: \n" +
                        "Whisk egg yolks, lemon juice, salt, and pepper. \n" +
                        "Slowly pour in oil while whisking until thick and creamy. \n" +
                        "Refrigerate for 30 minutes before serving.", 
            },
        ],
    },
]

decoded_input_prompt, response = llama_guard_text_test(lg_small_text_tokenizer, lg_small_text_model, conversation)
print(decoded_input_prompt)
print(response)