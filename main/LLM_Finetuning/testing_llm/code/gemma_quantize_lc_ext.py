from dotenv import load_dotenv
from llama_cpp import Llama, LlamaDiskCache
import time
import json  
import time
import pytesseract
from dotenv import load_dotenv
from collections import defaultdict
# from datetime import datetime
from llama_cpp import Llama, LlamaDiskCache
import pandas as pd
import json
from time import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import json  
import time
import pytesseract
from PIL import Image
from contextlib import contextmanager
import time
from llama_cpp import Llama, LlamaDiskCache
import gc
import re
# Load environment variables
load_dotenv()

# Initialize cache
cache = LlamaDiskCache(cache_dir="dex_llama_cache")


# image_path = '/home/azureuser/llm_/bni_indonesia/1.png'
# word_coordinates, all_text = get_ocr_tesseract(image_path)
ocr_file = '/home/ntlpt19/Downloads/lc 3/OCR/20886202_29_all_text.txt'
with open(ocr_file, 'r') as file:
    all_text = file.read()
    
print('>>>>>>>>>>>>>>>>>>.')
print('>>>>>>>>>>>>>>>>>>.')
print('>>>>>>>>>>>>>>>>>>.')
print(all_text)


all_text_data = all_text

print('all_text_data: ', all_text_data)


prompt_main1 = f""" You are an expert in extracting key-value pairs from OCR text. Your task is to analyze and extract all identifiable key-value pairs embedded between colons (e.g., `:46B:`, `:71D:`) from the provided Letter of Credit (LC) text. 

OCR text:
{all_text_data}

### Instructions:
1. Carefully analyze the OCR text to identify all instances of keys embedded between colons (e.g., `:46B:`, `:71D:`).
2. Recognize common field labels such as 
        `:52A:`,
        `:31C:`,
        `:26E:`,
        `:30:`,
        `:22A:`,
        `:45B:`,
        `:46B:`,
        `:71D:`, 
        `:72Z:`,
        Swift Created Date,
        Sender Code,
        Receiver BIC, 
        `:27:`,
        `:40A:`, 
        `:20:`, 
        `:40E:`, 
        `:31D:`
3. For each identified key, extract its corresponding value, which may span multiple lines until another key is encountered or the text ends.
4. Ensure the extracted key-value pairs are structured in valid JSON format.

### Output format:
The result must be in this JSON format:
{{
    ":52A:": "value1",
    ":31C:": "value2",
}}
"""

prompt_main = f""" You are an expert in extracting key-value pairs from OCR text. 
Your task is to analyze and extract all identifiable key-value pairs  from the provided Letter of Credit (LC) text. 

OCR text:
{all_text_data}

### Instructions:
1. Carefully analyze the OCR text to identify all instances of keys embedded between colons.
2. For each identified key, extract its corresponding value, which may span multiple lines until another key is encountered or the text ends.
3. Ensure the extracted key-value pairs are structured in valid JSON format.
4. give the direct responce, no need the python logic to extract the key-value pairs.
### Output format:
The result must be in this JSON format:
{{
    ":52A:": "value1",
    ":31C:": "value2",
}}
"""




# Model path
# model_path2 = '/home/azureuser/llm_/quantize_models/quantize_models/gemma-2-2b-it-Q3_K_L.gguf' #14.2 - 16.06 sec
# model_path2 = '/home/azureuser/llm_/quantize_models/quantize_models/gemma-2-2b-it-Q4_K_S.gguf' #14.2 - 16.06 sec
# model_path2 = '/home/azureuser/llm_/quantize_models/quantize_models/gemma-1.1-2b-it.Q2_K.gguf' #10 sec
model_path2 = '/media/ntlpt19/5250315B5031474F/quantize_models/gemma-2-2b-it-Q4_K_S.gguf'
model_path2 = '/media/ntlpt19/5250315B5031474F/quantize_models/Llama-3.2-3B-Instruct-Q4_K_M.gguf'
# Initialize model
model = Llama(
    model_path=model_path2,
    verbose=True,
    n_threads=None,
    n_ctx=1014,
    cache=cache,
    seed=63,
    chat_format="gemma"  # Specify the chat format as "gemma"
)
#top_k=40, top_p=0.7, temperature=0.5
def generate_response(prompt, temperature=0.5, top_k=40, top_p=0.7):
    start_time = time.time()
    
    # For Gemma, we'll use a simpler message format
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    try:
        output = model.create_chat_completion(
            messages=messages,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        
        response = output['choices'][0]['message']['content']
        print('prediction :')
        print('#############')
        print('#############')
        print(response)
        print('#############')
        print('#############')
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('elapsed_time:', elapsed_time)
        return {
            'response': response,
            'time_taken': elapsed_time
        }
    
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return None

# Example usage
if __name__ == "__main__":
    # prompt1_1 = "What is machine learning?"  # Replace with your actual prompt
    
    result = generate_response(prompt_main)
    
    if result:
        print('Prediction:', result['response'])
        print('Time taken:', result['time_taken'])
    
    model.reset()