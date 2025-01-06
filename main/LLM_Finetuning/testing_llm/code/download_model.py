import os
import requests
import time

from tqdm import tqdm
import os
import requests

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
cache = LlamaDiskCache(
    cache_dir="dex_llama_cache")




def download_model(model_name, save_path, download_url):
    """
    Downloads the model file if it does not exist locally and shows download progress with tqdm.
    """
    if not os.path.exists(save_path):
        print(f"Model {model_name} not found locally. Downloading from {download_url}...")
        
        # Make a request to get the file
        response = requests.get(download_url, stream=True)
        if response.status_code == 200:
            # Get total file size from headers
            total_size = int(response.headers.get('Content-Length', 0))
            
            # Create a tqdm progress bar
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=model_name, ascii=True) as pbar:
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))  # Update progress bar
            
            print(f"\nModel {model_name} downloaded successfully to {save_path}.")
        else:
            raise Exception(f"Failed to download the model: {response.status_code} {response.reason}")
    else:
        print(f"Model {model_name} already exists at {save_path}.")

prompt1 = 'what is quantization'
def run_model():
    model_name = 'Qwen2.5-0.5B-200K.Q4_K_M.gguf'
    model_path2 = '/home/ntlpt19/TF_testing_EXT/dummy_responces/bni_indonesia/testing'
    download_url = 'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q2_k.gguf?download=true'
    
    # Download the model if it doesn't exist
    download_model(model_name, model_path2, download_url)
    # Load and run the model
    try:
        model = Llama(model_path=model_path2,
                      verbose=True, n_threads=None, n_ctx=550, cache=cache, seed=63)
        
        start_time = time.time()
        
        output = model.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are assistant that helps answer questions."},
                {"role": "user", "content": prompt1}
            ],
            top_k=40, top_p=0.7, temperature=0.5
        )
        
        result = output['choices'][0]['message']['content']
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('prediction :')
        print('#############')
        print('#############')
        print(result)
        # final_result = extract_json_content(result)
        print("%%%%%%%%%%%%%%%%%%%%%%")
        # print(final_result)
        print('TIME TAKEN :', elapsed_time)
    except Exception as e:
        print(f"Error running the model: {e}")

run_model()