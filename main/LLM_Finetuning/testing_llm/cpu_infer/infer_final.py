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



def get_ocr_tesseract(image_path):
	"""
    Performs OCR (Optical Character Recognition) using Tesseract OCR engine.

    Args:
        image_path (str): Path to the image file.

    Returns:
        tuple: A tuple containing word coordinates (list of dictionaries) and all the extracted text (str).

    """
	img=None
	word_coordinates, all_text = [],""
	print("called Image OCR...", end="")
	try:
		img = Image.open(image_path)
		d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
		all_text = pytesseract.image_to_string(img)
		for i in range(len(d['text'])):
			word = d['text'][i]
			conf = float(d['conf'][i])
			if conf > 0:
				x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
				word_coordinates.append({
					"word": word,
					"confidence": conf,
					"left": x,
					"top": y,
					"width": w,
					"height": h,
					"x1": x,
					"y1": y,
					"x2": x + w,
					"y2": y + h
				})
	except Exception as e:
		print(f"exception: {e}")	
	finally:
		if hasattr(img,"close"):
			img.close()
	return word_coordinates, all_text


def extract_json_content(content):
    # Read the file content

    # Extract all JSON-like content
    json_pattern = r'\{([^{}]+)\}'
    matches = re.findall(json_pattern, content, re.DOTALL)
    
    results = []
    for match in matches:
        # Extract key-value pairs
        pair_pattern = r'"([\w_]+)":\s*"?([^",\n]+)"?'
        pairs = re.findall(pair_pattern, match)
        
        # Create a dictionary from the pairs
        result = {key: value.strip() for key, value in pairs}
        if result:  # Only add non-empty dictionaries
            results.append(result)
    
    return results



image_path = '/home/azureuser/llm_/bni_indonesia/7.png'
word_coordinates, all_text = get_ocr_tesseract(image_path)
print(all_text)
all_text_data = all_text

print('all_text_data: ', all_text_data)
prompt1_1 = f"""
You are an Expert in document key-value extraction.
Document Language: Indonesian (Bahasa Indonesia).
OCR text: 
{all_text_data}
Instructions:
- Carefully extract all identifiable key-value pairs from the provided text.
- "NIK" for Population Identification Number.
- "Gol Darah" for Blood Type, default to "null" if not provided.
- Each key should match the label found in the text (e.g., "Nama" for Name, "Tempat/Tgl Lahir" for Place/Date of Birth).
- For fields with hierarchical context, such as "PROVINSI" (Province) and "KABUPATEN" (Regency), specify the hierarchy in the JSON format.
- Include any missing values as "null" if not provided.
- "Kewarganegaraan" for Nationality.
- Ensure "Berlaku Hingga" (Valid Until) value.
  
Note: Return the result in JSON format:
"""

prompt1 = f"""
You are an Expert in document key-value extraction.
Document Language: Indonesian (Bahasa Indonesia).
OCR text: 
{all_text_data}
Instructions:
- Carefully extract all identifiable key-value pairs from the provided text.
- Each key should match the label found in the text (e.g., "Nama" for Name, "Tempat/Tgl Lahir" for Place/Date of Birth).
- For fields with hierarchical context, such as "PROVINSI" (Province) and "KABUPATEN" (Regency), specify the hierarchy in the JSON format.
- Include any missing values as "null" if not provided.
- Ensure "Berlaku Hingga" (Valid Until) has the value "SEUMUR HIDUP" if present.
  
Note: Return the result in JSON format:
"""

# def load_model(model_path):
# model = Llama(model_path="/datadrive/repo_code/llama-3.1-8b-instruct-q4_k_m.gguf",
#               verbose=True, n_threads=None, n_ctx=800, cache=cache, seed=42)

model_path1 = '/home/ntlpt19/LLM_training/quantize_models/Qwen2.5-0.5B-200K.F16.gguf'
model_path2 = "/home/ntlpt19/LLM_training/quantize_models/Qwen2.5-0.5B-200K.Q5_K_M.gguf"
# model_path2 = '/home/ntlpt19/LLM_training/quantize_models/gemma-2-2b-it-Q3_K_L.gguf'
model_path2 = '/home/azureuser/llm_/Qwen2.5-0.5B-200K.F16.gguf'
model_path2 ='/home/azureuser/llm_/quantize_models/Qwen2.5-0.5B-200K.Q5_K_M.gguf'


def run_model():
    model_path2 = '/home/azureuser/llm_/quantize_models/quantize_models/Qwen2.5-0.5B-200K.Q4_K_M.gguf'
    # model_path2 = '/home/azureuser/llm_/quantize_models/Qwen2.5-0.5B-200K.Q5_K_M.gguf'
    # model_path2 = '/home/azureuser/llm_/quantize_models/quantize_models/Qwen2.5-0.5B-200K.F16.gguf' #16bit
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
        final_result = extract_json_content(result)
        print("%%%%%%%%%%%%%%%%%%%%%%")
        print(final_result)
        print('TIME TAKEN :', elapsed_time)
        
    finally:
        if 'model' in locals():
            model.close()
            del model
            gc.collect()

if __name__ == "__main__":
    run_model()
    exit('OK')

