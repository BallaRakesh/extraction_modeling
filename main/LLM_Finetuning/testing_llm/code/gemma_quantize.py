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



image_path = '/home/azureuser/llm_/bni_indonesia/1.png'
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




# Model path
# model_path2 = '/home/azureuser/llm_/quantize_models/quantize_models/gemma-2-2b-it-Q3_K_L.gguf' #14.2 - 16.06 sec
model_path2 = '/home/azureuser/llm_/quantize_models/quantize_models/gemma-2-2b-it-Q4_K_S.gguf' #14.2 - 16.06 sec
model_path2 = '/home/azureuser/llm_/quantize_models/quantize_models/gemma-1.1-2b-it.Q2_K.gguf' #10 sec
# Initialize model
model = Llama(
    model_path=model_path2,
    verbose=True,
    n_threads=None,
    n_ctx=800,
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
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
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
    
    result = generate_response(prompt1_1)
    
    if result:
        print('Prediction:', result['response'])
        print('Time taken:', result['time_taken'])
    
    model.reset()