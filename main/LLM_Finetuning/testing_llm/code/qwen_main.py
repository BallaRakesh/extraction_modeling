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

import re
import re

def extract_json_content(content):
    # Define a pattern to match JSON-like objects
    json_pattern = r'\{([^{}]+)\}'
    matches = re.findall(json_pattern, content, re.DOTALL)
    
    results = []
    for match in matches:
        # Pattern to match key-value pairs
        # Supports keys with spaces, slashes, periods, and values in quotes or unquoted
        pair_pattern = r'"([\w\s\/.]+)":\s*"([^"]+)"|(\S+)'
        
        # Extract all key-value pairs from the current match
        pairs = re.findall(pair_pattern, match)
        
        # Create a dictionary from the pairs, excluding empty keys
        result = {}
        for key, quoted_val, unquoted_val in pairs:
            if key:  # Ensure the key is not empty
                value = (quoted_val or unquoted_val).strip()  # Use quoted if available, else unquoted
                result[key.strip()] = value
        
        # Add non-empty dictionaries only
        if result:
            results.append(result)
    
    return results




image_path = '/home/ntlpt19/TF_testing_EXT/dummy_responces/bni_indonesia/1.png' #prompt1
# image_path = '/home/ntlpt19/TF_testing_EXT/dummy_responces/bni_indonesia/4.jpg' #prompt1_1
image_path = '/home/ntlpt19/TF_testing_EXT/dummy_responces/bni_indonesia/2.png'

word_coordinates, all_text = get_ocr_tesseract(image_path)
print('>>>>>>>>>>>>>>>>>>.')
print('>>>>>>>>>>>>>>>>>>.')
print('>>>>>>>>>>>>>>>>>>.')
print(all_text)
# exit('OK')

all_text_data = all_text

print('all_text_data: ', all_text_data)


prompt_main = f""" You are an expert in document key-value extraction. Document Language: Indonesian (Bahasa Indonesia). 
OCR text: {all_text_data}

Instructions:
- Carefully extract all identifiable key-value pairs from the provided Indonesian national ID card (KTP) text.
- Recognize common field labels such as "NIK", "Nama", "Tempat/Tgl Lahir", "Jenis Kelamin", "Gol. Darah", "Alamat", "RT/RW", "Kel/Desa", "Kecamatan", "Agama", "Status Perkawinan", "Pekerjaan", "Berlaku Hingga", and "Kewarganegaraan".
- Handle variations in formatting, capitalization, and spacing.
- Use "null" for any missing or incomplete values.
- Normalize the data, such as converting dates to "YYYY-MM-DD" format.
- Ensure the output is in valid JSON format.

Note: Return the result in JSON format.
"""



def run_model():
    model_path2 = '/media/ntlpt19/5250315B5031474F/quantize_models/qwen2.5-0.5b-instruct-q2_k.gguf'     #2bit ##n_ctx=1000 #4.jpg = prompt1_1, 1.png = prompt1  #TIME FRIENDLY
    try:
        model = Llama(model_path=model_path2,
                     verbose=True, n_threads=None, n_ctx=1000, cache=cache, seed=63)
        
        start_time = time.time()
        
        output = model.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are assistant that helps answer questions."},
                {"role": "user", "content": prompt_main}
            ],
            top_k=40, top_p=0.7, temperature=0.5
        )
        
        result = output['choices'][0]['message']['content']
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('TIME TAKEN :', elapsed_time)
        
        print('prediction :')
        print('#############')
        print('#############')
        print(result)
        final_result = extract_json_content(result)
        print("%%%%%%%%%%%%%%%%%%%%%%")
        print(final_result)
        
    finally:
        if 'model' in locals():
            model.close()
            del model
            gc.collect()

if __name__ == "__main__":
    run_model()

