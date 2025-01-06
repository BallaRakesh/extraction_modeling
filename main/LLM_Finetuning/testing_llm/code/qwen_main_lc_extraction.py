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

# word_coordinates, all_text = get_ocr_tesseract(image_path)
# ocr_file = '/home/ntlpt19/Downloads/lc 3/OCR/20886202_29_all_text.txt'
ocr_file = '/home/ntlpt19/Desktop/TF_release/augmentation_llm/testing_data/outputs/chunk_wise_data/20886202_29..txt'
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

prompt_main2 = f""" You are an expert in extracting key-value pairs from OCR text. 
Your task is to extract only the values associated with the `:45B:` (Description of Goods and/or Services) and `:46B:` (Documents Required) keys from the provided Letter of Credit (LC) text. 

OCR text:
{all_text_data}
### Instructions:
1. Carefully analyze the OCR text to locate the keys `:45B:` and `:46B:`.
2. The values of these keys may span multiple lines and often start with specific keywords such as `/REPALL/` or `/ADD/`. Ensure these keywords are included in the extracted value.
3. Extract the entire value for each key, continuing until another key is encountered or the text ends.
4. Ignore all other keys and their values.
5. Ensure the extracted key-value pairs are structured in valid JSON format as follows:
   ```json
   {{
       "45B": "value for :45B: here",
       "46B": "value for :46B: here"
   }}
"""

prompt_main = f""" You are an expert in extracting the chunks for a required key.

I have a set of chunks from a document, and I need to extract specific chunks related to the following references:
1. **47A**: The value for `47A` begins with "Additional Conditions : 47A" and may span multiple chunks. Please extract the full description for `47A`.
2. **45B**: The description for the goods and/or services is denoted as `:45B:`. Please extract the full description associated with this reference.
3. **46B**: The documents required are denoted as `:46B:`. Please extract the full description associated with this reference.

Given the chunk-wise data below, please do the following:

1. Identify where each reference (`47A`, `45B`, `46B`) starts.
2. Extract all chunks that belong to each reference (including the chunk where the reference starts and all subsequent chunks that continue the description).
3. Provide the list of chunk numbers for each reference that contain the full description.

Here is the chunk-wise data:

```
{all_text_data}
```

"""
def run_model():
    # model_path2 = '/media/ntlpt19/5250315B5031474F/quantize_models/qwen2.5-0.5b-instruct-q2_k.gguf'     #2bit ##n_ctx=1000 #4.jpg = prompt1_1, 1.png = prompt1  #TIME FRIENDLY
    # model_path2 = '/media/ntlpt19/5250315B5031474F/quantize_models/qwen2.5-0.5b-instruct-q8_0.gguf'
    model_path2 = '/media/ntlpt19/5250315B5031474F/quantize_models/Llama-3.2-3B-Instruct-Q4_K_M.gguf'
    
    try:
        model = Llama(model_path=model_path2,
                     verbose=True, n_threads=None, n_ctx=2000, cache=cache, seed=63)
        
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

