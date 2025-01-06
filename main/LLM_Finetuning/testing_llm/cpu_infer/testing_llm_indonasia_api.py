from dotenv import load_dotenv
from time import time
import time
from llama_cpp import Llama, LlamaDiskCache
import gc
import re
# from client_code.llm_extraction.src.main.constant import model_path, NPWP_valid_keys, key_mapping_npwp
# from client_code.llm_extraction.src.main.prompts import prompt_main
from PIL import Image, ImageDraw
from fastapi import FastAPI
import base64
import requests
from pydantic import BaseModel
from io import BytesIO
import pytesseract
import os
import pandas as pd
import re
import json



model_path = '/media/ntlpt19/5250315B5031474F/quantize_models/qwen2.5-0.5b-instruct-q2_k.gguf'
model_path = '/home/ntlpt19/Downloads/qwen2.5-0.5b-instruct-q2_k.gguf'
# Define the set of keys you expect
NPWP_valid_keys = {"NIK", "NPWP", "Terdaftar", "name", "complete_address"}

key_mapping_npwp = {
    'Terdaftar': ['date', 'date_of_birth']
}




def prompt_main(all_text_data, task_name):
    if task_name == 'KTP':
        return f""" You are an expert in document key-value extraction. Document Language: Indonesian (Bahasa Indonesia). 
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
                
    elif task_name == 'NPWP':
        return f""" You are an expert in document key-value extraction. Document Language: Indonesian (Bahasa Indonesia). 
                OCR text: {all_text_data}
                Instructions:
                - Carefully extract all identifiable key-value pairs from the provided Indonesian national ID card (NPWP) text.
                - Recognize common field labels such as "NIK", "NPWP", "Terdaftar", "name" and "complete_address".
                - Handle variations in formatting, capitalization, and spacing.
                - Use "null" for any missing or incomplete values.
                - Normalize the data, such as converting dates to "YYYY-MM-DD" format.
                - Ensure the output is in valid JSON format.
                - Extract only the specified field labels exactly as mentioned above. Do not extract any other information, and do not include any additional response and do not include any repeated response.

                Note: Return the result in JSON format.
                """

# Load environment variables
load_dotenv()
cache = LlamaDiskCache(
    cache_dir="dex_llama_cache")


def is_valid_format(data):
    # Check if all top-level keys in the data have a dictionary with a "value" key
    return all(isinstance(info, dict) and "value" in info for info in data.values())

def convert_to_key_value_format(data):
    # Convert to a simpler JSON format if the format is valid
    if is_valid_format(data):
        # Extract each key and its 'value' field into a new dictionary
        simplified_data = {key: info['value'] for key, info in data.items()}
        return simplified_data
    else:
        return None
    

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
    if not any(results):
        results.append(extract_key_value_pairs(content))
        
    print('%%%% AFTER JSON FORMAT1 %s' % results)
    for res in range(len(results)):
        updated_result = convert_to_key_value_format(results[res])
        if updated_result:
            results[res] = updated_result
    print('%%%% AFTER JSON FORMAT2 %s' % results)
            
    return results


def extract_key_value_pairs(content):
    # Regex pattern to match "key": "value" pairs
    pair_pattern = r'"([\w\s\/.]+)":\s*"([^"]+)"'
    
    # Find all key-value pairs in the content
    pairs = re.findall(pair_pattern, content)

    # Build a dictionary, keeping only the last occurrence of each key
    result = {}
    for key, value in pairs:
        result[key.strip()] = value.strip()

    return result

import re

def normalize_key(key):
    # Convert to lowercase and remove spaces and underscores
    return re.sub(r'[\s_]+', '', key.lower())

def mapping_keys(final_res, required_mapping):
    for key, val in required_mapping.items():
        for value_ in val:
            if value_ in final_res and key not in final_res:
                final_res[key] = final_res[value_]
    return final_res
                
def post_process_results(results, valid_keys):
    processed_results = results.copy()
    for key, value in results.items():
        segments = re.split(r'[,\n]', value)
        for segment in segments:
            for valid_key in valid_keys:
                normalized_key = normalize_key(valid_key)
                match = re.search(rf"{re.escape(normalized_key)}\s*:\s*(.*?)(?:,|\n|$)", segment.strip(), re.IGNORECASE)
                if match:
                    if normalized_key not in {normalize_key(k) for k in processed_results}:
                        extracted_value = match.group(1).strip()
                        processed_results[valid_key] = extracted_value
    processed_results = mapping_keys(processed_results, key_mapping_npwp)
    
    return processed_results




# final_result = [{'NIK': '3275034111850030', 'Nama': 'EKA ROSANTI NOSAFRIA', 'Alamat': 'NIK: 3275034111850030', 'Terdaftar': 'NPWP: 54.152.428.6-407.000', 'KPP PRATAMA': 'PRATAMA'}]
# for final_res in range(len(final_result)):
#     final_result[final_res] = post_process_results(final_result[final_res], NPWP_valid_keys)
# print(final_result)

def get_llm_result(all_text, task_name):
    if task_name == 'KTP':
        n_ctx_val, seed_val, top_k_val, top_p_val, temperature_val = 1000, 63, 40, 0.7, 0.0
        # Completely fix the seed and set a very low temperature
        # n_ctx_val, seed_val, top_k_val, top_p_val, temperature_val = 1000, 42, 1, 0.1, 0.0
        
    else:
        # n_ctx_val, seed_val, top_k_val, top_p_val, temperature_val = 1000, 63, 40, 0.7, 0.0
        n_ctx_val, seed_val, top_k_val, top_p_val, temperature_val = 1000, 42, 1, 0.1, 0.0
        
        
    try:
        model = Llama(model_path=model_path,
                      verbose=True, n_threads=None, n_ctx=n_ctx_val, cache=cache, seed=seed_val)

        start_time = time.time()

        output = model.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are assistant that helps answer questions."},
                {"role": "user", "content": prompt_main(all_text, task_name)}
            ],
            top_k=top_k_val, top_p=top_p_val, temperature=temperature_val
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

        if task_name == 'NPWP':
            print('######## ENTERED INTO PP ###################')
            for final_res in range(len(final_result)):
                print('ITTER', final_res)
                print('######## BEFORE: ',final_result[final_res])
                final_result[final_res] = post_process_results(final_result[final_res], NPWP_valid_keys)
                print('######## AFTER: ', final_result[final_res])
    finally:
        if 'model' in locals():
            model.close()
            del model
            gc.collect()
    print("%%%%%%%%%%%%%%%%%%%%%%")
    print(final_result)
    return final_result, result

def get_ocr_tesseract_path(image_path):
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





# Dummy function definitions for reading and processing text
def read_all_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()




if __name__ == "__main__":
    # Directories and paths
    ocr_directory = '/home/ntlpt19/TF_testing_EXT/dummy_responces/bni_indonesia/Indentity/ocr'
    input_directory = '/home/ntlpt19/TF_testing_EXT/dummy_responces/bni_indonesia/Indentity/NPWP_samp'
    final_results_directory = '/home/ntlpt19/TF_testing_EXT/dummy_responces/bni_indonesia/Indentity/final_results'
    raw_results_directory = '/home/ntlpt19/TF_testing_EXT/dummy_responces/bni_indonesia/Indentity/raw_results'
    task_name = "NPWP"
    task_name = "KTP"
    # Ensure output directories exist
    output_csv_path = f'/home/ntlpt19/TF_testing_EXT/dummy_responces/bni_indonesia/Indentity/output_{task_name}.csv'
    
    raw_results_directory = raw_results_directory + task_name
    final_results_directory = final_results_directory + task_name
    os.makedirs(final_results_directory, exist_ok=True)
    os.makedirs(raw_results_directory, exist_ok=True)

    # Prepare list for storing results to be saved into the CSV file
    records = []
    # Process each file in the input directory
    for filename in os.listdir(input_directory):
        filename = os.path.splitext(filename)[0]
        file_path = os.path.join(ocr_directory, filename+"_all_text.txt")
        # file_path = '/home/ntlpt19/TF_testing_EXT/dummy_responces/bni_indonesia/Indentity/ocr/EDWIN EKO MARTHIN(1)_all_text.txt'
        file_path ='/home/ntlpt19/TF_testing_EXT/dummy_responces/bni_indonesia/Indentity/ocr/7_all_text.txt'
        all_text = read_all_text_file(file_path)
        
        # Obtain results from the LLM function
        final_result, raw_result = get_llm_result(all_text, task_name)
        # Save the final result as a text file in the specified folder
        final_result_path = os.path.join(final_results_directory, f"{filename}_final.txt")
        try:
            with open(final_result_path, 'w', encoding='utf-8') as final_file:
                for final_res_ in final_result:
                    for key, value in final_res_.items():
                        final_file.write(f"{key}: {value}\n")
        except:
            with open(final_result_path, 'w', encoding='utf-8') as final_file:
                final_file.write(str(final_result))
            
        # Save the raw result as a text file in the specified folder
        raw_result_path = os.path.join(raw_results_directory, f"{filename}_raw.txt")
        with open(raw_result_path, 'w', encoding='utf-8') as raw_file:
            raw_file.write(str(raw_result))
        
        # Append each field in final_result as a separate row in the records list
        for final_res in final_result:
            for field_name, predicted_value in final_res.items():
                records.append({
                    "File Name": filename,
                    "Field Name": field_name,
                    "Predicted": predicted_value
                })
        exit('PLLLLL')
    # Save all records to a CSV file with only three columns
    df = pd.DataFrame(records)
    df.to_csv(output_csv_path, index=False)

