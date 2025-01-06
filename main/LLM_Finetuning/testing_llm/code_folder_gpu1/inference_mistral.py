from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

import os
import pandas as pd
import time
import re
import json
import torch

def generate_instruction(doc_type="Airway bill"):
    return f'Trade Information Extraction: This text related to {doc_type} document and extract the key and value pair using this text\n\n'

def generate_prompt_inference(instruction, input_text):
    prefix_text = 'Below is an instruction that describes a task. Write a response that ' \
                  'appropriately completes the request.\n\n'
    return f"<s>[INST]{prefix_text} {instruction} here are the inputs {input_text} and Ensure that the response is in valid JSON format[/INST]</s>"

class config:
    TOKEN_LENGTH = 70
    DEVICE = 'cuda'
    DOCUMENT_CODE = {
    "COO": "certificate of origin", 
    "CS": "covering schedule",
    "BOL": "bill of landing",
    "CI": "commercial invoice",
    "AWB": "airway bill",
    "PI": "performa invoice",
    "IC":"insurance certificate",
    "PO": "performa invoice",
    "PL": "Packing List"
    }
    # THE LAST DIRECTORY YOU SHOULD BE PRESENT IN IS LLM_Finetuning
    CURRENT_DIR = '/home/gpu1admin/rakesh'
    SHEET_KEY = 'ground truth'
    SHEET_NAME = ''
    EXTENSION = '.xlsx'
    DOCUMENT_NAME = 'BOL'
    MODEL_PATH = '/datadrive/MistralModels/MergedModels/BOL'
    GROUND_TRUTH_SHEET = 'ground_truth_key_names_change'
    
    ROOT_PATH = "/home/gpu1admin/rakesh/BOL"
    GV_KEY = "/home/ntlpt19/Desktop/TF_release/training_code/ITF-Training/spheric-time-383904-f1b421d86eef.json"
    RESULT_FOLDER = '/home/ntlpt19/LLM_training/EVAL/CS/CS_results/text_files'
    
    
    EXCEL_FILE = '/home/gpu1admin/rakesh/BOL/ground_truth_key_names_change.xlsx'
    
def generate_responce(prompt_):
    
    inputs = tokenizer(prompt_, return_tensors="pt")
    input_length = inputs.input_ids.shape[1]
    
    output = model.generate(**inputs, max_new_tokens=2000, num_return_sequences=1, temperature=0.7)
                # do_sample=True,top_k=50,top_p=0.95, max_new_tokens=2000, max_length=2000
                
    generated_texts = tokenizer.decode(output[0][input_length:], skip_special_tokens=True)
    # generated_texts = tokenizer.batch_decode(output[0], skip_special_tokens=True)
    
    return generated_texts

def split_into_lists(words, length):
    return [words[i:i+length] for i in range(0, len(words), length)]


def combine_to_sentences(list_of_lists):
    return [' '.join(sublist) for sublist in list_of_lists]

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


if __name__=="__main__":

    print('started')
    device = config.DEVICE
    document_code = config.DOCUMENT_NAME
    token_length = config.TOKEN_LENGTH
    document_name = config.DOCUMENT_CODE[document_code]
    root_path = config.ROOT_PATH
    model_path = config.MODEL_PATH
    excel_file = config.EXCEL_FILE   #root_path + f'/{config.GROUND_TRUTH_SHEET}' + f'{config.EXTENSION}'
    results_path = root_path + '/results'
    results_path_txt = results_path + '/text_files'    
    results_path_json = results_path + '/json_files'
    os.makedirs(results_path,exist_ok=True)
    os.makedirs(results_path_txt,exist_ok=True)
    os.makedirs(results_path_json,exist_ok=True)
    print('start loading')
    
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print('loading')
    # model.to(device)
    df = pd.read_excel(excel_file, sheet_name=config.SHEET_KEY)
    print(df)
    for index, row in df.iterrows():
        input_text = row['text']
        file_name = row['file name']
        
        file_path = f'{file_name}.txt'
        if not os.path.exists(os.path.join(results_path_txt, file_path)):
            # for sentence_ in final_sentences:
            ins = generate_instruction(doc_type = document_name)
            prompt = generate_prompt_inference(ins, input_text)
            # Generate the response
            print(prompt)
            start_time = time.time()
            prediction = generate_responce(prompt)
            end_time = time.time()
            elapsed_time = end_time - start_time
            with open(os.path.join(results_path_txt, file_path), 'w') as file:
                print(prediction)
                file.write(prediction + '\n')

            json_result = extract_json_content(prediction)
            print(json_result)
            with open(os.path.join(results_path_json, file_path), 'w') as file:
                # Write the content to the file
                file.write(str(json_result))
                
            # Add the prediction to the "predicted" column
            df.at[index, 'predicted'] = str(json_result)
            df.at[index, 'time_taken'] = elapsed_time
        else:
            continue 
    df.to_excel(excel_file, sheet_name=config.SHEET_KEY, index=False)