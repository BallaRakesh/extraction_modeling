import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

from Batch_data_prep import generate_instruction, generate_prompt, generate_prompt_inference
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import time
import re
import json
import os


import re

def extract_key_value_pairs(input_text):
    pairs = {}
    lines = input_text.strip().split('\n')
    current_key = None
    current_value = ""

    for line in lines:
        line = line.strip().strip('"')
        match = re.match(r'(.+?):\s*(.+?)$', line)

        if match:
            if current_key:
                pairs[current_key] = current_value.strip()
            current_key, current_value = match.groups()
            current_key = current_key.strip('"')
            current_value = current_value.strip('"').rstrip(',')
        elif current_key:
            current_value += " " + line.strip('"').rstrip(',')

    if current_key:
        pairs[current_key] = current_value.strip()

    return pairs

def correct_response_creation(query):
        '''
        First we will extract the required key value pairs from the given query and then, send in a perfect JSON format query.
        '''
        pair_pattern = r'"([\w_]+)":\s*"?([^",\n]+)"?'
        pairs = re.findall(pair_pattern, query)
        # Create a dictionary from the pairs
        
        result = ["{"]
        for key,value in pairs:
            result.append(f'{key}:{value.strip()}')    
        result.append("}")
        res = "\n".join(result)
        print("The final response generated is:",res)
        exit('?')
        
        result_dict = {}
        for key, value in pairs:
            result_dict[key] = value#.strip()

        # Convert the dictionary to a JSON string
        json_str = json.dumps(result_dict, indent=4)

        # print("The final response generated is:", json_str)
        return json_str





def generate_responce(prompt_):
    
    inputs = tokenizer(prompt_, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=2000, num_return_sequences=1, temperature=0.7)
                # do_sample=True,top_k=50,top_p=0.95, max_new_tokens=2000, max_length=2000
    generated_texts = tokenizer.batch_decode(output, skip_special_tokens=True)

    return generated_texts[0]

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
    device = 'cuda'
    # Load the model and tokenizer
    model_path = "/home/gpu1admin/rakesh/ITF-Training/training/LLM_Finetuning/scripts/merged_model_coo"
    excel_file = "/home/gpu1admin/rakesh/COO/ground_truth_key_names_change.xlsx"
    sheet_name = "ground truth"
    document_name = "certificate of origin"
    results_path = '/home/gpu1admin/rakesh/COO/results'
    
    results_path_txt = '/home/gpu1admin/rakesh/COO/results/text_file'
    
    results_path_json = '/home/gpu1admin/rakesh/COO/results/json_files'
    token_length = 70

    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # model.to(device)
    df = pd.read_excel(excel_file, sheet_name=sheet_name)

    for index, row in df.iterrows():
        input_text = row['text']
        file_name = row['file name']
        
        # words = input_text.split(' ')
        # result = split_into_lists(words, token_length)
        # final_sentences = combine_to_sentences(result)
        
        file_path = f'{file_name}.txt'
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
        
    # Save the updated DataFrame back to Excel
    df.to_excel(excel_file, sheet_name=sheet_name, index=False)


######################## added inference code here #########################

exit('>>.>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Path to the checkpoint
checkpoint_path = "/home/nt-user1/rakesh/BOE/output/checkpoint-34"

# Load the base model and tokenizer
base_model_name = "meta-llama/Llama-2-7b-hf"  # Adjust this if you're using a different base model
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForCausalLM.from_pretrained(base_model_name)

# Load the PEFT configuration
peft_config = PeftConfig.from_pretrained(checkpoint_path)

# Load the fine-tuned model
model = PeftModel.from_pretrained(model, checkpoint_path)

# Set the model to evaluation mode
model.eval()

# Function for inference
def generate_text(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Example usage
prompt = "Once upon a time in a land far away,"
generated_text = generate_text(prompt)
print(f"Prompt: {prompt}")
print(f"Generated text: {generated_text}")


######################################################################

import csv
import re

def parse_text_file(content):
    # Extract key-value pairs
    pattern = r'(\w+):\s*(.*?)(?=\n\w+:|$)'
    matches = re.findall(pattern, content, re.DOTALL)
    # Create dictionary from matches
    predicted_data = {key.strip(): value.strip() for key, value in matches}
    return predicted_data

# Actual data
actual_data = {
    "bill_exchange_no": "4118",
    "bill_exchange_date": "04/09/12",
    "boe_currency": "GBP",
    "boe_amount": "36,760.00",
    "tenore_details": "AT 30 DAYS FROM THE DATE OF SHIPMENT",
    "signature": "LYNNE OWEN",
    "drawee_bank_name": "ICICI BANK LIMITED",
    "drawee_bank_address": "SCO.9-11 SECTOR - 9D , MADHYA MARG , CIBD CHANDIGARH 160017 , INDIA",
    "amount_in_words": "THIRTY SIX THOUSAND SEVEN HUNDRED AND SIXTY GB POUNDS AND ZERO GB PENCE .",
    "diclaration_by": "FOR AND ON BEHALF OF OXFORD INSTRUMENTS INDUSTRIAL PRODUCTS LIMITED",
    "ocean_transport_method": "PER AIRFREIGHT",
    "drawer_bank_name": "NOT APPLICABLE",
    "drawer_bank_address": "NOT APPLICABLE",
    "credit_ref_no": "0013MLC00005413",
    "credit_date": "22AUG12.",
    "drawer_name": "OXFORD INSTRUMENTS INDUSTRIAL PRODUCTS LIMITED TRIAL PRODUCTS LIMITED",
    "drawer_address": "Lower",
    "shipping_administer": "LYNNE OWEN",
    "bill_exchange_status": "ORIGINAL"
}

# Parse predicted data from text file
txt_file = '/home/ntlpt19/LLM_training/EVAL/BOE_EVAL_LATEST_39/results_filtered/Certificate_Of_Origin_115_page_14.txt'
    
with open(txt_file, 'r') as file:
    content = file.read()
    
predicted_data = parse_text_file(content)
print(predicted_data)
exit('????????/')
# Prepare data for CSV
csv_data = []

# Process matching keys
for key in set(actual_data.keys()) | set(predicted_data.keys()):
    actual_value = actual_data.get(key, "")
    predicted_value = predicted_data.get(key, "")
    csv_data.append([key, key, actual_value, predicted_value])

# Write to CSV
with open('mapped_data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Actual Key', 'Predicted Key', 'Actual Value', 'Predicted Value'])
    writer.writerows(csv_data)

print("CSV file 'mapped_data.csv' has been created.")