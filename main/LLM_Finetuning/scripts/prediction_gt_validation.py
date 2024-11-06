import os
from tqdm import tqdm
import json
import pandas as pd
from fuzzywuzzy import fuzz
import re
from config import config
from constants import key_mapping


def format_labels(labels: dict):
    label_dict = {}
    for key, value in labels.items():
        if key not in label_dict:
            label_dict[key] = value[0][0]
    return json.dumps(label_dict, indent=2)
        

def read_json_file(file_path):
    """Read a JSON file and return the data as a dictionary."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def read_json_with_name(file_path):
    """Read a JSON file and return the data as a dictionary."""
    with open(file_path, 'r') as file:
        data = json.load(file)["all_text"]
    return data

def read_file(file_path):
    """Read the entire content of the file."""
    with open(file_path, 'r') as file:
        content = file.read()
    return content

def normalize_text(text):
    """
    Normalize text by converting to lower case and stripping extra whitespace.
    """
    return re.sub(r'\s+', ' ', text.strip().lower())

def process_dict(input_dict, key_mapping, keys_not_to_consider, lables_train = [], threshold=20):
    """
    Process the input dictionary to combine values with fuzzy matching.

    Args:
    input_dict (dict): The dictionary with key-value pairs to be processed.
    threshold (int): The fuzzy matching threshold to consider values as similar.

    Returns:
    dict: A processed dictionary with combined values.
    """
    processed_dict = {}

    for key, value_list in input_dict.items():
        if key not in keys_not_to_consider:
            if not len(lables_train) or key in lables_train:
                if len(value_list) == 1:
                    # If there is only one value, add it directly to the processed dictionary
                    processed_dict[key] = value_list[0][0]
                else:
                    # Combine values if they are unique based on fuzzy matching
                    combined_value = value_list[0][0]
                    for value, coords in value_list[1:]:
                        normalized_value = normalize_text(value)
                        normalized_combined_value = normalize_text(combined_value)
                        match_ratio = fuzz.ratio(normalized_combined_value, normalized_value)
                        if match_ratio < threshold:
                            combined_value += " " + value

                    # Combine unique values into a single string, without any separator
                    processed_dict[key] = combined_value
    if any(key_mapping):
        processed_dict= change_key_names(processed_dict, key_mapping)
    
    return   json.dumps(processed_dict, indent=2), len(processed_dict)

def change_key_names(input_dict, key_mapping):
    """
    Change the key names in a dictionary according to the provided mapping.

    Args:
    input_dict (dict): The dictionary with original key-value pairs.
    key_mapping (dict): A dictionary mapping old key names to new key names.

    Returns:
    dict: A new dictionary with updated key names.
    """
    updated_dict = {}
    for old_key, value in input_dict.items():
        new_key = key_mapping.get(old_key, old_key)
        updated_dict[new_key] = value
    return updated_dict

import csv

# Specify the CSV file path

def get_train_keys(csv_file_path):
    column_name = 'Key'
    # Initialize a list to store the values under the specified column
    key_values = []
    # Step 1: Open the CSV file and read it
    with open(csv_file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        
        # Step 2: Iterate over each row and extract the values under the specified column
        for row in reader:
            key_values.append(row[column_name])

    # Step 3: Print or use the extracted list
    return key_values

def open_text_file(name_file):
    if os.path.exists(name_file):
        # Open the file in read mode
        with open(name_file, 'r') as file:
            # Read the content of the file
            content = file.read()
    else:
        content = ''
    return content
    # Print the content
    
    
if __name__=="__main__":
    root_path = config.ROOT_PATH
    results_folder = config.RESULT_FOLDER
    
    try:
        lables_analysis_path= os.path.join(root_path, "key_counts.csv")
        lables_train = get_train_keys(lables_analysis_path)
    except:
        lables_train = [] ####
    images_path= os.path.join(root_path, "Images")
    labels_path= os.path.join(root_path, "Labels")
    master_data_path= os.path.join(root_path, "Master_Data")
    img_lst=  [image.split(".png")[0] for image in os.listdir(images_path)]
    print(f"Num of images: {len(img_lst)}")


    keys_not_to_consider = ['stamp', 'signature', 'doc_settlement_instructions', 'signature', 'signed_stamp','signed_stamp', 'signed_by_carrier', 'signed_By_agent']
    label_lst=  [label.split(".txt")[0] for label in os.listdir(labels_path)]
    data=[]
    unprocessed_files= []
    for i, image in enumerate(tqdm(img_lst, desc= "master data prep...")):
        if image in label_lst and  os.path.exists(os.path.join(master_data_path, image+"_labels.txt")):
            print(f"File: {image} and count {i}")
            all_text_path= os.path.join(master_data_path, image+"_all_text.txt")
            text= read_json_with_name(all_text_path)
            result_content = open_text_file(os.path.join(results_folder, f'{image}.txt'))
            print(result_content)
            all_labels_path=  os.path.join(master_data_path, image+"_labels.txt")
            labels= read_json_file(all_labels_path)
            print(labels)
            gt_labels, len_gt = process_dict(labels, key_mapping, keys_not_to_consider, lables_train)
            # gt_labels= format_labels(labels)
            # print(gt_labels)
            data.append({'file name': image, 'ground truth': str(gt_labels), 'prediction': result_content, 'gt_count': len_gt})
        else:
            unprocessed_files.append(image)
    

    # print(all_data)
    print(f"unprocessed files : {len(unprocessed_files)}")
    print(f"ground truth count: {len(data)}")

    # Create a DataFrame from the list
    df = pd.DataFrame(data)

    # Write the DataFrame to an Excel file
    output_file = os.path.join(root_path,'ground_truth_key_names_change.xlsx')
    df.to_excel(output_file, index=False, sheet_name='ground truth')

