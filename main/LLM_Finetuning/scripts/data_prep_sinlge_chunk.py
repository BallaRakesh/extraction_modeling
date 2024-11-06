import os
import json
from tqdm import tqdm
import random
from fuzzywuzzy import fuzz
import re
from constants import documents_name, train_test_split_ratio
from config import config
from constants import key_mapping, keys_not_to_consider


def split_list(input_list, split_ratio=0.9):
    """Splits the input list into two parts based on the given ratio.
    
    Args:
        input_list (list): The list to be split.
        split_ratio (float): The ratio to split the list. Default is 0.9.
        
    Returns:
        tuple: A tuple containing two lists, the first with split_ratio% elements
               and the second with the remaining elements.
    """
    # Shuffle the list to ensure randomness
    random.shuffle(input_list)
    
    # Calculate the split index
    split_index = int(len(input_list) * split_ratio)
    
    # Split the list
    part_1 = input_list[:split_index]
    part_2 = input_list[split_index:]
    
    return part_1, part_2
def generate_prompt(instruction:str, input: str, output: str):
    """Gen. input text based on a prompt, task instruction, (context info.), and answer

    :param data_point: dict: Data point
    :return: dict: tokenzed prompt
    """
    prefix_text = 'Below is an instruction that describes a task. Write a response that ' \
                'appropriately completes the request.\n\n'
    text = f"""<s>[INST]{prefix_text} {instruction} here are the inputs {input} [/INST]{output}</s>"""

    return text
def generate_instruction(doc_type= "bill of ladding"):
    instruction= f'''Trade Information Extraction:
        This text related to {doc_type} document and extract the key and value pair using this text\n\n'''
    return instruction

def read_file(file_path):
    """Read the entire content of the file."""
    with open(file_path, 'r') as file:
        content = json.load(file)
    return content["all_text"]


def read_json_file(file_path):
    """Read a JSON file and return the data as a dictionary."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data
def format_label(labels: dict):
    label_dict = {}
    for key, value in labels.items():
        if key not in label_dict:
            label_dict[key] = value[0][0]
    return json.dumps(label_dict, indent=2)

def normalize_text(text):
    """
    Normalize text by converting to lower case and stripping extra whitespace.
    """
    return re.sub(r'\s+', ' ', text.strip().lower())

def process_dict(input_dict, key_mapping, keys_not_to_consider, threshold=20):
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
    return   json.dumps(processed_dict, indent=2)

def prepare(all_text_path: str, all_labels_path: str, file_name: str, keys_not_to_consider: list, key_mapping:dict={}):
    # read input text
    input_text= read_file(all_text_path)
    # read labels
    labels= read_json_file(all_labels_path)
    print(labels)
    output= process_dict(labels, key_mapping, keys_not_to_consider)
    print(output)
    # generat instruction
    instruction= generate_instruction(doc_type=documents_name)
    prompt= generate_prompt(instruction, input_text, output)
    json_object = {
            "File_Name":file_name,
            "text": prompt,
            "instruction": instruction,
            "input": input_text,
            "output": output,
            "prompt": prompt
        }
    return json_object

def data_prep(data_lst:list, data_prep: str,key_mapping:dict, keys_not_to_consider:list):
    all_data=[]
    for image in tqdm(data_lst, desc= data_prep):
       if image in label_lst and  os.path.exists(os.path.join(master_data_path, image+"_all_text.txt")) and os.path.exists(os.path.join(master_data_path, image+"_labels.txt")):
           # all text
           all_text_path= os.path.join(master_data_path, image+"_all_text.txt")
           # read labels
           all_labels_path=  os.path.join(master_data_path, image+"_labels.txt")
           row= prepare(all_text_path, all_labels_path, image,keys_not_to_consider,key_mapping)
           all_data.append(row)
    return all_data

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




if __name__=="__main__":
    root_path = config.ROOT_PATH
    images_path= os.path.join(root_path, "Images")
    labels_path= os.path.join(root_path, "Labels")
    master_data_path= os.path.join(root_path, "Master_Data")
    output_path= os.path.join(root_path, "extraction_data")
    os.makedirs(output_path, exist_ok=True)
    output_train_file_path= os.path.join(output_path, "train.json")
    output_test_file_path= os.path.join(output_path, "test.json")

    img_lst=  [image.split(".png")[0] for image in os.listdir(images_path)]

    label_lst=  [label.split(".txt")[0] for label in os.listdir(labels_path)]
    print(img_lst)
    train, test= split_list(img_lst, split_ratio=train_test_split_ratio)
    
    from constants import single_file
    # prepare tain data
    if single_file:
        train.extend(test)
        print(train)
        train_data= data_prep(train, "train", key_mapping, keys_not_to_consider)
        with open(output_train_file_path, "w") as output_jsonl_file:
            output_jsonl_file.write(json.dumps(train_data))
    else:
        #prepare val data 
        train_data= data_prep(train, "train", key_mapping, keys_not_to_consider)
        test_data= data_prep(test, "eval", key_mapping, keys_not_to_consider)

        with open(output_train_file_path, "w") as output_jsonl_file:
            output_jsonl_file.write(json.dumps(train_data))

        with open(output_test_file_path, "w") as output_jsonl_file:
            output_jsonl_file.write(json.dumps(test_data))
            
            






