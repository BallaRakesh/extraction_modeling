import os
import json
from tqdm import tqdm
import random
from fuzzywuzzy import fuzz
import re
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from config import config
from constants import key_mapping

# Define utility functions as given

def split_list(input_list, split_ratio=0.9):
    random.shuffle(input_list)
    split_index = int(len(input_list) * split_ratio)
    return input_list[:split_index], input_list[split_index:]


def generate_prompt_inference(instruction, input_text):
    prefix_text = 'Below is an instruction that describes a task. Write a response that ' \
                  'appropriately completes the request.\n\n'
    return f"<s>[INST]{prefix_text} {instruction} here are the inputs {input_text} and Ensure that the response is in valid JSON format[/INST]</s>"


def generate_prompt(instruction, input_text, output):
    prefix_text = 'Below is an instruction that describes a task. Write a response that ' \
                  'appropriately completes the request.\n\n'
    return f"<s>[INST]{prefix_text} {instruction} here are the inputs {input_text} [/INST]{output}</s>"

def generate_instruction(doc_type="Airway bill"):
    return f'Trade Information Extraction: This text related to {doc_type} document and extract the key and value pair using this text\n\n'

def read_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)["all_text"]

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def format_label(labels):
    return json.dumps({key: value[0][0] for key, value in labels.items()}, indent=2)

def normalize_text(text):
    return re.sub(r'\s+', ' ', text.strip().lower())

def process_dict(input_dict, key_mapping, threshold=20):
    processed_dict = {}
    for key, value_list in input_dict.items():
        combined_value = value_list[0][0]
        for value, coords in value_list[1:]:
            normalized_value = normalize_text(value)
            normalized_combined_value = normalize_text(combined_value)
            if fuzz.ratio(normalized_combined_value, normalized_value) < threshold:
                combined_value += " " + value
        processed_dict[key] = combined_value
    return change_key_names(processed_dict, key_mapping) if key_mapping else processed_dict

def change_key_names(input_dict, key_mapping):
    return {key_mapping.get(key, key): value for key, value in input_dict.items()}

# Custom Dataset Class
class TradeDataset(Dataset):
    def __init__(self, data_list, master_data_path, key_mapping, doc_type="Proforma Invoice"):
        self.data_list = data_list
        self.master_data_path = master_data_path
        self.key_mapping = key_mapping
        self.doc_type = doc_type

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_name = self.data_list[idx]
        all_text_path = os.path.join(self.master_data_path, file_name + "_all_text.txt")
        all_labels_path = os.path.join(self.master_data_path, file_name + "_labels.txt")

        input_text = read_file(all_text_path)
        labels = read_json_file(all_labels_path)
        output = process_dict(labels, self.key_mapping)

        instruction = generate_instruction(doc_type=self.doc_type)
        prompt = generate_prompt(instruction, input_text, json.dumps(output, indent=2))

        json_object = {
            "File_Name": file_name,
            "text": prompt,
            "instruction": instruction,
            "input": input_text,
            "output": json.dumps(output, indent=2),
            "prompt": prompt
        }

        return json_object

def prepare_datasets(img_lst, label_lst, master_data_path, key_mapping):
    train_list, test_list = split_list(img_lst)
    train_dataset = TradeDataset(train_list, master_data_path, key_mapping)
    test_dataset = TradeDataset(test_list, master_data_path, key_mapping)
    return train_dataset, test_dataset

def save_datasets(train_dataset, test_dataset, train_path, test_path):
    with open(train_path, 'wb') as f:
        pickle.dump(train_dataset, f)
    with open(test_path, 'wb') as f:
        pickle.dump(test_dataset, f)
        
def load_datasets(train_path, test_path):
    with open(train_path, 'rb') as f:
        train_dataset = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_dataset = pickle.load(f)
    return train_dataset, test_dataset

# Main code
if __name__ == "__main__":
    root_path = config.ROOT_PATH
    
    root_path = "/home/ntlpt-42/Documents/mani_projects/IDP/IDE/Extraction_using_llm/complete_data/PI"
    images_path = os.path.join(root_path, "Images")
    labels_path = os.path.join(root_path, "Labels")
    master_data_path = os.path.join(root_path, "Master_Data")
    output_path = os.path.join(root_path, "extraction_data")
    os.makedirs(output_path, exist_ok=True)

    train_pickle_path = os.path.join(output_path, "train_dataset.pkl")
    test_pickle_path = os.path.join(output_path, "test_dataset.pkl")


    img_lst = [image.split(".png")[0] for image in os.listdir(images_path)]
    label_lst = [label.split(".txt")[0] for label in os.listdir(labels_path)]
    
    train_dataset, test_dataset = prepare_datasets(img_lst, label_lst, master_data_path, key_mapping)
    
    # Save datasets to pickle files
    save_datasets(train_dataset, test_dataset, train_pickle_path, test_pickle_path)
    
    # Load datasets from pickle files (optional)
    train_dataset, test_dataset = load_datasets(train_pickle_path, test_pickle_path)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    for batch in train_loader:
        print("single batch")
        print(batch)
        break
