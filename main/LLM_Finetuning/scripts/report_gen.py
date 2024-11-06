import os
from tqdm import tqdm
import json
import pandas as pd
from config import config


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



if __name__=="__main__":
    root_path = config.ROOT_PATH
    images_path= os.path.join(root_path, "Images")
    labels_path= os.path.join(root_path, "Labels")
    master_data_path= os.path.join(root_path, "Master_Data")
    img_lst=  [image.split(".png")[0] for image in os.listdir(images_path)]
    print(f"Num of images: {len(img_lst)}")
    # exit('++++++++++++')
    label_lst=  [label.split(".txt")[0] for label in os.listdir(labels_path)]
    data=[]
    unprocessed_files= []
    for i, image in enumerate(tqdm(img_lst, desc= "master data prep...")):
        if image in label_lst and  os.path.exists(os.path.join(master_data_path, image+"_labels.txt")):
            print(f"File: {image} and count {i}")
            all_text_path= os.path.join(master_data_path, image+"_all_text.txt")
            text= read_json_with_name(all_text_path)
            all_labels_path=  os.path.join(master_data_path, image+"_labels.txt")
            labels= read_json_file(all_labels_path)
            gt_labels= format_labels(labels)
            # print(gt_labels)
            data.append({'file name': image, "text": text, 'ground truth': str(gt_labels)})
        else:
            unprocessed_files.append(image)
    

    # print(all_data)
    print(f"unprocessed files : {len(unprocessed_files)}")
    print(f"ground truth count: {len(data)}")

    # Create a DataFrame from the list
    df = pd.DataFrame(data)

    # Write the DataFrame to an Excel file
    output_file = os.path.join(root_path,'ground_truth.xlsx')
    df.to_excel(output_file, index=False, sheet_name='ground truth')

