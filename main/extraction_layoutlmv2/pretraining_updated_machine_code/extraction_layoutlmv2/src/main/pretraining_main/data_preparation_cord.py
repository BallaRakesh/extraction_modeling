import os
import json
import pickle
from typing import List, Dict, Tuple

import shutil
import os



# Normalization Function
def normalize(points: list, width: int, height: int, val: int, zero_val: int) -> list:
    """
    Normalizes bounding box coordinates to a scale defined by `val`.

    Args:
        points (list): [x1, y1, x2, y2] bounding box coordinates.
        width (int): Width of the image.
        height (int): Height of the image.
        val (int): Maximum value for normalization.
        zero_val (int): Minimum value for normalization.

    Returns:
        list: Normalized [x1, y1, x2, y2] coordinates.
    """
    x0, y0, x2, y2 = [int(p) for p in points]
    x0 = int(val * (x0 / width))
    x2 = int(val * (x2 / width))
    y0 = int(val * (y0 / height))
    y2 = int(val * (y2 / height))

    # Clamp values to [zero_val, val]
    x0 = min(max(x0, zero_val), val)
    x2 = min(max(x2, zero_val), val)
    y0 = min(max(y0, zero_val), val)
    y2 = min(max(y2, zero_val), val)

    return [x0, y0, x2, y2]



def normalize_custom(points: list, width: int, height: int) -> list:
    x0, y0, x2, y2 = [int(p) for p in points]
    val = 1000 #int(configur['PARAMS']['norm_val'])
    zero_val = 0 #int(configur['PARAMS']['nill_val'])
    x0 = int(val * (x0 / width))
    x2 = int(val * (x2 / width))
    y0 = int(val * (y0 / height))
    y2 = int(val * (y2 / height))
    if x0 > val:
        x0 = val
    if x0 < zero_val:
        x0 = zero_val
    if x2 > val:
        x2 = val
    if x2 < zero_val:
        x2 = zero_val
    if y0 > val:
        y0 = val
    if y0 < zero_val:
        y0 = zero_val
    if y2 > val:
        y2 = val
    if y2 < zero_val:
        y2 = zero_val
    return [x0, y0, x2, y2]


def data_convert(dataset: List = None) -> Tuple[List, List, List]:
    if dataset is None:
        return [], [], []

    words = []
    boxes = []
    labels = []

    for datas in dataset:
        words.append(datas['words'])
        boxes.append(datas['bbox'])
        labels.append(datas['labels'])
    return words, boxes, labels

def prepare_data(json_files: str, output_path: str, split_name: str) -> None:
    """
    Prepares data from the given JSON files and stores them in a .pkl file.

    Args:
        json_files (List[str]): List of JSON file paths containing word information.
        output_path (str): Path to store the final .pkl file.
        configur (dict): Configuration parameters containing normalization values.

    Returns:
        None
    """
    final_data = []
    norm_val = 1000 #int(configur['PARAMS']['norm_val'])
    zero_val = 0# int(configur['PARAMS']['nill_val'])

    for json_file in os.listdir(json_files):
        with open(os.path.join(json_files, json_file), 'r') as f:
            data = json.load(f)
        print(data)
        # Extract image dimensions
        image_width = data["meta"]["image_size"]["width"]
        image_height = data["meta"]["image_size"]["height"]
        
        words, bboxes, labels = [],[],[]
        final_name_ = os.path.basename(json_file.replace('.json', '.png').replace('label', 'image'))
        
        
        # if final_name_ in proper_images:
        #     source_path = os.path.join(source_dir, final_name_)
        #     destination_path = os.path.join(destination_dir, final_name_)
        #     shutil.copy2(source_path, destination_path)
        
        for item in data["valid_line"]:
            for word_info in item["words"]:
                words.append(word_info["text"])

                # Normalize bounding box coordinates
                quad = word_info["quad"]
                
                x1, y1 = quad["x1"], quad["y1"]
                x2, y2 = quad["x2"], quad["y2"]
                x3, y3 = quad["x3"], quad["y3"]
                x4, y4 = quad["x4"], quad["y4"]
                
                # Calculate rectangular bounding box
                x_min = min(x1, x2, x3, x4)
                y_min = min(y1, y2, y3, y4)
                x_max = max(x1, x2, x3, x4)
                y_max = max(y1, y2, y3, y4)
                
                x1_, y1_, x2_, y2_ = x_min, y_min, x_max, y_max #quad["x1"], quad["y1"], quad["x4"], quad["y4"]
                
                
                
                # bbox = [quad["x1"], quad["y1"], quad["x2"], quad["y3"]]
                bbox = [x1_, y1_, x2_, y2_]
                normalized_bbox = normalize_custom(bbox, image_width, image_height)
                # normalized_bbox = normalize(bbox, image_width, image_height, norm_val, zero_val)
                bboxes.append(normalized_bbox)

                # Modify label
                label = item["category"].replace(".", "_")
                label = "S-" + label
                labels.append(label)

        for symbol_group in data.get("repeating_symbol", []):
            for symbol in symbol_group:
                words.append(symbol["text"])
                quad = symbol["quad"]
                
                x1, y1 = quad["x1"], quad["y1"]
                x2, y2 = quad["x2"], quad["y2"]
                x3, y3 = quad["x3"], quad["y3"]
                x4, y4 = quad["x4"], quad["y4"]
                
                # Calculate rectangular bounding box
                x_min = min(x1, x2, x3, x4)
                y_min = min(y1, y2, y3, y4)
                x_max = max(x1, x2, x3, x4)
                y_max = max(y1, y2, y3, y4)
                x1_, y1_, x2_, y2_ = x_min, y_min, x_max, y_max #quad["x1"], quad["y1"], quad["x4"], quad["y4"]
                # bbox = [quad["x1"], quad["y1"], quad["x2"], quad["y3"]]
                bbox = [x1_, y1_, x2_, y2_]
                normalized_bbox = normalize_custom(bbox, image_width, image_height)
                # normalized_bbox = normalize(bbox, image_width, image_height, norm_val, zero_val)
                bboxes.append(normalized_bbox)
                # Modify label
                # label = item["category"].replace(".", "_")
                label = "O" #"S-" + label 
                labels.append(label)


            # Construct structured data for each file
        final_data.append({
            "words": words,
            "bbox": bboxes,
            "labels": labels,
            "filename": final_name_
        })
        
            
    test_samples = sorted(final_data, key=lambda x: x['filename'])
    words_test, boxes_test, labels_test = data_convert(test_samples)
    
    # if not os.path.exists(os.path.join(folder_path, 'test')):
    #     os.mkdir(os.path.join(folder_path, 'test'))
    
    with open(os.path.join(output_path, f'{split_name}.pkl'), 'wb') as t:
        pickle.dump([words_test, labels_test, boxes_test], t)

    # with open(os.path.join(output_path, 'train.pkl'), 'wb') as t:
    #     pickle.dump([words_test, labels_test, boxes_test], t)
        
    # Save the structured data into a .pkl file
    # with open(output_path, 'wb') as output_file:
    #     pickle.dump(final_data, output_file)
    print(f"Data preparation completed. Output saved to {output_path}")

# Example usage
split_name = 'train'
json_file_paths = "/home/data_science/geo_testing/lmv2_code/local_cord_v2_dataset/train/Labels" # Replace with actual file paths
output_pickle_path = "/home/data_science/geo_testing/COO_V3/CORD_DATA"

prepare_data(json_file_paths, output_pickle_path, split_name)
