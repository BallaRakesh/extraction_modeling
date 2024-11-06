"""
* *********************************************************************************
* Number Theory S/W Pvt. Ltd CONFIDENTIAL                                      *
* *
* [2016] - [2023] Number Theory S/W Pvt. Ltd Incorporated                       *
* All Rights Reserved.                                                          *
* *
* NOTICE:  All information contained herein is, and remains                     *
* the property of Number Theory S/W Pvt. Ltd Incorporated and its suppliers,    *
* if any.  The intellectual and technical concepts contained                    *
* herein are proprietary to Number Theory S/W Pvt. Ltd Incorporated             *
* and its suppliers and may be covered by India. and Foreign Patents,           *
* patents in process, and are protected by trade secret or copyright law.       *
* Dissemination of this information or reproduction of this material            *
* is strictly forbidden unless prior written permission is obtained             *
* from Number Theory S/W Pvt. Ltd Incorporated.                                 *
* *
* *********************************************************************************
"""
import os
import json
from PIL import Image
import pytesseract
import numpy as np

def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]

# Define a function to normalize the bounding box
# def normalize_box(box, img_width, img_height):
#     return box[0] / img_width, box[1] / img_height, box[2] / img_width, box[3] / img_height

# Specify the folder containing the image files
image_folder = "/home/ntlpt19/Downloads/TRADE_FINANCE_OTHERS/ROOT/data"
json_folder = '/home/ntlpt19/Downloads/TRADE_FINANCE_OTHERS/OCR'
json_list = os.listdir(json_folder) 

for i in range(len(json_list)):  
    json_list[i] = json_list[i].split("/")[-1].split(".")[0]
# exit()
lis = ['AIR_WAY', 'BOL', 'COO', 'CS', 'PL', 'IC', 'others']
for i in lis:
    print(i)
    ocr_gen_path = os.path.join(image_folder,i)
    print(ocr_gen_path) # image path needed to generate the dictnary
    # Iterate over image files in the folder
    for filename in os.listdir(ocr_gen_path):
        temp_filename = filename.split("/")[-1].split(".")[0]
        if temp_filename not in json_list:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):  # Check if it's an image file
                image_path = os.path.join(ocr_gen_path, filename)
                image = Image.open(image_path)
                width, height = image.size

                # Apply OCR to the image
                ocr_data = {}
                ocr_df = pytesseract.image_to_data(image, output_type='data.frame')
                float_cols = ocr_df.select_dtypes('float').columns
                ocr_df = ocr_df.dropna().reset_index(drop=True)
                ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
                ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True)
                ocr_df = ocr_df.dropna().reset_index(drop=True)

                words = list(ocr_df.text)
                words = [str(w) for w in words]
                coordinates = ocr_df[['left', 'top', 'width', 'height']]
                actual_boxes = []
                for idx, row in coordinates.iterrows():
                    x, y, w, h = tuple(row)
                    actual_box = [x, y, x + w, y + h]
                    actual_boxes.append(actual_box)

                boxes = []
                for box in actual_boxes:
                    boxes.append(normalize_box(box, width, height))

                # Iterate over the OCR words and boxes
                for word, box in zip(words, boxes):
                    box_list = [int(val) for val in box]
                    ocr_data[word] = box_list

                # Convert dictionary to JSON
                json_data = json.dumps(ocr_data)

                # Save JSON data to a file with the same name as the image
                json_filename = os.path.splitext(filename)[0] + ".json"
                json_file_path = os.path.join(json_folder, json_filename)

                with open(json_file_path, 'w') as json_file:
                    json_file.write(json_data)
