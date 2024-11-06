#/home/ntlpt-42/Documents/mani_projects/document_processing_retraining/demo/final_data
#/home/ntlpt-42/Documents/mani_projects/IDP/covering_schedule/test_data/code/draw_bounding_box.py

import cv2
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import shutil
import json
from functools import cmp_to_key
from google.cloud import vision
from base64 import b64encode
import time
from config import config

def draw_bounding_box(img_path, labels_list):
    image = cv2.imread(img_path)
    for item in labels_list:
        label= item['label']
        x1= item['x1']
        y1= item['y1']
        x2= item['x2']
        y2= item['y2']
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(
        image,
        label,
        (int(x1), int(y1)),
        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
        fontScale = 0.3,
        color = (255, 0, 0),
        thickness=1
    )

    return image


if __name__=="__main__":
    folder_path = config.ROOT_PATH
    images_path = os.path.join(folder_path, "Images")
    labels_path = os.path.join(folder_path, "Labels")
    master_path = os.path.join(folder_path, "Master_Data")
    ocr_path = os.path.join(folder_path, "OCR")
    bounding_box_path= os.path.join(folder_path,"bounding_box")
    if not os.path.exists(master_path):
        os.mkdir(master_path)
    if not os.path.exists(ocr_path):
        os.mkdir(ocr_path)
    if not os.path.exists(bounding_box_path):
        os.mkdir(bounding_box_path)

    with open(os.path.join(folder_path, "label.txt"), "r") as f:
        classes = (f.read())
        classes = classes.split("\n")
    labelled_files = os.listdir(labels_path)
    labelled_files = [x.split(".txt")[0] for x in labelled_files]

    annotation_data = []
    thresh = 300
    for file in labelled_files:
        if os.path.exists(os.path.join(images_path, file + ".png")):
            print('yes')
            print(file)
            image = cv2.imread(os.path.join(images_path, file + ".png"))                
            h, w, _ = image.shape
            with open(os.path.join(labels_path, file + ".txt"), "r") as f:
                label = (f.read())
            label = label.split("\n")
            labelled_data = []
            for l in label:
                l = l.split()
                if len(l) > 0:
                    l_class = classes[int(l[0])]
                    x_center = float(l[1]) * w
                    y_center = float(l[2]) * h
                    width = float(l[3]) * w
                    height = int(float(l[4]) * h)
                    x0 = int(x_center - (width/2))
                    x1 = int(x_center + (width/2))
                    y0 = int(y_center - (height / 2))
                    y1 = int(y_center + (height / 2))
                    cv2.rectangle(image, (x0, y0), (x1, y1), (0,255,0), 2)
                    labelled_data.append({
                        "label": l_class,
                        "x1": x0,
                        "y1": y0,
                        "x2": x1,
                        "y2": y1
                    })
            print(labelled_data)
            img_path= os.path.join(images_path, file + ".png")
            processed_img= draw_bounding_box(img_path,labelled_data)
            bounding_box= os.path.join(bounding_box_path, file+ ".png")
            cv2.imwrite(bounding_box, processed_img)


