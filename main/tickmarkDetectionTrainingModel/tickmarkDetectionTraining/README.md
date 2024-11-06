# Checkmark Detection Using YOLO-Based Approach

## Introduction
To identify checkboxes in a given document and determine whether they are checked, we use the YOLOv8n model. 

## Dataset Preparation
To train the YOLOv8n model, you need an image dataset and corresponding label text files. These text files store the class and coordinates for each checkbox in the images. Data annotation was performed using LabelImg.

The image dataset must be split into two folders: `train` and `val`, and stored within the `yolo_prepare_data` folder.

## Training the Model
1. Navigate to the `yolo_train` directory.
2. Execute the following command:
    ```bash
    python yolo_train.py
    ```
   **Note:** Ensure that the paths in the `config.yaml` file are correctly updated.