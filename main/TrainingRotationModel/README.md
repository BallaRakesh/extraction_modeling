# PULC Classification Model of Text Image Orientation

## Introduction
This guide provides step-by-step instructions to train the PULC Classification Model of Text Image Orientation on a custom dataset.

## Project Requirements Setup
A GPU is necessary for training this model. Before running any code, ensure that the requirements in the `requirements.txt` file are installed. It's recommended to use a Python virtual environment for this purpose.

### Python Virtual Environment Setup
1. Open terminal.
2. Navigate (`cd`) to the directory where you want to store the environment.
3. Run the command:
    ```bash
    python -m venv {environment_name}
    ```
   **Note:** You can specify a particular Python version if needed.
4. Activate your virtual environment:
    ```bash
    source {environment_name}/bin/activate
    ```

### Installing Requirements
1. In your terminal, run the command:
    ```bash
    python -m pip install -r requirements.txt
    ```

## Project Dataset Preparation Setup
After setting up the environment and installing the requirements, prepare the dataset. If you already have images, you can skip the PDF splitting step and proceed to the train-test split step.

### PDF to Image Conversion
If you have document PDFs, split them into images by executing `PDFSplitter.py`. Update the paths in the code to your own paths. Ensure filenames don't have extra "." characters before running the image rotation code.

### Train-Test Split Step
To properly train the model, perform a train-test split on the image dataset. Execute `ImageDuplicateRemoval.py` to get the required train and test split. Update the required paths and install necessary libraries using:
```bash
python -m pip install {library_name}
```

As a result, you will have two folders: Train and Test. Perform the remaining steps on these sets individually.

### Rotation Set Preparation
The model requires four image sets with specific naming conventions, each storing images in a particular orientation: 0°, 90°, 180°, 270°. Prepare this dataset by executing `ImageRotation.py`, ensuring paths are updated to match your image dataset storage path.

**Note:** Do not change the naming conventions defined in the code. Ensure all files are in the correct text-orientation folders before performing any label generation steps.

### Label, Train, Test .txt Files Setup
**Note:** Ensure images are correctly segregated into appropriate folders for proper model training.

Combine all four train datasets into one and similarly for test images. Generate `train_list.txt`, `test_list.txt`, `train_list.txt.debug`, and `test_list.txt.debug`. Additionally, you will need `label_list.txt`, which has predefined labels: 0, 1, 2, 3 corresponding to 0°, 90°, 180°, 270° respectively.

The structure for `train_list.txt` and `test_list.txt` is:
```
img_(degreetype)/img_name text_orientation_class
```

**Note:** `train_list.txt.debug` and `test_list.txt.debug` are subsets of `train_list.txt` and `test_list.txt`.

Execute `TextFileGeneration.py` to generate the necessary files.

### Intermediary Folder Setup
With all necessary text and image files ready, follow these steps for final setup:
1. Create a new directory `text_image_orientation` using the `mkdir` command.
2. The directory structure should be as follows:
```
text_image_orientation/
├── img_0
│   ├── img_rot0.jpg
│   ├── img_rot1.png
│   └── ...
├── img_90
│   ├── img_rot90.jpg
│   └── ...
├── img_180
│   ├── img_rot180.jpg
│   └── ...
├── img_270
│   ├── img_rot270.jpg
│   └── ...
├── train_list.txt
├── train_list.txt.debug
├── test_list.txt
├── test_list.txt.debug
└── label_list.txt
```

## Custom PaddleClas Training Steps
The first phase of training is complete. Now, set up PaddleClas as per your requirements. Detailed instructions are provided here, but you can also refer to the [PaddleClas GitHub page](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/en/PULC/PULC_text_image_orientation_en.md) for clarifications.

1. Place the `text_image_orientation` folder inside a new directory `dataset`, then place it inside `docs/en/PULC`.
2. Navigate to `ppcls/configs/PULC/text_image_orientation` and open `PPLCNet_x1_0.yaml`. Update paths to your corresponding file paths.

### Training Command
```bash
python -m paddle.distributed.launch .../tools/train.py -c .../PPLCNet_x1_0.yaml
```
Once this command successfully runs, the model will be trained.

### Evaluation Command
```bash
python -m paddle.distributed.launch .../tools/eval.py -c .../PPLCNet_x1_0.yaml -o Global.pretrained_model="output/PPLCNet_x1_0/best_model"
```

### Inference Command
```bash
python -m paddle.distributed.launch .../tools/infer.py -c .../PPLCNet_x1_0.yaml -o Global.pretrained_model="output/PPLCNet_x1_0/best_model"
```

### Save Inference Model Command
```bash
python3 tools/export_model.py -c ./ppcls/configs/PULC/text_image_orientation/PPLCNet_x1_0.yaml -o Global.pretrained_model=output/DistillationModel/best_model_student -o Global.save_inference_dir=deploy/models/PPLCNet_x1_0_text_image_orientation_infer
```