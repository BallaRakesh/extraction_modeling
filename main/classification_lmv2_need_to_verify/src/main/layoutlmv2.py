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
import time
import numpy as np
import pandas as pd
import pytesseract
from ocrpipeline import ApplyOcr
from torch.utils.data import random_split
from torch.optim import SGD, RMSprop
import torch
from PIL import Image
from datasets import Dataset, Features, Sequence, ClassLabel, Value, Array3D, Array2D
from transformers import AdamW, LayoutLMv2FeatureExtractor, LayoutLMv2ForSequenceClassification, LayoutLMv2Processor, \
    LayoutLMv2Tokenizer, LayoutLMv2Config
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json
import numpy as np
import pytesseract
from PIL import Image

# Global statements
# Step1: Reading configurations from ini files
parser = ConfigParser()

# custom change required here in below 2 lines
conf_folder_path: str = "/home/tarun/NumberTheory/trade-finance-delivery-code/src/main/training/extraction/mv2/config"
logs_folder_path: str = "/home/tarun/NumberTheory/trade-finance-delivery-code/src/main/training/extraction/mv2/logs"
config_file_name: str = "config/config.ini"

if os.path.exists(f"{conf_folder_path}/{config_file_name}"):
    parser.read(f"{conf_folder_path}/{config_file_name}")

ocr_engine_mode: str = parser['OCR']['OCR_ENGINE']
if ocr_engine_mode == "Vision":
    gv_key = parser['OCR']['gv_key']
elif ocr_engine_mode == "Tesseract":
    tessdata_dir = parser['OCR']['TESSDATA_DIR']

root_folder_path = str(parser['PATHS']['root_folder'])
debug_mode = str(parser["PARAMS"]["debug_mode"])
log_min_level = str(parser["LOG"]["LOG_FILTER_LEVELMIN"])

# Step 2: Setting up logger
# setting the log folder and file
set_basic_config_for_logging(folder_path=conf_folder_path, filename="data_preparation")  # will create
# data_preparation log file
# setting the logger object and log level
logger = get_logger_object_and_setting_the_loglevel(log_level=log_min_level)


def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]


# User will always have an option , whether to perform ocr manually or by automated layoutLMv2 model.
# LayoutLMv2 has OCR inbuilt unlike LayoutLM.
def a_ocr(example):
    if debug_mode:
        print(f"example is :{example}")
    file_name = example['image_path'].split("/")[-1].split(".")[0]
    try:
        ocr_gen = ocr_path + '/' + file_name + '.json'
        print(f"ocr gen path : {ocr_gen}")
        print("filename is: {file_name}")
        with open(ocr_gen, 'r') as f:
            data0 = json.load(f)
    except:
        j_data = ApplyOcr.apply_ocr(example)
        data0 = eval(j_data)
    dic = data0
    print(dic)
    keys = dic.keys()
    print(keys)
    # exit()
    words = []
    boxes = []
    for key in keys:
        words.append(key)
        boxes.append(dic[key])

    # add as extra columns
    assert len(words) == len(boxes)
    example['words'] = words
    example['bbox'] = boxes
    return example


def execute_optimizer_process(optimizer, model, lr, input_json_dict):
    switch = {
        "adam": lambda: AdamW(model.parameters(), lr),
        "sgd": lambda: SGD(model.parameters(), lr,
                           momentum=float(input_json_dict["layoutLMv2"]["generalConfig"]["momentum"])),
        "rmsprop": lambda: RMSprop(model.parameters(), lr,
                                   alpha=float(input_json_dict["layoutLMv2"]["generalConfig"]["alpha"]))
    }
    return switch.get(optimizer, lambda: print("Invalid optimizer"))()


def metric_calculation(u_true, u_pred):
    # Convert true labels and predicted labels to numpy arrays
    true_labels = np.array(u_true)
    predicted_labels = np.array(u_pred)

    if len(true_labels) != len(predicted_labels):
        raise ValueError(
            "Inconsistent numbers of samples: true_labels: {}, predicted_labels: {}".format(len(true_labels),
                                                                                            len(predicted_labels)))
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    print("Accuracy:", accuracy)

    # Calculate precision
    precision = precision_score(true_labels, predicted_labels, average="weighted")
    print("Precision:", precision)

    # Calculate recall
    recall = recall_score(true_labels, predicted_labels, average="weighted")
    print("Recall:", recall)

    # Calculate F1 score
    f1 = f1_score(true_labels, predicted_labels, average="weighted")
    print("F1 Score:", f1)

    # Calculate confusion matrix
    confusion_mat = confusion_matrix(true_labels, predicted_labels)
    print("Confusion Matrix:")
    print(confusion_mat)


if __name__ == "__main__":
    """________________________________________MAIN STARTS HERE______________________________________________________"""
    input_json_dict = {"layoutLMv2": {
        "modelConfig": {
            "vocabSize": "30522",
            "hiddenSize": "768",
            "numHiddenLayers": "12",
            "numAttentionHeads": "12",
            "intermediateSize": "3072",
            "hiddenAct": "gelu",
            "hiddenDropoutProb": "0.1",
            "attentionProbsDropoutProb": "0.1",
            "maxPositionEmbeddings": "512",
            "typeVocabSize": "2",
            "initializerRange": "0.02",
            "layerNormEps": "1e-12",
            "padTokenId": "0",
            "max2dPositionEmbeddings": "1024",
            "maxRelPos": "128",
            "relPosBins": "32",
            "fastQkv": "True",
            "maxRel2dPos": "256",
            "rel2dPosBins": "64",
            "convertSyncBatchnorm": "True",
            "imageFeaturePoolShape": [7, 7, 256],
            "coordinateSize": "128",
            "shapeSize": "128",
            "hasRelativeAttentionBias": "True",
            "hasSpatialAttentionBias": "True",
            "hasVisualSegmentEmbedding": "False",
            # "detectron2ConfigArgs": {}
        },
        "generalConfig": {
            "epoch": "40",
            "batchSize": "4",
            "shuffle": "True",
            "learningRate": "1e-5",
            "optimizer": "adam",
            "momentum": "0.9",
            "alpha": "0.9"
        },
        "device": "cuda"
    }}

    dataset_path = "/home/ntlpt19/Downloads/TRADE_FINANCE_OTHERS/ROOT_new"
    ocr_path = '/home/ntlpt19/Downloads/TRADE_FINANCE_OTHERS/OCR_GRN'
    labels = [label for label in os.listdir(dataset_path)]
    idx2label = {v: k for v, k in enumerate(labels)}
    label2idx = {k: v for v, k in enumerate(labels)}

    images = []
    labels = []
    for label in os.listdir(dataset_path):
        images.extend([
            f"{dataset_path}/{label}/{img_name}" for img_name in os.listdir(f"{dataset_path}/{label}")
        ])
        labels.extend([
            label for _ in range(len(os.listdir(f"{dataset_path}/{label}")))
        ])

    data = pd.DataFrame({'image_path': images, 'label': labels})

    feature_extractor = LayoutLMv2FeatureExtractor()
    tokenizer = LayoutLMv2Tokenizer.from_pretrained("microsoft/layoutlmv2-base-uncased")
    processor = LayoutLMv2Processor(feature_extractor, tokenizer)


    def encode_training_example(examples):
        images = [Image.open(path).convert("RGB") for path in examples['image_path']]
        encoded_inputs = processor(images, padding="max_length", truncation=True)
        encoded_inputs["labels"] = [label2idx[label] for label in examples["label"]]
        return encoded_inputs


    training_features = Features({
        'image': Array3D(dtype="int64", shape=(3, 224, 224)),
        'input_ids': Sequence(feature=Value(dtype='int64')),
        'attention_mask': Sequence(Value(dtype='int64')),
        'token_type_ids': Sequence(Value(dtype='int64')),
        'bbox': Array2D(dtype="int64", shape=(512, 4)),
        'labels': ClassLabel(num_classes=len(label2idx), names=list(label2idx.keys())),
    })


    def modify_llm_config(o_config):
        # Create a new configuration object based on the original configuration
        m_config = LayoutLMv2Config.from_dict(o_config.to_dict())

        # Modify the desired parameters
        m_config.num_labels = len(label2idx)
        m_config.vocab_size = int(input_json_dict["layoutLMv2"]["modelConfig"]["vocabSize"])
        m_config.hidden_size = int(input_json_dict["layoutLMv2"]["modelConfig"]["hiddenSize"])
        m_config.num_hidden_layers = int(input_json_dict["layoutLMv2"]["modelConfig"]["numHiddenLayers"])
        m_config.num_attention_heads = int(input_json_dict["layoutLMv2"]["modelConfig"]["numAttentionHeads"])
        m_config.intermediate_size = int(input_json_dict["layoutLMv2"]["modelConfig"]["intermediateSize"])
        m_config.hidden_act = str(input_json_dict["layoutLMv2"]["modelConfig"]["hiddenAct"])
        m_config.hidden_dropout_prob = float(input_json_dict["layoutLMv2"]["modelConfig"]["hiddenDropoutProb"])
        m_config.attention_probs_dropout_prob = float(
            input_json_dict["layoutLMv2"]["modelConfig"]["attentionProbsDropoutProb"])
        m_config.max_position_embeddings = int(input_json_dict["layoutLMv2"]["modelConfig"]["maxPositionEmbeddings"])
        m_config.type_vocab_size = int(input_json_dict["layoutLMv2"]["modelConfig"]["typeVocabSize"])
        m_config.initializer_range = float(input_json_dict["layoutLMv2"]["modelConfig"]["initializerRange"])
        m_config.layer_norm_eps = float(input_json_dict["layoutLMv2"]["modelConfig"]["layerNormEps"])
        m_config.pad_token_id = int(input_json_dict["layoutLMv2"]["modelConfig"]["padTokenId"])
        m_config.max_2d_position_embeddings = int(
            input_json_dict["layoutLMv2"]["modelConfig"]["max2dPositionEmbeddings"])
        m_config.max_rel_pos = int(input_json_dict["layoutLMv2"]["modelConfig"]["maxRelPos"])
        m_config.rel_pos_bins = int(input_json_dict["layoutLMv2"]["modelConfig"]["relPosBins"])
        m_config.fast_qkv = bool(input_json_dict["layoutLMv2"]["modelConfig"]["fastQkv"])
        m_config.max_rel_2d_pos = int(input_json_dict["layoutLMv2"]["modelConfig"]["maxRel2dPos"])
        m_config.rel_2d_pos_bins = int(input_json_dict["layoutLMv2"]["modelConfig"]["rel2dPosBins"])
        m_config.convert_sync_batchnorm = bool(input_json_dict["layoutLMv2"]["modelConfig"]["convertSyncBatchnorm"])
        m_config.image_feature_pool_shape = [int(val) for val in
                                             input_json_dict["layoutLMv2"]["modelConfig"]["imageFeaturePoolShape"]]
        m_config.coordinate_size = int(input_json_dict["layoutLMv2"]["modelConfig"]["coordinateSize"])
        m_config.shape_size = int(input_json_dict["layoutLMv2"]["modelConfig"]["shapeSize"])
        m_config.has_relative_attention_bias = bool(
            input_json_dict["layoutLMv2"]["modelConfig"]["hasRelativeAttentionBias"])
        m_config.has_spatial_attention_bias = bool(
            input_json_dict["layoutLMv2"]["modelConfig"]["hasSpatialAttentionBias"])
        m_config.has_visual_segment_embedding = bool(
            input_json_dict["layoutLMv2"]["modelConfig"]["hasVisualSegmentEmbedding"])
        # m_config.detectron2_config_args = dict(input_json_dict["layoutLMv2"]["modelConfig"]["detectron2ConfigArgs"])
        return m_config


    device = torch.device(input_json_dict["layoutLM"]["device"] if torch.cuda.is_available() else "cpu")
    dataset = Dataset.from_pandas(data)
    print(dataset)
    updated_dataset = dataset.map(a_ocr)
    print(dataset.column_names)

    encoded_dataset = updated_dataset.map(encode_training_example, remove_columns=updated_dataset.column_names,
                                          features=training_features,
                                          batched=True, batch_size=1)
    encoded_dataset.set_format(type='torch', device=device)

    encoded_dataset = encoded_dataset.shuffle()  # Shuffle the dataset before splitting
    '''splitting the data into training , testing and validation dataset.......................'''
    train_size = int(0.7 * len(encoded_dataset))
    test_size = int(0.2 * len(encoded_dataset))
    valid_size = len(encoded_dataset) - train_size - test_size
    train_data, test_data, valid_data = random_split(encoded_dataset, [train_size, test_size, valid_size])
    print(len(train_data), len(test_data), len(valid_data))

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=1, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)

    # print(encoded_dataset)

    batch = next(iter(train_dataloader))
    batch2 = next(iter(valid_dataloader))
    batch3 = next(iter(test_dataloader))

    # This is just for checking
    tokenizer.decode(batch['input_ids'][0].tolist())
    # print(idx2label[batch['label'][0].item()])

    '''------------------------------DEFINING THE MODEL---------------------------------'''

    original_config = LayoutLMv2Config.from_pretrained("microsoft/layoutlmv2-base-uncased")

    # Modify the configuration as needed
    modified_config = modify_llm_config(original_config)

    # Add more modifications as needed
    model = LayoutLMv2ForSequenceClassification.from_pretrained("microsoft/layoutlmv2-base-uncased",
                                                                config=modified_config)
    model.to(device)

    '''------------------------------TRAINING THE MODEL---------------------------------'''
    predicted_labels = []
    true_labels = []
    best_model = None
    best_metric = float('-inf')
    best_epoch = 0
    best_time = float('inf')
    best_loss = float('inf')
    best_performance = 0.0
    lr = float(input_json_dict["layoutLMv2"]["generalConfig"]["learningRate"])
    optimizer_str = str(input_json_dict["layoutLMv2"]["generalConfig"]["optimizer"])
    optimizer = execute_optimizer_process(optimizer_str, model, lr, input_json_dict)
    global_step = 0
    num_epochs = int(input_json_dict["layoutLMv2"]["generalConfig"]["epoch"])
    start_time = time.time()

    # put the model in training mode
    for epoch in range(num_epochs):
        print("Epoch:", epoch)
        # for training data
        training_loss = 0.0
        training_correct = 0
        # put the model in training mode
        model.train()

        for batch in train_dataloader:
            labels = batch["labels"].to(device)
            outputs = model(
                image=batch["image"].to(device),
                input_ids=batch["input_ids"].to(device), bbox=batch["bbox"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                token_type_ids=batch["token_type_ids"].to(device),
                labels=labels
            )
            # forward pass
            loss = outputs.loss

            training_loss += loss.item()
            predictions = outputs.logits.argmax(-1)
            training_correct += (predictions == labels).float().sum()

            # backward pass
            loss.backward()

            # update
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        print("Training Loss:", training_loss / len(train_data))
        training_accuracy = 100 * training_correct / len(train_data)
        print("Training accuracy:", training_accuracy.item())

        # For validation data
        validation_loss = 0.0
        validation_correct = 0
        for batch2 in valid_dataloader:
            labels = batch2["labels"].to(device)
            outputs = model(
                image=batch2["image"].to(device),
                input_ids=batch2["input_ids"].to(device), bbox=batch2["bbox"].to(device),
                attention_mask=batch2["attention_mask"].to(device),
                token_type_ids=batch2["token_type_ids"].to(device),
                labels=labels
            )
            # forward pass
            loss = outputs.loss
            validation_loss += loss.item()
            predictions = outputs.logits.argmax(-1)
            validation_correct += (predictions == labels).float().sum()

        print("Validation Loss:", validation_loss / len(valid_data))
        validation_accuracy = 100 * validation_correct / len(valid_data)
        print("Validation accuracy:", validation_accuracy.item())
        if validation_accuracy > best_metric:
            best_metric = validation_accuracy
            best_model = model
            best_loss = validation_loss
            best_epoch = epoch
            best_time = time.time() - start_time
            best_performance = validation_accuracy

        # For testing data
        testing_loss = 0.0
        testing_correct = 0
        for batch3 in test_dataloader:
            labels = batch3["labels"].to(device)
            t = labels.tolist()
            true_labels.append(t)
            outputs = model(
                image=batch3["image"].to(device),
                input_ids=batch3["input_ids"].to(device), bbox=batch3["bbox"].to(device),
                attention_mask=batch3["attention_mask"].to(device),
                token_type_ids=batch3["token_type_ids"].to(device),
                labels=labels
            )
            # forward pass
            loss = outputs.loss
            testing_loss += loss.item()
            predictions = outputs.logits.argmax(-1)
            x = predictions.tolist()
            predicted_labels.append(x)
            testing_correct += (predictions == labels).float().sum()

        print("Testing Loss:", testing_loss / len(test_data))
        testing_accuracy = 100 * testing_correct / len(test_data)
        print("Testing accuracy:", testing_accuracy.item())
        unlisted_true = [item[0] for item in true_labels]
        print(unlisted_true)
        unlisted_predicted = [item[0] for item in predicted_labels]
        print(unlisted_predicted)
        model.save_pretrained('saved_model')

    # to calculate metrics for the best model
    torch.save(best_model, 'best_model.pt')
    print(f"Best Model: Epoch {best_epoch}, "
          f"Time: {best_time:.2f} seconds, "
          f"Loss: {best_loss:.4f}, "
          f"Performance: {best_performance:.4f}")

    """_______________________________________________METRICS______________________________________________________"""
    metrics = metric_calculation(unlisted_true, unlisted_predicted)

    print(" CODE EXECUTED SUCCESSFULLY___________________________!!!!!!!!!!!!!!!!!!!!")
