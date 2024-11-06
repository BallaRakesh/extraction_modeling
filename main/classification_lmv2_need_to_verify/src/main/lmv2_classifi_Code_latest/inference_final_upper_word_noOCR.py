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


from seqeval.metrics import (
	classification_report)


def convert_values(input_list, value_mapping):
    converted_list = [f'"{value_mapping.get(value, value)}' for value in input_list]
    return converted_list

def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]


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
    return accuracy, precision, recall, f1, confusion_mat


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

dataset_path = "/home/ntlpt19/Downloads/TRADE_FINANCE_OTHERS/ROOT_samp"
label_path = '/home/ntlpt19/Downloads/TRADE_FINANCE_OTHERS/ROOT_MAIN_RESULTS/MODEL_2nd_itterate/MODEL_EPOCH/label.txt'
model_path = '/home/ntlpt19/Downloads/TRADE_FINANCE_OTHERS/ROOT_MAIN_RESULTS/MODEL_2nd_itterate/saved_model_24'

labels = [label for label in os.listdir(dataset_path)]
idx2label = {v: k for v, k in enumerate(labels)}
label2idx = {k: v for v, k in enumerate(labels)}

# with open('label.txt', 'w') as label_file:
# 	label_file.write(json.dumps(label2idx))
 


with open(label_path, "r") as f:
    label_train_gen = f.read().splitlines()
    
label_dict = json.loads(label_train_gen[0])
print(label_dict)
labeltoidx=label_dict    
#exit()

label_list = list(labeltoidx.keys())
classes = label_list
print(classes)


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
    encoded_inputs['image_path'] = examples['image_path']
    return encoded_inputs


training_features = Features({
    'image': Array3D(dtype="int64", shape=(3, 224, 224)),
    'input_ids': Sequence(feature=Value(dtype='int64')),
    'attention_mask': Sequence(Value(dtype='int64')),
    'token_type_ids': Sequence(Value(dtype='int64')),
    'bbox': Array2D(dtype="int64", shape=(512, 4)),
    'labels': ClassLabel(num_classes=len(label2idx), names=list(label2idx.keys())),
    'image_path': Value(dtype='string')
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
    m_config.max_2d_position_embeddings = int(input_json_dict["layoutLMv2"]["modelConfig"]["max2dPositionEmbeddings"])
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
    m_config.has_spatial_attention_bias = bool(input_json_dict["layoutLMv2"]["modelConfig"]["hasSpatialAttentionBias"])
    m_config.has_visual_segment_embedding = bool(
        input_json_dict["layoutLMv2"]["modelConfig"]["hasVisualSegmentEmbedding"])
    # m_config.detectron2_config_args = dict(input_json_dict["layoutLMv2"]["modelConfig"]["detectron2ConfigArgs"])
    return m_config


device = torch.device(input_json_dict["layoutLM"]["device"] if torch.cuda.is_available() else "cpu")
dataset = Dataset.from_pandas(data)
print(dataset)
# updated_dataset = dataset.map(a_ocr)
print(dataset.column_names)

encoded_dataset = dataset.map(encode_training_example, remove_columns=dataset.column_names, features=training_features,
                              batched=True, batch_size=1)
encoded_dataset.set_format(type='torch', device=device)
encoded_dataset = encoded_dataset.shuffle()  # Shuffle the dataset before splitting

model = LayoutLMv2ForSequenceClassification.from_pretrained(model_path, num_labels=len(label_dict))
test_dataloader = torch.utils.data.DataLoader(encoded_dataset, batch_size=1, shuffle=True)

correctpred = 0
wrongpred = 0
imglabel = []
imgname = []
imgpred = []
imgpredper = []

for batch3 in test_dataloader:
    image_path = str(batch3["image_path"])
    labels = batch3["labels"].to(device)
    outputs = model(
        image=batch3["image"].to(device),
        input_ids=batch3["input_ids"].to(device), bbox=batch3["bbox"].to(device),
        attention_mask=batch3["attention_mask"].to(device),
        token_type_ids=batch3["token_type_ids"].to(device)
        
    )
    print(image_path)
    classification_logits = outputs.logits
    classification_results = torch.softmax(classification_logits, dim=1).tolist()[0]
    cla = []
    clr = []
    val1 = 0
    for i in range(len(classes)):
        val = int(round(classification_results[i] * 100))
        if val1 < val:
            val1 = val
            classifiedas = f"{classes[i]}: {float((classification_results[i] * 100))}%"
    #print(img_name)        
    print('original:', idx2label[labels.item()], "********###**********", 'prediction :', classifiedas)
    if classes[i] == str(classifiedas).split(':')[0]:
        correctpred += 1
    else:
        wrongpred += 1
        # logging.info('Processing image: %s', image_path)
        # logging.info('Expected label: %s', lab1)
        # logging.info('Predicted label: %s', classifiedas)
    

    imgpred.append(str(classifiedas).split(':')[0])
    imgpredper.append(str(classifiedas).split(':')[1])




print(" CODE EXECUTED SUCCESSFULLY___________________________!!!!!!!!!!!!!!!!!!!!")

