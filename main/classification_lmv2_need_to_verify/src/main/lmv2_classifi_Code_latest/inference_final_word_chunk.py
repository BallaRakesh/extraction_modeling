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
from torch.nn.utils.rnn import pad_sequence


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


# User will always have an option , whether to perform ocr manually or by automated layoutLMv2 model.
# LayoutLMv2 has OCR inbuilt unlike LayoutLM.
def a_ocr(example):
    print(example)
    name = example['image_path'].split("/")[-1].split(".")[0]
    try:
        print(f'avalable:{name} ')
        ocr_gen = ocr_path+'/'+name+'.json'
        print(ocr_gen)
        print(name)
        with open(ocr_gen, 'r') as f:
            data0 = json.load(f)
    except:
        print('not avalable',example['image_path'])
        j_data = ApplyOcr.apply_ocr(example)
        data0 = eval(j_data)
        # print(data0)
        
    dic = data0
    #print(dic)
    #exit()
    keys = dic.keys()
    #print(keys)
    #exit()
    words = []
    boxes = []
    for key in keys:
        words.append(key)
        boxes.append(dic[key])

    # add as extra columns
    assert len(words) == len(boxes)
    example['words'] = words
    example['bbox'] = boxes
    # example[1]={'image_path':'abc/bvn', 'label':'cs', 'words':['a', 'b'],'bbox': [[19, 897, 25, 915]]}
    # print(example.keys())
    print(example)
    # exit("example   ***************************")
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
ocr_path = '/New_Volume/Rakesh/layoutlmv2/classification/OCR_PATH'
label_path = '/home/ntlpt19/Downloads/TRADE_FINANCE_OTHERS/ROOT_BACKUP/label.txt'
model_path = '/home/ntlpt19/Downloads/TRADE_FINANCE_OTHERS/ROOT_BACKUP/saved_model'

labels = [label for label in os.listdir(dataset_path)]
idx2label = {v: k for v, k in enumerate(labels)}
label2idx = {k: v for v, k in enumerate(labels)}


 
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




def encode_training_example(examples):
    # print(examples)
    
    images = [Image.open(path).convert("RGB") for path in examples['image_path']]
    encoded_inputs = processor(images, truncation=True)
    encoded_inputs["labels"] = [label2idx[label] for label in examples["label"]]
    encoded_inputs['image_path'] = examples['image_path']
    # print(encoded_inputs)#['input_ids'])
    # exit("&&&&&&&&&&&&&&&&&")
    
    return encoded_inputs

feature_extractor = LayoutLMv2FeatureExtractor()
tokenizer = LayoutLMv2Tokenizer.from_pretrained("microsoft/layoutlmv2-base-uncased")
# processor = LayoutLMv2Processor(feature_extractor, tokenizer)
processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")


def encode_example(example):
    images = [Image.open(path).convert("RGB") for path in example['image_path']]
    encoding = processor(images, example['words'], boxes=example['bbox'], padding="max_length", truncation=True)#, return_tensors="pt")#, truncation=True)#, apply_ocr=False)
    encoding["labels"] = [label2idx[label] for label in example["label"]]
    encoding['image_path'] = example['image_path']
    return encoding


training_features = Features({
    'image': Array3D(dtype="int64", shape=(3, 224, 224)),
    'input_ids': Sequence(feature=Value(dtype='int64')),
    'attention_mask': Sequence(Value(dtype='int64')),
    'token_type_ids': Sequence(Value(dtype='int64')),
    'bbox': Array2D(dtype="int64", shape=(512, 4)),
    'labels': ClassLabel(num_classes=len(label2idx), names=list(label2idx.keys())),
    'image_path': Value(dtype='string'),
    'label': Value(dtype='string'),
    'words': Sequence(feature=Value(dtype='string'))
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


'''df = dataset.to_pandas() #converted to data frame
print(df)
new_row = pd.DataFrame({'image_path': ['/path/to/new/image.jpeg'], 'label': ['NEW_LABEL']})

df = pd.concat([df, new_row], ignore_index=True)
print(df)'''

# exit("++++++++++++++++++++++++++++++++++")

updated_dataset = dataset.map(a_ocr)
print(dataset.column_names)


print(dataset.column_names)
print('this is updated data at zero index')
print(updated_dataset[0]['words'])
print(len(updated_dataset[0]['words']))

print(updated_dataset[0]['bbox'])
print(updated_dataset[0]['image_path'])

# exit("**************")
print(updated_dataset)



columns_names = ['image_path', 'label', 'words', 'bbox']  # Replace with your desired column names
new_updated_dataset = pd.DataFrame(columns=columns_names)

import shutil
print(len(updated_dataset)) 
for i in range(len(updated_dataset)):
    updated_dataset_len = len(updated_dataset[i]['words'])
    if updated_dataset_len > 250:
        print("+++++++++++++$$$$$$$$$$$$$$$$$$$")
        my_list_words = updated_dataset[i]['words']
        my_list_bbox = updated_dataset[i]['bbox']
        chunk_size = 250
        word_chunks = []
        bbox_chunks = []
        for k in range(0, len(my_list_words), chunk_size):
            chunk = my_list_words[k:k + chunk_size]
            word_chunks.append(chunk)
            chunk = my_list_bbox[k:k + chunk_size]
            bbox_chunks.append(chunk)
        print(word_chunks)
        print(len(word_chunks))
        # shutil.copy(updated_dataset[i]['image_path'], updated_dataset[i]['image_path'][0:-4]+'1'+'.png')
        # print(updated_dataset[i]['image_path'][0:-4]+'1'+'.png')
        # exit("********")
        for j in range(len(word_chunks)):
            new_row = {'image_path': updated_dataset[i]['image_path'], 'label': updated_dataset[i]['label'], 'words': word_chunks[j], 'bbox': bbox_chunks[j]}
            new_row_df = pd.DataFrame([new_row])
            new_updated_dataset = pd.concat([new_updated_dataset, new_row_df], ignore_index=True)
    else:
        new_row = {'image_path': updated_dataset[i]['image_path'], 'label': updated_dataset[i]['label'], 'words': updated_dataset[i]['words'], 'bbox': updated_dataset[i]['bbox']}
        new_row_df = pd.DataFrame([new_row])
        new_updated_dataset = pd.concat([new_updated_dataset, new_row_df], ignore_index=True)
        
        
print('chuncks', new_updated_dataset)   


new_updated_dataset = Dataset.from_pandas(new_updated_dataset)

# updated_dataset = updated_dataset.to_pandas()
print(updated_dataset)   
        

for i in range(len(updated_dataset)):
    input_id_chunks = list(updated_dataset[i]['words'])
    print(input_id_chunks)



#, remove_columns=updated_dataset.column_names
# encoded_dataset = new_updated_dataset.map(encode_training_example, features=training_features,
#                               batched=True, batch_size=1)                                   #actuallllllllllllllll
encoded_dataset = new_updated_dataset.map(lambda example: encode_example(example), features=training_features, batched=True, batch_size=1)

encoded_dataset.set_format(type='torch', device=device)

for example in encoded_dataset:
    for key, value in example.items():
        if hasattr(value, 'shape'):
            print(f"Feature: {key}, Shape: {value.shape}")


for i in range(len(encoded_dataset)):
    print(f"i==============={i}")
    print("#########################################")
    print("#########################################")
    print(encoded_dataset[i]['input_ids'].numpy().tolist())  
    print(len(encoded_dataset[i]['input_ids'].numpy().tolist())) 
    print(encoded_dataset[i]['words'])                                                                                                                                                                                                                                                                                                                                                    
                                                                                                                                                                                                                                                                                                                                                       
    print("#########################################")
    print("#########################################")                                                                                                                                                                                                     
# exit("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")


model = LayoutLMv2ForSequenceClassification.from_pretrained(model_path, num_labels=len(label_dict))                                                                
required_columns = ["image", "input_ids", "bbox", "attention_mask", "token_type_ids", "labels", "image_path"]  # List of column names to keep

new_modified_dataset = []
for example in encoded_dataset:
    new_example = {key: value for key, value in example.items() if key in required_columns}
    new_modified_dataset.append(new_example)
correctpred = 0
wrongpred = 0
imglabel = []
imgname = []
imgpred = []
imgpredper = []


test_dataloader = torch.utils.data.DataLoader(new_modified_dataset, batch_size=1, shuffle=True)


for batch3 in test_dataloader:
    image_path = str(batch3["image_path"])
    labels = batch3["labels"].to(device)
    print("###################################")
    print("###################################")
    print("###################################")
    print("###################################")
    print(batch3["input_ids"])
    outputs = model(
        image=batch3["image"].to(device),
        input_ids=batch3["input_ids"].to(device), bbox=batch3["bbox"].to(device),
        attention_mask=batch3["attention_mask"].to(device),
        token_type_ids=batch3["token_type_ids"].to(device)
        
    )

    print(image_path)
    classification_logits = outputs.logits
    classification_results = torch.softmax(classification_logits, dim=1).tolist()
    print(classification_results)
    print(classification_logits)
    print('??????????????????????????????????????????')
    # exit()
    cla = []
    clr = []
    final_prediction = []
    val1 = 0
    for results in classification_results:
        for i in range(len(classes)):
            val = int(round(results[i] * 100))
            if val1 < val:
                val1 = val
                classifiedas = f"{classes[i]}: {float((results[i] * 100))}%"
        final_prediction.append(classifiedas)

    labels_list = labels.tolist()
    for ground_truth, predictions in zip(labels_list, final_prediction):
        print('original:', idx2label[ground_truth], "********###**********", 'prediction :', predictions)   

    
    # if classes[i] == str(classifiedas).split(':')[0]:
    #     correctpred += 1
    # else:
    #     wrongpred += 1
    #     # logging.info('Processing image: %s', image_path)
    #     # logging.info('Expected label: %s', lab1)
    #     # logging.info('Predicted label: %s', classifiedas)
    

    # imgpred.append(str(classifiedas).split(':')[0])
    # imgpredper.append(str(classifiedas).split(':')[1])
