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
from sklearn.model_selection import train_test_split

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
        "epoch": "2",
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


labels = [label for label in os.listdir(dataset_path)]
idx2label = {v: k for v, k in enumerate(labels)}
label2idx = {k: v for v, k in enumerate(labels)}

with open('label.txt', 'w') as label_file:
	label_file.write(json.dumps(label2idx))
 
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
tokenizer = LayoutLMv2Tokenizer.from_pretrained("microsoft/layoutlmv2-large-uncased")
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
datasetall = Dataset.from_pandas(data)
print(type(datasetall))

training_set, testing_set = train_test_split(datasetall, test_size=0.2, random_state=42)
print(datasetall.column_names)

# training_df = training_set.to_pandas()
# testing_df = testing_set.to_pandas()

# Create an Excel writer object
df1 = pd.DataFrame(training_set)
df2 = pd.DataFrame(testing_set)

# Write the DataFrames to separate sheets in the Excel file
df1.to_csv('training_set.csv')
df2.to_csv('testing_set.csv')

def train_test_split_base(dataset, flag):
    dataset = Dataset.from_dict(dataset)
    if flag=='train':
        updated_dataset = dataset.map(a_ocr)  #needed to remove this if we only relay on lmv2 processor training
        encoded_dataset = updated_dataset.map(encode_training_example, remove_columns=updated_dataset.column_names, features=training_features,
                                    batched=True, batch_size=4)
        encoded_dataset.set_format(type='torch', device=device)

        encoded_dataset = encoded_dataset.shuffle()  # Shuffle the dataset before splitting
        '''splitting the data into training , testing and validation dataset.......................'''
        # train_size = int(0.8 * len(encoded_dataset))
        # valid_size = int(0.2 * len(encoded_dataset))
        #valid_size = len(encoded_dataset) - train_size - test_size
        train_data = encoded_dataset#, valid_data = random_split(encoded_dataset, [train_size, valid_size])
        print(len(train_data))
        return train_data
    elif flag=='test':
        updated_dataset = dataset.map(a_ocr)
        encoded_dataset = updated_dataset.map(encode_training_example, remove_columns=updated_dataset.column_names, features=training_features,
                                    batched=True, batch_size=4)#make it as 1
        encoded_dataset.set_format(type='torch', device=device)

        encoded_dataset = encoded_dataset.shuffle()  # Shuffle the dataset before splitting
        '''splitting the data into training , testing and validation dataset.......................'''
        # train_size = int(0.8 * len(encoded_dataset))
        # valid_size = int(0.2 * len(encoded_dataset))
        #valid_size = len(encoded_dataset) - train_size - test_size
        valid_data = encoded_dataset   #random_split(encoded_dataset, [train_size, valid_size])
        print(len(valid_data))    
        return valid_data    
        

train_data = train_test_split_base(training_set , 'train')
valid_data = train_test_split_base(testing_set , 'test')



train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=4, shuffle=True)
#test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=True)

# print(encoded_dataset)

batch = next(iter(train_dataloader))
batch2 = next(iter(valid_dataloader))
#batch3 = next(iter(test_dataloader))

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

best_model = None
best_metric = float('-inf')
best_epoch = 0
best_time = float('inf')
best_loss = None#float('inf')
best_pression = None
best_recall = None
best_f1 = None
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
    with open("final_result.txt", 'a') as f:
        print("Training Loss:", training_loss / len(train_data))
        f.write(f"Training Loss: :{training_loss / len(train_data)}\n")
        f.write(f"loss value after complete epoch : {loss}")
        training_accuracy = 100 * training_correct / len(train_data)
        print("Training accuracy:", training_accuracy.item())
        f.write(f"training_accuracy :{training_accuracy.item()}\n")

    
    predicted_labels = []
    true_labels = []
    # For validation data
    validation_loss = 0.0
    validation_correct = 0
    for batch2 in valid_dataloader:
        labels = batch2["labels"].to(device)
        t = labels.tolist()
        true_labels.append(t)
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
        x = predictions.tolist()
        predicted_labels.append(x)
        
    with open("final_result.txt", 'a') as f:
        
        print("Validation Loss:", validation_loss / len(valid_data))
        f.write(f"Validation Loss: :{validation_loss / len(valid_data)}\n")
        
        validation_accuracy = 100 * validation_correct / len(valid_data)
        print("Validation accuracy:", validation_accuracy.item())
        f.write(f"Validation accuracy: {validation_accuracy.item()}\n")
        validation_loss = validation_loss / len(valid_data)
    


    unlisted_true = [item[0] for item in true_labels]
    print(unlisted_true)
    unlisted_predicted = [item[0] for item in predicted_labels]
    print(unlisted_predicted)
    model.save_pretrained(f'saved_model_{epoch}')
    accuracy, precision, recall, f1, confusion_mat = metric_calculation(unlisted_true, unlisted_predicted)

    if best_loss is None:
        best_loss = validation_loss
    if best_pression is None:
        best_pression = precision
    if best_recall is None:
        best_recall = recall
    if best_f1 is None:
        best_f1 = f1
        #and f1 > best_f1 and recall > best_recall:
    if validation_loss < best_loss and f1 > best_f1 and recall > best_recall:
        best_metric = validation_accuracy
        best_model = model
        best_loss = validation_loss
        best_f1 = f1
        best_recall = recall
        best_epoch = epoch
        best_time = time.time() - start_time
        best_performance = validation_accuracy

    
    ground_truth = convert_values(unlisted_true, idx2label)
    pred_val = convert_values(unlisted_predicted, idx2label)

    with open("final_result.txt", 'a') as f:
        f.write(f"accuracy: {accuracy}\n")
        f.write(f"precision: {precision}\n")
        f.write(f"recall: {recall}\n")
        f.write(f"f1: {f1}\n")
        f.write(f"{confusion_mat}\n")
        f.write(classification_report([ground_truth], [pred_val]))

# to calculate metrics for the best model
# to calculate metrics for the best model
torch.save(best_model, 'best_model_large_model.pt')
best_model.save_pretrained('best_model_pre')
print(f"Best Model: Epoch {best_epoch}, "
      f"Time: {best_time:.2f} seconds, "
      f"Loss: {best_loss:.4f}, "
      f"Performance: {best_performance:.4f}")
with open("final_result.txt", 'a') as f:
    f.write(f"Best Model: Epoch {best_epoch}, "
      f"Time: {best_time:.2f} seconds, "
      f"Loss: {best_loss:.4f}, "
      f"Performance: {best_performance:.4f}\n")

"""_______________________________________________METRICS______________________________________________________"""

    #f.write(f"confusion_mat: {confusion_mat}\n")

print(" CODE EXECUTED SUCCESSFULLY___________________________!!!!!!!!!!!!!!!!!!!!")

