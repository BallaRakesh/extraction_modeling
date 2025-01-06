import torch
import pandas as pd
import os
import json
from glob import glob
import imagesize
from collections import Counter
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from transformers import BertTokenizer
from transformers import LayoutLMv2Config, LayoutLMv2ForTokenClassification
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import logging
from typing import List
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)

import torch
import pandas as pd
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from os import listdir

config = LayoutLMv2Config(
	vocab_size=30522,  # Match BERT vocabulary size
	hidden_size=768,
	num_hidden_layers=12,
	num_attention_heads=12,
	intermediate_size=3072,
	max_position_embeddings=512,
	max_2d_position_embeddings=1024,
	image_feature_pool_shape=[7, 7, 256],
	coordinate_size=128,
	shape_size=128,
	has_relative_attention_bias=True,
	has_spatial_attention_bias=True,
	has_visual_segment_embedding=True,
	num_labels=31#len(labels)  # Set based on your number of labels
)




import torch
import pandas as pd
import os
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
from os import listdir
import itertools


class CustomDataset(Dataset):
    def __init__(
        self,
        annotations,
        image_dir,
        tokenizer,
        max_seq_length=512,
        img_h=768,
        img_w=768,
    ):
        """
        Custom dataset aligned with GeoLayoutLM preprocessing
        
        Args:
            annotations (pd.DataFrame): DataFrame containing the annotations
            image_dir (str): Directory containing images
            tokenizer: BERT tokenizer
            max_seq_length (int): Maximum sequence length
            img_h (int): Image height to resize to
            img_w (int): Image width to resize to
        """
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.img_h = img_h
        self.img_w = img_w
        
        # Get image file names
        self.image_file_names = list(listdir(image_dir))
        self.words, self.labels, self.boxes = annotations
        
        # # Extract data from DataFrame
        # self.words = annotations['words'].tolist()
        # self.boxes = annotations['boxes'].tolist()
        # self.labels = annotations['labels'].tolist()
        
        # Set up tokenizer special tokens
        if getattr(self.tokenizer, "vocab", None) is not None:
            self.pad_token_id = self.tokenizer.vocab["[PAD]"]
            self.cls_token_id = self.tokenizer.vocab["[CLS]"]
            self.sep_token_id = self.tokenizer.vocab["[SEP]"]
            self.unk_token_id = self.tokenizer.vocab["[UNK]"]
        else:
            self.pad_token_id = self.tokenizer.pad_token_id
            self.cls_token_id = self.tokenizer.cls_token_id
            self.sep_token_id = self.tokenizer.sep_token_id
            self.unk_token_id = self.tokenizer.unk_token_id
            
        # Create label mappings
        unique_labels = set([label for label_list in self.labels for label in label_list])
        self.label2id = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        # Initialize return dictionary with proper padding
        return_dict = {
            "input_ids": np.ones(self.max_seq_length, dtype=int) * self.pad_token_id,
            "attention_mask": np.zeros(self.max_seq_length, dtype=int),
            "token_type_ids": np.zeros(self.max_seq_length, dtype=int),
            "bbox": np.zeros((self.max_seq_length, 4), dtype=np.float32),
            "labels": np.zeros(self.max_seq_length, dtype=int) - 100,  # -100 for ignored positions
        }
        
        # Load and process image
        image_path = os.path.join(self.image_dir, self.image_file_names[idx])
        image = cv2.resize(cv2.imread(image_path, 1), (self.img_w, self.img_h))
        image = image.astype("float32").transpose(2, 0, 1)  # CHW format
        
        # Process words and boxes
        words = self.words[idx]
        boxes = self.boxes[idx]
        labels = self.labels[idx]
        
        # Get original image dimensions for normalization
        orig_img = cv2.imread(image_path)
        width = orig_img.shape[1]
        height = orig_img.shape[0]
        
        # Tokenize and align boxes
        list_tokens = []
        list_boxes = []
        list_labels = []
        
        # Add CLS token
        list_tokens.append(self.cls_token_id)
        list_boxes.append([0, 0, 0, 0])
        list_labels.append(-100)
        
        # Process each word
        for word, box, label in zip(words, boxes, labels):
            if len(list_tokens) >= self.max_seq_length - 2:  # Account for [CLS] and [SEP]
                break
                
            # Tokenize word
            word_tokens = self.tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [self.unk_token_id]
                
            # Normalize box coordinates
            normalized_box = [
                int(box[0] / width * 1000),  # x1
                int(box[1] / height * 1000),  # y1
                int(box[2] / width * 1000),  # x2
                int(box[3] / height * 1000)   # y2
            ]
            
            # Add tokens and boxes
            for _ in word_tokens:
                list_tokens.append(self.tokenizer.convert_tokens_to_ids(word_tokens)[0])
                list_boxes.append(normalized_box)
                list_labels.append(self.label2id.get(label, -100))
        
        # Add SEP token
        list_tokens.append(self.sep_token_id)
        list_boxes.append([0, 0, 0, 0])
        list_labels.append(-100)
        
        # Calculate sequence length
        seq_length = len(list_tokens)
        ######################################
        ######################################
        # Update return dictionary
        return_dict["input_ids"][:seq_length] = list_tokens
        return_dict["attention_mask"][:seq_length] = 1
        return_dict["token_type_ids"][:seq_length] = 0
        return_dict["bbox"][:seq_length] = list_boxes
        return_dict["labels"][:seq_length] = list_labels
        
        # In CustomDataset.__getitem__
        # return_dict = {
        #     "input_ids": np.ones(self.max_seq_length, dtype=np.int64),  # Changed to int64
        #     "attention_mask": np.zeros(self.max_seq_length, dtype=np.int64),
        #     "token_type_ids": np.zeros(self.max_seq_length, dtype=np.int64),
        #     "bbox": np.zeros((self.max_seq_length, 4), dtype=np.float32),
        #     "labels": np.zeros(self.max_seq_length, dtype=np.int64) - 100,
        # }
        
        
        # Convert to tensors
        for k, v in return_dict.items():
            if isinstance(v, np.ndarray):
                return_dict[k] = torch.from_numpy(v)
        
        return_dict["image"] = torch.from_numpy(image)
        
        return return_dict

# Example usage:
def create_datasets(folder_path, tokenizer):
    """
    Create train and test datasets from pickle files
    """
    # Load pickle files
    train = pd.read_pickle(os.path.join(folder_path, 'train.pkl'))
    test = pd.read_pickle(os.path.join(folder_path, 'test.pkl'))
    
    # Create datasets
    train_dataset = CustomDataset(
        annotations=train,
        image_dir=os.path.join(folder_path, "train/"),
        tokenizer=tokenizer
    )
    
    test_dataset = CustomDataset(
        annotations=test,
        image_dir=os.path.join(folder_path, "test/"),
        tokenizer=tokenizer
    )
    
    return train_dataset, test_dataset

def train_model():
    output_dir = '/home/data_science/geo_testing/COO_V3'
    num_epochs=40
    # Set up logging
    logging.basicConfig(
        filename=os.path.join(output_dir, 'training.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    folder_path = '/home/data_science/geo_testing/COO_V3'
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Create datasets
    train_dataset, test_dataset = create_datasets(folder_path, tokenizer)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    print(train_loader)
    test_loader = DataLoader(test_dataset, batch_size=4)
    
    # Initialize model
    # config = LayoutLMv2Config.from_pretrained(config_path)
    model = LayoutLMv2ForTokenClassification(config)
    
    # Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    # labels = list(set(all_labels))
    global_step = 0
    num_train_epochs = 40
    preds_val = None
    out_label_ids = None
    best_loss=None
    best_precision=None
    best_recall=None
    best_f1=None
    steps = []
    losses = []
    training_loss = {}  
    validation_loss = {}
    # put the model in training mode
    model.train()
    best_model_flag_high = False
    best_model_flag_low = False
    for epoch in range(num_train_epochs):
        print("Epoch:", epoch)
        for batch in tqdm(train_loader):
            input_ids = batch['input_ids'].long().to(device)  # Modified
            bbox = batch['bbox'].to(device)
            image = batch['image'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            print(labels)
            print(type(labels))
            print(labels.shape)
            exit('OKKK')
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(input_ids=input_ids,
                            bbox=bbox,
                            image=image,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            labels=labels)
            loss = outputs.loss

            # print loss every epoch
            if (global_step + 1) % len(train_dataloader) == 0 or global_step == 0:
                print(f"Loss after {global_step} steps: {loss.item()}")
                steps.append(global_step)
                losses.append(float(loss.item()))
            loss.backward()
            optimizer.step()
            global_step += 1
        # model.eval()
        training_loss[epoch] = loss 
        val_loss = 0.0
        preds_val = None
        for batch in tqdm(test_loader, desc="Evaluating"):
            with torch.no_grad():
                input_ids = batch['input_ids'].to(device)
                bbox = batch['bbox'].to(device)
                image = batch['image'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids,
                                bbox=bbox,
                                image=image,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                labels=labels)
                val_los= outputs.loss
                print(f'batch validataion loss: {val_los}')
                val_loss += val_los.item()

                if preds_val is None:
                    preds_val = outputs.logits.detach().cpu().numpy()
                    out_label_ids = batch["labels"].detach().cpu().numpy()
                else:
                    preds_val = np.append(preds_val, outputs.logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(
                        out_label_ids, batch["labels"].detach().cpu().numpy(), axis=0)

        labels = list(set(all_labels))
        val_result, class_report = results_test(preds_val, out_label_ids, labels)

        print(f"precison: {val_result['precision']}")
        print(f"recall: {val_result['recall']}")
        print(f"f1: {val_result['f1']}")
        print('+++++++++++++++++++++++++++++++++++++++++++')
        val_loss= val_loss /len(test_dataloader)
        validation_loss[epoch] = val_loss
        print(f'final validation loss:{val_loss}')
        # print(val_result)
        # with train_writer.as_default():
        #     tf.summary.scalar("train loss ", loss.detach().cpu(), step=epoch)
        # with test_writer.as_default():
        #     tf.summary.scalar("Validation Loss", val_loss, step=epoch)   
        #precision, recall values need to log
        
        train_writer.add_scalar("train loss", loss.detach(), epoch)
        test_writer.add_scalar("val loss", val_los, epoch)
        
        
        precision = val_result['precision']
        recall = val_result['recall']
        f1= val_result['f1']
        
        # val metrics
        test_writer.add_scalar("val precision ", precision, epoch)
        test_writer.add_scalar("val f1",f1, epoch)
        test_writer.add_scalar("val recall ", recall, epoch)
        
        
        
        if  best_loss is None:
            best_loss=val_loss
        if best_precision is None:
            best_precision = precision
            best_recall = recall
            best_f1 = f1
        # print(f"best precison: {best_precision}")
        # print(f"best recall: {best_recall}")
        
        if val_loss < best_loss and f1 > best_f1 and recall > best_recall:
            best_model_flag_high = True
            best_loss = val_loss
            best_precision = precision
            best_recall = recall
            best_f1 = f1
            name = "Best_Model"

            if not os.path.exists(os.path.join(folder_path, name)):
                os.mkdir(os.path.join(folder_path, name))
            
            print(f'Model is {epoch} saving +++++++++++++++++++++++++++++++++')
            with open(os.path.join(folder_path, "model_saving_info.txt"), 'a') as f:
                f.write(f"Model is {epoch} saving +++++++++++++++++++++++++++++++++\n")

            # with best_train_test_writer.as_default():
            #     tf.summary.scalar("Best train loss ", best_loss, step=epoch)
            # with best_train_test_writer.as_default():
            #     tf.summary.scalar("best_precision ", best_precision, step=epoch) 
            # with best_train_test_writer.as_default():
            #     tf.summary.scalar("best_f1",best_f1, step=epoch) 
            # with best_train_test_writer.as_default():
            #     tf.summary.scalar("best_recall ", best_recall, step=epoch) 
            
            # best metrics 
            best_writer.add_scalar("Best loss ", best_loss, epoch)
            best_writer.add_scalar("best precision ", best_precision, epoch)
            best_writer.add_scalar("best f1",best_f1, epoch)
            best_writer.add_scalar("best recall ", best_recall, epoch)
            
            print(f"best Validation Loss: {best_loss}" )
            print("best Precision:", best_precision)
            print("best Recall:", best_recall) 
            print("best f1:", best_f1)
            model.save_pretrained(os.path.join(folder_path, name))

        if val_loss < best_loss:
            best_model_flag_low = True
            best_loss = val_loss
            best_writer.add_scalar("Best loss ", best_loss, epoch)
            best_writer.add_scalar("best precision ", best_precision, epoch)
            best_writer.add_scalar("best f1",best_f1, epoch)
            best_writer.add_scalar("best recall ", best_recall, epoch)
            name = "Best_Model_low"
            best_model_low = model
            
    if not best_model_flag_high and best_model_flag_low:
        best_model_low.save_pretrained(os.path.join(folder_path, name))
        


    #give best model path here
    if best_model_flag_high:
        model_path = f"{folder_path}/Best_Model"
    elif best_model_flag_low:
        model_path =  f"{folder_path}/Best_Model_low"
    else:
        exit("no best model exist")
    

 
def compute_metrics(predictions, true_labels):
    # Remove padding (-100)
    true_predictions = [p for p, l in zip(predictions, true_labels) if l != -100]
    true_labels = [l for l in true_labels if l != -100]
    
    return {
        'precision': precision_score(true_labels, true_predictions, average='weighted'),
        'recall': recall_score(true_labels, true_predictions, average='weighted'),
        'f1': f1_score(true_labels, true_predictions, average='weighted')
    }

if __name__ == "__main__":
    # # Configuration
    # config_path = "config.json"
    # train_dir = "data/train"
    # test_dir = "data/test"
    # output_dir = "output"
    
    # # Create output directory
    # os.makedirs(output_dir, exist_ok=True)
    
    # Train model
    # model = train_model(train_dir, test_dir, output_dir)
    model = train_model()