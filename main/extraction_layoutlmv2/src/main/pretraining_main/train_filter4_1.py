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
from torch.utils.tensorboard import SummaryWriter


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




from transformers import BatchEncoding
import torch
import pandas as pd
import os
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
from os import listdir
import itertools

doc_code = 'board'
doc_code_ = 'lc'
train_writer = SummaryWriter(log_dir=f'''logs/{doc_code}/{doc_code_}/{"_".join(str(datetime.now()).split(" "))}/train''')
test_writer = SummaryWriter(log_dir=f'''logs/{doc_code}/{doc_code_}/{"_".join(str(datetime.now()).split(" "))}/test''')
best_writer = SummaryWriter(log_dir=f'''logs/{doc_code}/{doc_code_}/{"_".join(str(datetime.now()).split(" "))}/best''')

def results_test(preds, out_label_ids, labels):
	preds = np.argmax(preds, axis=2)

	label_map = dict(enumerate(labels))

	out_label_list = [[] for _ in range(out_label_ids.shape[0])]
	preds_list = [[] for _ in range(out_label_ids.shape[0])]

	for i in range(out_label_ids.shape[0]):
		for j in range(out_label_ids.shape[1]):
			if out_label_ids[i, j] != -100:
				out_label_list[i].append(label_map[out_label_ids[i][j]])
				preds_list[i].append(label_map[preds[i][j]])

	results = {
		"precision": precision_score(out_label_list, preds_list),
		"recall": recall_score(out_label_list, preds_list),
		"f1": f1_score(out_label_list, preds_list)
	}
	return results, classification_report(out_label_list, preds_list)

from torchvision import transforms
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import BatchEncoding
import os
from os import listdir

import os

# Set environment variables
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

# Optional: Verify that the variables are set
print(f"CUDA_LAUNCH_BLOCKING = {os.environ.get('CUDA_LAUNCH_BLOCKING')}")
print(f"TORCH_USE_CUDA_DSA = {os.environ.get('TORCH_USE_CUDA_DSA')}")
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("CUDA device current:", torch.cuda.current_device())
print("CUDA device name:", torch.cuda.get_device_name(0))

class CustomDataset(Dataset):
    def __init__(
        self,
        label2id,
        id2label,
        annotations,
        image_dir,
        tokenizer,
        max_seq_length=512,
        img_size=224,
    ):
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.img_size = img_size
        
        self.image_file_names = list(listdir(image_dir))
        self.words, self.labels, self.boxes = annotations
        
        # Verify data
        assert len(self.words) == len(self.labels) == len(self.boxes) == len(self.image_file_names), \
            "Mismatch in data lengths"

        self.label2id = label2id
        self.id2label = id2label

        # Define special tokens
        self.pad_token_id = tokenizer.pad_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.unk_token_id = tokenizer.unk_token_id

        # Define transforms explicitly for debugging
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        # Basic input structure
        encoded_inputs = {
            "input_ids": torch.ones(self.max_seq_length, dtype=torch.long) * self.pad_token_id,
            "attention_mask": torch.zeros(self.max_seq_length, dtype=torch.long),
            "token_type_ids": torch.zeros(self.max_seq_length, dtype=torch.long),
            "bbox": torch.zeros((self.max_seq_length, 4), dtype=torch.long),
            "labels": torch.ones(self.max_seq_length, dtype=torch.long) * -100,
        }
        
        try:
            # 1. Process Image
            image_path = os.path.join(self.image_dir, self.image_file_names[idx])
            image = Image.open(image_path).convert("RGB")
            
            # Convert to tensor without normalization first
            image_tensor = self.transform(image)
            
            # Ensure it's the right shape and type
            image_tensor = image_tensor.to(torch.float32)
            image_tensor = torch.clamp(image_tensor, 0, 255)  # Ensure values are in valid range
            
            # 2. Process Tokens and Boxes
            words = self.words[idx][:self.max_seq_length-2]  # Leave room for CLS and SEP
            boxes = self.boxes[idx][:self.max_seq_length-2]
            labels = self.labels[idx][:self.max_seq_length-2]
            
            # Start with CLS token
            token_list = [self.cls_token_id]
            bbox_list = [[0, 0, 0, 0]]
            label_list = [-100]
            
            # Process each word
            for word, box, label in zip(words, boxes, labels):
                if len(token_list) >= self.max_seq_length - 1:  # Leave room for SEP
                    break
                
                # Convert word to tokens
                word_tokens = self.tokenizer.tokenize(str(word))
                if not word_tokens:
                    continue
                
                # Get token ids
                word_ids = self.tokenizer.convert_tokens_to_ids(word_tokens)
                
                # Ensure box coordinates are integers
                box = [int(coord) if isinstance(coord, (int, float)) else 0 for coord in box]
                
                # Extend lists
                token_list.extend(word_ids)
                bbox_list.extend([box] * len(word_ids))
                label_list.extend([self.label2id.get(label, -100)] * len(word_ids))
            
            # Add SEP token
            token_list.append(self.sep_token_id)
            bbox_list.append([0, 0, 0, 0])
            label_list.append(-100)
            
            # Truncate if necessary
            seq_length = min(len(token_list), self.max_seq_length)
            token_list = token_list[:seq_length]
            bbox_list = bbox_list[:seq_length]
            label_list = label_list[:seq_length]
            
            # Convert to tensors
            encoded_inputs["input_ids"][:seq_length] = torch.tensor(token_list, dtype=torch.long)
            encoded_inputs["attention_mask"][:seq_length] = 1
            encoded_inputs["bbox"][:seq_length] = torch.tensor(bbox_list, dtype=torch.long)
            encoded_inputs["labels"][:seq_length] = torch.tensor(label_list, dtype=torch.long)
            encoded_inputs["image"] = image_tensor
            
            # Verify final shapes
            assert encoded_inputs["input_ids"].shape == (self.max_seq_length,)
            assert encoded_inputs["attention_mask"].shape == (self.max_seq_length,)
            assert encoded_inputs["token_type_ids"].shape == (self.max_seq_length,)
            assert encoded_inputs["bbox"].shape == (self.max_seq_length, 4)
            assert encoded_inputs["labels"].shape == (self.max_seq_length,)
            assert encoded_inputs["image"].shape == (3, self.img_size, self.img_size)
            
            return BatchEncoding(encoded_inputs)
            
        except Exception as e:
            print(f"Error processing item {idx}: {str(e)}")
            raise e
        
# Example usage:
def create_datasets(folder_path, tokenizer):
    """
    Create train and test datasets from pickle files
    """
    # Load pickle files
    train = pd.read_pickle(os.path.join(folder_path, 'train.pkl'))
    test = pd.read_pickle(os.path.join(folder_path, 'test.pkl'))
    print(len(train[0]))
    all_labels = [item for sublist in train[1] for item in sublist] + [item for sublist in test[1] for item in sublist]
    Counter(all_labels)
    label_new = dict(Counter(all_labels))
    print(label_new)
    labels = list(set(all_labels))
    print(labels)
    print(len(labels))
    len_of_labels = len(labels)
    with open(os.path.join(folder_path, "classes.txt"), "w") as f:
        f.write(str(labels))
    f.close()
    #same count in labels and classes (+1 for others)
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for idx, label in enumerate(labels)}
    print(label2id)
    print(id2label)
    
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
	num_labels=len(labels)  # Set based on your number of labels
    )
    # Create datasets
    train_dataset = CustomDataset(
        label2id, id2label,
        annotations=train,
        image_dir=os.path.join(folder_path, "train/"),\
        tokenizer=tokenizer
    )
    
    test_dataset = CustomDataset(
        label2id, id2label,
        annotations=test,
        image_dir=os.path.join(folder_path, "test/"),\
        tokenizer=tokenizer
    )
    
    return train_dataset, test_dataset, len_of_labels, label2id, id2label, all_labels, config

def train_model():
    folder_path = '/home/data_science/geo_testing/COO_V3/testing'
    
    num_epochs=40
    # Set up logging
    logging.basicConfig(
        filename=os.path.join(folder_path, 'training.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Create datasets
    train_dataset, test_dataset, len_of_labels, label2id, id2label, all_labels, config = create_datasets(folder_path, tokenizer)
    print(test_dataset)
    # Create data loaders
    # train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Modify your data loader to include proper safeguards
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,  # Set to 0 for debugging
        pin_memory=True,
        drop_last=True  # Prevent issues with last batch
    )
    test_loader = DataLoader(test_dataset, batch_size=4)
    print(test_loader)
    # Initialize model
    # config = LayoutLMv2Config.from_pretrained(config_path)
    model = LayoutLMv2ForTokenClassification(config)
    # Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    
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
    for epoch in range(num_epochs):
        print('EPOCH',epoch)
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            try:
                batch = {k: v.cuda() for k, v in batch.items()}
                # for batch in tqdm(train_loader):
                input_ids = batch['input_ids'].long().to(device)
                bbox = batch['bbox'].long().to(device)  # Explicit cast to torch.int64
                image = batch['image'].float().to(device)  # Convert uint8 to float
                attention_mask = batch['attention_mask'].long().to(device)
                token_type_ids = batch['token_type_ids'].long().to(device)
                labels = batch['labels'].long().to(device)

                # print(f"input_ids dtype: {batch['input_ids'].dtype}")
                # print(f"input_ids dtype: {batch['input_ids']}")
                # print(f"bbox dtype: {batch['bbox'].dtype}")
                # print(f"bbox dtype: {batch['bbox']}")
                # print(f"image dtype: {batch['image'].dtype}")
                # print(f"image dtype: {batch['image']}")
                # print(f"attention_mask dtype: {batch['attention_mask'].dtype}")
                # print(f"token_type_ids dtype: {batch['token_type_ids'].dtype}")
                # print(f"labels dtype: {batch['labels'].dtype}")
                # print(f"labels dtype: {batch['labels']}")
                
                # print(type(input_ids))
                # print(input_ids.shape)
                # exit('OKKK')
                
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=input_ids,
                                    bbox=bbox,
                                    image=image,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    labels=labels)
                    loss = outputs.loss

                # print loss every epoch
                if (global_step + 1) % len(train_loader) == 0 or global_step == 0:
                    print(f"Loss after {global_step} steps: {loss.item()}")
                    steps.append(global_step)
                    losses.append(float(loss.item()))
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                global_step += 1
                print(f" Happy case {batch_idx}")
                print(f"input_ids dtype: {batch['input_ids'].dtype}")
                print(f"input_ids dtype: {batch['input_ids']}")
            except RuntimeError as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                print(f"input_ids dtype: {batch['input_ids'].dtype}")
                print(f"input_ids dtype: {batch['input_ids']}")
                print(f"image dtype: {batch['image'].dtype}")
                print(f"image dtype: {batch['image']}")
                # Print shapes for debugging
                for k, v in batch.items():
                    print(f"{k} shape: {v.shape}")
                    print('$$$$$$$$$$$$$$$$$$$$')
                continue
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()
        # model.eval()
        training_loss[epoch] = loss 
        val_loss = 0.0
        preds_val = None
        exit('OKKKK')
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
        val_loss= val_loss /len(test_loader)
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
        with open(os.path.join(folder_path, "metric_info.txt"), 'a') as fil:
            fil.write(f"Epoch: {epoch} \n")
            fil.write(f'final validation loss:{val_loss}\n')
            fil.write(f'final best loss till now:{best_loss}\n')
            fil.write(f'final train loss:{loss}\n')
            fil.write(f"precison: {val_result['precision']}\n")
            fil.write(f"recall: {val_result['recall']}\n")
            fil.write(f"f1: {val_result['f1']}\n")
            fil.write("\n")
            fil.write("\n")
            
                
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