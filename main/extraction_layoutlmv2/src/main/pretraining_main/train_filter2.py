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





class CustomDataset(Dataset):
    def __init__(self, annotations, image_dir, tokenizer, max_length=512):
        """
        Custom dataset for LayoutLMv2 that works with pickle files
        
        Args:
            annotations (pd.DataFrame): DataFrame containing the annotations (words, labels, boxes)
            image_dir (str): Directory containing images
            tokenizer: BERT tokenizer
            max_length (int): Maximum sequence length
        """
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Extract words, labels, and boxes from the DataFrame
        # Assuming the DataFrame has columns structured similar to SROIE
        # self.words = annotations['words'].tolist()
        # self.boxes = annotations['boxes'].tolist()
        # self.labels = annotations['labels'].tolist()
        self.words, self.labels, self.boxes = annotations
        
        # Get image file names
        self.image_file_names = list(listdir(image_dir))
        
        # Create label to id mapping from unique labels
        unique_labels = set([label for label_list in self.labels for label in label_list])
        self.label2id = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        # Load and process image
        image_path = os.path.join(self.image_dir, self.image_file_names[idx])
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224))
        image = torch.tensor(np.array(image)).permute(2, 0, 1)
        
        # Get words, boxes, and labels for this example
        words = self.words[idx]
        boxes = self.boxes[idx]
        labels = self.labels[idx]
        print('labels:', labels)
        # Tokenize words
        encoding = self.tokenizer(
            words,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Process boxes to match tokenized input
        processed_boxes = self._process_boxes(boxes, encoding.attention_mask)
        
        # Process labels to match tokenized input
        processed_labels = self._process_labels(labels, encoding.attention_mask)
        
        return {
            'input_ids': encoding.input_ids.squeeze(),
            'attention_mask': encoding.attention_mask.squeeze(),
            'token_type_ids': encoding.token_type_ids.squeeze(),
            'bbox': torch.tensor(processed_boxes),
            'labels': torch.tensor(processed_labels),
            'image': image
        }
        
    def _process_boxes(self, boxes, attention_mask):
        """
        Process bounding boxes to match the tokenized input length
        """
        processed_boxes = []
        for box in boxes:
            # Normalize box coordinates if needed
            normalized_box = [
                int(box[0]),  # x1
                int(box[1]),  # y1
                int(box[2]),  # x2
                int(box[3])   # y2
            ]
            processed_boxes.append(normalized_box)
            
        # Pad to max length
        padding_length = self.max_length - len(processed_boxes)
        processed_boxes.extend([[0, 0, 0, 0]] * padding_length)
        
        return processed_boxes[:self.max_length]
    
    def _process_labels(self, labels, attention_mask):
        """
        Process labels to match the tokenized input length
        """
        processed_labels = []
        for label in labels:
            if label in self.label2id:
                processed_labels.append(self.label2id[label])
            else:
                processed_labels.append(-100)  # Ignore index for loss
                
        # Pad to max length
        padding_length = self.max_length - len(processed_labels)
        processed_labels.extend([-100] * padding_length)
        
        return processed_labels[:self.max_length]

# Example usage:
def create_datasets(folder_path, tokenizer):
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
    exit('OKLLLLL')
    test_loader = DataLoader(test_dataset, batch_size=4)
    
    # Initialize model
    # config = LayoutLMv2Config.from_pretrained(config_path)
    model = LayoutLMv2ForTokenClassification(config)
    
    # Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    # Initialize tensorboard
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'runs'))
    
    # Training loop
    best_f1 = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch}'):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Evaluation
        model.eval()
        val_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()
                
                # Get predictions
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(batch['labels'].cpu().numpy())
        
        # Calculate metrics
        metrics = compute_metrics(predictions, true_labels)
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss / len(train_loader), epoch)
        writer.add_scalar('Loss/val', val_loss / len(test_loader), epoch)
        writer.add_scalar('Metrics/f1', metrics['f1'], epoch)
        
        # Save best model
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            model.save_pretrained(os.path.join(output_dir, 'best_model'))
            
        logging.info(f'Epoch {epoch}: Train Loss = {train_loss/len(train_loader):.4f}, '
                    f'Val Loss = {val_loss/len(test_loader):.4f}, F1 = {metrics["f1"]:.4f}')
    
    writer.close()
    return model

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