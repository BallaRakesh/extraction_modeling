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

class CustomDataset(Dataset):
    def __init__(self, annotations_dir, image_dir, tokenizer, max_length=512):
        """
        Custom dataset for LayoutLMv2 without using the processor
        
        Args:
            annotations_dir (str): Directory containing JSON annotations
            image_dir (str): Directory containing images
            tokenizer: BERT tokenizer
            max_length (int): Maximum sequence length
        """
        self.annotations_dir = annotations_dir
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load all JSON files
        self.json_files = glob(os.path.join(annotations_dir, "*.json"))
        
        # Initialize data structures
        self.words = []
        self.boxes = []
        self.labels = []
        self.images = []
        
        # Process all annotations
        self._load_annotations()
        
    def _load_annotations(self):
        for json_file in self.json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Get corresponding image
            image_path = os.path.join(self.image_dir, 
                                    os.path.basename(data['meta']['image_path']))
            
            # Process words and boxes
            words_list = []
            boxes_list = []
            labels_list = []
            
            for word_data in data['words']:
                words_list.append(word_data['text'])
                boxes_list.append(word_data['boundingBox'])
                
            # Process labels from parse section
            label_map = data['parse']['class']
            for label, indices in label_map.items():
                for idx_list in indices:
                    for idx in idx_list:
                        labels_list.append(label)
            
            self.words.append(words_list)
            self.boxes.append(boxes_list)
            self.labels.append(labels_list)
            self.images.append(image_path)

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        # Load and process image
        image = Image.open(self.images[idx]).convert("RGB")
        image = image.resize((224, 224))
        image = torch.tensor(np.array(image)).permute(2, 0, 1)
        
        # Process text
        words = self.words[idx]
        boxes = self.boxes[idx]
        labels = self.labels[idx]
        
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
        processed_boxes = []
        for box in boxes:
            processed_boxes.extend([[0, 0, 0, 0]] * len(box))  # Pad to token length
            
        # Pad to max length
        padding_length = self.max_length - len(processed_boxes)
        processed_boxes.extend([[0, 0, 0, 0]] * padding_length)
        
        return processed_boxes[:self.max_length]
    
    def _process_labels(self, labels, attention_mask):
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

def train_model(config_path, train_dir, test_dir, output_dir, num_epochs=40):
    # Set up logging
    logging.basicConfig(
        filename=os.path.join(output_dir, 'training.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    
    # Create datasets
    train_dataset = CustomDataset(
        os.path.join(train_dir, 'annotations'),
        os.path.join(train_dir, 'images'),
        tokenizer
    )
    
    test_dataset = CustomDataset(
        os.path.join(test_dir, 'annotations'),
        os.path.join(test_dir, 'images'),
        tokenizer
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4)
    
    # Initialize model
    config = LayoutLMv2Config.from_pretrained(config_path)
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
    # Configuration
    config_path = "config.json"
    train_dir = "data/train"
    test_dir = "data/test"
    output_dir = "output"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Train model
    model = train_model(config_path, train_dir, test_dir, output_dir)