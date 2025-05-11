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

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import LayoutLMv2PreTrainedModel, LayoutLMv2Model
from typing import Optional, Tuple, Union
import torch
from transformers import LayoutLMv2PreTrainedModel, LayoutLMv2Model
from transformers import LayoutLMv2ForTokenClassification
from torch import nn


import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import LayoutLMv2PreTrainedModel, LayoutLMv2Model
from typing import Optional, Tuple, Union
import os
import json
from transformers.modeling_outputs import TokenClassifierOutput

class LayoutLMv2ForTokenClassificationCustom(LayoutLMv2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.layoutlmv2 = LayoutLMv2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    @classmethod
    def from_pretrained_custom(cls, pretrained_path, num_labels):
        """
        Load a custom pretrained model from local directory
        
        Args:
            pretrained_path: Path to the pretrained model directory
            num_labels: Number of labels for the classifier
        """
        # Load config
        config = LayoutLMv2PreTrainedModel.config_class.from_pretrained(pretrained_path)
        config.num_labels = num_labels
        
        # Load training info
        with open(os.path.join(pretrained_path, 'training_info.json'), 'r') as f:
            training_info = json.load(f)
            for key, value in training_info.items():
                setattr(config, key, value)
        
        # Initialize model with config
        model = cls(config)
        
        # Load backbone weights
        backbone_path = os.path.join(pretrained_path, 'backbone.pth')
        if os.path.exists(backbone_path):
            backbone_state_dict = torch.load(backbone_path, map_location='cpu')
            
            # Filter out classifier weights if present
            backbone_state_dict = {k: v for k, v in backbone_state_dict.items() 
                                 if 'classifier' not in k}
            
            # Load the weights
            missing_keys, unexpected_keys = model.load_state_dict(backbone_state_dict, strict=False)
            print(f"Missing keys: {missing_keys}")
            print(f"Unexpected keys: {unexpected_keys}")
        
        return model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        image: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, dict]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Remove labels from kwargs before passing to layoutlmv2
        outputs = self.layoutlmv2(
            input_ids=input_ids,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Get sequence length from input_ids or inputs_embeds
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        
        # Only take the text part of the output representations
        sequence_output = outputs[0][:, :seq_length]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # Ensure labels tensor is properly shaped
            if labels.size(0) != logits.size(0):
                raise ValueError(f"Batch size mismatch: labels batch size ({labels.size(0)}) "
                               f"!= logits batch size ({logits.size(0)})")
            
            loss_fct = CrossEntropyLoss()
            # Make sure logits and labels have the same shape before loss calculation
            if logits.size(-1) != self.num_labels:
                raise ValueError(f"Logits shape mismatch: expected {self.num_labels} classes, "
                               f"got {logits.size(-1)}")
                
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def train_model_finetuining_temp(folder_path, model, **kwargs):
    """
    Fine-tune the model with proper batch handling
    """
    # Ensure all inputs have the same batch size
    def validate_batch_sizes(**inputs):
        batch_sizes = {k: v.size(0) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        if len(set(batch_sizes.values())) > 1:
            raise ValueError(f"Inconsistent batch sizes: {batch_sizes}")
        return True

    try:
        # Validate input batch sizes
        validate_batch_sizes(
            input_ids=kwargs.get('input_ids'),
            bbox=kwargs.get('bbox'),
            attention_mask=kwargs.get('attention_mask'),
            labels=kwargs.get('labels')
        )
        
        # Forward pass
        outputs = model(**kwargs)
        
        return outputs
        
    except ValueError as e:
        print(f"Batch size validation failed: {str(e)}")
        raise

# Modified training function with backbone saving

# Example usage for new dataset with different labels
def train_on_new_dataset(pretrained_backbone_path, new_num_labels):
    """
    Train on a new dataset using pretrained backbone
    
    Args:
        pretrained_backbone_path: Path to the pretrained backbone
        new_num_labels: Number of labels in the new dataset
    """
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Load pretrained backbone with new classification head
    model = load_pretrained_backbone(pretrained_backbone_path, new_num_labels)
    
    # Freeze backbone layers (optional)
    for param in model.layoutlmv2.parameters():
        param.requires_grad = False
    
    # Only train the classification head
    trainable_params = model.classifier.parameters()
    optimizer = AdamW(trainable_params, lr=5e-5)
    
    # Create datasets
    folder_path = '/home/data_science/geo_testing/COO_V3'
    train_dataset, test_dataset, _, label2id, id2label, all_labels, _ = create_datasets(
        folder_path, 
        tokenizer
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_loader):
            # ... [Your existing training loop code] ...
            pass
    
    return model


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
        """
        Custom dataset for LayoutLMv2 with proper shape handling
        
        Args:
            annotations (pd.DataFrame): DataFrame containing the annotations
            image_dir (str): Directory containing images
            tokenizer: BERT tokenizer
            max_seq_length (int): Maximum sequence length (512 for LayoutLMv2)
            img_size (int): Image size for resizing (224 for LayoutLMv2)
        """
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.img_size = img_size
        
        # Get image file names
        self.image_file_names = list(listdir(image_dir))
        self.words, self.labels, self.boxes = annotations
        print(len(self.words))
        print(len(self.labels))
        print(len(self.boxes))
        print(len(self.image_file_names))
        assert len(self.words) == len(self.image_file_names), \
        f"Mismatch: {len(self.words)} annotations vs {len(self.image_file_names)} images"

        # Set up tokenizer special tokens
        if hasattr(tokenizer, "vocab"):
            self.pad_token_id = tokenizer.vocab["[PAD]"]
            self.cls_token_id = tokenizer.vocab["[CLS]"]
            self.sep_token_id = tokenizer.vocab["[SEP]"]
            self.unk_token_id = tokenizer.vocab["[UNK]"]
        else:
            self.pad_token_id = tokenizer.pad_token_id
            self.cls_token_id = tokenizer.cls_token_id
            self.sep_token_id = tokenizer.sep_token_id
            self.unk_token_id = tokenizer.unk_token_id
            
        # Create label mappings
        unique_labels = set([label for label_list in self.labels for label in label_list])
        self.label2id = label2id #{label: idx for idx, label in enumerate(sorted(unique_labels))}
        self.id2label = id2label #{idx: label for label, idx in self.label2id.items()}

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        # Initialize tensors with correct shapes and types
        encoded_inputs = {
            "input_ids": torch.ones(self.max_seq_length, dtype=torch.long) * self.pad_token_id,
            "attention_mask": torch.zeros(self.max_seq_length, dtype=torch.long),
            "token_type_ids": torch.zeros(self.max_seq_length, dtype=torch.long),
            "bbox": torch.zeros((self.max_seq_length, 4), dtype=torch.long),
            "labels": torch.ones(self.max_seq_length, dtype=torch.long) * -100,
        }
        
        # Load and process image
        image_path = os.path.join(self.image_dir, self.image_file_names[idx])
        # print('>>>>>>>>>>>?', idx, '>>>>>>$$$$>>>', image_path)
        
        image = Image.open(image_path).convert("RGB")
        # Resize to required dimensions (3, 224, 224)
        image = image.resize((self.img_size, self.img_size))
        image = torch.FloatTensor(np.array(image)).permute(2, 0, 1)  # Convert to CHW format
        
        # image = image / 255.0  # Normalize to [0, 1] range if needed
    
        # Get original image dimensions
        # width, height = Image.open(image_path).size
        
        # Process tokens and boxes
        words = self.words[idx]
        boxes = self.boxes[idx]
        labels = self.labels[idx]
        
        # Initialize lists for collecting tokens and alignments
        token_list = []
        bbox_list = []
        label_list = []
        
        # Add CLS token
        token_list.append(self.cls_token_id)
        bbox_list.append([0, 0, 0, 0])
        label_list.append(-100)
        
        # Process each word
        for word, box, label in zip(words, boxes, labels):
            if len(token_list) >= self.max_seq_length - 2:  # Leave room for [CLS] and [SEP]
                break
                
            # Tokenize word
            word_tokens = self.tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [self.unk_token_id]
            
            # Normalize box coordinates to 1000x1000
            # normalized_box = [
            #     int(1000 * box[0] / width),
            #     int(1000 * box[1] / height),
            #     int(1000 * box[2] / width),
            #     int(1000 * box[3] / height),
            # ]
            
            # Add tokens and boxes
            word_ids = self.tokenizer.convert_tokens_to_ids(word_tokens)
            token_list.extend(word_ids)
            # bbox_list.extend([normalized_box] * len(word_ids))
            # bbox_list.extend([box] * len(word_ids))
            bbox_list.extend([list(map(int, box))] * len(word_ids))
            label_list.extend([self.label2id.get(label, -100)] * len(word_ids))
        
        # Add SEP token
        token_list.append(self.sep_token_id)
        bbox_list.append([0, 0, 0, 0])
        label_list.append(-100)
        
        # Truncate if necessary and pad sequences
        seq_length = min(len(token_list), self.max_seq_length)
        encoded_inputs["input_ids"][:seq_length] = torch.tensor(token_list[:seq_length])
        encoded_inputs["attention_mask"][:seq_length] = 1
        encoded_inputs["bbox"][:seq_length] = torch.tensor(bbox_list[:seq_length])
        encoded_inputs["labels"][:seq_length] = torch.tensor(label_list[:seq_length])
        
        # Add image to encoded inputs
        # encoded_inputs["image"] = image
        encoded_inputs["image"] = image.type(torch.uint8) 
        # Verify shapes
        assert encoded_inputs["input_ids"].shape == torch.Size([512])
        assert encoded_inputs["attention_mask"].shape == torch.Size([512])
        assert encoded_inputs["token_type_ids"].shape == torch.Size([512])
        assert encoded_inputs["bbox"].shape == torch.Size([512, 4])
        assert encoded_inputs["image"].shape == torch.Size([3, 224, 224])
        assert encoded_inputs["labels"].shape == torch.Size([512])
        return BatchEncoding(encoded_inputs)
        # return encoded_inputs
        
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

def train_model_first(folder_path):
    
    num_epochs=50
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
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=4)
    print(test_loader)
    # Initialize model
    # config = LayoutLMv2Config.from_pretrained(config_path)
    model = LayoutLMv2ForTokenClassification(config)
    
    # os.makedirs(os.path.join(folder_path, 'base_model'), exist_ok=True)
    os.makedirs(os.path.join(folder_path, 'base_model', 'save_pretrain'), exist_ok=True)
    os.makedirs(os.path.join(folder_path, 'base_model', 'save_backbone_classifier'), exist_ok=True)
    
    model.save_pretrained(os.path.join(folder_path, 'base_model', 'save_pretrain'))
    save_model_components(model, os.path.join(folder_path, 'base_model', 'save_backbone_classifier'))
    # Setup training
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    # labels = list(set(all_labels))
    global_step = 0
    num_train_epochs = 50
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
        for batch in tqdm(train_loader, desc="TRaining"):
            input_ids = batch['input_ids'].long().to(device)
            bbox = batch['bbox'].long().to(device)  # Explicit cast to torch.int64
            image = batch['image'].float().to(device)  # Convert uint8 to float
            attention_mask = batch['attention_mask'].long().to(device)
            token_type_ids = batch['token_type_ids'].long().to(device)
            labels = batch['labels'].long().to(device)

            # print(f"input_ids dtype: {batch['input_ids'].dtype}")
            # print(f"bbox dtype: {batch['bbox'].dtype}")
            # print(f"image dtype: {batch['image'].dtype}")
            # print(f"attention_mask dtype: {batch['attention_mask'].dtype}")
            # print(f"token_type_ids dtype: {batch['token_type_ids'].dtype}")
            # print(f"labels dtype: {batch['labels'].dtype}")
            
            # print(type(input_ids))
            # print(input_ids.shape)
            # exit('OKKK')
            
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
            if (global_step + 1) % len(train_loader) == 0 or global_step == 0:
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
            
        os.makedirs(os.path.join(folder_path, 'pretrained_model'), exist_ok=True)
        
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
            save_model_components(model, os.path.join(folder_path, 'pretrained_model'))
            
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
        # save_pretrained_backbone(best_model_low, os.path.join(folder_path, 'pretrained_model'))


    #give best model path here
    if best_model_flag_high:
        model_path = f"{folder_path}/Best_Model"
    elif best_model_flag_low:
        model_path =  f"{folder_path}/Best_Model_low"
    else:
        exit("no best model exist")
    


def save_pretrained_backbone(model, save_path):
    """
    Save only the backbone (feature extractor) of the model
    
    Args:
        model: Trained LayoutLMv2 model
        save_path: Path to save the backbone
    """
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Save the backbone configuration
    model.layoutlmv2.config.save_pretrained(save_path)
    
    # Save only the backbone weights
    backbone_state_dict = {
        k: v for k, v in model.state_dict().items()
        if 'classifier' not in k
    }
    
    torch.save(backbone_state_dict, os.path.join(save_path, 'backbone.pth'))
    
    # Save additional training info
    training_info = {
        'hidden_size': model.config.hidden_size,
        'max_position_embeddings': model.config.max_position_embeddings,
        'max_2d_position_embeddings': model.config.max_2d_position_embeddings,
    }
    
    with open(os.path.join(save_path, 'training_info.json'), 'w') as f:
        json.dump(training_info, f)
 


def save_model_components(model, save_path):
    """
    Save the backbone (feature extractor) and classifier of the model separately
    
    Args:
        model: Trained LayoutLMv2 model
        save_path: Base path to save the components
    """
    # Create directories for backbone and classifier
    backbone_path = os.path.join(save_path, 'backbone')
    classifier_path = os.path.join(save_path, 'classifier')
    os.makedirs(backbone_path, exist_ok=True)
    os.makedirs(classifier_path, exist_ok=True)
    
    # Save the backbone configuration
    model.layoutlmv2.config.save_pretrained(backbone_path)
    
    # Separate and save backbone weights
    backbone_state_dict = {
        k: v for k, v in model.state_dict().items()
        if 'classifier' not in k
    }
    torch.save(backbone_state_dict, os.path.join(backbone_path, 'backbone.pth'))
    
    # Separate and save classifier weights
    classifier_state_dict = {
        k: v for k, v in model.state_dict().items()
        if 'classifier' in k
    }
    torch.save(classifier_state_dict, os.path.join(classifier_path, 'classifier.pth'))
    
    # Save backbone training info
    backbone_info = {
        'hidden_size': model.config.hidden_size,
        'max_position_embeddings': model.config.max_position_embeddings,
        'max_2d_position_embeddings': model.config.max_2d_position_embeddings,
    }
    
    # Save classifier info with optional attributes
    classifier_info = {
        'num_labels': model.config.num_labels,
        'hidden_size': model.config.hidden_size,  # needed for classifier input dim
    }
    
    # Attempt to add classifier_dropout if available
    try:
        # Check if classifier_dropout exists as an attribute of the model's config
        dropout = getattr(model.config, 'classifier_dropout', None)
        if dropout is not None:
            classifier_info['classifier_dropout'] = dropout
    except Exception:
        # If there's any issue accessing the attribute, we'll simply skip it
        pass
    
    # Save info files
    with open(os.path.join(backbone_path, 'backbone_info.json'), 'w') as f:
        json.dump(backbone_info, f, indent=2)
        
    with open(os.path.join(classifier_path, 'classifier_info.json'), 'w') as f:
        json.dump(classifier_info, f, indent=2)
        
    # Save a README with loading instructions
    readme_content = """
    Model components saved separately:
    
    1. Backbone (Feature Extractor):
       - Configuration: config.json
       - Weights: backbone.pth
       - Info: backbone_info.json
    
    2. Classifier:
       - Weights: classifier.pth
       - Info: classifier_info.json
    
    To load:
    ```python
    # Load model components
    model = load_model_components(
        backbone_path='path/to/backbone', 
        classifier_path='path/to/classifier',
        num_labels=num_labels
    )
    ```
    """
    
    with open(os.path.join(save_path, 'README.md'), 'w') as f:
        f.write(readme_content.strip())

def compute_metrics(predictions, true_labels):
    # Remove padding (-100)
    true_predictions = [p for p, l in zip(predictions, true_labels) if l != -100]
    true_labels = [l for l in true_labels if l != -100]
    
    return {
        'precision': precision_score(true_labels, true_predictions, average='weighted'),
        'recall': recall_score(true_labels, true_predictions, average='weighted'),
        'f1': f1_score(true_labels, true_predictions, average='weighted')
    }


# First, modify the train_model function to use our custom class:
def train_model(save_backbone=True):
    output_dir = '/home/data_science/geo_testing/COO_V3'
    backbone_dir = os.path.join(output_dir, 'pretrained_backbone')
    num_epochs = 40
    
    # Set up logging
    logging.basicConfig(
        filename=os.path.join(output_dir, 'training.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    folder_path = '/home/data_science/geo_testing/COO_V3'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Create datasets
    train_dataset, test_dataset, len_of_labels, label2id, id2label, all_labels, config = create_datasets(folder_path, tokenizer)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4)
    
    # Initialize custom model instead of standard LayoutLMv2
    model = LayoutLMv2ForTokenClassificationCustom(config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    global_step = 0
    best_loss = None
    best_precision = None
    best_recall = None
    best_f1 = None
    best_model_flag_high = False
    best_model_flag_low = False

    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_loader):
            input_ids = batch['input_ids'].long().to(device)
            bbox = batch['bbox'].long().to(device)
            image = batch['image'].float().to(device)
            attention_mask = batch['attention_mask'].long().to(device)
            token_type_ids = batch['token_type_ids'].long().to(device)
            labels = batch['labels'].long().to(device)

            optimizer.zero_grad()

            # Modified forward pass using custom model
            logits, sequence_output = model(
                input_ids=input_ids,
                bbox=bbox,
                image=image,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            # Calculate loss manually since we're not using the standard forward pass
            loss_fct = nn.CrossEntropyLoss()
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, model.num_labels)
            active_labels = labels.view(-1)
            loss = loss_fct(active_logits, active_labels)

            # Rest of training logic remains the same
            loss.backward()
            optimizer.step()
            global_step += 1
            
            # Log training progress
            if (global_step + 1) % len(train_loader) == 0:
                print(f"Loss after {global_step} steps: {loss.item()}")
                train_writer.add_scalar("train loss", loss.detach(), epoch)

        # Evaluation loop
        model.eval()
        val_loss = 0.0
        preds_val = None
        out_label_ids = None
        
        for batch in tqdm(test_loader, desc="Evaluating"):
            with torch.no_grad():
                input_ids = batch['input_ids'].to(device)
                bbox = batch['bbox'].to(device)
                image = batch['image'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels = batch['labels'].to(device)

                # Modified forward pass for evaluation
                logits, _ = model(
                    input_ids=input_ids,
                    bbox=bbox,
                    image=image,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                
                # Calculate validation loss
                loss_fct = nn.CrossEntropyLoss()
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, model.num_labels)
                active_labels = labels.view(-1)
                val_batch_loss = loss_fct(active_logits, active_labels)
                val_loss += val_batch_loss.item()

                if preds_val is None:
                    preds_val = logits.detach().cpu().numpy()
                    out_label_ids = labels.detach().cpu().numpy()
                else:
                    preds_val = np.append(preds_val, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

        # Calculate metrics and save best model
        val_result, class_report = results_test(preds_val, out_label_ids, list(set(all_labels)))
        
        # Save model logic
        if save_backbone and (best_model_flag_high or best_model_flag_low):
            # Save custom model backbone
            save_pretrained_backbone(model, backbone_dir)
            
            # Also save the classification head separately if needed
            classifier_state = {
                'classifier': model.classifier.state_dict(),
                'num_labels': model.num_labels
            }
            torch.save(classifier_state, os.path.join(backbone_dir, 'classifier.pth'))
            
    return model, backbone_dir


def train_model_finetuining(folder_path, pre_model_path):
    
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
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=4)
    print(test_loader)
    # Initialize model
    # config = LayoutLMv2Config.from_pretrained(config_path)
    # model = LayoutLMv2ForTokenClassification(config)
    
    
    # model = quick_load_model(
    #     base_save_path=pre_model_path,
    #     # num_labels=30
    # )
    
    
    # model = LayoutLMv2ForTokenClassificationCustom.from_pretrained_custom(
    # pretrained_path= pre_model_path,
    # num_labels=len_of_labels
    # )
    
    model = quick_load_model_test(
        base_save_path=pre_model_path, #'/home/data_science/geo_testing/COO_V3/CORD_DATA_iter2/pretrained_model',
        num_labels=len_of_labels,
        ignore_classifier=True
    )
    
    print(model)
    # exit('>>>')
    # os.makedirs(os.path.join(folder_path, 'base_model'), exist_ok=True)
    os.makedirs(os.path.join(folder_path, 'base_model', 'save_pretrain'), exist_ok=True)
    os.makedirs(os.path.join(folder_path, 'base_model', 'save_backbone_classifier'), exist_ok=True)
    
    model.save_pretrained(os.path.join(folder_path, 'base_model', 'save_pretrain'))
    save_model_components(model, os.path.join(folder_path, 'base_model', 'save_backbone_classifier'))
    # Setup training
    # exit('')
    # model = model_pre
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
        for batch in tqdm(train_loader, desc="TRaining"):
            input_ids = batch['input_ids'].long().to(device)
            bbox = batch['bbox'].long().to(device)  # Explicit cast to torch.int64
            image = batch['image'].float().to(device)  # Convert uint8 to float
            attention_mask = batch['attention_mask'].long().to(device)
            token_type_ids = batch['token_type_ids'].long().to(device)
            labels = batch['labels'].long().to(device)

            # print(f"input_ids dtype: {batch['input_ids'].dtype}")
            # print(f"bbox dtype: {batch['bbox'].dtype}")
            # print(f"image dtype: {batch['image'].dtype}")
            # print(f"attention_mask dtype: {batch['attention_mask'].dtype}")
            # print(f"token_type_ids dtype: {batch['token_type_ids'].dtype}")
            # print(f"labels dtype: {batch['labels'].dtype}")
            
            # print(type(input_ids))
            # print(input_ids.shape)
            # exit('OKKK')
            
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
            print(loss)
            # print loss every epoch
            if (global_step + 1) % len(train_loader) == 0 or global_step == 0:
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
            
        os.makedirs(os.path.join(folder_path, 'pretrained_model'), exist_ok=True)
        # save_pretrained_backbone(model, os.path.join(folder_path, 'pretrained_model'))
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


# Function to load the custom model
def load_custom_model(model_path, num_labels):
    """
    Load custom model with pretrained weights
    """
    config = LayoutLMv2Config.from_pretrained(model_path)
    config.num_labels = num_labels
    
    # Initialize custom model
    model = LayoutLMv2ForTokenClassificationCustom(config)
    
    # Load backbone weights
    backbone_state_dict = torch.load(os.path.join(model_path, 'backbone.pth'))
    model.load_state_dict(backbone_state_dict, strict=False)
    
    # Load classifier if available and if number of labels matches
    classifier_path = os.path.join(model_path, 'classifier.pth')
    if os.path.exists(classifier_path):
        classifier_state = torch.load(classifier_path)
        if classifier_state['num_labels'] == num_labels:
            model.classifier.load_state_dict(classifier_state['classifier'])
    
    return model




import os
import json
import torch
from transformers import LayoutLMv2Config, LayoutLMv2ForTokenClassification

def load_model_components(backbone_path, classifier_path, num_labels=None):
    """
    Load backbone and classifier components and reconstruct the full model for token classification
    
    Args:
        backbone_path (str): Path to the saved backbone components
        classifier_path (str): Path to the saved classifier components
        num_labels (int, optional): Number of labels if not found in classifier info
    
    Returns:
        LayoutLMv2ForTokenClassification: Reconstructed model
    """
    # Load backbone configuration
    backbone_config = LayoutLMv2Config.from_pretrained(backbone_path)
    
    # Load backbone and classifier info
    with open(os.path.join(backbone_path, 'backbone_info.json'), 'r') as f:
        backbone_info = json.load(f)
    
    with open(os.path.join(classifier_path, 'classifier_info.json'), 'r') as f:
        classifier_info = json.load(f)
    
    # Determine number of labels
    num_labels = num_labels or classifier_info.get('num_labels')

    if num_labels is None:
        raise ValueError("Number of labels must be provided either in classifier info or as an argument")
    
    # Set configuration
    backbone_config.num_labels = num_labels
    
    # Create model instance
    model = LayoutLMv2ForTokenClassification(backbone_config)
    
    # Load weights
    backbone_weights = torch.load(os.path.join(backbone_path, 'backbone.pth'))
    classifier_weights = torch.load(os.path.join(classifier_path, 'classifier.pth'))
    
    # Load backbone weights
    state_dict = model.state_dict()
    for k, v in backbone_weights.items():
        if k in state_dict and state_dict[k].shape == v.shape:
            state_dict[k].copy_(v)
    
    # Get shapes for classifier weights
    current_classifier_shape = state_dict['classifier.weight'].shape  # Should be [num_labels, hidden_size]
    classifier_w = classifier_weights['classifier.weight']
    classifier_b = classifier_weights['classifier.bias']
    
    if current_classifier_shape[1] != classifier_w.shape[1]:
        print(f"Adapting classifier weights from shape {classifier_w.shape} to {current_classifier_shape}")
        
        # For TokenClassification, we need to handle the hidden size properly
        # The default hidden size for LayoutLMv2 token classification is 768
        if current_classifier_shape[1] == 768:
            # If the target model expects 768 dimensions, we need to potentially resize
            if classifier_w.shape[1] > 768:
                # Take only the first 768 dimensions if source is larger
                adapted_w = classifier_w[:, :768]
            else:
                # Pad with zeros if source is smaller
                adapted_w = torch.zeros(current_classifier_shape, device=classifier_w.device)
                adapted_w[:, :classifier_w.shape[1]] = classifier_w
        else:
            # For other cases, we'll need to resize appropriately
            adapted_w = torch.zeros(current_classifier_shape, device=classifier_w.device)
            # Copy the weights to the first section
            min_dim = min(classifier_w.shape[1], current_classifier_shape[1])
            adapted_w[:, :min_dim] = classifier_w[:, :min_dim]
    else:
        adapted_w = classifier_w

    # Update state dict with adapted weights
    state_dict['classifier.weight'].copy_(adapted_w)
    state_dict['classifier.bias'].copy_(classifier_b)
    
    # Load the complete state dict
    model.load_state_dict(state_dict)
    return model


def quick_load_model(base_save_path, num_labels=None):
    """
    Convenience method to load model components from a base directory
    
    Args:
        base_save_path (str): Base path containing backbone and classifier subdirectories
        num_labels (int, optional): Number of labels for the model
    
    Returns:
        LayoutLMv2ForTokenClassification: Loaded model
    """
    backbone_path = os.path.join(base_save_path, 'backbone')
    classifier_path = os.path.join(base_save_path, 'classifier')
    return load_model_components(backbone_path, classifier_path, num_labels)



import os
import json
import torch
from transformers import LayoutLMv2Config, LayoutLMv2ForTokenClassification

def load_model_components_test(backbone_path, classifier_path, num_labels=None, ignore_classifier=False):
    """
    Load backbone and classifier components and reconstruct the full model for token classification,
    supporting different numbers of labels between source and target models
    
    Args:
        backbone_path (str): Path to the saved backbone components
        classifier_path (str): Path to the saved classifier components
        num_labels (int, optional): Number of labels for target model
        ignore_classifier (bool): If True, initialize new classifier weights instead of loading old ones
    
    Returns:
        LayoutLMv2ForTokenClassification: Reconstructed model
    """
    # Load backbone configuration
    backbone_config = LayoutLMv2Config.from_pretrained(backbone_path)
    
    # Load backbone and classifier info
    with open(os.path.join(backbone_path, 'backbone_info.json'), 'r') as f:
        backbone_info = json.load(f)
    
    with open(os.path.join(classifier_path, 'classifier_info.json'), 'r') as f:
        classifier_info = json.load(f)
    
    # Get source number of labels
    source_num_labels = classifier_info.get('num_labels')
    
    # Use target number of labels if provided, otherwise use source
    target_num_labels = num_labels if num_labels is not None else source_num_labels

    if target_num_labels is None:
        raise ValueError("Number of labels must be provided either in classifier info or as an argument")
    
    # Set configuration with target number of labels
    backbone_config.num_labels = target_num_labels
    
    # Create model instance
    model = LayoutLMv2ForTokenClassification(backbone_config)
    
    # Load weights
    backbone_weights = torch.load(os.path.join(backbone_path, 'backbone.pth'))
    classifier_weights = torch.load(os.path.join(classifier_path, 'classifier.pth'))
    
    # Load backbone weights
    state_dict = model.state_dict()
    for k, v in backbone_weights.items():
        if k in state_dict and 'classifier' not in k:  # Skip classifier weights
            if state_dict[k].shape == v.shape:
                state_dict[k].copy_(v)
            else:
                print(f"Skipping {k} due to shape mismatch: {state_dict[k].shape} vs {v.shape}")

    if not ignore_classifier and source_num_labels is not None:
        print('YES')
        # Handle classifier weights
        classifier_w = classifier_weights['classifier.weight']  # Shape: [source_num_labels, hidden_size]
        classifier_b = classifier_weights['classifier.bias']    # Shape: [source_num_labels]
        
        # Get current classifier weights
        current_w = state_dict['classifier.weight']  # Shape: [target_num_labels, hidden_size]
        current_b = state_dict['classifier.bias']    # Shape: [target_num_labels]
        
        # Copy weights for the minimum number of labels
        min_labels = min(source_num_labels, target_num_labels)
        
        # Copy weights and biases for common labels
        state_dict['classifier.weight'][:min_labels].copy_(classifier_w[:min_labels])
        state_dict['classifier.bias'][:min_labels].copy_(classifier_b[:min_labels])
        
        # Initialize remaining weights randomly if target has more labels
        if target_num_labels > source_num_labels:
            print(f"Initializing weights for {target_num_labels - source_num_labels} new labels")
            # The remaining weights/biases will keep their random initialization
    else:
        print("Initializing new classifier weights for all labels")
        print('NO')
        # The classifier weights will keep their random initialization
    # Load the complete state dict
    model.load_state_dict(state_dict)
    return model

def quick_load_model_test(base_save_path, num_labels=None, ignore_classifier=False):
    """
    Convenience method to load model components from a base directory
    
    Args:
        base_save_path (str): Base path containing backbone and classifier subdirectories
        num_labels (int, optional): Number of labels for the target model
        ignore_classifier (bool): If True, initialize new classifier weights instead of loading old ones
    
    Returns:
        LayoutLMv2ForTokenClassification: Loaded model
    """
    backbone_path = os.path.join(base_save_path, 'backbone')
    classifier_path = os.path.join(base_save_path, 'classifier')
    return load_model_components_test(backbone_path, classifier_path, num_labels, ignore_classifier)



# model2 = quick_load_model_test(
#     base_save_path='/home/data_science/geo_testing/COO_V3/CORD_DATA_iter2/pretrained_model',
#     num_labels=38,
#     ignore_classifier=True
# )
# exit('OKKKKKKKKKKKKK')
if __name__ == "__main__":
    
    folder_path = '/home/data_science/geo_testing/COO_V3/CORD_DATA_iter3'
    pre_model_path = '/home/data_science/geo_testing/COO_V3/CORD_DATA_iter2/pretrained_model'
    # # Configuration
    # config_path = "config.json"
    # train_dir = "data/train"
    # test_dir = "data/test"
    # output_dir = "output"
    
    # # Create output directory
    # os.makedirs(output_dir, exist_ok=True)
    
    # Train model
    # model = train_model(train_dir, test_dir, output_dir)
    
    # model = train_model(folder_path)
    ############################################
    ############################################
    ############################################
    ############################################
    # train_model_first(folder_path)
    # exit('DONE')
    ############################################
    ############################################
    ############################################
    # # new_num_labels = 65
    # # backbone_path = '/home/data_science/geo_testing/COO_V3/pretrained_model'
    # new_model = load_custom_model(backbone_path, new_num_labels)
    model = train_model_finetuining(folder_path, pre_model_path)
    
    exit('>>>>>OKKKK>>>>>>>')