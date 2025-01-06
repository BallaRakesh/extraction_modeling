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

import torch
import pandas as pd
import os
from collections import Counter
from torch.utils.data import DataLoader
from os import listdir
from torch.utils.data import Dataset
from PIL import Image
from transformers import LayoutLMv2Processor
from transformers import LayoutLMv2ForTokenClassification, AdamW
import torch
from tqdm import *
from torch.utils.tensorboard import SummaryWriter
# import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
# import tensorflow as tf
import logging
from typing import List
from configparser import ConfigParser
from datetime import datetime
warnings.filterwarnings("ignore")
from seqeval.metrics import (
	classification_report,
	f1_score,
	precision_score,
	recall_score,
accuracy_score)
import numpy as np

from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from transformers import LayoutLMv2Config, LayoutLMv2ForTokenClassification
from transformers import LayoutLMv2Processor
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import OneCycleLR
import torch
from tqdm import tqdm
import numpy as np

#################################################################
# from config.prod_mapping import product_code_map, document_code_map

def denormalize_bboxes(bboxes, width, height, norm_val=1000, zero_val=0):
    """
    Denormalize bounding boxes back to original image coordinates.
    
    Args:
    - bboxes: List of bounding boxes, each bbox is [x0, y0, x2, y2]
    - width: Original image width
    - height: Original image height
    - norm_val: Normalization value used (default 100)
    - zero_val: Minimum value used in normalization (default 0)
    
    Returns:
    List of denormalized bounding boxes
    """
    denormalized_bboxes = []
    
    for bbox in bboxes:
        # Denormalize each coordinate
        x0 = int((bbox[0] / norm_val) * width)
        y0 = int((bbox[1] / norm_val) * height)
        x2 = int((bbox[2] / norm_val) * width)
        y2 = int((bbox[3] / norm_val) * height)
        
        denormalized_bboxes.append([x0, y0, x2, y2])
    
    return denormalized_bboxes

# # Example usage
# image_width = 800  # replace with your actual image width
# image_height = 640  # replace with your actual image height

# normalized_bboxes = [[97, 604, 212, 635], 
#                      [337, 595, 462, 626], 
#                      [467, 595, 513, 626]]


# To draw on the image
import cv2

def draw_bboxes(image, bboxes, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes on an image.
    
    Args:
    - image: Input image
    - bboxes: List of bounding boxes [x0, y0, x2, y2]
    - color: BGR color of bounding box
    - thickness: Line thickness
    """
    for bbox in bboxes:
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
    return image


from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import numpy as np
from torchvision.transforms import ToTensor
import torch
import torch.nn as nn
from transformers.models.layoutlm import LayoutLMModel, LayoutLMConfig
from transformers.modeling_outputs import TokenClassifierOutput
import torchvision
from torchvision.ops import RoIAlign
import os
import torch
import json

class LayoutLMForTokenClassification(nn.Module):
	def __init__(self, output_size=(3,3), 
					spatial_scale=14/224, 
					sampling_ratio=2
		): 
		super().__init__()
		
		# LayoutLM base model + token classifier
		# self.num_labels = len(label2idx)#label2id
		self.num_labels = len(label2id)#label2id
		self.layoutlm = LayoutLMModel.from_pretrained("microsoft/layoutlm-base-uncased", num_labels=self.num_labels)
		self.dropout = nn.Dropout(self.layoutlm.config.hidden_dropout_prob)
		self.classifier = nn.Linear(self.layoutlm.config.hidden_size, self.num_labels)

		# backbone + roi-align + projection layer
		model = torchvision.models.resnet101(pretrained=True)
		self.backbone = nn.Sequential(*(list(model.children())[:-3]))
		self.roi_align = RoIAlign(output_size, spatial_scale=spatial_scale, sampling_ratio=sampling_ratio)
		self.projection = nn.Linear(in_features=1024*3*3, out_features=self.layoutlm.config.hidden_size)
  
  
	def save_pretrained(self, save_directory):
		"""
		Save the model and its configuration to a directory.
		
		:param save_directory: Directory to save the model and configuration
		"""
		# Create the save directory if it doesn't exist
		os.makedirs(save_directory, exist_ok=True)
		
		# Save the model's state dictionary
		model_path = os.path.join(save_directory, 'pytorch_model.bin')
		torch.save(self.state_dict(), model_path)
		
		# Create a configuration dictionary
		config = {
			"output_size": self.roi_align.output_size,
			"spatial_scale": self.roi_align.spatial_scale,
			"sampling_ratio": self.roi_align.sampling_ratio,
			"num_labels": self.num_labels,
			"layoutlm_model_name": "microsoft/layoutlm-base-uncased"
		}
		
		# Save the configuration as a JSON file
		config_path = os.path.join(save_directory, 'config.json')
		with open(config_path, 'w') as f:
			json.dump(config, f, indent=4)
		
		# Optionally, save the LayoutLM model
		layoutlm_save_path = os.path.join(save_directory, 'layoutlm')
		self.layoutlm.save_pretrained(layoutlm_save_path)


	def forward(
		self,
		input_ids,
		bbox,
		attention_mask,
		token_type_ids,
		position_ids=None,
		head_mask=None,
		inputs_embeds=None,
		labels=None,
		resized_images=None, # shape (N, C, H, W), with H = W = 224
		resized_and_aligned_bounding_boxes=None, # single torch tensor that also contains the batch index for every bbox at image size 224
		output_attentions=None,
		output_hidden_states=None,
		return_dict=None,
	):
		r"""
		labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
			Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
			1]``.

		"""
		return_dict = return_dict if return_dict is not None else self.layoutlm.config.use_return_dict

		# first, forward pass on LayoutLM
		outputs = self.layoutlm(
			input_ids=input_ids,
			bbox=bbox,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		sequence_output = outputs[0]

		# next, send resized images of shape (batch_size, 3, 224, 224) through backbone to get feature maps of images 
		# shape (batch_size, 1024, 14, 14)
		feature_maps = self.backbone(resized_images)
		
		# next, use roi align to get feature maps of individual (resized and aligned) bounding boxes
		# shape (batch_size*seq_len, 1024, 3, 3)
		device = input_ids.device
		resized_bounding_boxes_list = []
		for i in resized_and_aligned_bounding_boxes:
			resized_bounding_boxes_list.append(i.float().to(device))
                
		feat_maps_bboxes = self.roi_align(input=feature_maps, 
										# we pass in a list of tensors
										# We have also added -0.5 for the first two coordinates and +0.5 for the last two coordinates,
										# see https://stackoverflow.com/questions/60060016/why-does-roi-align-not-seem-to-work-in-pytorch
										rois=resized_bounding_boxes_list
							)  
		
		# next, reshape  + project to same dimension as LayoutLM. 
		batch_size = input_ids.shape[0]
		seq_len = input_ids.shape[1]
		feat_maps_bboxes = feat_maps_bboxes.view(batch_size, seq_len, -1) # Shape (batch_size, seq_len, 1024*3*3)
		projected_feat_maps_bboxes = self.projection(feat_maps_bboxes) # Shape (batch_size, seq_len, hidden_size)

		# add those to the sequence_output - shape (batch_size, seq_len, hidden_size)
		sequence_output += projected_feat_maps_bboxes

		sequence_output = self.dropout(sequence_output)
		logits = self.classifier(sequence_output)

		loss = None
		if labels is not None:
			loss_fct = nn.CrossEntropyLoss()

			if attention_mask is not None:
				active_loss = attention_mask.view(-1) == 1
				active_logits = logits.view(-1, self.num_labels)[active_loss]
				active_labels = labels.view(-1)[active_loss]
				loss = loss_fct(active_logits, active_labels)
			else:
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


def resize_and_align_bounding_box(bbox, original_image, target_size):
	x_, y_ = original_image.size

	x_scale = target_size / x_ 
	y_scale = target_size / y_

	origLeft, origTop, origRight, origBottom = tuple(bbox)

	x = int(np.round(origLeft * x_scale))
	y = int(np.round(origTop * y_scale))
	xmax = int(np.round(origRight * x_scale))
	ymax = int(np.round(origBottom * y_scale)) 

	return [x-0.5, y-0.5, xmax+0.5, ymax+0.5]


def denormalize(normalized_points: list, width: int, height: int) -> list:
    """
    Convert normalized coordinates back to original image dimensions.
    
    :param normalized_points: List of 4 normalized coordinates [x0, y0, x2, y2]
    :param width: Original image width
    :param height: Original image height
    :return: List of original bbox coordinates
    """
    x0, y0, x2, y2 = normalized_points
    val = 1000  # This matches the normalization constant
    
    # Convert back to original scale
    x0 = int((x0 / val) * width)
    x2 = int((x2 / val) * width)
    y0 = int((y0 / val) * height)
    y2 = int((y2 / val) * height)
    
    return [x0, y0, x2, y2]

class FUNSDDataset(Dataset):
	"""LayoutLM dataset with visual features."""

	def __init__(self, annotations,image_dir, tokenizer, max_length, target_size, train=True):
		# self.image_file_names = image_file_names
		self.image_file_names = list(listdir(image_dir))
		self.tokenizer = tokenizer
		self.max_seq_length = max_length
		self.target_size = target_size
		self.pad_token_box = [0, 0, 0, 0]
		self.train = train
		self.words, self.labels, self.boxes = annotations
		self.image_dir = image_dir
  
	def __len__(self):
		return len(self.image_file_names)

	def __getitem__(self, idx):

		# first, take an image
		item = self.image_file_names[idx]
		words = self.words[idx]
		boxes = self.boxes[idx]
		word_labels = self.labels[idx]
  
		
		# if self.train:
		#   base_path = "/content/data/training_data"
		# else:
		#   base_path = "/content/data/testing_data"
			
		original_image = Image.open(self.image_dir + item).convert("RGB")
		
		# original_image = Image.open(base_path + "/images/" + item).convert("RGB")
		# resize to target size (to be provided to the pre-trained backbone)
		resized_image = original_image.resize((self.target_size, self.target_size))
		
		# # first, read in annotations at word-level (words, bounding boxes, labels)
		# with open(base_path + '/annotations/' + item[:-4] + '.json') as f:
		#   data = json.load(f)
		# words = []
		# unnormalized_word_boxes = []
		# word_labels = []
		# for annotation in data['form']:
		#   # get label
		#   label = annotation['label']
		#   # get words
		#   for annotated_word in annotation['words']:
		#       if annotated_word['text'] == '':
		#         continue
		#       words.append(annotated_word['text'])
		#       unnormalized_word_boxes.append(annotated_word['box'])
		#       word_labels.append(label)

		width, height = original_image.size
		normalized_word_boxes = boxes #[normalize_box(bbox, width, height) for bbox in unnormalized_word_boxes]
		unnormalized_word_boxes = [denormalize(bbox, width, height) for bbox in normalized_word_boxes]
		# assert len(words) == len(normalized_word_boxes)
		assert len(words) == len(boxes)

		# next, transform to token-level (input_ids, attention_mask, token_type_ids, bbox, labels)
		token_boxes = []
		unnormalized_token_boxes = []
		# token_labels = []
		token_labels = word_labels
		# token_labels = [label2id[label] for label in word_labels]
  
		for word, unnormalized_box, box, label in zip(words, unnormalized_word_boxes, normalized_word_boxes, word_labels):
			word_tokens = self.tokenizer.tokenize(word)
			unnormalized_token_boxes.extend(unnormalized_box for _ in range(len(word_tokens)))
			token_boxes.extend(box for _ in range(len(word_tokens)))
			# label first token as B-label (beginning), label all remaining tokens as I-label (inside)
			# ?????????????????
			# for i in range(len(word_tokens)):
			# 	if i == 0:
			# 		token_labels.extend(['B-' + label])
			# 	else:
			# 		token_labels.extend(['I-' + label])
		
		# Truncation of token_boxes + token_labels
		special_tokens_count = 2 
		if len(token_boxes) > self.max_seq_length - special_tokens_count:
			token_boxes = token_boxes[: (self.max_seq_length - special_tokens_count)]
			unnormalized_token_boxes = unnormalized_token_boxes[: (self.max_seq_length - special_tokens_count)]
			token_labels = token_labels[: (self.max_seq_length - special_tokens_count)]
		token_labels = token_labels[: (self.max_seq_length - special_tokens_count)] ########???
		# add bounding boxes and labels of cls + sep tokens
		token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]
		unnormalized_token_boxes = [[0, 0, 0, 0]] + unnormalized_token_boxes + [[1000, 1000, 1000, 1000]]
		token_labels = [-100] + token_labels + [-100]
		# print(len(token_labels))
		encoding = self.tokenizer(' '.join(words), padding='max_length', truncation=True)
		# Padding of token_boxes up the bounding boxes to the sequence length.
		input_ids = self.tokenizer(' '.join(words), truncation=True)["input_ids"]
		padding_length = self.max_seq_length - len(input_ids)
		token_boxes += [self.pad_token_box] * padding_length
		unnormalized_token_boxes += [self.pad_token_box] * padding_length
		# token_labels += [-100] * padding_length
		token_labels += [-100] * (self.max_seq_length - len(token_labels))
		encoding['bbox'] = token_boxes
		encoding['labels'] = token_labels
		assert len(encoding['input_ids']) == self.max_seq_length
		assert len(encoding['attention_mask']) == self.max_seq_length
		assert len(encoding['token_type_ids']) == self.max_seq_length
		assert len(encoding['bbox']) == self.max_seq_length
		assert len(encoding['labels']) == self.max_seq_length

		encoding['resized_image'] = ToTensor()(resized_image)
		# rescale and align the bounding boxes to match the resized image size (typically 224x224) 
		encoding['resized_and_aligned_bounding_boxes'] = [resize_and_align_bounding_box(bbox, original_image, self.target_size) 
															for bbox in unnormalized_token_boxes]

		encoding['unnormalized_token_boxes'] = unnormalized_token_boxes
		
		# finally, convert everything to PyTorch tensors 
		for k,v in encoding.items():
			if k == 'labels':
				label_indices = []
				# convert labels from string to indices
				for label in encoding[k]:
					if label != -100:
						label_indices.append(label2id[label])
					else:
						label_indices.append(label)
				encoding[k] = label_indices
			encoding[k] = torch.as_tensor(encoding[k])
		
		return encoding




class SROIEDataset(Dataset):
	"""CORD dataset."""

	def __init__(self, annotations, image_dir, processor=None, max_length=512):
		"""
		Args:
			annotations (List[List]): List of lists containing the word-level annotations (words, labels, boxes).
			image_dir (string): Directory with all the document images.
			processor (LayoutLMv2Processor): Processor to prepare the text + image.
		"""
		self.words, self.labels, self.boxes = annotations
		self.image_dir = image_dir
		self.image_file_names = list(listdir(image_dir))
		self.processor = processor

		print(f"len of words: {len(self.words)}, labels: {len(self.labels)}, boxes: {len(self.boxes)}")
		# exit("+++++++++++")
	def __len__(self):
		return len(self.image_file_names)

	def __getitem__(self, idx):
		# print("Index:", idx)
		# first, take an image
		item = self.image_file_names[idx]
		image = Image.open(self.image_dir + item).convert("RGB")
		# image_width, image_height = image.size
		# get word-level annotations
		words = self.words[idx]
		boxes = self.boxes[idx]
		word_labels = self.labels[idx]
		# image = np.array(image)
		# print((self.image_dir + item))
		# print(words)
		# # print(boxes)
		# # Draw each bounding box
		# for box, word in zip(boxes, words):
		# 	# Scale the bounding box to image dimensions if necessary
		# 	x1, y1, x2, y2 = box
		# 	# Draw the rectangle on the image
		# 	cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
		# 	# Put the corresponding word near the box
		# 	cv2.putText(image, word, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
		# cv2.imwrite('output_path.png', image)
		# exit('OLLLL')
		assert len(words) == len(boxes) == len(word_labels)

		word_labels = [label2id[label] for label in word_labels]
		# use processor to prepare everything
		encoded_inputs = self.processor(image, words, boxes=boxes, word_labels=word_labels,
										padding="max_length", truncation=True,
										return_tensors="pt")

		# remove batch dimension
		for k, v in encoded_inputs.items():
			encoded_inputs[k] = v.squeeze()

		assert encoded_inputs.input_ids.shape == torch.Size([512])
		assert encoded_inputs.attention_mask.shape == torch.Size([512])
		assert encoded_inputs.token_type_ids.shape == torch.Size([512])
		assert encoded_inputs.bbox.shape == torch.Size([512, 4])
		assert encoded_inputs.image.shape == torch.Size([3, 224, 224])
		assert encoded_inputs.labels.shape == torch.Size([512])
		return encoded_inputs


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


def results_train(preds, out_label_ids, labels):
	preds = np.argmax(preds, axis=2)

	label_map = {i: label for i, label in enumerate(labels)}

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
		"f1": f1_score(out_label_list, preds_list),
	}
	return results, classification_report(out_label_list, preds_list)


import csv
from itertools import zip_longest
def write_to_csv_train(data, file_path):
	max_length = max(len(column) for column in data)
	rows = zip_longest(*data, fillvalue='')
	with open(file_path, 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerows(rows)
csv_file_path = 'trainin_viz.csv'

def set_basic_config_for_logging(folder_path, filename: str = None):
	"""    
	Set the basic config for logging python program.   
	:return: None   
	"""    
	# Create and configure logger    
	log_file_path = os.path.join(folder_path, f"{filename}.log")
	logging.basicConfig(filename=log_file_path, format='%(asctime)s %(message)s',
						filemode='w')
	
def get_logger_object_and_setting_the_loglevel():
	"""    get the logger object and set the loglevel for the logger object    
	:return: Logger Object    
	"""    
	# Creating an object    
	logger_object = logging.getLogger()
	# Setting the threshold of logger to DEBUG    
	logger_object.setLevel(logging.DEBUG)
	return logger_object





log_dir = "logs"  # Directory to store the TensorBoard logs

# from config.prod_mapping import product_code_map, document_code_map



# # product config
# product_config = ConfigParser()
# product_config.read("config/config.ini")

# prod_code = product_code_map[product_config["Product"]["code"]]
# doc_code = product_config["Product"]["document_code"]
# if '[' in doc_code:
# 	doc_elements = doc_code[1:-1].split(', ')
# 	# Convert elements to a Python list
# 	doc_code_list = [element.strip() for element in doc_elements]
# print(doc_code)
# # data folder path
# product_wise_folder = ConfigParser()
# product_wise_folder.read("config/prod.ini")



folder_path = '/home/data_science/geo_testing/COO_V3/CI_train'
logger = get_logger_object_and_setting_the_loglevel()

# for doc_code_ in doc_code_list:

# 	doc_code = document_code_map[doc_code_]
# 	folder_path = product_wise_folder[prod_code][doc_code]
# 	# folder_path = '/New_Volume/Rakesh/DATA_LMV2/LMV2_BASE/AWB'
# 	print("==================Trade Finance Solutions===================")
# 	# print("Product Code: {product_code}")
# 	# print("Documenry Code: {doc_code}")
# 	print(f"folder_path: {folder_path}")
# 	set_basic_config_for_logging(folder_path, filename="train_words_count")

train = pd.read_pickle(os.path.join(folder_path, 'train.pkl'))
test = pd.read_pickle(os.path.join(folder_path, 'test.pkl'))


# print(f"test data : {type(test)}")
# print(test[:1])
# exit("+++++++++++++")



###################### not required ################################
# train_writer = tf.summary.create_file_writer("logs/train/")
# test_writer = tf.summary.create_file_writer("logs/test/")
# best_train_test_writer= tf.summary.create_file_writer("logs/best")
########################################################################
doc_code = 'board'
doc_code_ = 'lc'
train_writer = SummaryWriter(log_dir=f'''logs/{doc_code}/{doc_code_}/{"_".join(str(datetime.now()).split(" "))}/train''')
test_writer = SummaryWriter(log_dir=f'''logs/{doc_code}/{doc_code_}/{"_".join(str(datetime.now()).split(" "))}/test''')
best_writer = SummaryWriter(log_dir=f'''logs/{doc_code}/{doc_code_}/{"_".join(str(datetime.now()).split(" "))}/best''')

print("version of the cuda")
print(torch.__version__)
print(f"cuda available: {torch.cuda.is_available()}")

train_samples = len(train[0])
test_samples = len(test[0])


train_text, train_label, train_bb = ['TEXT'], ['LABELS'], ['BOUNDING_BOXES']

for j in range(len(train[0])):
	train_text = train_text + train[0][j] + [' ' for _ in range(512-len(train[0][j]))]
	train_label = train_label + train[1][j] + [' ' for _ in range(512-len(train[1][j]))]
	train_bb = train_bb + train[2][j] + [' ' for _ in range(512-len(train[2][j]))]

all_labels = [item for sublist in train[1] for item in sublist] + [item for sublist in test[1] for item in sublist]
Counter(all_labels)
label_new = dict(Counter(all_labels))
print(label_new)
labels = list(set(all_labels))
print(labels)
print(len(labels))



with open(os.path.join(folder_path, "classes.txt"), "w") as f:
	f.write(str(labels))
f.close()

#same count in labels and classes (+1 for others)
label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for idx, label in enumerate(labels)}
print(label2id)
print(id2label)
from transformers import BertTokenizer


# processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", 
# 												revision="no_ocr")

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_dataset = FUNSDDataset(annotations=train,
								image_dir=os.path.join(folder_path
													, "train/"), tokenizer=tokenizer, max_length=512, target_size=224)
test_dataset = FUNSDDataset(annotations=test,
								image_dir=os.path.join(folder_path
													, "test/"), tokenizer=tokenizer, max_length=512, target_size=224, train=False)

# exit('exit111111111111')
# train_dataset = SROIEDataset(annotations=train,
# 								image_dir=os.path.join(folder_path
# 													, "train/"),
# 								processor=processor)

# test_dataset = SROIEDataset(annotations=test,
# 							image_dir=os.path.join(folder_path, "test/"),
# 							processor=processor)

token_ids  = ['token_ids']
encode_ids = ['tokenized_text']
enco_lab_ids = ['tokenized_text_label_ids']
zip_id2labels = ['tokenized_text_label']

'''for i in range(len(train_dataset)):
	encoding = train_dataset[i]
	print(len(encoding['labels']),encoding['labels'])
	#exit()
	encoding_ids = []
	enco_label_ids = []
	tokens = []
	for id, label in zip(encoding['input_ids'], encoding['labels']):
		tokens.append(id.item())
		encoding_ids.append(processor.tokenizer.decode(id.item()))
		enco_label_ids.append(label.item())
	
	#print(enco_lab_ids)
	ids2lab = []
	for label in enco_label_ids:
		if label != -100:
			ids2lab.append(id2label[label])
		else:
			ids2lab.append('O')

	encode_ids = encode_ids + encoding_ids
	enco_lab_ids = enco_lab_ids + enco_label_ids  
	
	token_ids = token_ids+tokens
	zip_id2labels = zip_id2labels+ids2lab			
data = [train_text, train_label, train_bb, token_ids, encode_ids, enco_lab_ids, zip_id2labels]
write_to_csv_train(data, csv_file_path)
'''

#creating the log file for having the count of words
'''sample = 0
for cou1, cou2 in zip(range(len(train[0])), range(len(train_dataset))):
	sample+=1
	word_count = 0
	padding_count = 0
	print('sample =>', sample)
	print("============")
	print('train', len(train[0][cou1]))
	# print(len(train_dataset[cou2]))
	encoding = train_dataset[cou2]
	print('embidding', len(encoding['input_ids']))
	for id in encoding['input_ids']:
		if id.item()==0:
			padding_count+=1
		else:
			word_count+=1
		
	print('word_coun', word_count)
	print('padding_count', padding_count)
	print("***********")
	logger.info(f"sample: {sample}; train_chunk_count: {len(train[0][cou1])}; embedding: {len(encoding['input_ids'])}; word_count: {word_count}; padding_count: {padding_count}")
'''
# with open(os.path.join(folder_path, "label.txt"), "r") as file:
# 	class_names: List = file.readlines()
# 	class_names = list(map(lambda x: x.strip(), class_names))
# 	dict_mapping = dict(enumerate(class_names))
# file.close()
# logger.info(f"actual_label_length: {len(dict_mapping)}; train_gen_classes:{len(id2label)}")


encoding = train_dataset[0]
encoding.keys()
for k, v in encoding.items():
	print(k, v.shape)

# print(processor.tokenizer.decode(encoding['input_ids']))
print(tokenizer.decode(encoding['input_ids']))
print(train[0][0])
print(train[1][0])
print([id2label[label] for label in encoding['labels'].tolist() if label != -100])

for id, label in zip(encoding['input_ids'][:30], encoding['labels'][:30]):
	# print(processor.tokenizer.decode([id]), label.item())
	print(tokenizer.decode([id]), label.item())

batch_size = 4
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
# exit('YES')
'''
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

'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = LayoutLMv2ForTokenClassification(config)
# print(model)
# exit('OK')

# model = LayoutLMv2ForTokenClassification.from_pretrained('microsoft/layoutlmv2-base-uncased',
# 															num_labels=len(labels))

model = LayoutLMForTokenClassification()
# exit('*************')
print(device)
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)
labels = list(set(all_labels))
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





from transformers import AdamW
# from tqdm.notebook import tqdm

optimizer = AdamW(model.parameters(), lr=5e-5)

global_step = 0
num_train_epochs = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#put the model in training mode
model.to(device)
model.train()
for epoch in range(num_train_epochs):
	print("Epoch:", epoch)
	for batch in tqdm(train_dataloader):
		# forward pass
		input_ids=batch['input_ids'].to(device)
		bbox=batch['bbox'].to(device)
		attention_mask=batch['attention_mask'].to(device)
		token_type_ids=batch['token_type_ids'].to(device)
		labels=batch['labels'].to(device)
		resized_images = batch['resized_image'].to(device) 
		resized_and_aligned_bounding_boxes = batch['resized_and_aligned_bounding_boxes'].to(device) 

		outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids, 
						labels=labels, resized_images=resized_images, resized_and_aligned_bounding_boxes=resized_and_aligned_bounding_boxes)
		loss = outputs.loss

		# print loss every 10 steps
		if global_step % 10 == 0:
			print(f"Loss after {global_step} steps: {loss.item()}")

		# backward pass to get the gradients 
		loss.backward()

		# update
		optimizer.step()
		optimizer.zero_grad()
		global_step += 1

	# model.eval()
	training_loss[epoch] = loss 
	val_loss = 0.0
	preds_val = None
	for batch in tqdm(test_dataloader):
		# forward pass
		input_ids=batch['input_ids'].to(device)
		bbox=batch['bbox'].to(device)
		attention_mask=batch['attention_mask'].to(device)
		token_type_ids=batch['token_type_ids'].to(device)
		labels=batch['labels'].to(device)
		resized_images = batch['resized_image'].to(device) 
		resized_and_aligned_bounding_boxes = batch['resized_and_aligned_bounding_boxes'].to(device) 

		outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids, 
						labels=labels, resized_images=resized_images, resized_and_aligned_bounding_boxes=resized_and_aligned_bounding_boxes)
		# loss = outputs.loss
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



eval_loss = 0.0
nb_eval_steps = 0
preds = None
out_label_ids = None
labels = list(set(all_labels))
label_map_updated = {i: label for i, label in enumerate(labels)}

# put model in evaluation mode
model.eval()
for batch in tqdm(test_dataloader, desc="Evaluating"):
    with torch.no_grad():
        input_ids=batch['input_ids'].to(device)
        bbox=batch['bbox'].to(device)
        attention_mask=batch['attention_mask'].to(device)
        token_type_ids=batch['token_type_ids'].to(device)
        labels=batch['labels'].to(device)
        resized_images = batch['resized_image'].to(device) 
        resized_and_aligned_bounding_boxes = batch['resized_and_aligned_bounding_boxes'].to(device) 

        # forward pass
        outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids, 
                        labels=labels, resized_images=resized_images, resized_and_aligned_bounding_boxes=resized_and_aligned_bounding_boxes)

        # get the loss and logits
        tmp_eval_loss = outputs.loss
        logits = outputs.logits

        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1

        # compute the predictions
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, labels.detach().cpu().numpy(), axis=0
            )

# compute average evaluation loss
eval_loss = eval_loss / nb_eval_steps
preds = np.argmax(preds, axis=2)

out_label_list = [[] for _ in range(out_label_ids.shape[0])]
preds_list = [[] for _ in range(out_label_ids.shape[0])]

for i in range(out_label_ids.shape[0]):
    for j in range(out_label_ids.shape[1]):
        if out_label_ids[i, j] != -100:
            out_label_list[i].append(label_map_updated[out_label_ids[i][j]])
            preds_list[i].append(label_map_updated[preds[i][j]])

results = {
    "loss": eval_loss,
    "precision": precision_score(out_label_list, preds_list),
    "recall": recall_score(out_label_list, preds_list),
    "f1": f1_score(out_label_list, preds_list),
}
print(results)

exit('DONE')
















# model = LayoutLMv2ForTokenClassification.from_pretrained(
# 		pretrained_model_name_or_path=os.path.join(model_path, 'pytorch_model.bin'),
# 		config=os.path.join(model_path, 'config.json'))

# model = LayoutLMv2ForTokenClassification.from_pretrained(model_path)

print(training_loss)
with open(os.path.join(folder_path, "training_loss.txt"), 'w') as f:
	for key, value in training_loss.items():
		f.write(f"{key}: {value}\n") 

print(validation_loss)
with open(os.path.join(folder_path, "validation_loss.txt"), 'w') as f:
	for key, value in validation_loss.items():
		f.write(f"{key}: {value}\n")        
		
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
encoding = test_dataset[0]
# processor.tokenizer.decode(encoding['input_ids'])
tokenizer.decode(encoding['input_ids'])
ground_truth_labels = [id2label[label] for label in encoding['labels'].squeeze().tolist() if label != -100]
print(ground_truth_labels)
exit("DONE")
for k, v in encoding.items():
	encoding[k] = v.unsqueeze(0).to(device)

preds_val = None
out_label_ids = None
# put model in evaluation mode
model.eval()
for batch in tqdm(test_dataloader, desc="Evaluating"):
	with torch.no_grad():
		input_ids = batch['input_ids'].to(device)
		bbox = batch['bbox'].to(device)
		image = batch['image'].to(device)
		attention_mask = batch['attention_mask'].to(device)
		token_type_ids = batch['token_type_ids'].to(device)
		labels = batch['labels'].to(device)

		# forward pass
		outputs = model(input_ids=input_ids, bbox=bbox, image=image, attention_mask=attention_mask,
						token_type_ids=token_type_ids, labels=labels)

		if preds_val is None:
			preds_val = outputs.logits.detach().cpu().numpy()
			out_label_ids = batch["labels"].detach().cpu().numpy()
		else:
			preds_val = np.append(preds_val, outputs.logits.detach().cpu().numpy(), axis=0)
			out_label_ids = np.append(
				out_label_ids, batch["labels"].detach().cpu().numpy(), axis=0)

labels = list(set(all_labels))
val_result, class_report = results_test(preds_val, out_label_ids, labels)
test_result = val_result
test_all = class_report
print("Overall results:", val_result)
print(class_report)
with open(os.path.join(folder_path, "test_report.txt"), 'w') as f:
	f.write(str(val_result))
	f.write('\n')
	f.write(class_report)
f.close()

# put model in evaluation mode
preds_val = None
out_label_ids = None
model.eval()
for batch in tqdm(train_dataloader, desc="Evaluating"):
	with torch.no_grad():
		input_ids = batch['input_ids'].to(device)
		bbox = batch['bbox'].to(device)
		image = batch['image'].to(device)
		attention_mask = batch['attention_mask'].to(device)
		token_type_ids = batch['token_type_ids'].to(device)
		labels = batch['labels'].to(device)

		# forward pass
		outputs = model(input_ids=input_ids, bbox=bbox, image=image, attention_mask=attention_mask,
						token_type_ids=token_type_ids, labels=labels)

		if preds_val is None:
			preds_val = outputs.logits.detach().cpu().numpy()
			out_label_ids = batch["labels"].detach().cpu().numpy()
		else:
			preds_val = np.append(preds_val, outputs.logits.detach().cpu().numpy(), axis=0)
			out_label_ids = np.append(
				out_label_ids, batch["labels"].detach().cpu().numpy(), axis=0
			)

labels = list(set(all_labels))
val_result, class_report = results_train(preds_val, out_label_ids, labels)
train_result = val_result
train_all = class_report
print("Overall results:", val_result)
print(class_report) 
print('woo!,Model Training has done successfully')
