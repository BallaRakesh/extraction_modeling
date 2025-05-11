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
from configparser import ConfigParser
from os import listdir
import torch
from torch.utils.data import Dataset
from PIL import Image


# Global imports
# Step1: Reading configurations from ini files
parser = ConfigParser()
conf_folder_path: str = "/media/tarun/D1/Trade-Finance/src/main/extraction/config"
config_file_name: str = "config.ini"

if os.path.exists(f"{conf_folder_path}/{config_file_name}"):
	parser.read(f"{conf_folder_path}/{config_file_name}")

debug_mode: str = str(parser["PARAMS"]["debug_mode"])


class TradeFinanceDataset(Dataset):
	def __init__(self, annotations, image_dir, processor=None, max_length=512,
	             label_id_mapping: dict = None):
		"""
		Args:
			annotations (List[List]): List of lists containing the word-level annotations
			(words, labels, boxes).
			image_dir (string): Directory with all the document images.
			processor (LayoutLMv2Processor): Processor to prepare the text + image.
		"""
		self.words, self.labels, self.boxes = annotations
		self.image_dir = image_dir
		self.image_file_names = [image_file for image_file in listdir(image_dir)]
		self.processor = processor
		self.label2idmapping = label_id_mapping
		self.max_length = max_length

	def __len__(self):
		"""
		:return: Length of all images
		"""
		return len(self.image_file_names)

	def __getitem__(self, idx):
		"""
		return idx wise annotations
		:param idx:
		:return:
		"""
		item = self.image_file_names[idx]
		try:
			image = Image.open(fp=self.image_dir + item, mode="r").convert(mode="RGB")
		except IOError as io:
			print("I/O error({0}): {1}".format(io.errno, io.strerror))
			exit(f"error in reading a file name: {item} at path: {self.image_dir}")

		# get word-level annotations
		words = self.words[idx]
		boxes = self.boxes[idx]
		word_labels = self.labels[idx]

		if len(words) == len(boxes) == len(word_labels):
			print("length matched!!!!!!!!")
		else:
			exit("Length mismatch error")

		# convert word labels to id while training
		word_labels = [self.label2idmapping[label] for label in word_labels]

		if image is not None:
			"""
            `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
            acceptable input length for the model if that argument is not provided.
			"""
			# use processor to prepare everything
			encoded_inputs = self.processor(image, words, boxes=boxes, word_labels=word_labels,
			                                padding=self.max_length, truncation=True,
			                                return_tensors="pt")

		# remove batch dimension
		for k, v in encoded_inputs.items():
			encoded_inputs[k] = v.squeeze()
		if debug_mode:
			assert encoded_inputs.input_ids.shape == torch.Size([512]), "input ids shape not matching"
			assert encoded_inputs.attention_mask.shape == torch.Size([512]), "attention masks shape is not matching"
			assert encoded_inputs.token_type_ids.shape == torch.Size([512]), "token type ids shape not matching"
			assert encoded_inputs.bbox.shape == torch.Size([512, 4]), "bbox shape not matching"
			assert encoded_inputs.image.shape == torch.Size([3, 224, 224]), "image shape is not matching"
			assert encoded_inputs.labels.shape == torch.Size([512]), "labels shape is not matching"
		return encoded_inputs
