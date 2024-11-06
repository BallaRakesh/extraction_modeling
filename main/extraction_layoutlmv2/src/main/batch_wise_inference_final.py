from torch.utils.data import Dataset
from os import listdir
from transformers import LayoutLMv2Processor
from torch.utils.data import DataLoader
import tqdm
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
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
import tensorflow as tf
import logging
from typing import List
from configparser import ConfigParser
from datetime import datetime
warnings.filterwarnings("ignore")

class SROIEDataset(Dataset):
	"""CORD dataset."""

	def __init__(self, annotations, image_dir, processor=None, max_length=512):
		"""
		Args:
			annotations (List[List]): List of lists containing the word-level annotations (words, labels, boxes).
			image_dir (string): Directory with all the document images.
			processor (LayoutLMv2Processor): Processor to prepare the text + image.
		"""
		self.words, self.boxes = annotations
		self.image_dir = image_dir
		self.image_file_names = list(listdir(image_dir))
		self.processor = processor

		print(f"len of words: {len(self.words)}, boxes: {len(self.boxes)}")
		# exit("+++++++++++")
	def __len__(self):
		return len(self.image_file_names)

	def __getitem__(self, idx):
		# print("Index:", idx)
		# first, take an image
		item = self.image_file_names[idx]
		image = Image.open(self.image_dir + item).convert("RGB")

		# get word-level annotations
		words = self.words[idx]
		boxes = self.boxes[idx]

		assert len(words) == len(boxes)

		# use processor to prepare everything
		encoded_inputs = self.processor(image, words, boxes=boxes,
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
		return encoded_inputs


def normalize(points: list, width: int, height: int) -> list:
	x0, y0, x2, y2 = [int(p) for p in points]

	x0 = int(1000 * (x0 / width))
	x2 = int(1000 * (x2 / width))
	y0 = int(1000 * (y0 / height))
	y2 = int(1000 * (y2 / height))

	if x0 > 1000:
		x0 = 1000
	# print(">")
	if x0 < 0:
		x0 = 0
	# print("<")
	if x2 > 1000:
		x2 = 1000
	# print(">")
	if x2 < 0:
		x2 = 0
	# print("<")
	if y0 > 1000:
		y0 = 1000
	# print(">")
	if y0 < 0:
		y0 = 0
	# print("<")
	if y2 > 1000:
		y2 = 1000
	# print(">")
	if y2 < 0:
		y2 = 0
	# print("<")
	return [x0, y0, x2, y2]


def unnormalize_box(bbox, width, height):
	return [
		int(width * (bbox[0] / 1000)),
		int(height * (bbox[1] / 1000)),
		int(width * (bbox[2] / 1000)),
		int(height * (bbox[3] / 1000)),
	]
def most_common(lst):
	return max(set(lst), key=lst.count)

 
import torchvision.transforms as transforms
import json
import random
from scipy.special import softmax
from PIL import Image, ImageDraw, ImageFont

transform2 = transforms.ToPILImage()
transform = transforms.ToTensor()

processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_dir_testing = '/home/ntlpt19/Downloads/Final_Delivery_Training_itter_2/testing_finetuning/PI/testing_images/Images'
model_path = '/home/ntlpt19/Downloads/Final_Delivery_Training_itter_2/testing_finetuning/PI/pi_model/Best_Model'
ocr_path = '/home/ntlpt19/Downloads/Final_Delivery_Training_itter_2/testing_finetuning/PI/testing_images/OCR'
output_path = "/home/ntlpt19/Downloads/Final_Delivery_Training_itter_2/testing_finetuning/PI/testing_images/testing_out_put"
classes_txt = '/home/ntlpt19/Downloads/Final_Delivery_Training_itter_2/testing_finetuning/PI/pi_model/classes.txt'
sample_image_path = '/home/ntlpt19/Downloads/Final_Delivery_Training_itter_2/testing_finetuning/PI/testing_images/Images/Proforma_Invoice(2013_01_10_13_51_15_4180)_493_0.png'
batch_size_ = 1

model = LayoutLMv2ForTokenClassification.from_pretrained(
        pretrained_model_name_or_path=os.path.join(model_path),
        config=os.path.join(model_path, 'config.json'))

testing = []
master_words = []
master_bbox= []
for img_test in listdir(img_dir_testing):
    # word_coordinates, all_text = get_ocr_vision_api(os.path.join(img_dir_testing, img_test))
    
    ocr_path_word = os.path.join(ocr_path, img_test[:-4]+'_text.txt')
    with open(ocr_path_word, "r") as wc_file:
            ocr_data = json.load(wc_file)
    wc_file.close()
    image_ = Image.open(os.path.join(img_dir_testing, img_test))
    w, h = image_.size
    words: list = []
    bboxes: list = []
    bounding_boxes: list = []
    for t in ocr_data['word_coordinates']:
        if 'right' in list(t.keys()):
            t['x1'] = t['left']
            t['y1'] = t['top']
            t['x2'] = t['right']
            t['y2'] = t['bottom']
        words.append(t['word'])
        bounding_boxes.append([t['x1'], t['y1'], t['x2'], t['y2']])
        bboxes.append(normalize([t['x1'], t['y1'], t['x2'], t['y2']], w, h))
    master_words.append(words)
    master_bbox.append(bboxes)
testing = [master_words, master_bbox]

train_dataset = SROIEDataset(annotations=testing,image_dir=f'{img_dir_testing}/',processor=processor)




test_dataloader1 = DataLoader(train_dataset, batch_size=batch_size_)

t_start = datetime.now()

batch_count = 0
for batch in tqdm(test_dataloader1, desc="inference"):
    with torch.no_grad(): 
        input_ids = batch['input_ids'].to(device)
        bbox = batch['bbox'].to(device)
        image = batch['image'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        print(">>>>>>>>>>>>>>>>>>>>>>>> entered into the model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        outputs = model(input_ids=input_ids,
                                bbox=bbox,
                                image=image,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids)

        #######################################################
        all_predictions: list = []
        all_boxes: list = []
        all_confidences: list = []
        all_text: list = []
        ########################################################

        # this is the list of classes that will be given to us to be extracted.
        with open(classes_txt, "r") as f:
            labels = eval(f.read())
        f.close()

        # Creating two dictionaries labels2id and id2labels
        labels = [x.replace("S-", "") for x in labels]
        label2id = {label: idx for idx, label in enumerate(labels)}
        id2label = {idx: label for idx, label in enumerate(labels)}

        # fixed 80 unique hexcodes has been created if more than 80 classes will be there
        # this needs to change
        ##############################################################################
        number_of_colors: int = 200
        # creating random hexcodes
        color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                    for i in range(number_of_colors)]
        ##############################################################################

        # color for each label
        label2color = {}
        for i, l in enumerate(labels):
            label2color[l] = color[i]
        image_png = Image.open(sample_image_path)
        arr = transform(image_png)
        ########################################################################3
        # bbox_chunks = list(encoded_inputs['bbox'][0].split(510))
        bbox_chunks = bbox
        for i, output in enumerate(outputs.logits):
            # print(i, output)
            # converting back into PIL image
            new_img = transform2(arr)
            # loading the image font
            font = ImageFont.truetype(font="/home/ntlpt19/Downloads/Evaluation_Data/updated_code/src/main/extraction/arial.ttf", size=20)

            predictions = output.argmax(-1).squeeze().tolist()
            # print('the predictions', predictions)

            confidences = softmax(output.cpu().detach().numpy(), axis=1)
            # print(confidences)

            max_confidences = np.max(confidences, axis=1).reshape(confidences.shape[0], -1)
            # print(max_confidences)

            all_confidences += [x[0] for x in max_confidences]
            # print(all_confidences)

            token_boxes = bbox_chunks[i].squeeze().tolist()
            width, height = new_img.size
            true_predictions = [id2label[prediction] for prediction in predictions]
            all_predictions += true_predictions
            # print('all_predictions',all_predictions)

            true_boxes = [unnormalize_box(box, width, height) for box in token_boxes]
            all_boxes += true_boxes
            # print(all_boxes)

            for id in input_ids[i]:
                all_text.append(processor.tokenizer.decode(id))
            print("+++++++++++++++++++>>>>>>>>>>>>>>>", i)

        del outputs

        new_img = transform2(arr)
        draw = ImageDraw.Draw(new_img)
        # print("%20s - %30s - %12s - %30s" % ("Text", "Prediction", "Confidence", "Bounding Box"))

        curr_box: list = []
        results_pred: list = []
        results_conf: list = []
        results_bbox: list = []
        results_text: list = []
        temp_preds: list = []
        temp_confs: list = []
        temp_text: list = []
        sep_index: list = all_text.index('[SEP]')

        if len(all_text) > 512:
            if '[PAD]' in all_text:
                sep_index = all_text.index('[PAD]') - 2
            else:
                sep_index = len(all_text) - 3

        # print(sep_index)
        for i in range(len(all_text)):
            if all_text[i] not in ['[CLS]', '[SEP]', '[PAD]']:  # and all_predictions[i] != 'O':
                # print(i)
                if (curr_box != all_boxes[i] and \
                    len(temp_text) > 0) or \
                        (i == sep_index - 1 and len(temp_text) > 0):

                    # print("1: ", all_text[i])
                    if i == sep_index - 1:
                        temp_text.append(all_text[i])
                        temp_confs.append(all_confidences[i])
                        temp_preds.append(all_predictions[i])
                    text = ""
                    pred = "O"
                    preds = [x for x in temp_preds if x != 'O']
                    conf = 0
                    if len(preds) > 0:
                        pred = most_common(preds)

                    for j in range(len(temp_text)):
                        text += temp_text[j].replace("##", "")
                        if temp_preds[j] == pred:
                            conf += temp_confs[j]

                    conf = float(np.round(conf * 100 / len(temp_text), 2))

                    results_text.append(text)
                    results_conf.append(conf)
                    results_pred.append(pred)
                    results_bbox.append(curr_box)

                    temp_text = []
                    temp_confs = []
                    temp_preds = []
                    if i != sep_index - 1:
                        temp_text.append(all_text[i])
                        temp_confs.append(all_confidences[i])
                        temp_preds.append(all_predictions[i])
                        curr_box = all_boxes[i]
                elif curr_box == all_boxes[i]:
                    temp_text.append(all_text[i])
                    temp_confs.append(all_confidences[i])
                    temp_preds.append(all_predictions[i])
                elif len(temp_text) == 0:
                    temp_text.append(all_text[i])
                    temp_confs.append(all_confidences[i])
                    temp_preds.append(all_predictions[i])
                    curr_box = all_boxes[i]
        # print('#',results_text)
        # print('##',results_conf)
        # print(results_pred)
        # print(results_bbox)
        # exit()
        result_set = {}

        for i in range(len(results_pred)):
            if results_pred[i] != 'O':
                if results_pred[i] not in list(result_set.keys()):
                    result_set[results_pred[i]] = []
                result_set[results_pred[i]].append([results_text[i],
                                                    results_bbox[i], results_conf[i]])
        # print(result_set)
        model_output = result_set.copy()
        # print(f'model output++++++++++++++++++++++++++++=={model_output}')
        # exit()
        print("+++++++++++++++++++reached here+++++++++++++++++")
        # exit("+++++++++")
        with open(os.path.join(output_path, f"model_output.txt{batch_count}"), "w") as f:
            json.dump(result_set, f)
        f.close()
        batch_count+=1 
        
t_end = datetime.now()
print("Time Taken:", t_end - t_start)