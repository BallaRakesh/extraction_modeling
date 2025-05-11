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

from configparser import ConfigParser
from datetime import datetime
import pandas as pd
import os
from collections import Counter
import psutil
from torch.utils.data import DataLoader
from transformers import LayoutLMv2Processor
from transformers import LayoutLMv2ForTokenClassification, AdamW
import torch
from tqdm.notebook import tqdm
import numpy as np
import warnings
import training_utility as tu
import socket

from src.main.extraction.dataset import TradeFinanceDataset
from src.main.extraction.utility import get_gpu_memory_usage

warnings.filterwarnings("ignore")
from seqeval.metrics import (
	classification_report,
	f1_score,
	precision_score,
	recall_score, )


def results_test(preds, out_label_ids, labels):
	preds = np.argmax(preds, axis=2)
	label_map: dict = {i: label for i, label in enumerate(labels)}

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


if __name__ == "__main__":
	# get hostname on linux by writing hostname
	hostname: str = socket.gethostname()
	print("===================Training started==================")
	# Step1: Reading configurations from ini files
	parser = ConfigParser()

	# Hard dependency
	# Change according to your system
	conf_folder_path = "src/main/extraction/config"
	if hostname == "NTLPT25":
		conf_folder_path: str = "/src/main/extraction/config/config.ini"

	config_file_name: str = "config.ini"

	try:
		if os.path.exists(f"{conf_folder_path}/{config_file_name}"):
			parser.read(f"{conf_folder_path}/{config_file_name}")
	except Exception as e:
		exit("Not able to read the config file! Challenge in reading the file")

	gv_key = parser['OCR']['gv_key']
	# all files will be dumped in this folder only
	root_folder_path = str(parser['PATHS']['root_folder'])
	debug_mode = str(parser["PARAMS"]["debug_mode"])
	log_folder_path = str(parser['LOG']['LOG_PATH_FOLDER'])
	training_log_filename: str = str(parser['LOG']['TRAIN_LOG_FILENAME'])
	log_level: str = str(parser['LOG']['LOG_FILTER_LEVELMIN'])

	# Step 2: Setting up logger
	# setting the log folder and file
	tu.set_basic_config_for_logging(folder_path=f"{root_folder_path}/{log_folder_path}", filename="training")

	# setting the logger object and log level
	logger = tu.get_logger_object_and_setting_the_loglevel(log_level=log_level)

	# info logs
	logger.info(f"Root folder path: {root_folder_path}")
	logger.info(f"debug mode: {debug_mode}")
	logger.info(f"log folder path: {log_folder_path}")
	logger.info(f"training log file name: {training_log_filename}")
	logger.info(f"log level: {log_level}")

	# initial stats
	process_memory = psutil.Process()
	start_time = datetime.now()
	cpu_utilization_start = psutil.cpu_percent()
	before_memory = process_memory.memory_info().rss

	# info logs
	logger.info("version of the cuda")
	logger.info(torch.__version__)
	logger.info(f"cuda available: {torch.cuda.is_available()}")

	train = pd.read_pickle(os.path.join(root_folder_path, 'train.pkl'))
	test = pd.read_pickle(os.path.join(root_folder_path, 'test.pkl'))

	if debug_mode:
		print(f"output after reading the pickle:")
		print(f"Number of elements in pickle file after extraction: {len(test)}")
		print(f"type check of the pickle output: {type(test)}")

	train_samples = len(train[0])
	test_samples = len(test[0])

	all_labels = [item for sublist in train[1] for item in sublist] + [item for sublist in test[1] for item in sublist]

	if debug_mode:
		# logging the counter object for all labels
		logger.info(f"counter for all the labels => \n {Counter(all_labels)}")

	label_new = dict(Counter(all_labels))
	labels = list(set(all_labels))

	# index order mapping in  classes_list.txt file while training
	with open(os.path.join(root_folder_path, "classes_list.txt"), "w") as f:
		f.write(str(labels))
	f.close()

	label2id = {label: idx for idx, label in enumerate(labels)}
	id2label = {idx: label for idx, label in enumerate(labels)}

	if debug_mode:
		print(label2id)
		print(id2label)
		logger.info(f"label2id: {label2id}")
		logger.info(f"id2label: {id2label}")

	"""
	Ref Link: https://huggingface.co/docs/transformers/v4.31.0/en/model_doc/layoutlmv2#transformers.LayoutLMv2Processor
	It first uses LayoutLMv2ImageProcessor to resize document images to a fixed size, 
	and optionally applies OCR to get words and normalized bounding boxes. These are then 
	provided to LayoutLMv2Tokenizer or LayoutLMv2TokenizerFast, which turns the words and bounding 
	boxes into token-level input_ids, attention_mask, token_type_ids, bbox.
	"""

	processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")

	train_dataset = TradeFinanceDataset(annotations=train,
	                                    image_dir=os.path.join(root_folder_path
	                                                           , "train/"),
	                                    processor=processor)
	test_dataset = TradeFinanceDataset(annotations=test,
	                                   image_dir=os.path.join(root_folder_path, "test/"),
	                                   processor=processor)

	# validating the input coming from dataset code
	encoding = train_dataset[0]
	encoding.keys()
	for k, v in encoding.items():
		print(k, v.shape)
	if debug_mode:
		"""
		Ref Link: https://huggingface.co/docs/transformers/v4.31.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer
		"""
		print(processor.tokenizer.decode(token_ids=encoding['input_ids']))
		print(train[0][0])
		print(train[1][0])
		print([id2label[label] for label in encoding['labels'].tolist() if label != -100])
		if debug_mode:
			for id, label in zip(encoding['input_ids'][:30], encoding['labels'][:30]):
				print(processor.tokenizer.decode([id]), label.item())

	train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
	test_dataloader = DataLoader(test_dataset, batch_size=4)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	model = LayoutLMv2ForTokenClassification.from_pretrained('microsoft/layoutlmv2-base-uncased',
	                                                         num_labels=len(labels))

	print(f"device : {device}")
	if device.type == 'cuda':
		print("memory before:")
		get_gpu_memory_usage()

	# transferring model to device
	model.to(device)

	if device.type == 'cuda':
		print("memory after:")
		get_gpu_memory_usage()

	optimizer = AdamW(params=model.parameters(),
	                  lr=5e-5)

	labels = list(set(all_labels))

	# training essesntial variables
	global_step = 0
	num_train_epochs = 50
	val_loss = 0.0
	preds_val = None
	out_label_ids = None
	best_loss = None
	best_precision = None
	best_recall = None
	best_f1 = None
	steps = []
	losses = []

	# put the model in training mode
	model.train()
	for epoch in tqdm(num_train_epochs):
		print("Epoch:", epoch)
		for batch in tqdm(train_dataloader):
			input_ids = batch['input_ids'].to(device)
			bbox = batch['bbox'].to(device)
			image = batch['image'].to(device)
			attention_mask = batch['attention_mask'].to(device)
			token_type_ids = batch['token_type_ids'].to(device)
			labels = batch['labels'].to(device)

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
		for batch in tqdm(test_dataloader, desc="Evaluating"):
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
				val_loss = outputs.loss
				print(f'validation loss: {val_loss}')
				val_loss += val_loss.item()

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
		val_loss = val_loss / len(test_dataloader)
		print(f'final validation loss:{val_loss}')
		precision = val_result['precision']
		recall = val_result['recall']
		f1 = val_result['f1']
		if best_loss is None:
			best_loss = loss
		if best_precision is None:
			best_precision = precision
			best_recall = recall
			best_f1 = f1
		if loss < best_loss and f1 > best_f1 and recall > best_recall:
			best_loss = loss
			best_precision = precision
			best_recall = recall
			name = "Best_Model"

			if not os.path.exists(os.path.join(root_folder_path, name)):
				os.mkdir(os.path.join(root_folder_path, name))
			print(f'Model is {epoch} saving +++++++++++++++++++++++++++++++++')
			print(f"best Validation Loss: {best_loss}")
			print("best Precision:", best_precision)
			print("best Recall:", best_recall)
			model.save_pretrained(os.path.join(root_folder_path, name))

	model_path = "/New_Volume/handover_doc_extract/model_training" \
	             "/jul_13_certificate_of_origin_training_on_best_model_code/training_using_new_code/internal_data" \
	             "/Best_Model "

	model = LayoutLMv2ForTokenClassification.from_pretrained(
		pretrained_model_name_or_path=os.path.join(model_path, 'pytorch_model.bin'),
		config=os.path.join(model_path, 'config.json'))

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model.to(device)
	encoding = test_dataset[0]
	processor.tokenizer.decode(encoding['input_ids'])
	ground_truth_labels = [id2label[label] for label in encoding['labels'].squeeze().tolist() if label != -100]
	print(ground_truth_labels)

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
	with open(os.path.join(root_folder_path, "test_report.txt"), 'w') as f:
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
