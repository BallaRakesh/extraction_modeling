import torch
import pandas as pd
import os
from collections import Counter
from torch.utils.data import DataLoader
from os import listdir
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoProcessor
from transformers import AutoProcessor, AutoModelForTokenClassification, AdamW
import torch
from tqdm.notebook import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from seqeval.metrics import (
	classification_report,
	f1_score,
	precision_score,
	recall_score)


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
		self.image_file_names = [f for f in listdir(image_dir)]
		self.processor = processor

	def __len__(self):
		return len(self.image_file_names)

	def __getitem__(self, idx):
		# first, take an image
		item = self.image_file_names[idx]
		# print(item)
		image = Image.open(self.image_dir + item).convert("RGB")

		# get word-level annotations
		words = self.words[idx]
		boxes = self.boxes[idx]
		word_labels = self.labels[idx]

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
		assert encoded_inputs.bbox.shape == torch.Size([512, 4])
		assert encoded_inputs.pixel_values.shape == torch.Size([3, 224, 224])
		assert encoded_inputs.labels.shape == torch.Size([512])
		# print(encoded_inputs.labels)
		# print(encoded_inputs.bbox)

		return encoded_inputs


def results_test(preds, out_label_ids, labels):
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
	print("version of the cuda")
	print(torch.__version__)
	print(f"cuda available: {torch.cuda.is_available()}")

	folder_path = "/content/data/step_3_training_data"

	train = pd.read_pickle(os.path.join(folder_path, 'train.pkl'))
	test = pd.read_pickle(os.path.join(folder_path, 'test.pkl'))

	train_samples = len(train[0])
	test_samples = len(test[0])

	all_labels = [item for sublist in train[1] for item in sublist] + [item for sublist in test[1] for item in sublist]
	Counter(all_labels)
	label_new = dict(Counter(all_labels))

	labels = list(set(all_labels))
	print(labels)
	print(len(labels))

	with open(os.path.join(folder_path, "classes.txt"), "w") as f:
		f.write(str(labels))

	label2id = {label: idx for idx, label in enumerate(labels)}
	id2label = {idx: label for idx, label in enumerate(labels)}
	print(label2id)
	print(id2label)

	# exit("++++++++++++++")
	processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

	train_dataset = SROIEDataset(annotations=train,
	                             image_dir=os.path.join(folder_path
	                                                    , "train/"),
	                             processor=processor)
	test_dataset = SROIEDataset(annotations=test,
	                            image_dir=os.path.join(folder_path, "test/"),
	                            processor=processor)

	encoding = train_dataset[0]
	encoding.keys()
	for k, v in encoding.items():
		print(k, v.shape)

	print(processor.tokenizer.decode(encoding['input_ids']))

	print(train[0][0])
	print(train[1][0])
	print([id2label[label] for label in encoding['labels'].tolist() if label != -100])

	for id, label in zip(encoding['input_ids'][:30], encoding['labels'][:30]):
		print(processor.tokenizer.decode([id]), label.item())

	batch_size = 4
	train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
	test_dataloader = DataLoader(test_dataset, batch_size=4)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	model = AutoModelForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=len(labels))
	print(device)
	model.to(device)
	optimizer = AdamW(model.parameters(), lr=5e-5)
	labels = list(set(all_labels))
	global_step = 0
	num_train_epochs = 40
	steps = []
	losses = []
	# put the model in training mode
	model.train()
	for epoch in range(num_train_epochs):
		print("Epoch:", epoch)
		for batch in tqdm(train_dataloader):
			input_ids = batch['input_ids'].to(device)
			bbox = batch['bbox'].to(device)
			pixel_values = batch['pixel_values'].to(device)
			attention_mask = batch['attention_mask'].to(device)
			labels = batch['labels'].to(device)

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = model(input_ids=input_ids,
			                bbox=bbox,
			                pixel_values=pixel_values,
			                attention_mask=attention_mask,
			                labels=labels)
			# outputs = model(**batch)
			loss = outputs.loss

			# print loss every epoch
			if (global_step + 1) % len(train_dataloader) == 0 or global_step == 0:
				print(f"Loss after {global_step} steps: {loss.item()}")
				steps.append(global_step)
				losses.append(float(loss.item()))
			loss.backward()
			optimizer.step()
			global_step += 1
		if (epoch + 1) % 4 == 0:
			name = "Model_" + str(epoch + 1) + "_epochs"
			os.mkdir(os.path.join(folder_path, name))
			model.save_pretrained(os.path.join(folder_path, name))
			sns.lineplot(x=steps, y=losses)
			plt.show()

	encoding = test_dataset[0]
	processor.tokenizer.decode(encoding['input_ids'])
	ground_truth_labels = [id2label[label] for label in encoding['labels'].squeeze().tolist() if label != -100]
	print(ground_truth_labels)

	for k, v in encoding.items():
		encoding[k] = v.unsqueeze(0).to(device)

	model.eval()

	# forward pass
	outputs = model(input_ids=encoding['input_ids'], attention_mask=encoding['attention_mask'], bbox=encoding['bbox'],
	                pixel_values=encoding['pixel_values'])

	prediction_indices = outputs.logits.argmax(-1).squeeze().tolist()
	print(prediction_indices)

	# prediction_indices = outputs.logits.argmax(-1).squeeze().tolist()
	predictions = [id2label[label] for gt, label in zip(encoding['labels'].squeeze().tolist(), prediction_indices) if
	               gt != -100]
	print(predictions)

	preds_val = None
	out_label_ids = None
	# put model in evaluation mode
	model.eval()
	for batch in tqdm(test_dataloader, desc="Evaluating"):
		with torch.no_grad():
			input_ids = batch['input_ids'].to(device)
			bbox = batch['bbox'].to(device)
			pixel_values = batch['pixel_values'].to(device)
			attention_mask = batch['attention_mask'].to(device)
			labels = batch['labels'].to(device)

			# forward pass
			outputs = model(input_ids=input_ids, bbox=bbox, pixel_values=pixel_values, attention_mask=attention_mask,
			                 labels=labels)

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

	# put model in evaluation mode
	preds_val = None
	out_label_ids = None
	model.eval()
	for batch in tqdm(train_dataloader, desc="Evaluating"):
		with torch.no_grad():
			input_ids = batch['input_ids'].to(device)
			bbox = batch['bbox'].to(device)
			pixel_values = batch['pixel_values'].to(device)
			attention_mask = batch['attention_mask'].to(device)
			labels = batch['labels'].to(device)

			# forward pass
			outputs = model(input_ids=input_ids, bbox=bbox, pixel_values=pixel_values, attention_mask=attention_mask, labels=labels)

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