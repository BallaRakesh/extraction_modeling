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
import tensorflow as tf

warnings.filterwarnings("ignore")
from seqeval.metrics import (
	classification_report,
	f1_score,
	precision_score,
	recall_score,
  accuracy_score)

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

# t_start = datetime.now()
# cpu_utilization_start = psutil.cpu_percent()
# before_memory = process_memory.memory_info().rss
log_dir = "logs"  # Directory to store the TensorBoard logs
train_writer = tf.summary.create_file_writer("logs/train/")
test_writer = tf.summary.create_file_writer("logs/test/")
best_train_test_writer= tf.summary.create_file_writer("logs/best")


print("version of the cuda")
print(torch.__version__)
print(f"cuda available: {torch.cuda.is_available()}")

folder_path = "/content/data/training_on_colab"
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
f.close()

label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for idx, label in enumerate(labels)}
print(label2id)
print(id2label)

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
num_train_epochs = 4
val_loss = 0.0
preds_val = None
out_label_ids = None
best_loss=None
best_precision=None
best_recall=None
best_f1=None
best_val_loss = None
steps = []
losses = []
my_dict = {}
valida_loss = {}
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
        loss = outputs.loss
        my_dict[epoch] = loss
        with train_writer.as_default():
            tf.summary.scalar("Train Loss", loss.item(), step=epoch)
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
        pixel_values = batch['pixel_values'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids,
                        bbox=bbox,
                        pixel_values=pixel_values,
                        attention_mask=attention_mask,

                        labels=labels)
        val_loss= outputs.loss
        validation_loss = outputs.loss
        valida_loss[epoch] = validation_loss
        print(f'validataion loss: {val_loss}')
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
    val_loss= val_loss /len(test_dataloader)   # final validation loss
    print(f'final validation loss:{val_loss}')
    with test_writer.as_default():
        tf.summary.scalar("Validation Loss", val_loss.cpu(), step=epoch)
    
    # print(val_result)
    precision = val_result['precision']
    recall = val_result['recall']
    f1= val_result['f1']
    print(f'The train loss : {loss}')
    if  best_loss is None:
      best_loss=loss
    if best_val_loss is None:
        best_val_loss = validation_loss  
    if best_precision is None:
      best_precision= precision
      best_recall= recall
      best_f1= f1
    # print(f"best precison: {best_precision}")
    # print(f"best recall: {best_recall}")
    if loss< best_loss:
        best_loss = loss
        best_precision = precision
        best_recall = recall

        with best_train_test_writer.as_default():
            tf.summary.scalar("Best train loss ", best_loss.detach().cpu(), step=epoch)
            tf.summary.scalar("Best val loss", val_loss.cpu(), step=epoch)
        print(f"best Validation Loss: {best_loss}" )
        print("best Precision:", best_precision)
        print("best Recall:", best_recall)
        
        
    if validation_loss < best_val_loss:
        best_val_loss = validation_loss
        name = "Best_Model"
        if not os.path.exists(os.path.join(folder_path, name)):
          os.mkdir(os.path.join(folder_path, name))
        print(f'Model is {epoch} saving +++++++++++++++++++++++++++++++++')
        model.save_pretrained(os.path.join(folder_path, name))
                 
print(my_dict)
with open(os.path.join(folder_path, "model_loss.txt"), 'w') as f:
    for key, value in my_dict.items():
        f.write(f"{key}: {value}\n")        
        
print(valida_loss)
with open(os.path.join(folder_path, "validation_loss.txt"), 'w') as f:
    for key, value in valida_loss.items():
        f.write(f"{key}: {value}\n")                    

model_path= "/content/data/training_on_colab"    #provide the best_model path
model = AutoModelForTokenClassification.from_pretrained(model_path)

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
f.close()

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
        outputs = model(input_ids=input_ids, bbox=bbox, pixel_values=pixel_values, attention_mask=attention_mask,
                         labels=labels)

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