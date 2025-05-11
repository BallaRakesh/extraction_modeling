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
from tqdm.notebook import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
import tensorflow as tf
import logging
from typing import List
import csv
from itertools import zip_longest
from torch.optim.lr_scheduler import StepLR
import optuna
from torch.optim.lr_scheduler import StepLR
from transformers import AdamW
from loguru import logger
import sys
from datetime import datetime
from numba import cuda


from transformers import AdamW, LayoutLMv2FeatureExtractor, LayoutLMv2ForSequenceClassification, LayoutLMv2Processor, \
    LayoutLMv2Tokenizer, LayoutLMv2Config

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
		assert encoded_inputs.token_type_ids.shape == torch.Size([512])
		assert encoded_inputs.bbox.shape == torch.Size([512, 4])
		assert encoded_inputs.image.shape == torch.Size([3, 224, 224])
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




log_dir = "logs"  # Directory to store the TensorBoard logs
train_writer = tf.summary.create_file_writer("logs/train/")
test_writer = tf.summary.create_file_writer("logs/test/")
best_train_test_writer= tf.summary.create_file_writer("logs/best")



current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
logger.add(f"hyper_param_tuning_{current_datetime}.log", rotation="500 MB", retention="7 days")
# logger.add(f"hyper_param_tuning{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.log", rotation="500 MB", retention="7 days")  # Log to a file
logger.add(sys.stdout, format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}", level="INFO")  # Log to the console
itter = 0
training_loss = {}  
validation_loss = {}
folder_path = "/New_Volume/Rakesh/LMV2/IC_MASTER_V3"
train = pd.read_pickle(os.path.join(folder_path, 'train.pkl'))
test = pd.read_pickle(os.path.join(folder_path, 'test.pkl'))
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
#label2id, id2label, labels, all_labels
def objective(trial, itter, training_loss, validation_loss, train_writer, test_writer, best_train_test_writer, train, test, label2id, id2label, labels, all_labels): #hyper
    import torch


    print("version of the cuda")
    print(torch.__version__)
    print(f"cuda available: {torch.cuda.is_available()}")
    # torch.cuda.set_device(0)
    processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")

    train_dataset = SROIEDataset(annotations=train,
                                    image_dir=os.path.join(folder_path
                                                        , "train/"),
                                    processor=processor)
    test_dataset = SROIEDataset(annotations=test,
                                image_dir=os.path.join(folder_path, "test/"),
                                processor=processor)


    # encoding = train_dataset[0]
    # encoding.keys()
    # for k, v in encoding.items():
    #     print(k, v.shape)

    # print(processor.tokenizer.decode(encoding['input_ids']))

    # print(train[0][0])
    # print(train[1][0])
    # print([id2label[label] for label in encoding['labels'].tolist() if label != -100])

    # for id, label in zip(encoding['input_ids'][:30], encoding['labels'][:30]):
    #     print(processor.tokenizer.decode([id]), label.item())

    batch_size = 1
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    o_config = LayoutLMv2Config()
    m_config = LayoutLMv2Config.from_dict(o_config.to_dict())
    print(m_config)

    import torch
    itter+=1
    # Define hyperparameter search spaces
    lr = trial.suggest_loguniform('lr', 5e-5, 1e-3)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)
    step_size = trial.suggest_int('step_size', 1, 10)
    gamma = trial.suggest_float('gamma', 0.1, 0.9)
    hidden_dropout_prob = trial.suggest_float('hidden_dropout_prob', 0.1, 0.3)

    m_config.hidden_dropout_prob = hidden_dropout_prob
    m_config.num_labels = len(labels)
    # Create and configure the optimizer
    print('###########################', itter)
    print('###########################')
    logger.info(f"at the itteration: {itter}; learining rate:{lr}; weight decay: {weight_decay}; step size:{step_size}; gamma value{gamma}; hidden drop prob_value{hidden_dropout_prob}")

    # Create and configure the learning rate scheduler
    # scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Train and evaluate your model with the current hyperparameters
    # Compute a metric (e.g., validation loss) that you want to minimize

    # Return the metric for optimization (e.g., negative validation loss)

    model = LayoutLMv2ForTokenClassification.from_pretrained('microsoft/layoutlmv2-base-uncased', config = m_config) #hyper
    # model = LayoutLMv2ForTokenClassification.from_pretrained('microsoft/layoutlmv2-base-uncased',
    #                                                         num_labels=len(labels))
    print(device)
    model.to(device)

    # optimizer = AdamW(model.parameters(), lr=5e-5)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) #hyper
    # optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=1e-2)
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.5) 
    # scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma) #hyper


    labels = list(set(all_labels))
    global_step = 0
    num_train_epochs = 2
    preds_val = None
    out_label_ids = None
    best_loss=None
    best_precision=None
    best_recall=None
    best_f1=None
    steps = []
    losses = []
    # put the model in training mode
    model.train()
    for epoch in range(num_train_epochs):
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
            current_learning_rate = optimizer.param_groups[0]['lr']
            # print("###############current learinig rate###########", current_learning_rate)
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
            # scheduler.step()  #Update the learning rate using the scheduler###########
        # model.eval()
        # training_loss[f"training_loss_{i}"] = current_training_loss
        training_loss[f'tuning_itteration_{itter}_epoch_{epoch}'] = loss
        training_loss[f'current_learing_rate_tuning_itteration_{itter}_epoch_{epoch}'] = current_learning_rate
        val_loss = 0.0
        preds_val = None
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
        print('################################')
        print('################################')
        print('################################')
        labels = list(set(all_labels))
        print(preds_val)
        print(out_label_ids)
        print(labels)
        val_result, class_report = results_test(preds_val, out_label_ids, labels)

        print(f"precison: {val_result['precision']}")
        print(f"recall: {val_result['recall']}")
        print(f"f1: {val_result['f1']}")



        print('+++++++++++++++++++++++++++++++++++++++++++')
        val_loss= val_loss /len(test_dataloader)
        validation_loss[f'tuning_itteration_{itter}_epoch_{epoch}'] = val_loss
        print(f'final validation loss:{val_loss}')
        # print(val_result)
        with train_writer.as_default():
            tf.summary.scalar(f"train_loss_at_itteration{itter}", loss.detach().cpu(), step=epoch)
        with test_writer.as_default():
            tf.summary.scalar(f"Validation Loss_at_itteration{itter}", val_loss, step=epoch)   
        #precision, recall values need to log
        precision = val_result['precision']
        recall = val_result['recall']
        f1= val_result['f1']
        if  best_loss is None:
            best_loss=val_loss
        if best_precision is None:
            best_precision= precision
            best_recall= recall
            best_f1= f1
        # print(f"best precison: {best_precision}")
        # print(f"best recall: {best_recall}")
        if val_loss< best_loss:
            best_loss = val_loss
            best_precision = precision
            best_recall = recall
            best_f1 = f1
            name = f"Best_Model{itter}"

            if not os.path.exists(os.path.join(folder_path, name)):
                os.mkdir(os.path.join(folder_path, name))
            print(f'Model is {epoch} saving +++++++++++++++++++++++++++++++++')
            with open(os.path.join(folder_path, "model_saving_info.txt"), 'a') as f:
                f.write(f"at itteration {itter} ,Model is saving at {epoch} saving +++++++++++++++++++++++++++++++++\n")
            with best_train_test_writer.as_default():
                tf.summary.scalar(f"Best_train_at_itteration{itter}", best_loss, step=epoch)
            with best_train_test_writer.as_default():
                tf.summary.scalar(f"best_precision_at_itteration{itter}", best_precision, step=epoch) 
            with best_train_test_writer.as_default():
                tf.summary.scalar(f"best_f1_at_itteration{itter}",best_f1, step=epoch) 
            with best_train_test_writer.as_default():
                tf.summary.scalar(f"best_recall_at_itteration{itter}", best_recall, step=epoch) 
            print(f"best Validation Loss: {best_loss}" )
            print("best Precision:", best_precision)
            print("best Recall:", best_recall) 
            print("best f1:", best_f1)
            model.save_pretrained(os.path.join(folder_path, name))


    #give best model path here
    name = f"Best_Model{itter}"
    model_path= f"/New_Volume/Rakesh/LMV2/IC_MASTER_V3/{name}"
    model = LayoutLMv2ForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=os.path.join(model_path, 'pytorch_model.bin'),
            config=os.path.join(model_path, 'config.json'))

    print(training_loss)
    with open(os.path.join(folder_path, "training_loss.txt"), 'w') as f:
        f.write(f"running itteration_{itter}")
        for key, value in training_loss.items():
            f.write(f"{key}: {value}\n") 

    print(validation_loss)
    with open(os.path.join(folder_path, "validation_loss.txt"), 'w') as f:
        f.write(f"running itteration_{itter}")
        for key, value in validation_loss.items():
            f.write(f"{key}: {value}\n")        
            
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
    with open(os.path.join(folder_path, "test_report.txt"), 'w') as f:
        f.write(f"running itteration_{itter}")
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
    
    logger.info(f"the best f1 score {best_loss}")
    var_list = [input_ids, bbox, image, attention_mask, token_type_ids, current_learning_rate, outputs, loss, global_step, preds_val, out_label_ids, val_result, class_report, val_loss, precision, recall, f1, processor, train_dataset, test_dataset, train_dataloader, test_dataloader, device]
    for i in range(len(var_list)):
        if torch.is_tensor(var_list[i]):
            var_list[i] = var_list[i].detach()
    
    del model      #first delete it and then go for the model torch.cuda.empty_cache()
    del optimizer
    del input_ids
    del bbox
    del image
    del attention_mask
    del token_type_ids
    del current_learning_rate
    del outputs
    del loss
    del global_step
    del preds_val
    del out_label_ids
    del val_result
    del class_report
    del val_loss
    del precision
    del recall
    del f1
    del processor
    del train_dataset
    del test_dataset
    del train_dataloader
    del test_dataloader
    del device
    
    gpu_id = 1
    
    torch.cuda.reset_max_memory_allocated()
    
    torch.cuda.empty_cache()
    import gc
    # del variables
    gc.collect()
    torch.cuda.memory_summary(device=None, abbreviated=False)
    # cuda.select_device(0)
    # cuda.close()
    print('woo!,Model Training has done successfully')
    return best_loss



# Create an Optuna study and optimize hyperparameters
# search_space = define_search_space(trial)
# sampler = optuna.samplers.GridSampler(list(search_space),seed=42)
# sampler = optuna.samplers.GridSampler(seed=42)
study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=2)  # You can adjust the number of trials
study.optimize(lambda trial: objective(trial, itter, training_loss, validation_loss, train_writer, test_writer, best_train_test_writer, train, test, label2id, id2label, labels, all_labels), n_trials=50)
best_params = study.best_params
best_lr = best_params['lr']
best_weight_decay = best_params['weight_decay']
best_step_size = best_params['step_size']
best_gamma = best_params['gamma']
best_hidden_dropout_prob= best_params['hidden_dropout_prob']
print("Best Learning Rate:", best_lr)
print("Best Weight Decay:", best_weight_decay)
print("Best Step Size:", best_step_size)
print("Best Gamma:", best_gamma)
print('Best hidden_dropout_prob:', best_hidden_dropout_prob)
