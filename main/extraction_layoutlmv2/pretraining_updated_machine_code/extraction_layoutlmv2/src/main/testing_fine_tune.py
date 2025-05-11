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
from seqeval.metrics import (
	classification_report,
	f1_score,
	precision_score,
	recall_score,
accuracy_score)
#################################################################
# from config.prod_mapping import product_code_map, document_code_map



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
    label_map = dict(enumerate(labels))
    print(label_map)
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

from config.prod_mapping import product_code_map, document_code_map



# product config
product_config = ConfigParser()
product_config.read("/home/ntlpt19/Downloads/Evaluation_Data/updated_code/src/main/extraction/config/config.ini")

prod_code = product_code_map[product_config["Product"]["code"]] #tp
doc_code = product_config["Product"]["document_code"] #pi
if '[' in doc_code:
	doc_elements = doc_code[1:-1].split(', ')
	# Convert elements to a Python list
	doc_code_list = [element.strip() for element in doc_elements]
print(doc_code_list)
print(prod_code)
# data folder path
product_wise_folder = ConfigParser()
product_wise_folder.read("/home/ntlpt19/Downloads/Evaluation_Data/updated_code/src/main/extraction/config/prod.ini")
# doc_code = 'lc'
for doc_code_ in doc_code_list:
    doc_code = document_code_map[doc_code_]
    folder_path = product_wise_folder[prod_code][doc_code]
    # folder_path = '/New_Volume/Rakesh/DATA_LMV2/LMV2_BASE/AWB'
    print("==================Trade Finance Solutions===================")
    # print("Product Code: {product_code}")
    # print("Documenry Code: {doc_code}")
    print(f"folder_path: {folder_path}")
    set_basic_config_for_logging(folder_path, filename="train_words_count")
    logger = get_logger_object_and_setting_the_loglevel()
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

    doc_code = 'lc'
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
    print(all_labels)
    Counter(all_labels)
    label_new = dict(Counter(all_labels))
    print(label_new)
    labels = list(set(all_labels))
    print(labels)
    print(len(labels))
    

    # fine_tune_labels = ['S-vessel_name', 'S-usance_tenor', 'S-awb_number', 'S-bill_of_lading_no', 'S-consignor_name', 'S-transaction_amonunt_value', 'S-transaction_currency', 'S-beneficiary_bank_identifier', 'S-beneficiary_bank_name', 'S-indicator_date', 'S-signature_of_the_issuer', 'S-stamp', 'S-beneficiary_account_number', 'S-dimension', 'S-invoice_discount_ccy', 'S-shipper_address', 'S-invoice_tax_amount', 'S-page_no', 'S-rate_per_unit', 'S-notify_party_country', 'S-signature', 'S-original_or_copy', 'S-pre-carriage-by', 'S-incoterm', 'S-beneficiary_country', 'S-delivery_terms', 'S-lc_date', 'S-total_quantity_of_goods', 'S-country_of_final_destination', 'O', 'S-consignee_address', 'S-invoice_due_date', 'S-invoice_tax_ccy', 'S-consignor_address', 'S-beneficiary_address', 'S-cosignee_name', 'S-remitter_drawee_applicant_importer_buyer_name', 'S-net_weight', 'S-tenor_type', 'S-invoice_amount', 'S-consignor_country', 'S-awb_date', 'S-date_of_invoice', 'S-invoice_currency', 'S-invoice_no', 'S-transaction_date', 'S-beneficiary_iban_number', 'S-beneficiary_drawer_exporter_seller_supplier_name', 'S-gross_weight', 'S-beneficiary_drawer_exporter_seller_supplier_tin', 'S-hs_code_no', 'S-invoice_amount_in_words', 'S-notify_party_name', 'S-notify_party_address', 'S-invoice_discount_amount', 'S-declaration_by_issuer', 'S-beneficiary_bank_country', 'S-port_of_loading', 'S-goods_description', 'S-tenor_indicator', 'S-shipper_name', 'S-indicator_type', 'S-country_of_origin_of_goods', 'S-lc_ref_no', 'S-vessel_flight_no', 'S-bill_of_lading_date', 'S-port_of_discharge', 'S-remitter_address']
    fine_tune_labels =  ['S-tenor_type', 'S-invoice_due_date', 'S-diclaration_by', 'S-invoice_amount', 'S-original_or_copy', 'S-boe_currency', 'S-lc_ref_no', 'S-boe_amount', 'S-indicator_date', 'S-tenore_details', 'S-lc_date', 'S-issuing_bank', 'S-drawer_name', 'S-stamp', 'S-drawer_bank_address', 'S-drawer_address', 'S-drawee_name', 'S-country_of_origin', 'S-tenor_indicator', 'S-bill_exchange_no', 'S-drawee_bank_address', 'S-issuing_bank_address', 'S-bill_exchange_date', 'S-goods_discription', 'S-drawer_bank_name', 'S-indicator_type', 'S-drawee_bank_name', 'S-usance_tenor', 'S-signature', 'S-invoice_no', 'S-original_number', 'S-drawee_address', 'S-invoice_date', 'S-issue_place', 'S-invoice_currency', 'O', 'S-amount_in_words']
    with open(os.path.join(folder_path, "classes.txt"), "w") as f:
        f.write(str(labels))
    f.close()
    
    #same count in labels and classes (+1 for others)
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for idx, label in enumerate(labels)}
    print(label2id)
    print(id2label)
    processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", 
                                                    revision="no_ocr")

    train_dataset = SROIEDataset(annotations=train,
                                    image_dir=os.path.join(folder_path
                                                        , "train/"),
                                    processor=processor)
    test_dataset = SROIEDataset(annotations=test,
                                image_dir=os.path.join(folder_path, "test/"),
                                processor=processor)

    # token_ids  = ['token_ids']
    # encode_ids = ['tokenized_text']
    # enco_lab_ids = ['tokenized_text_label_ids']
    # zip_id2labels = ['tokenized_text_label']

    # for i in range(len(train_dataset)):
    #     encoding = train_dataset[i]
    #     print(len(encoding['labels']),encoding['labels'])
    #     #exit()
    #     encoding_ids = []
    #     enco_label_ids = []
    #     tokens = []
    #     for id, label in zip(encoding['input_ids'], encoding['labels']):
    #         tokens.append(id.item())
    #         encoding_ids.append(processor.tokenizer.decode(id.item()))
    #         enco_label_ids.append(label.item())
        
    #     #print(enco_lab_ids)
    #     ids2lab = []
    #     for label in enco_label_ids:
    #         if label != -100:
    #             ids2lab.append(id2label[label])
    #         else:
    #             ids2lab.append('O')

    #     encode_ids = encode_ids + encoding_ids
    #     enco_lab_ids = enco_lab_ids + enco_label_ids  
        
    #     token_ids = token_ids+tokens
    #     zip_id2labels = zip_id2labels+ids2lab			
    # data = [train_text, train_label, train_bb, token_ids, encode_ids, enco_lab_ids, zip_id2labels]
    # write_to_csv_train(data, csv_file_path)


    # #creating the log file for having the count of words
    # sample = 0
    # for cou1, cou2 in zip(range(len(train[0])), range(len(train_dataset))):
    #     sample+=1
    #     word_count = 0
    #     padding_count = 0
    #     print('sample =>', sample)
    #     print("============")
    #     print('train', len(train[0][cou1]))
    #     # print(len(train_dataset[cou2]))
    #     encoding = train_dataset[cou2]
    #     print('embidding', len(encoding['input_ids']))
    #     for id in encoding['input_ids']:
    #         if id.item()==0:
    #             padding_count+=1
    #         else:
    #             word_count+=1
            
    #     print('word_coun', word_count)
    #     print('padding_count', padding_count)
    #     print("***********")
    #     logger.info(f"sample: {sample}; train_chunk_count: {len(train[0][cou1])}; embedding: {len(encoding['input_ids'])}; word_count: {word_count}; padding_count: {padding_count}")

    with open(os.path.join(folder_path, "label.txt"), "r") as file:
        class_names: List = file.readlines()
        class_names = list(map(lambda x: x.strip(), class_names))
        dict_mapping = dict(enumerate(class_names))
    file.close()
    logger.info(f"actual_label_length: {len(dict_mapping)}; train_gen_classes:{len(id2label)}")


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
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model = LayoutLMv2ForTokenClassification.from_pretrained('microsoft/layoutlmv2-base-uncased',
    #                                                             num_labels=len(labels)) #37

    # model_path = '/home/ntlpt19/Downloads/Final_Delivery_Training_itter_2/testing_finetuning/PI/Best_Model_boe'
    model_path = '/home/ntlpt19/Downloads/Final_Delivery_Training_itter_2/testing_finetuning/PI/Best_Model_boe'
     
    '''custom_weights = torch.load(os.path.join(model_path, 'pytorch_model.bin'))
    model = LayoutLMv2ForTokenClassification.from_pretrained(pretrained_model_name_or_path=os.path.join(model_path, 'pytorch_model.bin'),state_dict=custom_weights, 
        config=os.path.join(model_path, 'config.json'), num_labels=len(labels), ignore_mismatched_sizes = True)'''
    
    #,  from_tf=True)
    
    
    
    model = LayoutLMv2ForTokenClassification.from_pretrained(
        pretrained_model_name_or_path=os.path.join(model_path, 'pytorch_model.bin'),
        config=os.path.join(model_path, 'config.json'), num_labels=len(labels), ignore_mismatched_sizes = True)
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
    for epoch in range(num_train_epochs):
        print("Epoch:", epoch)
        for batch in tqdm(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            bbox = batch['bbox'].to(device)
            image = batch['image'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            print(labels)
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
        training_loss[epoch] = loss 
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

        labels = list(set(all_labels))
        # val_result, class_report = results_test(preds_val, out_label_ids, fine_tune_labels)
        val_result, class_report = results_test(preds_val, out_label_ids, labels)

        print(f"precison: {val_result['precision']}")
        print(f"recall: {val_result['recall']}")
        print(f"f1: {val_result['f1']}")
        print('+++++++++++++++++++++++++++++++++++++++++++')
        val_loss = val_loss /len(test_dataloader)
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
        # model.save_pretrained(os.path.join(folder_path, 'saving_model'), state_dict = model.state_dict())
        file_path = os.path.join(folder_path, 'saving_model.pt')
        torch.save(model.state_dict(), file_path)
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
        # print(model.state_dict())

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


    model = LayoutLMv2ForTokenClassification.from_pretrained(
    		pretrained_model_name_or_path=os.path.join(model_path, 'pytorch_model.bin'),
    		config=os.path.join(model_path, 'config.json'))


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
    # val_result, class_report = results_test(preds_val, out_label_ids, fine_tune_labels)
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
    # val_result, class_report = results_train(preds_val, out_label_ids, fine_tune_labels)
    val_result, class_report = results_train(preds_val, out_label_ids, labels)
    train_result = val_result
    train_all = class_report
    print("Overall results:", val_result)
    print(class_report) 
    print('woo!,Model Training has done successfully')