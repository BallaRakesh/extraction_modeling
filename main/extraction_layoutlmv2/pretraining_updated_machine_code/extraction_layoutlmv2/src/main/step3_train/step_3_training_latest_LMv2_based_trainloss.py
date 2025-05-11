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

"""
Review internal_data
Aug 10 , 2023 => by Tarun Sharma
"""

from configparser import ConfigParser
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
import numpy as np
import warnings
import tensorflow as tf
import training_utility as tu

warnings.filterwarnings("ignore")
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score, )


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

        # step1: convert into RGB
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


if __name__ == "__main__":
    # Step1: Reading configurations from ini files
    parser = ConfigParser()
    conf_folder_path: str = "/media/tarun/D1/Trade-Finance/src/main/extraction/internal_data"
    config_file_name: str = "config.ini"

    if os.path.exists(f"{conf_folder_path}/{config_file_name}"):
        parser.read(f"{conf_folder_path}/{config_file_name}")

    gv_key = parser['OCR']['gv_key']
    root_folder_path = str(parser['PATHS']['root_folder'])
    debug_mode = str(parser["PARAMS"]["debug_mode"])
    folder_path = str(parser['PATHS']['folder_path'])

    # Step 2: Setting up logger
    # setting the log folder and file
    tu.set_basic_config_for_logging(folder_path=conf_folder_path, filename="data_preparation")
    # setting the logger object and log level
    logger = tu.get_logger_object_and_setting_the_loglevel()

    log_dir = "logs"  # Directory to store the TensorBoard logs
    train_writer = tf.summary.create_file_writer("logs/train/")
    test_writer = tf.summary.create_file_writer("logs/test/")
    best_train_test_writer = tf.summary.create_file_writer("logs/best/")
    print("version of the cuda")
    print(torch.__version__)
    print(f"cuda available: {torch.cuda.is_available()}")

    folder_path = "/New_Volume/handover_doc_extract/model_training" \
                  "/jul_13_certificate_of_origin_training_on_best_model_code/training_using_new_code/internal_data"

    # Step1: reading train and test data
    try:
        train = pd.read_pickle(os.path.join(folder_path, 'train.pkl'))
        test = pd.read_pickle(os.path.join(folder_path, 'test.pkl'))
    except Exception as e:
        exit(" code exited as pickle is not readed properly")

    train_samples: int = len(train[0])
    test_samples: int = len(test[0])

    logger.info(f"length of the training sample is {train_samples}")
    logger.info(f"length of the test sample is {test_samples}")

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

    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for idx, label in enumerate(labels)}
    print(label2id)
    print(id2label)

    # ModelName: microsoft/layoutlmv2-base-uncased
    processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")

    train_dataset = SROIEDataset(annotations=train,
                                 image_dir=os.path.join(folder_path
                                                        , "train/"),
                                 processor=processor)
    test_dataset = SROIEDataset(annotations=test,
                                image_dir=os.path.join(folder_path, "test/"),
                                processor=processor)

    encoding = train_dataset[0]
    print(encoding.keys())

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

    model = LayoutLMv2ForTokenClassification.from_pretrained('microsoft/layoutlmv2-base-uncased',
                                                             num_labels=len(labels))

    print(device)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    labels = list(set(all_labels))
    global_step = 0
    num_train_epochs = 50
    preds_val = None
    out_label_ids = None
    best_loss = None
    best_precision = None
    best_recall = None
    best_f1 = None
    steps = []
    losses = []
    training_loss = {}
    validation_loss = {}
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
                val_los = outputs.loss
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

        print(f"validation precision: {val_result['precision']}")
        print(f"validation recall: {val_result['recall']}")
        print(f"validation f1: {val_result['f1']}")

        logger.info(f"validation precison: {val_result['precision']}")
        logger.info(f"validation recall: {val_result['recall']}")
        logger.info(f"validation f1: {val_result['f1']}")

        print('+++++++++++++++++++++++++++++++++++++++++++')
        val_loss = val_loss / len(test_dataloader)
        validation_loss[epoch] = val_loss
        print(f'final validation loss:{val_loss}')
        # print(val_result)
        with train_writer.as_default():
            tf.summary.scalar("train loss ", loss.detach().cpu(), step=epoch)
        with test_writer.as_default():
            tf.summary.scalar("Validation Loss", val_loss, step=epoch)
        precision = val_result['precision']
        recall = val_result['recall']
        f1 = val_result['f1']
        if best_loss is None:
            best_loss = loss
        if best_precision is None:
            best_precision = precision
            best_recall = recall
            best_f1 = f1
        # print(f"best precison: {best_precision}")
        # print(f"best recall: {best_recall}")
        if loss < best_loss and f1 > best_f1 and recall > best_recall:
            best_loss = loss
            best_precision = precision
            best_recall = recall
            name = "Best_Model"

            if not os.path.exists(os.path.join(folder_path, name)):
                os.mkdir(os.path.join(folder_path, name))
            print(f'Model is {epoch} saving +++++++++++++++++++++++++++++++++')
            with best_train_test_writer.as_default():
                tf.summary.scalar("Best train loss ", best_loss.detach().cpu(), step=epoch)
            print(f"best Validation Loss: {best_loss}")
            print("best Precision:", best_precision)
            print("best Recall:", best_recall)
            model.save_pretrained(os.path.join(folder_path, name))

    # give best model path here
    model_path = "/New_Volume/handover_doc_extract/model_training/jul_13_certificate_of_origin_training_on_best_model_code/training_using_new_code/internal_data/Best_Model"
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
