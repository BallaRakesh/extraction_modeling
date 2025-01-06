import torch
import pandas as pd
import os
from collections import Counter
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from transformers import LayoutLMTokenizer, AdamW
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import LayoutLMForTokenClassification

# Ensure 'results_train' and 'results_test' are defined somewhere in your code
# Assuming they return the results and classification report.
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

# Custom encode function
def encode(example, max_seq_length=512, pad_token_box=None):
    if pad_token_box is None:
        pad_token_box = [0, 0, 0, 0]

    tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
    words = example['words']
    normalized_word_boxes = example['bbox']

    # Ensure words and boxes length match
    assert len(words) == len(normalized_word_boxes), "Mismatch between number of words and bounding boxes."

    # Initialize lists for tokens, boxes, and labels
    tokenized_words = []
    token_boxes = []
    labels = []

    for i,(word, box) in enumerate(zip(words, normalized_word_boxes)):
        # Tokenize the word
        word_tokens = tokenizer.tokenize(word)
        # Extend the tokenized words list
        tokenized_words.extend(word_tokens)
        
        # For each subword, append the bounding box of the original word
        token_boxes.extend([box] * len(word_tokens))

        # Assign the label for the first subword and -100 for subsequent subwords
        token_labels = [example["label"][i]] + [-100] * (len(word_tokens) - 1)  # Only the first subword gets the actual label
        labels.extend(token_labels)

    # Truncate token_boxes, tokenized_words, and labels if necessary
    special_tokens_count = 2  # CLS and SEP tokens
    total_length = len(token_boxes) + special_tokens_count

    if total_length > max_seq_length:
        excess_length = total_length - max_seq_length
        tokenized_words = tokenized_words[:-excess_length]
        token_boxes = token_boxes[:-excess_length]
        labels = labels[:-excess_length]

    # Add bounding boxes for CLS and SEP tokens
    token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]
    labels = [-100] + labels + [-100]

    # Encoding input_ids
    encoding = tokenizer(' '.join(tokenized_words), padding='max_length', truncation=True, return_tensors="pt", max_length=max_seq_length)
    
    input_ids = encoding['input_ids'].squeeze(0)  # Squeeze to remove batch dimension

    # Pad token_boxes to max_seq_length
    padding_length = max_seq_length - len(token_boxes)
    token_boxes += [pad_token_box] * padding_length
    token_boxes = token_boxes[:max_seq_length]  # Ensure the size does not exceed max_seq_length

    # Convert labels to tensors
    labels += [-100] * padding_length  # Use -100 for padding in CrossEntropy loss
    labels = labels[:max_seq_length]  # Ensure labels match max_seq_length
    encoding['bbox'] = torch.tensor(token_boxes)
    encoding['labels'] = torch.tensor(labels)

    # Assertions to verify dimensions
    assert input_ids.shape[0] == max_seq_length, f"input_ids shape {input_ids.shape[0]} does not match max_seq_length {max_seq_length}."
    assert encoding['bbox'].shape[0] == max_seq_length, f"bbox shape {encoding['bbox'].shape[0]} does not match max_seq_length {max_seq_length}."
    assert encoding['labels'].shape[0] == max_seq_length, f"labels shape {encoding['labels'].shape[0]} does not match max_seq_length {max_seq_length}."

    return encoding


# Dataset class
class SROIEDataset(Dataset):
    def __init__(self, annotations, image_dir):
        self.words, self.labels, self.boxes = annotations
        self.image_dir = image_dir
        self.image_file_names = [f for f in os.listdir(image_dir)]

    def __len__(self):
        return len(self.image_file_names)

    def __getitem__(self, idx):
        words = self.words[idx]
        boxes = self.boxes[idx]
        word_labels = self.labels[idx]

        assert len(words) == len(boxes) == len(word_labels)

        word_labels = [label2id[label] for label in word_labels]
        encoded_inputs = encode({'words': words, 'bbox': boxes, 'label': word_labels})
        
        for k, v in encoded_inputs.items():
            encoded_inputs[k] = v.squeeze()  # Ensure tensors are of the correct shape
        assert encoded_inputs.input_ids.shape == torch.Size([512])
        assert encoded_inputs.attention_mask.shape == torch.Size([512])
        assert encoded_inputs.token_type_ids.shape == torch.Size([512])
        assert encoded_inputs.bbox.shape == torch.Size([512, 4])
        assert encoded_inputs.labels.shape == torch.Size([512])
        return encoded_inputs

# Training configuration
if __name__ == "__main__":
    folder_path = "/home/data_science/geo_testing/COO_V3/CI_train"

    train = pd.read_pickle(os.path.join(folder_path, 'train.pkl'))
    test = pd.read_pickle(os.path.join(folder_path, 'test.pkl'))

    all_labels = [item for sublist in train[1] for item in sublist] + [item for sublist in test[1] for item in sublist]
    labels = list(set(all_labels))
    with open(os.path.join(folder_path, "classes.txt"), "w") as f:
        f.write(str(labels))
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for idx, label in enumerate(labels)}
    print("label2id mapping:", label2id)

    train_dataset = SROIEDataset(annotations=train, image_dir=os.path.join(folder_path, "train/"))
    test_dataset = SROIEDataset(annotations=test, image_dir=os.path.join(folder_path, "test/"))
    print(train_dataset[0])
    print(train_dataset[0].keys())
    # exit('++++++++++++++')
    batch_size = 4
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_id = "microsoft/layoutlm-base-uncased"
    model = LayoutLMForTokenClassification.from_pretrained(model_id,
                                                           num_labels=len(labels), 
                                                           label2id=label2id, 
                                                           id2label=id2label)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    num_train_epochs = 40
    steps = []
    losses = []

    model.train()
    global_step = 0
    for epoch in range(num_train_epochs):
        print(f"Epoch: {epoch}")
        for batch in tqdm(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            bbox = batch['bbox'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            if (global_step + 1) % len(train_dataloader) == 0 or global_step == 0:
                print(f"Loss after {global_step} steps: {loss.item()}")
                steps.append(global_step)
                losses.append(float(loss.item()))
            global_step += 1

        if (epoch + 1) % 10 == 0:
            model_dir = os.path.join(folder_path, f"Model_{epoch + 1}_epochs")
            os.makedirs(model_dir, exist_ok=True)
            model.save_pretrained(model_dir)
            sns.lineplot(x=steps, y=losses)
            plt.show()

        # Evaluating model on test dataset
        preds_val, out_label_ids = None, None
        model.eval()
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            with torch.no_grad():
                input_ids = batch['input_ids'].to(device)
                bbox = batch['bbox'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask,
                                token_type_ids=token_type_ids, labels=labels)

                if preds_val is None:
                    preds_val = outputs.logits.detach().cpu().numpy()
                    out_label_ids = labels.detach().cpu().numpy()
                else:
                    preds_val = np.append(preds_val, outputs.logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
        labels = list(set(all_labels))     
        val_result, class_report = results_test(preds_val, out_label_ids, labels)
        print("Overall results:", val_result)
        print(class_report)
        with open(os.path.join(folder_path, "test_report.txt"), 'a') as f:
            f.write(str(val_result))
            f.write('\n')
            f.write(class_report)

    # Evaluate on training set as well
    preds_val, out_label_ids = None, None
    model.eval()
    for batch in tqdm(train_dataloader, desc="Evaluating"):
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            bbox = batch['bbox'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, labels=labels)

            if preds_val is None:
                preds_val = outputs.logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:
                preds_val = np.append(preds_val, outputs.logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
    labels = list(set(all_labels))
    train_result, class_report = results_train(preds_val, out_label_ids, labels)
    print("Overall results (Train):", train_result)
    print(class_report)
    with open(os.path.join(folder_path, "train_report.txt"), 'w') as f:
        f.write(str(train_result))
        f.write('\n')
        f.write(class_report)
