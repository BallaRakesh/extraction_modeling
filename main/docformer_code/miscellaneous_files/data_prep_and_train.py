# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.
import kagglehub
pdavpoojan_the_rvlcdip_dataset_test_path = kagglehub.dataset_download('pdavpoojan/the-rvlcdip-dataset-test')

print('Data source import complete.')
## Importing the libraries

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

import torch.nn.functional as F
import torchvision.models as models

## Adding the path of docformer to system path
import sys
# sys.path.append('./docformer/src/docformer/')
sys.path.append('./src/docformer')

## Importing the functions from the DocFormer Repo
# from dataset import create_features
from modeling import DocFormerEncoder,ResNetFeatureExtractor,DocFormerEmbeddings,LanguageFeatureExtractor
from transformers import BertTokenizerFast


## Hyperparameters

seed = 42
target_size = (500, 384) # (1280, 960)

## Setting some hyperparameters

device = 'cuda' if torch.cuda.is_available() else 'cpu'

## One can change this configuration and try out new combination
config = {
  "coordinate_size": 96,              ## (768/8), 8 for each of the 8 coordinates of x, y
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "image_feature_pool_shape": [7, 7, 256],
  "intermediate_ff_size_factor": 4,
  "max_2d_position_embeddings": 1024,
  "max_position_embeddings": 128,
  "max_relative_positions": 8,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "shape_size": 96,
  "vocab_size": 30522,
  "layer_norm_eps": 1e-12,
}


from datasets import load_dataset

import os
from typing import List, Dict, Union
import numpy as np
from PIL import Image
from datasets import Dataset, DatasetDict
import datasets
import os
from PIL import Image


def convert_image_mode(img: Image.Image, target_mode: str = 'L') -> Image.Image:
    """
    Convert image to the specified mode, preserving as much original information as possible.
    
    Args:
        img (PIL.Image.Image): Input image
        target_mode (str): Desired image mode (default 'L' for grayscale)
    
    Returns:
        PIL.Image.Image: Converted image
    """
    # Map of conversion strategies
    conversion_strategies = {
        '1': lambda x: x.convert('L'),    # 1-bit pixels (black and white)
        'L': lambda x: x,                 # Grayscale 
        'P': lambda x: x.convert('L'),    # Palette-mapped 
        'RGB': lambda x: x.convert('L'),  # Color to grayscale
        'RGBA': lambda x: x.convert('L'), # Color with alpha to grayscale
    }
    
    # Get the current mode
    current_mode = img.mode
    
    # Choose conversion strategy
    if current_mode in conversion_strategies:
        return conversion_strategies[current_mode](img)
    
    # Fallback to direct conversion
    return img.convert(target_mode)

def create_custom_dataset(root_folder: str) -> Dataset:
    """
    Create a custom dataset from a root folder with class-specific subfolders.
    
    Args:
        root_folder (str): Path to the root folder containing class subfolders
    
    Returns:
        Dataset: A single dataset with all images
    """
    # Collect images and labels
    images = []
    labels = []
    class_names = []
    image_names = []
    # Iterate through class folders
    for class_name in sorted(os.listdir(root_folder)):
        class_path = os.path.join(root_folder, class_name)
        
        # Skip if not a directory
        if not os.path.isdir(class_path):
            continue
        
        # Collect images for this class
        class_images = []
        image_name = []
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            print(img_name)
            # Check if it's an image file
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                # Open and convert image mode
                img = Image.open(img_path)
                img = convert_image_mode(img, target_mode='L')  # Ensure grayscale mode
                img.load()  # Ensure the image is valid
                print(img)
                class_images.append(img)
                image_name.append(img_name)
        # Add images and their corresponding labels
        images.extend(class_images)
        image_names.extend(image_name)
        labels.extend([len(class_names)] * len(class_images))
        class_names.append(class_name)

    # Create dataset
    dataset = Dataset.from_dict({
        'image': images,
        'label': labels,
        'image_names':image_names
    })

    # Set features with class names
    dataset = dataset.cast_column(
        'label', 
        datasets.ClassLabel(names=class_names)
    )

    return dataset




# dataset = load_dataset("nielsr/rvl_cdip_10_examples_per_class")

##############################################
##############################################
##############################################

train_root_folder = "/home/data_science/geo_testing/Classification_root/BILLS"
test_root_folder = "/home/data_science/geo_testing/Classification_root/BILLS"

train_dataset_main = create_custom_dataset(train_root_folder)
test_dataset_main = create_custom_dataset(test_root_folder)

# If you want to combine them into a DatasetDict
dataset = datasets.DatasetDict({
    'train': train_dataset_main,
    'test': test_dataset_main
})


print(dataset)
print(dataset['train'].features)
print(dataset['train']['image'][0])
print(dataset['train']['label'][0])
print(dataset['train'].features['label'].names)
print('########$$$$$$$$#########')
# exit('OKOOKOK')

print(dataset['train'].features['label'].names)
id2label = {id: label for id, label in enumerate(dataset['train'].features['label'].names)}
print(id2label)
from tqdm.auto import tqdm


dataset_train = dataset['train']

#DEBUG
# =======++++++++++++++++++++++++=======
# =======++++++++++++++++++++++++=======
# # Ensure a directory exists to save the images
# output_dir = "/home/data_science/geo_testing/docformer/testing_docfrm"

# # Iterate through the dataset and save each image
# for idx, item in enumerate(dataset_train):
#     image = item['image']  # Access the PIL image object
#     print('LABEL:', item['label'])  # Access the PIL image object
#     image_path = os.path.join(output_dir, f"image_{idx}.png")  # Name the file
#     image.save(image_path)  # Save the image
#     print(f"Image saved at: {image_path}")
#     exit('OLOLOL')

dataset_test = dataset['test']
print(dataset_test)
###############################
###############################
###############################
###############################
import pandas as pd
train_df = pd.DataFrame(dataset_train)
valid_df = pd.DataFrame(dataset_test)

train_df = train_df.rename(columns={"image":"img"})
valid_df = valid_df.rename(columns={"image":"img"})
print(valid_df.head())
train_df = train_df.reset_index().drop(columns = ['index'], axis = 1)
valid_df = valid_df.reset_index().drop(columns = ['index'], axis = 1)

print((valid_df['img'][0]).size)

def get_topleft_bottomright_coordinates(df_row):
    left, top, width, height = df_row["left"], df_row["top"], df_row["width"], df_row["height"]
    return [left, top, left + width, top + height]


def get_ocr_tesseract(image_):
	"""
    Performs OCR (Optical Character Recognition) using Tesseract OCR engine.

    Args:
        image_path (str): Path to the image file.

    Returns:
        tuple: A tuple containing word coordinates (list of dictionaries) and all the extracted text (str).

    """
	# img=None
	word_coordinates, all_text = [],""
	print("called Image OCR...", end="")
	try:
		# img = Image.open(image_path)
		d = pytesseract.image_to_data(image_, output_type=pytesseract.Output.DICT)
		all_text = pytesseract.image_to_string(image_)
		for i in range(len(d['text'])):
			word = d['text'][i]
			conf = float(d['conf'][i])
			if conf > 0:
				x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
				word_coordinates.append({
					"word": word,
					"confidence": conf,
					"left": x,
					"top": y,
					"width": w,
					"height": h,
					"x1": x,
					"y1": y,
					"x2": x + w,
					"y2": y + h
				})
	except Exception as e:
		print(f"exception: {e}")	
	# finally:
	# 	if hasattr(img,"close"):
	# 		img.close()
	return word_coordinates, all_text

custom_pytesseract = True
def apply_ocr(image):
    """
    Returns words and its bounding boxes from an image
    """
#     image = Image.open(image_fp)
    width, height = image.size
    if custom_pytesseract:
        words = []
        actual_bboxes = []
        word_coordinates, _ = get_ocr_tesseract(image)
        print(word_coordinates)
        for wrd_ in word_coordinates:
            words.append(wrd_['word'])
            actual_bboxes.append([wrd_['x1'], wrd_['y1'], wrd_['x2'], wrd_['y2']])
        # print(words)
        # print(actual_bboxes)
    else:
        ocr_df = pytesseract.image_to_data(image, output_type="data.frame")
        # print(ocr_df)
        ocr_df = ocr_df.dropna().reset_index(drop=True)
        float_cols = ocr_df.select_dtypes("float").columns
        ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
        ocr_df = ocr_df.replace(r"^\s*$", np.nan, regex=True)
        ocr_df = ocr_df.dropna().reset_index(drop=True)
        words = list(ocr_df.text.apply(lambda x: str(x).strip()))
        print(words)
        actual_bboxes = ocr_df.apply(get_topleft_bottomright_coordinates, axis=1).values.tolist()
        print(actual_bboxes)
    
    # add as extra columns
    assert len(words) == len(actual_bboxes)
    return {"words": words, "bbox": actual_bboxes}


def normalize_box(box, width, height, size=1000):
    """
    Takes a bounding box and normalizes it to a thousand pixels. If you notice it is
    just like calculating percentage except takes 1000 instead of 100.
    """
    return [
        int(size * (box[0] / width)),
        int(size * (box[1] / height)),
        int(size * (box[2] / width)),
        int(size * (box[3] / height)),
    ]

def get_tokens_with_boxes(unnormalized_word_boxes, pad_token_box, word_ids,max_seq_len = 512):

    # assert len(unnormalized_word_boxes) == len(word_ids), this should not be applied, since word_ids may have higher
    # length and the bbox corresponding to them may not exist

    unnormalized_token_boxes = []

    for i, word_idx in enumerate(word_ids):
        if word_idx is None:
            break
        unnormalized_token_boxes.append(unnormalized_word_boxes[word_idx])

    # all remaining are padding tokens so why add them in a loop one by one
    num_pad_tokens = len(word_ids) - i - 1
    if num_pad_tokens > 0:
        unnormalized_token_boxes.extend([pad_token_box] * num_pad_tokens)


    if len(unnormalized_token_boxes)<max_seq_len:
        unnormalized_token_boxes.extend([pad_token_box] * (max_seq_len-len(unnormalized_token_boxes)))

    return unnormalized_token_boxes[:max_seq_len] ## maybe in case the length is higher than max_seq_len



def resize_align_bbox(bbox, orig_w, orig_h, target_w, target_h):
    x_scale = target_w / orig_w
    y_scale = target_h / orig_h
    orig_left, orig_top, orig_right, orig_bottom = bbox
    x = int(np.round(orig_left * x_scale))
    y = int(np.round(orig_top * y_scale))
    xmax = int(np.round(orig_right * x_scale))
    ymax = int(np.round(orig_bottom * y_scale))
    return [x, y, xmax, ymax]

def get_centroid(actual_bbox):
    centroid = []
    for i in actual_bbox:
        width = i[2] - i[0]
        height = i[3] - i[1]
        centroid.append([i[0] + width / 2, i[1] + height / 2])
    return centroid


def get_pad_token_id_start_index(words, encoding, tokenizer):
#     assert len(words) < len(encoding["input_ids"])  This condition, was creating errors on some sample images
    for idx in range(len(encoding["input_ids"])):
        if encoding["input_ids"][idx] == tokenizer.pad_token_id:
            break
    return idx


def get_relative_distance(bboxes, centroids, pad_tokens_start_idx):

    a_rel_x = []
    a_rel_y = []

    for i in range(0, len(bboxes)-1):
        if i >= pad_tokens_start_idx:
            a_rel_x.append([0] * 8)
            a_rel_y.append([0] * 8)
            continue

        curr = bboxes[i]
        next = bboxes[i+1]

        a_rel_x.append(
            [
                curr[0],  # top left x
                curr[2],  # bottom right x
                curr[2] - curr[0],  # width
                next[0] - curr[0],  # diff top left x
                next[0] - curr[0],  # diff bottom left x
                next[2] - curr[2],  # diff top right x
                next[2] - curr[2],  # diff bottom right x
                centroids[i+1][0] - centroids[i][0],
            ]
        )

        a_rel_y.append(
            [
                curr[1],  # top left y
                curr[3],  # bottom right y
                curr[3] - curr[1],  # height
                next[1] - curr[1],  # diff top left y
                next[3] - curr[3],  # diff bottom left y
                next[1] - curr[1],  # diff top right y
                next[3] - curr[3],  # diff bottom right y
                centroids[i+1][1] - centroids[i][1],
            ]
        )

    # For the last word

    a_rel_x.append([0]*8)
    a_rel_y.append([0]*8)


    return a_rel_x, a_rel_y



# -*- coding: utf-8 -*-
import os
import pickle
from functools import lru_cache
import pytesseract
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import ToTensor

PAD_TOKEN_BOX = [0, 0, 0, 0]
GRID_SIZE = 1000

def create_features(
        image,
        tokenizer,
        add_batch_dim=False,
        target_size=(500, 384),  # This was the resolution used by the authors
        max_seq_length=512,
        path_to_save=None,
        save_to_disk=False,
        apply_mask_for_mlm=False,
        extras_for_debugging=False,
        use_ocr = True,
        bounding_box = None,
        words = None
):

    # step 1: read original image and extract OCR entries
    try:
#         original_image = Image.open(image).convert("RGB")
        original_image = image.convert("RGB")
    except:
        original_image = Image.new(mode = "RGB", size = ((500, 500)), color = (255, 255, 255))
    if (use_ocr == False) and (bounding_box == None or words == None):
        raise Exception('Please provide the bounding box and words or pass the argument "use_ocr" = True')

    if use_ocr == True:
      entries = apply_ocr(image)
      bounding_box = entries["bbox"]
      words = entries["words"]

    CLS_TOKEN_BOX = [0, 0, *original_image.size]    # Can be variable, but as per the paper, they have mentioned that it covers the whole image
    # step 2: resize image
    resized_image = original_image.resize(target_size)
#     display(resized_image)

    # step 3: normalize image to a grid of 1000 x 1000 (to avoid the problem of differently sized images)
    width, height = original_image.size
    normalized_word_boxes = [
        normalize_box(bbox, width, height, GRID_SIZE) for bbox in bounding_box
    ]
    assert len(words) == len(normalized_word_boxes), "Length of words != Length of normalized words"

    # step 4: tokenize words and get their bounding boxes (one word may split into multiple tokens)
    encoding = tokenizer(words,
                         padding="max_length",
                         max_length=max_seq_length,
                         is_split_into_words=True,
                         truncation=True,
                         add_special_tokens=False)

    unnormalized_token_boxes = get_tokens_with_boxes(bounding_box,
                                                                  PAD_TOKEN_BOX,
                                                                  encoding.word_ids())

    # step 5: add special tokens and truncate seq. to maximum length
    unnormalized_token_boxes = [CLS_TOKEN_BOX] + unnormalized_token_boxes[:-1]
    # add CLS token manually to avoid autom. addition of SEP too (as in the paper)
    encoding["input_ids"] = [tokenizer.cls_token_id] + encoding["input_ids"][:-1]

    # step 6: Add bounding boxes to the encoding dict
    encoding["unnormalized_token_boxes"] = unnormalized_token_boxes

    # step 7: apply mask for the sake of pre-training
    if apply_mask_for_mlm:
        encoding["mlm_labels"] = encoding["input_ids"]
        encoding["input_ids"] = apply_mask(encoding["input_ids"], tokenizer)
        assert len(encoding["mlm_labels"]) == max_seq_length, "Length of mlm_labels != Length of max_seq_length"

    assert len(encoding["input_ids"]) == max_seq_length, "Length of input_ids != Length of max_seq_length"
    assert len(encoding["attention_mask"]) == max_seq_length, "Length of attention mask != Length of max_seq_length"
    assert len(encoding["token_type_ids"]) == max_seq_length, "Length of token type ids != Length of max_seq_length"

    # step 8: normalize the image
    encoding["resized_scaled_img"] = ToTensor()(resized_image)

    # step 9: apply mask for the sake of pre-training
    if apply_mask_for_mlm:
        encoding["mlm_labels"] = encoding["input_ids"]
        encoding["input_ids"] = apply_mask(encoding["input_ids"], tokenizer)

    # step 10: rescale and align the bounding boxes to match the resized image size (typically 224x224)
    resized_and_aligned_bboxes = []

    for bbox in unnormalized_token_boxes:
        # performing the normalization of the bounding box
        resized_and_aligned_bboxes.append(resize_align_bbox(tuple(bbox), *original_image.size, *target_size))

    encoding["resized_and_aligned_bounding_boxes"] = resized_and_aligned_bboxes

    # step 11: add the relative distances in the normalized grid
    bboxes_centroids = get_centroid(resized_and_aligned_bboxes)
    pad_token_start_index = get_pad_token_id_start_index(words, encoding, tokenizer)
    a_rel_x, a_rel_y = get_relative_distance(resized_and_aligned_bboxes, bboxes_centroids, pad_token_start_index)

    # step 12: convert all to tensors
    for k, v in encoding.items():
        encoding[k] = torch.as_tensor(encoding[k])

    encoding.update({
        "x_features": torch.as_tensor(a_rel_x, dtype=torch.int32),
        "y_features": torch.as_tensor(a_rel_y, dtype=torch.int32),
        })

    # step 13: add tokens for debugging
    if extras_for_debugging:
        input_ids = encoding["mlm_labels"] if apply_mask_for_mlm else encoding["input_ids"]
        encoding["tokens_without_padding"] = tokenizer.convert_ids_to_tokens(input_ids)
        encoding["words"] = words


    # step 14: add extra dim for batch
    if add_batch_dim:
        encoding["x_features"].unsqueeze_(0)
        encoding["y_features"].unsqueeze_(0)
        encoding["input_ids"].unsqueeze_(0)
        encoding["resized_scaled_img"].unsqueeze_(0)

    # step 15: save to disk
    if save_to_disk:
        os.makedirs(path_to_save, exist_ok=True)
        image_name = os.path.basename(image)
        with open(f"{path_to_save}{image_name}.pickle", "wb") as f:
            pickle.dump(encoding, f)

    # step 16: keys to keep, resized_and_aligned_bounding_boxes have been added for the purpose to test if the bounding boxes are drawn correctly or not, it maybe removed

    keys = ['resized_scaled_img', 'x_features','y_features','input_ids','resized_and_aligned_bounding_boxes']

    if apply_mask_for_mlm:
        keys.append('mlm_labels')

    final_encoding = {k:encoding[k] for k in keys}

    del encoding
    return final_encoding



## Creating the dataset

class RVLCDIPData(Dataset):

    def __init__(self, image_list, label_list, target_size, tokenizer, max_len = 512, transform = None):



        self.image_list = image_list
        self.label_list = label_list
        # print('before')
        # print(self.image_list)
        # print(type(self.image_list))
        # print(self.image_list[0])
        # print(type(self.image_list[0]))
        # self.image_list = image_list if isinstance(image_list[0], str) else [img for sublist in image_list for img in sublist]
        # self.label_list = label_list if isinstance(label_list[0], (int, torch.Tensor)) else [label for sublist in label_list for label in sublist]
        self.target_size = target_size
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        print('????????????????????????')
        print(self.image_list)
        print(type(self.image_list))
        print(idx)
        if isinstance(idx, list):
            if len(idx)>1:
                exit('INDEX LEN ISSUE')
            idx = idx[0]
        print('$$$$$$$$$$$$$$$$$$$')
        img_path = self.image_list[idx]
        
        label = self.label_list[idx]
        print(idx, '##########>>>>>>>>', img_path, '##########>>>>>>>', label)
        ## More on this, in the repo mentioned previously
        final_encoding = create_features(
            img_path,
            self.tokenizer,
            add_batch_dim=False,
            target_size=self.target_size,
            max_seq_length=self.max_len,
            path_to_save=None,
            save_to_disk=False,
            apply_mask_for_mlm=False,
            extras_for_debugging=False,
            use_ocr = True
        )
        # exit('DONE final_encoding')
        if self.transform is not None:
            ## Note that, ToTensor is already applied on the image
            final_encoding['resized_scaled_img'] = self.transform(final_encoding['resized_scaled_img'])


        keys_to_reshape = ['x_features', 'y_features', 'resized_and_aligned_bounding_boxes']
        for key in keys_to_reshape:
            final_encoding[key] = final_encoding[key][:self.max_len]

        final_encoding['label'] = torch.as_tensor(label).long()
        return final_encoding
    
    
    
    ## Defining the tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


from torchvision import transforms

## Normalization to these mean and std (I have seen some tutorials used this, and also in image reconstruction, so used it)
transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

train_ds = RVLCDIPData(train_df['img'].tolist(), train_df['label'].tolist(),
                      target_size, tokenizer, config['max_position_embeddings'], transform)
val_ds = RVLCDIPData(valid_df['img'].tolist(), valid_df['label'].tolist(),
                      target_size, tokenizer,config['max_position_embeddings'],  transform)

print(train_ds[0])

def collate_fn(data_bunch):

  '''
  A function for the dataloader to return a batch dict of given keys

  data_bunch: List of dictionary
  '''

  dict_data_bunch = {}

  for i in data_bunch:
    for (key, value) in i.items():
      if key not in dict_data_bunch:
        dict_data_bunch[key] = []
      dict_data_bunch[key].append(value)

  for key in list(dict_data_bunch.keys()):
      dict_data_bunch[key] = torch.stack(dict_data_bunch[key], axis = 0)

  return dict_data_bunch

  
  
import pytorch_lightning as pl

class DataModule(pl.LightningDataModule):

  def __init__(self, train_dataset, val_dataset,  batch_size = 1): #เดิม batch_size=4

    super(DataModule, self).__init__()
    self.train_dataset = train_dataset
    self.val_dataset = val_dataset
    self.batch_size = batch_size

  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size = self.batch_size,
                      collate_fn = collate_fn, shuffle = False)

  def val_dataloader(self):
    return DataLoader(self.val_dataset, batch_size = self.batch_size,
                                  collate_fn = collate_fn, shuffle = False)
    
    
# datamodule = DataModule(train_ds, val_ds) #?????????

print(train_ds[0])
print('DONE1')

# # print(train_ds)
# for i in train_ds:
#     print(i.keys())
#     print(i['label'])
#     print(type(i['label']))
#     break
# exit('READY')

class DocFormerForClassification(nn.Module):

    def __init__(self, config):
      super(DocFormerForClassification, self).__init__()

      self.resnet = ResNetFeatureExtractor(hidden_dim = config['max_position_embeddings'])
      self.embeddings = DocFormerEmbeddings(config)
      self.lang_emb = LanguageFeatureExtractor()
      self.config = config
      self.dropout = nn.Dropout(config['hidden_dropout_prob'])
      self.linear_layer = nn.Linear(in_features = config['hidden_size'], out_features = len(id2label))  ## Number of Classes
      self.encoder = DocFormerEncoder(config)

    def forward(self, batch_dict):

      x_feat = batch_dict['x_features']
      y_feat = batch_dict['y_features']

      token = batch_dict['input_ids']
      img = batch_dict['resized_scaled_img']

      v_bar_s, t_bar_s = self.embeddings(x_feat,y_feat)
      v_bar = self.resnet(img)
      t_bar = self.lang_emb(token)
      out = self.encoder(t_bar,v_bar,t_bar_s,v_bar_s)
      out = self.linear_layer(out)
      out = out[:, 0, :]
      return out
  
  

## Defining pytorch lightning model
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torchmetrics

class DocFormer(pl.LightningModule):

  def __init__(self, config , lr = 1e-5): # เดิม lr=5e-5
    super(DocFormer, self).__init__()

    self.save_hyperparameters()
    self.config = config
    self.docformer = DocFormerForClassification(config)

    self.num_classes = len(id2label)
    print(self.num_classes)
    print(type(self.num_classes))
    print('>?>>>>>>>>>>>>>>>>>>>>>>>>')
    self.train_accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
    self.val_accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
    self.f1_metric = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes)
    print(self.num_classes)
    print(type(self.num_classes))
    print('???????????????')
    self.precision_macro_metric = torchmetrics.Precision(task="multiclass",
            average="macro", num_classes=self.num_classes
        )
    print('DONE1')
    self.recall_macro_metric = torchmetrics.Recall(task="multiclass",
            average="macro", num_classes=self.num_classes
        )
    print('DONE2')
    self.precision_micro_metric = torchmetrics.Precision(task="multiclass", average="micro", num_classes=self.num_classes)
    print('DONE3')
    self.recall_micro_metric = torchmetrics.Recall(task="multiclass", average="micro", num_classes=self.num_classes)
    print('DONE4')
  def forward(self, batch_dict):
    logits = self.docformer(batch_dict)
    return logits

  def training_step(self, batch, batch_idx):
    logits = self.forward(batch)

    loss = nn.CrossEntropyLoss()(logits, batch['label'])
    preds = torch.argmax(logits, 1)

    ## Calculating the accuracy score
    train_acc = self.train_accuracy_metric(preds, batch["label"])

    ## Logging
    self.log('train/loss', loss,prog_bar = True, on_epoch=True, logger=True, on_step=True)
    self.log('train/acc', train_acc, prog_bar = True, on_epoch=True, logger=True, on_step=True)

    return loss

  def validation_step(self, batch, batch_idx):
    logits = self.forward(batch)
    loss = nn.CrossEntropyLoss()(logits, batch['label'])
    preds = torch.argmax(logits, 1)

    labels = batch['label']
    # Metrics
    valid_acc = self.val_accuracy_metric(preds, labels)
    precision_macro = self.precision_macro_metric(preds, labels)
    recall_macro = self.recall_macro_metric(preds, labels)
    precision_micro = self.precision_micro_metric(preds, labels)
    recall_micro = self.recall_micro_metric(preds, labels)
    f1 = self.f1_metric(preds, labels)

    # Logging metrics
    self.log("valid/loss", loss, prog_bar=True, on_step=True, logger=True)
    self.log("valid/acc", valid_acc, prog_bar=True, on_epoch=True, logger=True, on_step=True)
    self.log("valid/precision_macro", precision_macro, prog_bar=True, on_epoch=True, logger=True, on_step=True)
    self.log("valid/recall_macro", recall_macro, prog_bar=True, on_epoch=True, logger=True, on_step=True)
    self.log("valid/precision_micro", precision_micro, prog_bar=True, on_epoch=True, logger=True, on_step=True)
    self.log("valid/recall_micro", recall_micro, prog_bar=True, on_epoch=True, logger=True, on_step=True)
    self.log("valid/f1", f1, prog_bar=True, on_epoch=True)

    return {"label": batch['label'], "logits": logits}

  # def validation_epoch_end(self, outputs): # commented this !!!!!!!!!!
  # def on_validation_epoch_end(self, outputs):
  #       labels = torch.cat([x["label"] for x in outputs])
  #       logits = torch.cat([x["logits"] for x in outputs])
  #       preds = torch.argmax(logits, 1)

  #       wandb.log({"cm": wandb.sklearn.plot_confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy())})
  #       self.logger.experiment.log(
  #           {"roc": wandb.plot.roc_curve(labels.cpu().numpy(), logits.cpu().numpy())}
  #       )

  def configure_optimizers(self):
    return torch.optim.AdamW(self.parameters(), lr = self.hparams['lr'])

# ckpt_path = "/kaggle/input/downloading-docformer-weights/docformer_best_ckpt.ckpt"
# ckpt_path = "/content/drive/MyDrive/docformer_best_ckpt.ckpt"

# model = DocFormer.load_from_checkpoint(ckpt_path, map_location = "cpu", config = config)

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb

def main():
    datamodule = DataModule(train_ds, val_ds)
    docformer = DocFormer(config)

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor="valid/loss", mode="min"
    )
    early_stopping_callback = EarlyStopping(
        monitor="valid/loss", patience=3, verbose=True, mode="min"
    )

    # wandb.init(config=config, project="RVL CDIP with DocFormer New Version")
    # wandb_logger = WandbLogger(project="RVL CDIP with DocFormer New Version", entity="iakarshu")
    ## https://www.tutorialexample.com/implement-reproducibility-in-pytorch-lightning-pytorch-lightning-tutorial/
    pl.seed_everything(seed, workers=True)
    trainer = pl.Trainer(
        default_root_dir="logs",
        # devices='auto',#1 if torch.cuda.is_available() else 0,
        # accelerator = 'auto',
        # gpus=(1 if torch.cuda.is_available() else 0),
        # devices = 1,
        devices = 1 if torch.cuda.is_available() else 0,
        # accelerator = 'cpu',
        accelerator = 'cuda',
        min_epochs=2,
        max_epochs=2,
        fast_dev_run=False,
        # logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        deterministic=True
    )
    trainer.fit(docformer, datamodule)

    return docformer, datamodule


# if __name__ == "__main__":
model, datamodule = main()