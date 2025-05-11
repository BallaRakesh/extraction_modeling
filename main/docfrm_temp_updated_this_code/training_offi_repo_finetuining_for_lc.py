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
from dataset import create_features
from modeling import DocFormerEncoder,ResNetFeatureExtractor,DocFormerEmbeddings,LanguageFeatureExtractor
from transformers import BertTokenizerFast


## Hyperparameters

seed = 42
target_size = (500, 384)

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


from tqdm.auto import tqdm

## For the purpose of prediction
id2label = []
# id2label = {0: 'AWB', 1: 'BOE', 2: 'BOL', 3: 'CI', 4: 'COO', 5: 'CS', 6: 'IC', 7: 'OTHERS', 8: 'PL'}
# label2id = {'AWB':0, 'BOE':1, 'BOL':2, 'CI':3, 'COO':4, 'CS':5, 'IC':6, 'OTHERS':7, 'PL':8}
label2id = {}
curr_class = 0
## Preparing the Dataset
base_directory = '/home/data_science/geo_testing/Classification_root/LC'
dict_of_img_labels = {'img':[], 'label':[]}

max_sample_per_class = 2350

for label in tqdm(os.listdir(base_directory)):
    img_path = os.path.join(base_directory, label)
    
    count = 0
    if label not in label2id:
        label2id[label] = curr_class
        curr_class+=1
        id2label.append(label)
        
    for img in os.listdir(img_path):
        if count>max_sample_per_class:
            break
            
        curr_img_path = os.path.join(img_path, img)
        dict_of_img_labels['img'].append(curr_img_path)
        dict_of_img_labels['label'].append(label2id[label])
        count+=1
        
        
# print(dict_of_img_labels['img'])
# print(len(dict_of_img_labels['img']))
# exit('OKOKOK')
print(label2id)
with open('modeling_label2id.txt', 'w') as file:
    file.write(str(label2id))
file.close()
    
print(label2id)
id2label = {v: k for k, v in label2id.items()}
print(id2label)

import pandas as pd
df = pd.DataFrame(dict_of_img_labels)

from sklearn.model_selection import train_test_split as tts
train_df, valid_df = tts(df, random_state = seed, stratify = df['label'], shuffle = True)
# Save the training set  
train_df.to_csv('train_lc.csv', index=False)  
valid_df.to_csv('valid_lc.csv', index=False)


train_df = train_df.reset_index().drop(columns = ['index'], axis = 1)
valid_df = valid_df.reset_index().drop(columns = ['index'], axis = 1)

## Creating the dataset

class RVLCDIPData(Dataset):
    
    def __init__(self, image_list, label_list, target_size, tokenizer, max_len = 512, transform = None):
        
        self.image_list = image_list
        self.label_list = label_list
        self.target_size = target_size
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transform
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        label = self.label_list[idx]
        
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
# tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")

from torchvision import transforms

## Normalization to these mean and std (I have seen some tutorials used this, and also in image reconstruction, so used it)
transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              
train_ds = RVLCDIPData(train_df['img'].tolist(), train_df['label'].tolist(),
                      target_size, tokenizer, config['max_position_embeddings'], transform)
val_ds = RVLCDIPData(valid_df['img'].tolist(), valid_df['label'].tolist(),
                      target_size, tokenizer,config['max_position_embeddings'],  transform)

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

  def __init__(self, train_dataset, val_dataset,  batch_size = 1):

    super(DataModule, self).__init__()
    self.train_dataset = train_dataset
    self.val_dataset = val_dataset
    self.batch_size = batch_size

  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size = self.batch_size, 
                      collate_fn = collate_fn, shuffle = True)
  
  def val_dataloader(self):
    return DataLoader(self.val_dataset, batch_size = self.batch_size,
                                  collate_fn = collate_fn, shuffle = False)
    
    
    
datamodule = DataModule(train_ds, val_ds)

class DocFormerForClassificationLoad(nn.Module):
  
    def __init__(self, config):
      super(DocFormerForClassificationLoad, self).__init__()

      self.resnet = ResNetFeatureExtractor(hidden_dim = config['max_position_embeddings'])
      self.embeddings = DocFormerEmbeddings(config)
      self.lang_emb = LanguageFeatureExtractor()
      self.config = config
      self.dropout = nn.Dropout(config['hidden_dropout_prob'])
      self.linear_layer = nn.Linear(in_features = config['hidden_size'], out_features = 9)# len(id2label))  ## Number of Classes for BILLS
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


class DocFormerLoad(pl.LightningModule):

  def __init__(self, config , lr = 5e-5):
    super(DocFormerLoad, self).__init__()
    self.save_hyperparameters()
    self.config = config
    self.docformer = DocFormerForClassificationLoad(config)
    self.num_classes = len(id2label)
    
  def forward(self, batch_dict):
    logits = self.docformer(batch_dict)
    return logits
  
  
  

model_path = '/home/data_science/geo_testing/docformer/models_bills/model-epoch=04.ckpt'
# Load the entire model
model = DocFormerLoad.load_from_checkpoint(model_path, config=config)#, strict=False)
# print(model)
# state_dict = model.state_dict()  # Get all weights from the loaded model
# print(state_dict.keys())  # Inspect available keys to locate the weights you need

class DocFormerForClassification(nn.Module):
  
    def __init__(self, config):
      super(DocFormerForClassification, self).__init__()
      ################################################################
      ################################################################
      # self.resnet = ResNetFeatureExtractor(hidden_dim = config['max_position_embeddings'])
      # self.embeddings = DocFormerEmbeddings(config)
      # self.lang_emb = LanguageFeatureExtractor()
      # self.config = config
      # self.dropout = nn.Dropout(config['hidden_dropout_prob'])
      # self.linear_layer = nn.Linear(in_features = config['hidden_size'], out_features = len(id2label))  ## Number of Classes
      # self.encoder = DocFormerEncoder(config)
      ################################################################
      ################################################################
      
      # self.resnet = ResNetFeatureExtractor()
      self.resnet = ResNetFeatureExtractor(hidden_dim = config['max_position_embeddings'])
      # self.resnet.load_state_dict(model.extract_feature.visual_feature.state_dict()) #????
      self.resnet.load_state_dict(model.docformer.resnet.state_dict())#,  strict=False)
      self.embeddings = DocFormerEmbeddings(config)
      # self.embeddings.load_state_dict(model.extract_feature.spatial_feature.state_dict())
      self.embeddings.load_state_dict(model.docformer.embeddings.state_dict())
      self.lang_emb = LanguageFeatureExtractor()
      # self.lang_emb.load_state_dict(model.extract_feature.language_feature.state_dict())
      self.lang_emb.load_state_dict(model.docformer.lang_emb.state_dict())
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

  def __init__(self, config , lr = 5e-5):
    super(DocFormer, self).__init__()
    
    self.save_hyperparameters()
    self.config = config
    self.docformer = DocFormerForClassification(config)
    
    self.num_classes = len(id2label)
    self.train_accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
    self.val_accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
    self.f1_metric = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes)
    self.precision_macro_metric = torchmetrics.Precision(
            task="multiclass",average="macro", num_classes=self.num_classes
        )
    self.recall_macro_metric = torchmetrics.Recall(
            task="multiclass", average="macro", num_classes=self.num_classes
        )
    self.precision_micro_metric = torchmetrics.Precision(task="multiclass", average="micro",  num_classes=self.num_classes)
    self.recall_micro_metric = torchmetrics.Recall(task="multiclass", average="micro",  num_classes=self.num_classes)

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

#   def validation_epoch_end(self, outputs):
#         labels = torch.cat([x["label"] for x in outputs])
#         logits = torch.cat([x["logits"] for x in outputs])
#         preds = torch.argmax(logits, 1)

#         # wandb.log({"cm": wandb.sklearn.plot_confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy())})
#         # self.logger.experiment.log(
#         #     {"roc": wandb.plot.roc_curve(labels.cpu().numpy(), logits.cpu().numpy())}
#         # )
        
  def configure_optimizers(self):
    return torch.optim.AdamW(self.parameters(), lr = self.hparams['lr'])

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

def main():
    datamodule = DataModule(train_ds, val_ds)
    docformer = DocFormer(config)

    # checkpoint_callback = ModelCheckpoint(
    #     dirpath="./models", monitor="valid/loss", mode="min"
    # )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor="valid/loss", mode="min", filename='model-{epoch:02d}',save_top_k=-1, save_on_train_epoch_end=True
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
        # gpus=(1 if torch.cuda.is_available() else 0),
        max_epochs=50,
        devices = 1 if torch.cuda.is_available() else 0,
        accelerator = 'cuda',
        fast_dev_run=False,
        # logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        deterministic=True
    )
    trainer.fit(docformer, datamodule)
    
if __name__ == "__main__":
    main()


