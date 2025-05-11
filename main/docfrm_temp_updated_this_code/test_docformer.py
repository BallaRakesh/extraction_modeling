import sys
import torch
sys.path.extend(['src/docformer/'])
import modeling, dataset
from transformers import BertTokenizerFast


config = {
  "coordinate_size": 96,
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "image_feature_pool_shape": [7, 7, 256],
  "intermediate_ff_size_factor": 4,
  "max_2d_position_embeddings": 1000,
  "max_position_embeddings": 512,
  "max_relative_positions": 8,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "shape_size": 96,
  "vocab_size": 30522,
  "layer_norm_eps": 1e-12,
}


fp = "/home/ntlpt19/Downloads/pdf2tiff/IM-000000016510466-AP.tiff"

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
encoding = dataset.create_features(fp, tokenizer, add_batch_dim=True)

feature_extractor = modeling.ExtractFeatures(config)
docformer = modeling.DocFormerEncoder(config)
v_bar, t_bar, v_bar_s, t_bar_s = feature_extractor(encoding)
output = docformer(v_bar, t_bar, v_bar_s, t_bar_s)  # shape (1, 512, 768)
print('$$$$$$$$$$$$$$$$$$$$$$$$$')
print(output)
# Option 1: Mean Pooling across sequence length (dim=1)
mean_pooled_output = torch.mean(output, dim=1)
# Option 2: Max Pooling across sequence length (dim=1)
max_pooled_output, _ = torch.max(output, dim=1)
# Option 3: Use CLS token (first token in the sequence)
cls_output = output[:, 0, :]
print("Mean pooled output:", mean_pooled_output)
print("Max pooled output:", max_pooled_output)
print("CLS token output:", cls_output)
