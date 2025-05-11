base_directory = '/home/ntlpt19/Downloads/Classification_final_training/V2_ROOT/LC'
split_ocr_files = '/home/ntlpt19/Downloads/Classification_final_training/V4_ROOT/LC/ocr_chunks'
train_data_csv = '/home/ntlpt19/Downloads/Classification_final_training/V4_ROOT/LC/training_set_1.csv'
test_data_csv = '/home/ntlpt19/Downloads/Classification_final_training/V4_ROOT/LC/testing_set_1.csv'
custom_label2id = {'PO': 0, 'PI': 1, 'OTHERS': 2}

custom_split = True

## For the purpose of prediction

## Preparing the Dataset
eavl_directory = '/home/ntlpt19/Downloads/Eval_classification/LC'

eval_split_ocr_files = '/home/ntlpt19/Downloads/Classification_final_training/V4_ROOT/LC/eval/ocr_chunk'
label2id_infer = {'PO': 0, 'PI': 1, 'OTHERS': 2}
id2label_infer = {0:'PO', 1:'PI', 2:'OTHERS'}
model_path = '/home/ntlpt19/Downloads/Classification_final_training/docformer_model/LC_org/itr2/model-epoch=03.ckpt'
# Load the entire model
