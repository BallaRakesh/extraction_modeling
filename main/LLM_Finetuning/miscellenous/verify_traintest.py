import os




def verify_train(eval_data, train_data):
    for file_ in os.listdir(eval_data):
        if file_.endswith('_labels.txt'):
            if file_ in os.listdir(train_data):
                print(file_)
            

eval_data = '/home/ntlpt19/LLM_training/TRAIN/COO/TEST/Master_Data'
train_data = '/home/ntlpt19/LLM_training/TRAIN/COO/TRAIN/Master_Data'
verify_train(eval_data, train_data)