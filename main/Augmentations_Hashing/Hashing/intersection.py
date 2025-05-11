import os

folder_path = "/home/ntlpt39/work/TradeFinance/Final_Data/IC"

train_folder_name = "train/Images"
test_folder_name = "test/Images"
val_folder_name = "val/Images"

train_list = os.listdir(os.path.join(folder_path,train_folder_name))
test_list = os.listdir(os.path.join(folder_path,test_folder_name))
val_list = os.listdir(os.path.join(folder_path,val_folder_name))

for i in train_list:
    if i not in test_list and i not in val_list:
        pass
    else:
        if i in test_list:
            print("{} is present in common in test".format(i))
            if  i in val_list:
                print("{} is present in common in val".format(i))

