import os
import shutil

all_imgs_dir = '/home/ntlpt39/work/TradeFinance/New_Data/CI/Master_images'
train_imgs_dir = "/home/ntlpt39/work/TradeFinance/New_Data/CI/Test_Val/DeDup/unique_docs"
all_labels_txt_folder = "/home/ntlpt39/work/TradeFinance/New_Data/CI/Master_Labels"

train_image_folder = "/home/ntlpt39/work/TradeFinance/New_Data/CI/Code_Test/Train/Images"
train_label_folder = "/home/ntlpt39/work/TradeFinance/New_Data/CI/Code_Test/Train/Labels"


test_val_image_folder = "/home/ntlpt39/work/TradeFinance/New_Data/CI/Code_Test/Test/Images"
test_val_label_folder = "/home/ntlpt39/work/TradeFinance/New_Data/CI/Code_Test/Test/Labels"

if not os.path.exists(train_image_folder):
    os.makedirs(train_image_folder)
if not os.path.exists(train_label_folder):
    os.makedirs(train_label_folder)
if not os.path.exists(test_val_image_folder):
    os.makedirs(test_val_image_folder)
if not os.path.exists(test_val_label_folder):
    os.makedirs(test_val_label_folder)


all_labels_list = os.listdir(all_labels_txt_folder)
train_img_list = os.listdir(train_imgs_dir)
print(len(train_img_list))

#Copy annotation.txt and images to the folder for training images or unique images
for i in train_img_list:
    img_path = os.path.join(train_imgs_dir,i)
    label_path = os.path.join(all_labels_txt_folder,i.replace(".png",".txt"))
    if i.replace(".png",".txt") in all_labels_list:
        shutil.copy(label_path,train_label_folder)
        shutil.copy(img_path,train_image_folder)

#COPY remaining test and val in a folder
for i in os.listdir(all_imgs_dir):
    if i not in train_img_list:
        if i.replace(".png",".txt") in all_labels_list:
            img_path = os.path.join(all_imgs_dir,i)
            shutil.copy(img_path,test_val_image_folder)
            label_path = os.path.join(all_labels_txt_folder,i.replace(".png",".txt"))
            shutil.copy(label_path,test_val_label_folder)

