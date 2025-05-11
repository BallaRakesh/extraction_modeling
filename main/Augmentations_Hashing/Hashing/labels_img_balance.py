import os
import shutil
from random import sample 




imgs_ = '/home/ntlpt19/Downloads/Final_Delivery_Training_itter_5/Train_data/CS_797/CS/Images'
labels = '/home/ntlpt19/Downloads/Final_Delivery_Training_itter_5/Train_data/CS_797/CS/Labels'
cou = 0
for i in os.listdir(imgs_):
    if i.replace(".png",".txt") in os.listdir(labels):
        # print(i)
        cou+=1
print(cou)
exit('lplplplp')
    
    




master_dir = "/home/ntlpt39/work/TradeFinance/Final_Data/IC"
all_imgs_dir =master_dir+ "/"+"Images"
all_labels_txt_folder = master_dir+"/"+"Labels"

path_dir = "/home/ntlpt39/work/TradeFinance/Final_Data/IC/val"
imgs_dir = path_dir+"/"+"Images"
lbl_dir = path_dir+"/"+"Labels"

# def copy_labels_based_imgs(all_labels_txt_folder)
for i in os.listdir(imgs_dir):
    if i.replace(".png",".txt") in os.listdir(all_labels_txt_folder):
        shutil.copy(os.path.join(all_labels_txt_folder,i.replace(".png",".txt")),lbl_dir)








