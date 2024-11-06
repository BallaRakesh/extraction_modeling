root_imgs_path = "/home/ntlpt19/Downloads/Final_Delivery_Training_itter_2/PO_actual/Images"
imgs_path = '/home/ntlpt19/Downloads/Final_Delivery_Training_itter_2/PO_actual/PO_training_300_images/Images'
root_lables_path = "/home/ntlpt19/Downloads/Final_Delivery_Training_itter_2/PO_actual/Labels"
lables_path = '/home/ntlpt19/Downloads/Final_Delivery_Training_itter_2/PO_actual/PO_training_300_images/labels'
out_put  = '/home/ntlpt19/Downloads/Final_Delivery_Training_itter_2/PO_actual/PO_pending_validation/Labels'
import shutil
import os


'''imga_list = os.listdir(imgs_path)
for img in os.listdir(root_imgs_path):
    if img not in imga_list:
        shutil.copy(os.path.join(root_imgs_path, img), os.path.join(out_put, img))

'''


imga_list = os.listdir(lables_path)
for img in os.listdir(root_lables_path):
    if img not in imga_list:
        shutil.copy(os.path.join(root_lables_path, img), os.path.join(out_put, img))


