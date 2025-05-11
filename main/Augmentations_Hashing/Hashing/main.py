import json
import shutil
from imagededup.methods import *
from imagededup.utils import plot_duplicates
import os
import moment
import time
import random
import time

def remove_files_with_no_annotation_or_img(image_directory,labels_directory,ocr_directory,path_of_folder_containing_images_labels_folder,IMAGE_EXTENSTION):
    images_list = os.listdir(image_directory)
    labels_list = os.listdir(labels_directory)

    new_master_img_path = path_of_folder_containing_images_labels_folder+"/Master/Images"
    new_master_label_path = path_of_folder_containing_images_labels_folder+"/Master/Labels"
    new_master_ocr_path = path_of_folder_containing_images_labels_folder+"/Master/OCR"

    if not os.path.exists(new_master_img_path):
            os.makedirs(new_master_img_path)
    if not os.path.exists(new_master_label_path):
            os.makedirs(new_master_label_path)
    if not os.path.exists(new_master_ocr_path):
            os.makedirs(new_master_ocr_path)


    for image in images_list:
        if image.replace(IMAGE_EXTENSTION,".txt") in labels_list:
            image_path = os.path.join(image_directory,image)
            label_path = os.path.join(labels_directory,image.replace(IMAGE_EXTENSTION,".txt"))
            ocr_path = os.path.join(ocr_directory,image.replace(IMAGE_EXTENSTION,"_text.txt"))

            shutil.copy(image_path,new_master_img_path)
            shutil.copy(label_path,new_master_label_path)
            shutil.copy(ocr_path,new_master_ocr_path)
    shutil.copy(path_of_folder_containing_images_labels_folder+"/label.txt",path_of_folder_containing_images_labels_folder+"/Master")
    return path_of_folder_containing_images_labels_folder+"/Master"

def hashing_split(hasher_function,master_image_directory):
    encodings = hasher_function.encode_images(image_dir=master_image_directory)
    duplicates = hasher_function.find_duplicates(encoding_map=encodings)
    unique_images=[]
    duplicate_images=[]
    for key in duplicates.keys():
        if key not in duplicate_images:
            unique_images.append(key)
            for image in duplicates[key]:
                if image not in unique_images and image not in duplicate_images:
                    duplicate_images.append(image)
    return unique_images, duplicate_images

def create_temp_master(list_of_imgs,master_image_directory):
    temp_folder = os.getcwd()+"/temp_images"
    if not os.path.exists(temp_folder):
            os.mkdir(temp_folder)
    for i in list_of_imgs:
        shutil.copy(os.path.join(master_image_directory,i),temp_folder)
    return temp_folder

def select_random_images(list_of_imgs,num_values):
    print('list_of_imgs',list_of_imgs,'num_values',num_values)
    if len(list_of_imgs) < num_values:
        num_values = len(list_of_imgs)
        print("Length of test {} is smaller than value {}".format(len(list_of_imgs), num_values))
        # return 
    print("---------------------------",num_values)
    selected_values = random.sample(list_of_imgs, num_values)
    for value in selected_values:
        list_of_imgs.remove(value)
    return selected_values, list_of_imgs

def check_lists_for_consistency(train_list,test_list,val_list):
    set1 = set(train_list)
    set2 = set(test_list)
    set3 = set(val_list)
    # Check for intersections
    common_12 = set1.intersection(set2)
    common_13 = set1.intersection(set3)
    common_23 = set2.intersection(set3)
    # If there are any common elements, the intersection will not be empty
    if common_12 or common_13 or common_23:
        return False
    return True


def file_moving(list_img,master_image_directory,master_labels_directory,master_ocr_directory,final_set_folder,IMAGE_EXTENSTION,counter):
     with open(final_set_folder+"/"+counter+".txt", 'w') as file:
        for img in list_img:
            file.write(img + '\n')
            image_path = os.path.join(master_image_directory,img)
            label_path = os.path.join(master_labels_directory,img.replace(IMAGE_EXTENSTION,".txt"))
            ocr_path = os.path.join(master_ocr_directory,img.replace(IMAGE_EXTENSTION,"_text.txt"))
            image_folder=final_set_folder+"/Images"
            label_folder = final_set_folder+"/Labels"
            ocr_folder = final_set_folder+"/OCR"
            if not os.path.exists(image_folder):
                os.mkdir(image_folder)
            if not os.path.exists(label_folder):
                os.mkdir(label_folder)
            if not os.path.exists(ocr_folder):
                os.mkdir(ocr_folder)
            shutil.copy(image_path,image_folder)
            shutil.copy(label_path,label_folder)
            if os.path.exists(ocr_path):
                shutil.copy(ocr_path,ocr_folder)


def create_train_test_split(train_list,test_list,val_list,master_folder,IMAGE_EXTENSTION):
    final_train_folder = master_folder+"/Final_Train/"
    final_test_folder = master_folder+"/Final_Test/"
    final_val_folder = master_folder+"/Final_Val/"

    master_image_directory = master_folder+"/Images/"
    master_labels_directory = master_folder+"/Labels/"
    master_ocr_directory = master_folder+"/OCR/"

    if not os.path.exists(final_train_folder):
            os.mkdir(final_train_folder)
    if not os.path.exists(final_test_folder):
            os.mkdir(final_test_folder)
    if not os.path.exists(final_val_folder):
            os.mkdir(final_val_folder)
    file_moving(train_list,master_image_directory,master_labels_directory,master_ocr_directory,final_train_folder,IMAGE_EXTENSTION,"train")
    file_moving(test_list,master_image_directory,master_labels_directory,master_ocr_directory,final_test_folder,IMAGE_EXTENSTION,"test")
    file_moving(val_list,master_image_directory,master_labels_directory,master_ocr_directory,final_val_folder,IMAGE_EXTENSTION,"val")
    
if __name__=="__main__":
    
    TRAIN_COUNT = 200
    TEST_COUNT = 196
    VAL_COUNT = 200

    TRAIN_PERCENTAGE = 0.75
    TEST_PERCENTAGE = 0.10
    VAL_PERCENTAGE = 0.15

    PERMISSIBLE_ERROR = 5

    HASHER_NAME = "Phash"

    IMAGE_EXTENSTION = ".png"
    hasher_functions = {
                    "Phash":PHash(),
                    "CNNhash":CNN(),
                    "Whash":WHash(),
                    "Ahash":AHash(),
                    "Dhash":DHash()
                        }
    path_of_folder_containing_images_labels_folder = "/home/ntlpt19/Downloads/Final_Delivery_Training_itter_5/Train_data/PL/Master/Final_Val"


    hasher_function = hasher_functions[HASHER_NAME]

    image_directory = path_of_folder_containing_images_labels_folder+"/Images"
    labels_directory = path_of_folder_containing_images_labels_folder+"/Labels"
    ocr_directory = path_of_folder_containing_images_labels_folder+"/OCR"

    final_path_of_master_data = remove_files_with_no_annotation_or_img(image_directory,labels_directory,ocr_directory,path_of_folder_containing_images_labels_folder,IMAGE_EXTENSTION)


    master_image_directory = final_path_of_master_data+"/Images"
    master_labels_directory = final_path_of_master_data+"/Labels"
    master_ocr_directory = final_path_of_master_data+"/OCR"
    total_count = len(os.listdir(master_image_directory))

    train_set,test_val_set =hashing_split(hasher_function,master_image_directory)
    train_count = int(TRAIN_PERCENTAGE*(total_count))
    test_count = int(TEST_PERCENTAGE*(total_count))
    val_count = int(VAL_PERCENTAGE*(total_count))

    temp_folder_path_test_val_set = create_temp_master(test_val_set,master_image_directory)
    val_set,test_set =hashing_split(hasher_function,temp_folder_path_test_val_set)

    print(len(os.listdir(temp_folder_path_test_val_set)))
    print(len(val_set))
    print(len(test_set))

    if len(val_set) > len(test_set) or abs(len(val_set)-val_count) < PERMISSIBLE_ERROR or abs(len(val_set)-VAL_COUNT) < PERMISSIBLE_ERROR:
        validation_set = val_set
    else:
        print("VALIDATION SET LENGTH IS {} , SPlitting Data Again".format(len(val_set)))
        selected_values, list_of_imgs = select_random_images(test_set,abs(len(val_set)-VAL_COUNT))
        validation_set = val_set+selected_values
        print("FINAL VALIDATION SET LENGTH IS {}".format(len(validation_set)))
        print("FINAL TEST SET LENGTH IS {}".format(len(list_of_imgs)))

    if abs(len(train_set)-train_count) < PERMISSIBLE_ERROR or abs(len(train_set)-TRAIN_COUNT) < PERMISSIBLE_ERROR:
        training_set = train_set
    else:
        print("TRAIN SET LENGTH IS {} , SPlitting Data Again".format(len(train_set)))
        count=abs(len(train_set)-train_count)
        selected_values, list_of_imgs = select_random_images(test_set,count)
        training_set = train_set+selected_values
        print("FINAL TRAIN SET LENGTH IS {}".format(len(training_set)))
        print("FINAL TEST SET LENGTH IS {}".format(len(list_of_imgs)))
        testing_set = list_of_imgs

    if check_lists_for_consistency(training_set,testing_set,validation_set) == False:
        print("Data not split correctly")
    else:
        create_train_test_split(training_set,testing_set,validation_set,final_path_of_master_data,IMAGE_EXTENSTION)

    shutil.rmtree(temp_folder_path_test_val_set)
    time.sleep(2)