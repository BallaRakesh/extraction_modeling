import cv2
import numpy as np
import os
from utility import draw_random_lines_across_image, masking, angle_mask, toggle_mask_all_corners
from PIL import Image
import os
from tqdm import tqdm
from constants import omitted_doc_type, random_agumentation, important_agumentation
import random
from augraphy import *
import cv2
import numpy as np
import random
from augraphy_agument import AugraphyAgument
import os
import shutil
from new_augraphy_augment import Extraction_AugraphyAgument,shifting_annotation
import time
import os  
from collections import defaultdict  


def check_image_exists(image_name_, folder_path):
    prefixes = [
    "Shifted_",
    "hollow_op_",
    "watermark__",
    "dirtydrum_op_",
    "dotmatrix_op_",
    "colorpaper_op_",
    "colorshift_op_",
    "salt_n_pepper_",
    "bleedthrough_op_",
    "dirty_rollers_op_",
    "BadPhotoCopy_new_op_",
    "binder_punch_holes_op_",
    "depthsimulatedblur_op_",
    "brightness_texturize_op_",
    ]
    for prefix in prefixes:
        if os.path.exists(os.path.join(folder_path, prefix + image_name_)):
            return True
    return False
# from memory_profiler import profile
# @profile
def main():
    doc_type="train_data"
    root_path ="/home/ntlpt19/Downloads/Final_Delivery_Training_itter_5/CI_verified_annotations"
    image_folder_name = "Images"
    labels_folder_name = "Labels"
    image_folder = os.path.join(root_path,doc_type,image_folder_name)
    labels_folder = os.path.join(root_path,doc_type,labels_folder_name)
    field_counts = defaultdict(int)  
    field_images = defaultdict(set)  # Use a set to avoid duplicate image entries  

    for filename in os.listdir(labels_folder):  
        if filename.endswith('.txt'):  
            file_path = os.path.join(labels_folder, filename)  
            base_name = os.path.splitext(filename)[0]+".png"  # Get the base name to match with images  
            with open(file_path, 'r') as file:  
                for line in file:  
                    parts = line.strip().split()  
                    if len(parts) > 0:  
                        field = parts[0]  # Get the first part as the field  
                        field_counts[field] += 1  # Increment the count for this field  
                        field_images[field].add(base_name)  # Append base_name to the set for the corresponding field 
    total_images = len([f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))])  
    threshold = total_images * 0.2  
    less_occuring_fields = [field for field, count in field_counts.items() if count < threshold]  

    images_with_less_fields = []  
    for field in less_occuring_fields:  
        images_with_less_fields.extend(field_images[field])  
    final_images = []
    # Step 6: Print results  
    print(f"Total images: {total_images}")  
    print(f"Threshold (20%): {threshold}")  
    print(f"Less occurring fields (count < {threshold}): {less_occuring_fields}")  
    print("Images corresponding to the less occurring fields:")  
    for image in set(images_with_less_fields):  # Using set to avoid duplicates  
        final_images.append(image)
    print(final_images)
    augmented_image_folder = os.path.join(root_path,doc_type,"Augmented_Master_images")
    augmented_labels_folder = os.path.join(root_path,doc_type,"Augmented_Master_Labels")
    os.makedirs(augmented_image_folder, exist_ok=True)
    os.makedirs(augmented_labels_folder, exist_ok=True)
    for image_file in tqdm(os.listdir(image_folder)):
        if image_file in final_images:
            print('$$$$$$$$$$$$$$$$$', image_file)
            if not check_image_exists(image_file, augmented_image_folder):
                if image_file.replace(".png",".txt") in os.listdir(labels_folder):
                    label_file_name = image_file.replace(".png",".txt")
                    random_number1 = random.randint(0, 4)
                    image_path = os.path.join(image_folder,image_file)
                    original_image = cv2.imread(image_path)
                    xshift=10
                    yshift=10
                    augraphy_agument_obj = Extraction_AugraphyAgument(original_image)
                    start_time = time.time()
                    ################### SALT N PEPPER NOISE ADDITION ###################
                    salt_and_pepper_ratio = 0.06  # Adjust the ratio as needed

                    # Generate salt-and-pepper noise
                    salt_and_pepper_mask = np.random.rand(*original_image.shape[:2])
                    salt_pixels = salt_and_pepper_mask < salt_and_pepper_ratio / 2.0
                    pepper_pixels = salt_and_pepper_mask > 1 - salt_and_pepper_ratio / 2.0

                    # Add salt-and-pepper noise to the image
                    
                    noisy_image = original_image.copy()
                    noisy_image[salt_pixels] = 255  # Set salt pixels to white (255)
                    noisy_image[pepper_pixels] = 0  # Set pepper pixels to black (0)
                    cv2.imwrite(os.path.join(augmented_image_folder,"salt_n_pepper_"+image_file),noisy_image)
                    shutil.copy(os.path.join(labels_folder,label_file_name),os.path.join(augmented_labels_folder,"salt_n_pepper_"+label_file_name))
                    endtime = time.time()
                    print("SALT N PEPER TIME: ",endtime - start_time)
                    ############################################################################
                    start_time = time.time()
                    augmented_image, name = augraphy_agument_obj.watermark_()
                    cv2.imwrite(os.path.join(augmented_image_folder,name+"_"+image_file), augmented_image)
                    shutil.copy(os.path.join(labels_folder,label_file_name),os.path.join(augmented_labels_folder,name+"_"+label_file_name))
                    endtime = time.time()
                    print("watermark_ TIME: ",endtime - start_time)
                    ############################################################################
                    start_time = time.time()
                    augmented_image, name = augraphy_agument_obj.BadPhotoCopy_new_op()
                    cv2.imwrite(os.path.join(augmented_image_folder,name+"_"+image_file), augmented_image)
                    shutil.copy(os.path.join(labels_folder,label_file_name),os.path.join(augmented_labels_folder,name+"_"+label_file_name))
                    endtime = time.time()
                    print("BadPhotoCopy_new_op TIME: ",endtime - start_time)
                    ############################################################################
                    start_time = time.time()
                    augmented_image, name = augraphy_agument_obj.binder_punch_holes_op()
                    cv2.imwrite(os.path.join(augmented_image_folder,name+"_"+image_file), augmented_image)
                    shutil.copy(os.path.join(labels_folder,label_file_name),os.path.join(augmented_labels_folder,name+"_"+label_file_name))
                    endtime = time.time()
                    print("binder_punch_holes_op TIME: ",endtime - start_time)
                    ############################################################################
                    start_time = time.time()
                    augmented_image, name = augraphy_agument_obj.bleedthrough_op()
                    cv2.imwrite(os.path.join(augmented_image_folder,name+"_"+image_file), augmented_image)
                    shutil.copy(os.path.join(labels_folder,label_file_name),os.path.join(augmented_labels_folder,name+"_"+label_file_name))
                    endtime = time.time()
                    print("bleedthrough_op TIME: ",endtime - start_time)
                    ############################################################################
                    start_time = time.time()
                    augmented_image, name = augraphy_agument_obj.brightness_texturize_op()
                    cv2.imwrite(os.path.join(augmented_image_folder,name+"_"+image_file), augmented_image)
                    shutil.copy(os.path.join(labels_folder,label_file_name),os.path.join(augmented_labels_folder,name+"_"+label_file_name))
                    endtime = time.time()
                    print("brightness_texturize_op TIME: ",endtime - start_time)
                    ############################################################################
                    start_time = time.time()
                    augmented_image, name = augraphy_agument_obj.colorpaper_op()
                    cv2.imwrite(os.path.join(augmented_image_folder,name+"_"+image_file), augmented_image)
                    shutil.copy(os.path.join(labels_folder,label_file_name),os.path.join(augmented_labels_folder,name+"_"+label_file_name))
                    endtime = time.time()
                    print("colorpaper_op TIME: ",endtime - start_time)
                    ############################################################################
                    start_time = time.time()
                    augmented_image, name = augraphy_agument_obj.colorshift_op()
                    cv2.imwrite(os.path.join(augmented_image_folder,name+"_"+image_file), augmented_image)
                    shutil.copy(os.path.join(labels_folder,label_file_name),os.path.join(augmented_labels_folder,name+"_"+label_file_name))
                    endtime = time.time()
                    print("colorshift_op TIME: ",endtime - start_time)
                    ############################################################################
                    start_time = time.time()
                    augmented_image, name = augraphy_agument_obj.depthsimulatedblur_op()
                    cv2.imwrite(os.path.join(augmented_image_folder,name+"_"+image_file), augmented_image)
                    shutil.copy(os.path.join(labels_folder,label_file_name),os.path.join(augmented_labels_folder,name+"_"+label_file_name))
                    endtime = time.time()
                    print("depthsimulatedblur_op TIME: ",endtime - start_time)
                    ############################################################################
                    start_time = time.time()
                    augmented_image, name = augraphy_agument_obj.dirtydrum_op()
                    cv2.imwrite(os.path.join(augmented_image_folder,name+"_"+image_file), augmented_image)
                    shutil.copy(os.path.join(labels_folder,label_file_name),os.path.join(augmented_labels_folder,name+"_"+label_file_name))
                    endtime = time.time()
                    print("dirtydrum_op TIME: ",endtime - start_time)
                    ############################################################################
                    start_time = time.time()
                    augmented_image, name = augraphy_agument_obj.dirty_rollers_op()
                    cv2.imwrite(os.path.join(augmented_image_folder,name+"_"+image_file), augmented_image)
                    shutil.copy(os.path.join(labels_folder,label_file_name),os.path.join(augmented_labels_folder,name+"_"+label_file_name))
                    endtime = time.time()
                    print("dirty_rollers_op TIME: ",endtime - start_time)
                    ############################################################################
                    start_time = time.time()
                    augmented_image, name = augraphy_agument_obj.dotmatrix_op()
                    cv2.imwrite(os.path.join(augmented_image_folder,name+"_"+image_file), augmented_image)
                    shutil.copy(os.path.join(labels_folder,label_file_name),os.path.join(augmented_labels_folder,name+"_"+label_file_name))
                    endtime = time.time()
                    print("dotmatrix_op TIME: ",endtime - start_time)
                    ############################################################################
                    start_time = time.time()
                    augmented_image, name = augraphy_agument_obj.hollow_op()
                    cv2.imwrite(os.path.join(augmented_image_folder,name+"_"+image_file), augmented_image)
                    shutil.copy(os.path.join(labels_folder,label_file_name),os.path.join(augmented_labels_folder,name+"_"+label_file_name))
                    endtime = time.time()
                    print("hollow_op TIME: ",endtime - start_time)
                    ############################################################################
                    start_time = time.time()
                    annotation_shift_obj = shifting_annotation(augmented_labels_folder, augmented_image_folder,xshift,yshift)
                    annotation_shift_obj.shifting_op(image_path,os.path.join(labels_folder,label_file_name))
                    endtime = time.time()
                    print("shifting_annotation TIME: ",endtime - start_time)
                    ############################################################################
if __name__=="__main__":
    main()
