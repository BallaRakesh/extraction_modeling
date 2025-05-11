import json
import shutil
from imagededup.methods import *
from imagededup.utils import plot_duplicates
import os
import moment
import time
 
 
def remove_duplicate_images(image_directory: str, unique_docs_dir: str, unique_duplicates_images_directory: str, log_dir: str):
    phasher = PHash()
    encodings = phasher.encode_images(image_dir=image_directory)
    duplicates = phasher.find_duplicates(encoding_map=encodings)
    unique_images=[]
    duplicate_images=[]
    SameDuplicatesInMoreThanOneImage={}
    for key in duplicates.keys():
        if key not in duplicate_images:
            unique_fodler_path=unique_duplicates_images_directory+"/"+key.split(".")[0]+"/Unique"
            duplicates_folder_path=unique_duplicates_images_directory+"/"+key.split(".")[0]+"/Duplicates"
 
            if not os.path.isdir(unique_fodler_path):
                os.makedirs(unique_fodler_path)
            if not os.path.isdir(duplicates_folder_path):
                os.makedirs(duplicates_folder_path)
            shutil.copy(image_directory+"/"+key, unique_fodler_path+"/"+key)
            shutil.copy(image_directory+"/"+key, unique_docs_dir)
            unique_images.append(key)
            for image in duplicates[key]:
                shutil.copy(image_directory+"/"+image, duplicates_folder_path+"/"+image)
                if image in duplicate_images:
                    if image not in SameDuplicatesInMoreThanOneImage.keys():
                        SameDuplicatesInMoreThanOneImage[image]=[key]
                    else:
                        SameDuplicatesInMoreThanOneImage[image].append(key)
                else:
                    duplicate_images.append(image)
 
 
if __name__== "__main__":
 
    # phasher = CNN()
    # phasher = WHash()
    # phasher = AHash()
 
    # phasher = DHash()
    image_directory='/home/ntlpt39/work/TradeFinance/Final_Data/IC/Test/Test/Test/Images'
    output_path = '/home/ntlpt39/work/TradeFinance/Final_Data/IC/Test/Test/Test/Images/DeDup'
    unique_duplicates_images_directory = os.path.join(output_path, 'duplicates_removal_res')
    unique_docs_dir= os.path.join(output_path, "unique_docs")
    if not os.path.isdir(unique_duplicates_images_directory):
        os.makedirs(unique_duplicates_images_directory)
    os.makedirs(unique_docs_dir, exist_ok=True)
    todays_date = moment.unix(time.time(), utc=True).locale('Asia/Kolkata').format("YYYY-MM-DD_HH-mm-ss")
    log_dir= os.path.join(output_path, "logs")
    if os.path.isdir(log_dir):
        lst = os.listdir(log_dir)
        number_files = len(lst)
    else :
        os.makedirs(log_dir)
        number_files=0
    remove_duplicate_images(image_directory, unique_docs_dir, unique_duplicates_images_directory, log_dir)