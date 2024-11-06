#<----------------------------------------------------------For Two Images with ImageHash-------------------------------------------------------------------------
# from PIL import Image
# import imagehash
# hash0 = imagehash.average_hash(Image.open('')) 
# hash1 = imagehash.average_hash(Image.open('')) 
# cutoff = 5  # maximum bits that could be different between the hashes. 
# print("hash0",hash0)
# print("hash1",hash1)
# print("Difference",hash0-hash1)
# if hash0 - hash1 < cutoff:
#   print('images are similar')
# else:
#   print('images are not similar')
#-----------------------------------------------------------for whole dataset--------------------------------------------------------------------------------------------
import json
import shutil
from imagededup.methods import *
from imagededup.utils import plot_duplicates
# from logging_file import *
import os
import moment
import time

def __logging__(log_dir,image_directory, duplicates: list, unique_images:str, duplicate_images: str, SameDuplicatesInMoreThanOneImage:str):
    todays_date = moment.unix(time.time(), utc=True).locale('Asia/Kolkata').format("YYYY-MM-DD_HH-mm-ss")
    if os.path.isdir(log_dir):
        lst = os.listdir(log_dir)
        number_files = len(lst)
    else :
        os.makedirs(log_dir)
        number_files=0
    log_file_name="{}_{}_{}_{}".format(image_directory.split("/")[-1],"Phash",todays_date,number_files+1)
    log_file_path=os.path.join(log_dir,log_file_name)
    set_basic_config_for_logging(log_file_path)
    logger=get_logger_object_and_setting_the_loglevel()
    logger.info("Number of images in directory = "+str(len(os.listdir(log_dir))))
    logger.info('Duplicates = '+json.dumps(duplicates))
    logger.info("Unique Images = "+str(unique_images))
    logger.info("Length of Unique Images = "+str(len(unique_images)))
    logger.info("Duplicate Images = "+str(duplicate_images))
    logger.info("Length of Duplicate Images = "+str(len(duplicate_images)))
    logger.info("Same duplicates in more than One Images : "+json.dumps(SameDuplicatesInMoreThanOneImage))
    logger.debug("-----------------------------------------------------")
    logger.debug("/n/n")


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

    # __logging__(log_dir, image_directory, duplicates, unique_images, duplicate_images, SameDuplicatesInMoreThanOneImage)

if __name__== "__main__":


    # phasher = CNN()
    # phasher = WHash()
    # phasher = AHash()

    # phasher = DHash()
    image_directory='/home/ntlpt19/Downloads/Evaluation_Data/FinalEvaluationEvalData/COO/Images'
    output_path = '/home/ntlpt19/Downloads/Evaluation_Data/FinalEvaluationEvalData/COO/dedup_testing'
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
    
            # plot_duplicates(image_dir='/home/ntlpt-52/work/IDP/Auto_Label/Trade_Finance/Trade_Finance_Actual_Data/link_trade_finance/0_79',
            #         duplicate_map=duplicates,
            #         filename=key)



    #-------------------------------------------------------------Logging-------------------------------------------------------------------------------------------------

    # log_file_name="{}_{}_{}_{}".format(image_directory.split("/")[-1],"Phash",todays_date,number_files+1)
    # log_file_path=os.path.join(logs_directory,log_file_name)
    # set_basic_config_for_logging(log_file_path)
    # logger=get_logger_object_and_setting_the_loglevel()
    # logger.info("Number of images in directory = "+str(len(os.listdir(image_directory))))
    # logger.info('Duplicates = '+json.dumps(duplicates))
    # logger.info("Unique Images = "+str(unique_images))
    # logger.info("Length of Unique Images = "+str(len(unique_images)))
    # logger.info("Duplicate Images = "+str(duplicate_images))
    # logger.info("Length of Duplicate Images = "+str(len(duplicate_images)))
    # logger.info("Same duplicates in more than One Images : "+json.dumps(SameDuplicatesInMoreThanOneImage))
    # logger.debug("-----------------------------------------------------")
    # logger.debug("/n/n")