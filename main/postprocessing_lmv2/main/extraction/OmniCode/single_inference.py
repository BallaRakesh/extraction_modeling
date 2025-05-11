"""
* *********************************************************************************
* Number Theory S/W Pvt. Ltd CONFIDENTIAL                                      *
* *
* [2016] - [2023] Number Theory S/W Pvt. Ltd Incorporated                       *
* All Rights Reserved.                                                          *
* *
* NOTICE:  All information contained herein is, and remains                     *
* the property of Number Theory S/W Pvt. Ltd Incorporated and its suppliers,    *
* if any.  The intellectual and technical concepts contained                    *
* herein are proprietary to Number Theory S/W Pvt. Ltd Incorporated             *
* and its suppliers and may be covered by India. and Foreign Patents,           *
* patents in process, and are protected by trade secret or copyright law.       *
* Dissemination of this information or reproduction of this material            *
* is strictly forbidden unless prior written permission is obtained             *
* from Number Theory S/W Pvt. Ltd Incorporated.                                 *
* *
* *********************************************************************************
"""

"""
Author: Tarun Sharma
Date: Nov 21, 2023
"""

from transformers import LayoutLMv2ForTokenClassification
from transformers import LayoutLMv2Processor
from datetime import datetime
import os
from src.main.extraction.inference_utility_testing import image_result, pdf_result
from configparser import ConfigParser
import argparse
import glob
import psutil
from datetime import datetime

from src.main.extraction.utility import get_logger_object_and_setting_the_loglevel, \
set_basic_config_for_logging
from src.main.extraction.config.prod_mapping import product_code_map, document_code_map

logger = get_logger_object_and_setting_the_loglevel()
process_memory = psutil.Process()

if __name__ == "__main__":

    # Setting configuration for logging purposes   
    #####################################################################
    set_basic_config_for_logging(folder_path = "src/main", 
                                 filename = f'''inference_{"_".join(str(datetime.now()).split(" "))}''')
    logger = get_logger_object_and_setting_the_loglevel()

    process_memory = psutil.Process()
    start_time = datetime.now()
    cpu_utilization_start = psutil.cpu_percent()
    before_memory = process_memory.memory_info().rss
    logger.info("checkpoint 1 => setting basis cofiguration for logging purposes")
    # exit("+++++++++++++++++++")
    #####################################################################  


    ######################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--path', type =str, required = False, 
                        default='Images', help = "provide the folder name")
    parser.add_argument('-p', '--image', type =str, required = False, 
                        help = "provide the image path")
    parser.add_argument('-rt', '--result_type',type=str, required = True, 
                        default=True, help = "provide the result type")
    #######################################################################

    #######################################################################
    args = parser.parse_args()
    # product config
    product_config = ConfigParser()
    product_config.read("src/main/extraction/config/config.ini")

    prod_codes = product_config["OmniParameters"]["products"].split(",")
    doc_codes = [a.translate(str.maketrans({"(": "", 
                              ")": ""})).split(",") for a in 
                 product_config["OmniParameters"]["document_code"].split("||")]
    
    print(f"product codes: {prod_codes}")
    print(f"document codes: {doc_codes}")
    
    assert len(prod_codes) == len(doc_codes)
    
    
    for i, prod_code in enumerate(prod_codes):
        print(f"Product code: {prod_code}")
        for doc_code in doc_codes[i]:
            print(f"Document code: {doc_code}")
                    
        # data folder path
        product_wise_folder = ConfigParser()
        product_wise_folder.read("src/main/extraction/config/prod.ini")
        folder_path = product_wise_folder[prod_code][doc_code]

        print("==================Trade Finance Solutions===================")
        print("Product Code: {product_code}")
        print("Documenry Code: {doc_code}")
        print(f"folder_path: {folder_path}")
        #######################################################################

        # setting up some initial variables
        count = 0

        print("Loading Model...")
        t_start = datetime.now()
        logger.info(f"start time: {t_start}")
        cpu_utilization_start = psutil.cpu_percent()
        before_memory = process_memory.memory_info().rss

        # model_path = os.path.join(folder_path,"Best_Model_Airway")
        # model = LayoutLMv2ForTokenClassification.from_pretrained(
        #     pretrained_model_name_or_path=os.path.join(model_path, 'pytorch_model.bin'),
        #     config=os.path.join(model_path, 'config.json'), from_tf=True)

        model_path = glob.glob(f"{folder_path}/Best_Model*")
        print(f"Model path: {model_path}")
        assert len(model_path) == 1

        model_path = model_path[0]
        # model_path= os.path.join(folder_path, 'Best_Model')#"/home/ntlpt19/Downloads/MERGED_DATA/AIR_WAY/AIRWAY_v2/Best_Model_Airway"


        model = LayoutLMv2ForTokenClassification.from_pretrained(
                pretrained_model_name_or_path=os.path.join(model_path, 'pytorch_model.bin'),
                config=os.path.join(model_path, 'config.json'))

        processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", apply_ocr=False)
        #processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")
        device = 'cpu'
        model.to(device)

        t_end = datetime.now()
        cpu_utilization_end = psutil.cpu_percent()
        after_memory = process_memory.memory_info().rss
        cpu_utt = cpu_utilization_end - cpu_utilization_start
        memory_consumption = after_memory - before_memory
        logger.info("Time Taken for loading Model:", str(t_end - t_start))
        logger.info('RAM memory for Model loading used: % a', psutil.virtual_memory()[2])
        logger.info(f"cpu_utilization % for Model Loading:{str(cpu_utt)}")
        logger.info(
            f"memory_consumption in bytes for Loading Model:{str(memory_consumption)}"
        )
        logger.info("Loaded Model successfully")
        if args.image:
            file = args.image
            print(f"file name : {file}")
            print("Time Taken:", t_end - t_start)
            img_name = list(file.split('/'))[-1]
            logger.info("is img_name is instance of list? %s", isinstance(img_name, list))
            file_extension = os.path.splitext(img_name)[1]
            print(f"file_extension: {file_extension}")
            path01 = ''

            if file_extension == '.pdf':
                print('yes its a pdf')
                ans = pdf_result(file, path01, model, processor, device)
            else:
                ans = image_result(file, path01, model, processor, device)
            count += 1
        else:
            result_type = args.result_type
            path01 = f"{folder_path}/{result_type}_images"
            print("Time Taken:", t_end - t_start)

            for file in os.listdir(os.path.join(folder_path, path01)):
                file_extension = os.path.splitext(file)[1]
                print(f"file_extension: {file_extension}")

                if file_extension == '.pdf':
                    print('yes its a pdf')
                    ans = pdf_result(file, path01, model, processor, device)
                else:
                    ans = image_result(file, path01, model, processor, device)
                count += 1


        print("Total files processed:", count)
        print("***************Processed all the files!****************")
