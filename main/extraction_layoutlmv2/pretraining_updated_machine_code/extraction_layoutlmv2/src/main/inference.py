"""
* -----------------------------------------------------------------------------------------
*                              NEWGEN SOFTWARE TECHNOLOGIES LIMITED
*
* Group: Number Theory
* Product/Project: Intelligent-Trade-Finance
* Module:
* File Name:
* Author: Tarun Sharma
* Date(DD/MM/YYYY):
* Description: Create dynamic component and layout
*
* -----------------------------------------------------------------------------------------
*                              CHANGE HISTORY
* -----------------------------------------------------------------------------------------
* Date(DD/MM/YYYY)               Change By              Change Description(Bug No.(If Any))
* -----------------------------------------------------------------------------------------
* 24/05/2020                     Shailesh Bist           create dynamic component and layoutlinkData Array
* 06/03/2020                     Shailesh Bist           sidebar Option implemented according to user or admin
"""

from transformers import LayoutLMv2ForTokenClassification
from transformers import LayoutLMv2Processor
import os
import shutil

# from inference_utility_testing import get_logger_object_and_setting_the_loglevel, \
# set_basic_config_for_logging, image_result

from inference_utility_without_processor import get_logger_object_and_setting_the_loglevel, \
set_basic_config_for_logging, image_result

# from step4_result_generation import image_result
from configparser import ConfigParser
import argparse
import glob
import psutil
from datetime import datetime
# from training.lmv2code.src.main.extraction.utility import get_logger_object_and_setting_the_loglevel, \
# set_basic_config_for_logging
#from training.lmv2code.src.main.extraction.config.prod_mapping import product_code_map, document_code_map

from config.prod_mapping import product_code_map, document_code_map

logger = get_logger_object_and_setting_the_loglevel()
process_memory = psutil.Process()

def segregating_test_imgs(root_folder):
    file_names_file = os.path.join(root_folder, 'test.txt')
    with open(file_names_file, 'r') as file:
        file_names = [line.strip() for line in file]
    print(file_names)
    source_folder = os.path.join(root_folder, 'Images')
    destination_folder = os.path.join(root_folder, 'test_images')
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        
    for file_name in file_names:
        base_name = os.path.splitext(file_name)[0]
        image_file_name = base_name + ".png"
        source_file_path = os.path.join(source_folder, image_file_name)
        destination_file_path = os.path.join(destination_folder, image_file_name)
        if os.path.exists(source_file_path):
            shutil.copy(source_file_path, destination_file_path)
            print(f"Copied {image_file_name} to {destination_folder}")
        else:
            print(f"Image file {image_file_name} not found for {file_name}")
    return 'test_images'



import os
import json
import torch
from transformers import LayoutLMv2Config, LayoutLMv2ForTokenClassification

def load_model_components(backbone_path, classifier_path, num_labels=None):
    """
    Load backbone and classifier components and reconstruct the full model for token classification
    
    Args:
        backbone_path (str): Path to the saved backbone components
        classifier_path (str): Path to the saved classifier components
        num_labels (int, optional): Number of labels if not found in classifier info
    
    Returns:
        LayoutLMv2ForTokenClassification: Reconstructed model
    """
    # Load backbone configuration
    backbone_config = LayoutLMv2Config.from_pretrained(backbone_path)
    
    # Load backbone and classifier info
    with open(os.path.join(backbone_path, 'backbone_info.json'), 'r') as f:
        backbone_info = json.load(f)
    
    with open(os.path.join(classifier_path, 'classifier_info.json'), 'r') as f:
        classifier_info = json.load(f)
    
    # Determine number of labels
    num_labels = num_labels or classifier_info.get('num_labels')

    if num_labels is None:
        raise ValueError("Number of labels must be provided either in classifier info or as an argument")
    
    # Set configuration
    backbone_config.num_labels = num_labels
    
    # Create model instance
    model = LayoutLMv2ForTokenClassification(backbone_config)
    
    # Load weights
    backbone_weights = torch.load(os.path.join(backbone_path, 'backbone.pth'))
    classifier_weights = torch.load(os.path.join(classifier_path, 'classifier.pth'))
    
    # Load backbone weights
    state_dict = model.state_dict()
    for k, v in backbone_weights.items():
        if k in state_dict and state_dict[k].shape == v.shape:
            state_dict[k].copy_(v)
    
    # Get shapes for classifier weights
    current_classifier_shape = state_dict['classifier.weight'].shape  # Should be [num_labels, hidden_size]
    classifier_w = classifier_weights['classifier.weight']
    classifier_b = classifier_weights['classifier.bias']
    
    if current_classifier_shape[1] != classifier_w.shape[1]:
        print(f"Adapting classifier weights from shape {classifier_w.shape} to {current_classifier_shape}")
        
        # For TokenClassification, we need to handle the hidden size properly
        # The default hidden size for LayoutLMv2 token classification is 768
        if current_classifier_shape[1] == 768:
            # If the target model expects 768 dimensions, we need to potentially resize
            if classifier_w.shape[1] > 768:
                # Take only the first 768 dimensions if source is larger
                adapted_w = classifier_w[:, :768]
            else:
                # Pad with zeros if source is smaller
                adapted_w = torch.zeros(current_classifier_shape, device=classifier_w.device)
                adapted_w[:, :classifier_w.shape[1]] = classifier_w
        else:
            # For other cases, we'll need to resize appropriately
            adapted_w = torch.zeros(current_classifier_shape, device=classifier_w.device)
            # Copy the weights to the first section
            min_dim = min(classifier_w.shape[1], current_classifier_shape[1])
            adapted_w[:, :min_dim] = classifier_w[:, :min_dim]
    else:
        adapted_w = classifier_w

    # Update state dict with adapted weights
    state_dict['classifier.weight'].copy_(adapted_w)
    state_dict['classifier.bias'].copy_(classifier_b)
    
    # Load the complete state dict
    model.load_state_dict(state_dict)
    return model

def quick_load_model(base_save_path, num_labels=None):
    """
    Convenience method to load model components from a base directory
    
    Args:
        base_save_path (str): Base path containing backbone and classifier subdirectories
        num_labels (int, optional): Number of labels for the model
    
    Returns:
        LayoutLMv2ForTokenClassification: Loaded model
    """
    backbone_path = os.path.join(base_save_path, 'backbone')
    classifier_path = os.path.join(base_save_path, 'classifier')
    return load_model_components(backbone_path, classifier_path, num_labels)


if __name__ == "__main__":

    # Setting configuration for logging purposes   
    #####################################################################
    set_basic_config_for_logging(filename = f'''inference_{"_".join(str(datetime.now()).split(" "))}''', folder_path = "src/main")
    logger = get_logger_object_and_setting_the_loglevel()

    process_memory = psutil.Process()
    start_time = datetime.now()
    cpu_utilization_start = psutil.cpu_percent()
    before_memory = process_memory.memory_info().rss
    logger.info("checkpoint 1 => setting basis cofiguration for logging purposes")
    #####################################################################


    ######################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--path', type =str, required = False, 
                        default='Images', help = "provide the folder name")
    parser.add_argument('-t', '--test', type =str, required = False, 
                    default = False, help = "provide the folder name")
    parser.add_argument('-p', '--image', type =str, required = False, 
                        help = "provide the image path")
    #######################################################################

    #######################################################################
    args = parser.parse_args()
    # product config
    product_config = ConfigParser()
    product_config.read("./config/config.ini")

    prod_code = product_code_map[product_config["Product"]["code"]]
    doc_code = product_config["Product"]["document_code"]
    if '[' in doc_code:
        doc_elements = doc_code[1:-1].split(', ')
        # Convert elements to a Python list
        doc_code_list = [element.strip() for element in doc_elements]
    print(doc_code)
    print(doc_code_list)
    print(prod_code)
    # data folder path
    product_wise_folder = ConfigParser()
    product_wise_folder.read("./config/prod.ini")

    # doc_code = 'lc'


    for doc_code_ in doc_code_list:
        doc_code = document_code_map[doc_code_]
        folder_path = product_wise_folder[prod_code][doc_code]

        print("==================Trade Finance Solutions===================")
        print("Product Code: {prod_code}")
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



        # model_path = glob.glob(f"{folder_path}/Best_Model*")
        folder_path = '/home/data_science/geo_testing/COO_V3/eval_data'
        model_path1 = '/home/data_science/geo_testing/COO_V3'
        model_path2 = '/home/data_science/geo_testing/COO_V3/Best_Model'
        classes_txt_path = '/home/data_science/geo_testing/COO_V3/classes.txt'
        print(f"Model path: {model_path1}")
        # assert len(model_path) == 1
        # model_path = model_path[0]
        # model_path= os.path.join(folder_path, 'Best_Model')#"/home/ntlpt19/Downloads/MERGED_DATA/AIR_WAY/AIRWAY_v2/Best_Model_Airway"


        # model = LayoutLMv2ForTokenClassification.from_pretrained(
        #         pretrained_model_name_or_path=os.path.join(model_path, 'pytorch_model.bin'),
        #         config=os.path.join(model_path, 'config.json'))
        ###############################################################
        ###############################################################
        ###############################################################
        model = LayoutLMv2ForTokenClassification.from_pretrained(model_path2)
        ###############################################################
        ###############################################################
        # model = quick_load_model(model_path1)
        ###############################################################
        ###############################################################
        processor = ''
        # processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", apply_ocr=False)
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
                exit("+++++++++++++++")
                ans = pdf_result(file, path01, model, processor, device)
            else:
                ans = image_result(file, path01, model, processor, device, folder_path,label_path = classes_txt_path)
            count += 1
        else:
            if args.test:
                path01 = segregating_test_imgs(folder_path)
            else:
                path01 = args.path
            print("Time Taken:", t_end - t_start)

            for file in os.listdir(os.path.join(folder_path, path01)):
                file_extension = os.path.splitext(file)[1]
                print(f"file_extension: {file_extension}")

                if file_extension == '.pdf':
                    print('yes its a pdf')
                    exit("++++++++++++")
                    ans = pdf_result(file, path01, model, processor, device)
                else:
                    ans = image_result(file, path01, model, processor, device, folder_path, label_path = classes_txt_path)
                count += 1


        print("Total files processed:", count)
        print("***************Processed all the files!****************")
