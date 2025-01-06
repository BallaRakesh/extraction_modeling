import os
import pandas as pd

import json
# from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from datetime import datetime
import re
import time
from time import sleep

# load_dotenv()
# from langchain_ollama import ChatOllama
# llm = ChatOllama(model = "qwen2.5:7b",
#                 temperature=0)
                


# llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2-7B-Instruct"
# model_name = "Qwen/Qwen2.5-7B-Instruct"
# model_name = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4" #half download
# model_name = "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4" # not working
# model_name = "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8" # not working
 #"Qwen/Qwen2.5-7B-Instruct-GGUF"

# model_name =  '/datadrive/MistralModels/Base_Model/model_base'
from huggingface_hub import login
login("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")
 


device = "cuda"


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cuda"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def merge(query: str, structured_response: str, unstructured_response: str, doc_class):
    prompt_template = prompt_template_merge_extraction(query, structured_response, unstructured_response, doc_class)
        
    messages = [
        {"role": "system", "content": "You are an expert at Extraction of data form Trade Finance Document"},
        {"role": "user", "content": prompt_template}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=2000,
        do_sample=False, 
        temperature=0.0  # Adjust the temperature value here
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)

    # response = llm.invoke(prompt_template).content
    # print(response)
    
    return response


def prompt_template_merge_extraction(query: str, structured_response: str, unstructured_response: str, doc_class):
    return f"""
    'You are an expert in Trade-Finance document analysis. You will do analysis and answer question related to Trade-Finance documents {doc_class}'
    Given the context in the structured response, provide the 
    final response with respect to the unstructured query. If both responses do not have any 
    information within it, inform that you are unable to provide an answer as the 
    information is not available.

    Here's the query: {query}

    Sturctured responses: {structured_response}
    Unstructures responses: {unstructured_response}
    Give me the response in the following structure:
    {{
        "key1" : "value",
        "key2" : "value",
        "key3" : "value",
        // add more keys as per the requirement
    }}
    Response:
    """


def final_report(csv01, folder_path, doc_code, category_name, sheet_type, pandas_data = False):
    match_name = "Match/No_Match"
    if not pandas_data.empty:
        df = pandas_data
    else:
        df = pd.read_csv(os.path.join(csv01))
    report = {}
    label_names = []
    label_counts = []
    label_detected = []
    detection_accuracy = []
    average_accuracy = []
    matched_labels = []
    matched_detected = []
    total_match_percentage = []
    g = df.groupby("label_name")
    for name, name_df in g:
        label_names.append(name)
        label_counts.append(len(name_df.index))
        average_accuracy.append(round(name_df["Accuracy"].mean(), 2))
        matched = sum(list(name_df[match_name]))
        print(f'matched numbers: {matched}')
        matched_labels.append(matched)
        total_match_percentage.append(round((matched / len(name_df.index)) * 100, 2))
        not_detected = name_df["filter_prediction"].isnull().sum()
        detected = len(name_df.index) - not_detected
        print(f'detected numbers: {detected}')
        label_detected.append(detected)
        matched_detected.append(matched / detected)
        print(matched_detected)
        detection_accuracy.append((detected / len(name_df.index)) * 100)
    data = {'Label_Name': label_names, 'Label_Count': label_counts, "Labels_Detected": label_detected,
            "Detection_Accuracy": detection_accuracy, "Fuzzy_Match_Percentage": average_accuracy,
            "Complete_Match_Count_Detected": matched_detected, "Complete_Match_Count": matched_labels,
            "Complete_Match_Percentage": total_match_percentage}

    # Just to calculate detection accuracy
    detected = sum(label_detected)
    all_matched = sum(matched_labels)
    avg_matched_detected = all_matched / detected
    all_labels = sum(label_counts)
    overall_detection_accuracy = (detected / all_labels) * 100
    # dataframe and csv file generation
    report = pd.DataFrame(data)
    li = ["OVERALL", sum(list(report["Label_Count"])), sum(list(report["Labels_Detected"])), overall_detection_accuracy,
        df["Accuracy"].mean(), avg_matched_detected, sum(list(report["Complete_Match_Count"])),
        sum(list(report["Complete_Match_Count"])) / sum(list(report["Label_Count"]))]
    report.loc[len(report.index)] = li

    # if not os.path.exists(os.path.join(folder_path, 'result_path', category_name, f"{doc_code}_{datetime.now().date()}_{datetime.now().hour}")):
    #     os.makedirs(os.path.join(folder_path, 'result_path', category_name, f"{doc_code}_{datetime.now().date()}_{datetime.now().hour}"))

    if not os.path.exists(os.path.join(folder_path, f"{doc_code}_{datetime.now().date()}_{datetime.now().hour}")):
        os.makedirs(os.path.join(folder_path, f"{doc_code}_{datetime.now().date()}_{datetime.now().hour}"))


    name = f"final_report_{doc_code}_pre_{str(datetime.now())}_{sheet_type}.csv"

    # file_path_csv2 = os.path.join(folder_path, 'result_path', category_name, f"{doc_code}_{datetime.now().date()}_{datetime.now().hour}", name)
    file_path_csv2 = os.path.join(folder_path, f"{doc_code}_{datetime.now().date()}_{datetime.now().hour}", name)
    report.to_csv(file_path_csv2)

    # txt file generation
    # name = f"final_report_{datetime.now()}" + ".txt"
    # name_path = os.path.join(folder_path, 'result_path',category_name, f"{doc_code}_{datetime.now().date()}_{datetime.now().hour}",  name)
    
    # text_report = {
    #     row["Label_Name"]: {
    #         "Label_Count": row["Label_Count"],
    #         "Detection_Accuracy": row["Detection_Accuracy"],
    #         "Fuzzy_Match_Percentage": row["Fuzzy_Match_Percentage"],
    #         "Complete_Match_Count_Detected": row["Complete_Match_Count_Detected"],
    #         "Complete_Match_Count": row["Complete_Match_Count"],
    #         "Complete_Match_Percentage": row["Complete_Match_Percentage"],
    #     }
    #     for index, row in report.iterrows()
    # }
    
    # with open(name_path, "w") as f:
    #     json.dump(text_report, f)
    # f.close()

    return report

def extract_key_value_pairs(input_text):
    pairs = {}
    lines = input_text.strip().split('\n')
    current_key = None
    current_value = ""

    for line in lines:
        line = line.strip().strip('"')
        match = re.match(r'(.+?):\s*(.+?)$', line)

        if match:
            if current_key:
                pairs[current_key] = current_value.strip()
            current_key, current_value = match.groups()
            current_key = current_key.strip('"')
            current_value = current_value.strip('"').rstrip(',')
        elif current_key:
            current_value += " " + line.strip('"').rstrip(',')

    if current_key:
        pairs[current_key] = current_value.strip()

    return pairs

def extract_text_from_word_coords(word_coordinates: list):
    all_text = ' '.join([item['word'] for item in word_coordinates])
    return all_text

import logging

def setup_logger():
    # Create a logger
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)  # Set loggig level

    # Create a file handler for writing the logs
    file_handler = logging.FileHandler('log_file.log')
    file_handler.setLevel(logging.DEBUG)  # Log level for handler

    # Create a formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(file_handler)

    return logger

logger = setup_logger()

def write_log(message, level="info"):
    if level == "info":
        logger.info(message)
    elif level == "warning":
        logger.warning(message)
    elif level == "error":
        logger.error(message)
    elif level == "critical":
        logger.critical(message)





def get_low_match_labels(csv_file, threshold_val = 75, skip_keys = []):
    # Step 1: Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Step 2: Initialize an empty list to store the labels
    labels_below_threshold = []

    # Step 3: Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Check if the "Complete_Match_Percentage" is less than 70
        if row['Label_Name'] not in skip_keys:
            if row['Complete_Match_Percentage'] < threshold_val:
                # Append the corresponding "Label_Name" to the list
                labels_below_threshold.append(row['Label_Name'])
    
    # Step 4: Return the list of labels
    return labels_below_threshold


def open_text_file(file_path):
    # Open the file in read mode ('r')
    with open(file_path, 'r') as file:
        # Read the entire content of the file
        content = file.read()
    return content

def remove_special_chars_check(input_text):
    characters_to_remove = ".'·!'|:()/-%;'*,"
    cleaned_text = re.sub(f'^[{re.escape(characters_to_remove)}\s]+|[{re.escape(characters_to_remove)}\s]+$', '', str(input_text))
    return cleaned_text


def verify_prediction(actual_val, merged_result = None, mistral_key_value = None, mistral_flag = False, gpt_flag = False, both_results = False):
    if mistral_flag:
        if actual_val.lower() in str(merged_result).lower():
            return True
        else:
            return False
    elif gpt_flag:
        if actual_val.lower() in str(mistral_key_value).lower():
            return True
        else:
            return False
    else:
        if actual_val.lower() in str(merged_result).lower() or actual_val.lower() in str(mistral_key_value).lower():
            return True
        else:
            return False

def filter_keys(dataframe_, file_name_, all_keys, executed_json_file_data_):
    print('>>>>>>>>>>>>>>>>>>>', file_name_)
    print(dataframe_)
    filter_keys_list = []
    for keys_ in all_keys:
        filtered_df = dataframe_[(dataframe_['File_Name'] == file_name_) & (dataframe_['label_name'] == keys_)]
        if not filtered_df.empty:
            match_no_match = filtered_df['Match/No_Match'].iloc[0]
            print(match_no_match) #img_name+'_'+req_keys
            preprocessed_key = keys_.replace('_', ' ').replace('-', ' ')
            if match_no_match == 0 and file_name_ + '_' + preprocessed_key not in executed_json_file_data_:
                filter_keys_list.append(preprocessed_key)
    return filter_keys_list

# Save the prompt template to a text file
def save_gpt_result(img_, folder_path, info):
    with open(os.path.join(folder_path, img_+'.txt'), 'w') as file:
        file.write(info)

def open_gpt_result(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content

def get_value_case_insensitive(json_obj, key, key2, img_name):
    # Normalize the key
    key = key.lower()
    # Create a normalized key mapping
    # normalized_keys = {k.lower().replace('\', ').replace('/', ''): v for k, v in json_obj.items()}
    normalized_keys = {k.lower().replace('\\', '').replace('/', ''): v for k, v in json_obj.items()}

    # Fetch the value using the normalized key

    with open(os.path.join(result_save_filter, img_name+'.txt'), 'w') as file:
        file.write(str(normalized_keys))
    normalized_keys.get(key, '')

    return 


if __name__ == '__main__':
    # df_temp = pd.read_csv(os.path.join('/home/ntlpt19/itf_results_field_wise_report/BOL/filtered_file_combined_manual_analysis.csv'))
    # out_put_df = final_report('', '/home/ntlpt19/itf_results_field_wise_report/BOL/manual_analysis', 'bol', 'label_wise', 'combined', pandas_data=df_temp)
    
    # exit('OK')
    #COO
    # base_folder = '/home/gpu1admin/rakesh/COO'
    # mistral_results = ''
    # csv_file_path = '/home/gpu1admin/rakesh/COO/Raw_files/final_report_coo_pre_2024-02-22 11:28:26.538452_after_fuzzy_match_change.csv'
    # label_wise_keys = '/home/gpu1admin/rakesh/COO/Raw_files/coo_analysis_pre_valid_after_fuzzy_match_post-processing_latest.csv'
    # ocr_folder = '/home/gpu1admin/rakesh/COO/Raw_files/OCR'
    
    # # CS
    # base_folder = '/home/gpu1admin/rakesh/CS'
    # mistral_results = ''
    # csv_file_path = '/home/gpu1admin/rakesh/CS/Raw_files/final_report_cs_pre_2024-02-22 11:38:43.043781_after_fuzzy_match_change.csv'
    # label_wise_keys = '/home/gpu1admin/rakesh/CS/Raw_files/cs_analysis_pre_valid_after_fuzzy_match_post-processing_latest.csv'
    # ocr_folder = '/home/gpu1admin/rakesh/CS/Raw_files/OCR'
    
    #PL
    # base_folder = '/home/gpu1admin/rakesh/PL'
    # mistral_results = ''
    # csv_file_path = '/home/gpu1admin/rakesh/PL/Raw_files/final_report_pl_pre_2024-02-22 11:56:36.030137_after_fuzzy_match_change.csv'
    # label_wise_keys = '/home/gpu1admin/rakesh/PL/Raw_files/pl_analysis_pre_valid_after_fuzzy_match_post-processing_latest.csv'
    # ocr_folder = '/home/gpu1admin/rakesh/PL/Raw_files/OCR'
    
    #BOL
    # base_folder = '/home/gpu1admin/rakesh/BOL'
    # mistral_results = ''
    # csv_file_path = '/home/gpu1admin/rakesh/BOL/Raw_files/final_report_bol_pre_2024-03-04 18:34:04.019791_after_fuzzy_match_change.csv'
    # label_wise_keys = '/home/gpu1admin/rakesh/BOL/Raw_files/bol_analysis_pre_valid_after_fuzzy_match_post-processing_latest.csv'
    # ocr_folder = '/home/gpu1admin/rakesh/BOL/Raw_files/OCR'
    
    #boe
    # base_folder = '/home/gpu1admin/rakesh/BOE'
    # mistral_results = '/home/ntlpt19/LLM_training/EVAL/PL/all_results/results/text_files'
    # csv_file_path = '/home/gpu1admin/rakesh/BOE/Raw_Files/final_report.csv'
    # label_wise_keys = '/home/gpu1admin/rakesh/BOE/Raw_Files/boe_analysis_pre_valid_after_fuzzy_match_post-processing_latest.csv'
    # ocr_folder = '/home/gpu1admin/rakesh/BOE/Raw_Files/OCR'
    
    #PI
    # base_folder = '/home/gpu1admin/rakesh/PI'
    # mistral_results = ''
    # csv_file_path = '/home/gpu1admin/rakesh/PI/Raw_files/final_report_pi_pre_2024_06_20_15_08_54_441573_after_fuzzy_match_change.csv'
    # label_wise_keys = '/home/gpu1admin/rakesh/PI/Raw_files/pi_analysis_pre_valid_after_fuzzy_match_post_processing_latest.csv'
    # ocr_folder = '/home/gpu1admin/rakesh/PI/Raw_files/OCR'

    #PO
    # base_folder = '/home/gpu1admin/rakesh/PO'
    # mistral_results = ''
    # csv_file_path = '/home/gpu1admin/rakesh/PO/Raw_files/final_report_po_pre_2024_06_10_19_11_16_928624_after_fuzzy_match_change.csv'
    # label_wise_keys = '/home/gpu1admin/rakesh/PO/Raw_files/po_analysis_pre_valid_after_fuzzy_match_post_processing_latest.csv'
    # ocr_folder = '/home/gpu1admin/rakesh/PO/Raw_files/OCR'
    
    #IC
    # base_folder = '/home/gpu1admin/rakesh/IC'
    # mistral_results = ''
    # csv_file_path = '/home/gpu1admin/rakesh/IC/Raw_files/final_report_ic_pre_2024_06_21 11_38_53.349156_after_fuzzy_match_change.csv'
    # label_wise_keys = '/home/gpu1admin/rakesh/IC/Raw_files/ic_analysis_pre_valid_after_fuzzy_match_post_processing_latest.csv'
    # ocr_folder = '/home/gpu1admin/rakesh/IC/Raw_files/OCR'
    #AWB
    base_folder = '/home/gpu1admin/rakesh/AWB'
    mistral_results = ''
    csv_file_path = '/home/gpu1admin/rakesh/AWB/Raw_files/final_awb_base_label.csv'
    label_wise_keys = '/home/gpu1admin/rakesh/AWB/Raw_files/awb_analysis_pre_valid_after_fuzzy_match_post-processing_latest.csv'
    ocr_folder = '/home/gpu1admin/rakesh/AWB/Raw_files/OCR'
    
    model_using = 'llama_7B'
    itter_number = 'v2'
    root_folder = os.path.join(base_folder, f'results_{model_using}_{itter_number}')
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    
    result_save_path = os.path.join(root_folder,'results')
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)
        
    result_save_filter = os.path.join(root_folder,'results_filter')
    if not os.path.exists(result_save_filter):
        os.makedirs(result_save_filter)
    
    reports_gen_path = os.path.join(root_folder,'Reports')
    if not os.path.exists(reports_gen_path):
        os.makedirs(reports_gen_path)
        
    only_mistral_flag = False
    only_gpt_flag = False
    both_results = True
    executed_json_file = ''
    if os.path.exists(executed_json_file):
        with open(executed_json_file, 'r') as json_file:
            executed_json_file_data = json.load(json_file)
    else:
        executed_json_file_data = {}
    
    master_exception_keys = ['signature', 'OVERALL', 'stamp', 'signed_by_carrier', 'signed_By_agent', 'certificate_stamped', 'signed_stamp']
    exception_keys = {
        'CI': ['signature']
    }
    threshold_ = 70
    document_type = 'CI' #no need to change
    
    
    # document_class = 'Commercial Invoice'
    # document_class = 'Bill of Exchange'
    # document_class = 'Bill of Lading'
    # document_class = 'Perfoma Invoice'
    
    # document_class = 'Certificate of Origin'
    # document_class = 'Covering Schedule'
    # document_class = 'Packing List'
    
    # document_class = 'Purchase Order'
    # document_class = 'Insurance Certificate'
    document_class = 'AirWay Bill'
    
    
    document_class_com = document_class.replace(' ', '_')
    executed_file = {}
    
    filtered_dfs = []
  
    
    df = pd.read_csv(label_wise_keys)
    

    df_mistral = df.copy() 
    df_gpt = df.copy() 
    
    df['mistral_result'] = None
    df['gpt_result'] = None
    
    df_mistral['mistral_result'] = None
    df_gpt['gpt_result'] = None

    df['filter_prediction'] = df['predicted']
    df_mistral['filter_prediction'] = df['predicted']
    df_gpt['filter_prediction'] = df['predicted']
    
    low_match_labels = get_low_match_labels(csv_file_path, threshold_val = threshold_, skip_keys = exception_keys.get(document_type, [])+ master_exception_keys)
    print(low_match_labels)
    df_keys_match = pd.read_csv(csv_file_path)
    required_df_base = df_keys_match[df_keys_match['Label_Name'].isin(low_match_labels)]
    filtered_dfs.append(required_df_base[['Label_Name', 'Label_Count', 'Labels_Detected', 'Complete_Match_Count', 'Complete_Match_Percentage']])
    
    filename = os.path.join(root_folder, f"{document_class_com}_low_match_labels.txt")
    with open(filename, 'w') as file:
        for label in low_match_labels:
            file.write(f"{label}\n")
            
    
    unique_values_list = df['File_Name'].unique().tolist()
    print(unique_values_list)

    # for req_keys in low_match_labels:
    #     print(req_keys)
    #     filtered_df = df[df['label_name'] == req_keys]
    
    missing_ocr = {}
    for img_name in unique_values_list: #filtered_df['File_Name']:
        filename_without_extension = os.path.splitext(img_name)[0]
        print(filename_without_extension)
        keys_with_filter = filter_keys(df, img_name, low_match_labels, executed_json_file_data)
        print(keys_with_filter)
        
        # match_no_Match = df[(df['File_Name'] == img_name) & (df['label_name'] == req_keys)]['Match/No_Match'].iloc[0]
        # if match_no_Match == 0 and img_name+'_'+req_keys not in executed_json_file_data:
        # try:
        if any(keys_with_filter):
            ocr_file = os.path.join(ocr_folder, filename_without_extension + '_text.txt')
            print(ocr_file)
            if os.path.exists(ocr_file):
                all_text = extract_text_from_word_coords(eval(open_text_file(ocr_file)).get('word_coordinates', []))
                print(all_text)
            else:
                print(f'File {ocr_file} does not exist.')
                missing_ocr[str(ocr_file)] = True
                continue
            # preprocessed_key = req_keys.replace('_', ' ').replace('-', ' ')
            
            query = f'Find the ocr information for the {img_name} and extract the {keys_with_filter} ?'
            structured_response = all_text
            unstructured_response = f'extract the {keys_with_filter}'
            if os.path.exists(os.path.join(mistral_results, filename_without_extension+'.txt')):
                mistral_sample_result = open_text_file(os.path.join(mistral_results, filename_without_extension+'.txt'))
                # print(mistral_sample_result)
                mistral_sample_result_json = extract_key_value_pairs(mistral_sample_result)
                
                # mistral_key_value = mistral_sample_result_json.get(req_keys, '') + mistral_sample_result_json.get(preprocessed_key, '')
                # print(mistral_key_value)
            else:
                mistral_sample_result_json = {}
                # Find the ocr information for the document name pdf10_4.png and extract the your order no?
            if os.path.exists(os.path.join(result_save_path, filename_without_extension+".txt")):
                merged_result_str = open_gpt_result(os.path.join(result_save_path, filename_without_extension+".txt"))
            else:
                start_time = time.time()
                merged_result_str = merge(query, structured_response, unstructured_response, document_class)
                
                end_time = time.time()
                elapsed_time = end_time - start_time
                write_log(f"Yes Execuated for Image {img_name}", level="info")
                write_log(f"Time taken for Generating the result: {elapsed_time}", level="info")
                print(img_name)
                print(merged_result_str)
                # exit('OK')
                save_gpt_result(filename_without_extension, result_save_path, merged_result_str)
                # merged_result_json = eval(merged_result_str)
                # exit('OK')
                
                
            try:
                merged_result_json = eval(merged_result_str)
            except:
                merged_result_json = extract_key_value_pairs(merged_result_str)
                
            print(merged_result_json)
            for required_keys in keys_with_filter:
                
                req_keys = required_keys.replace(' ', '_')
                
                # merged_result = gpt_df[(gpt_df['File_Name'] == img_name) & (gpt_df['label_name'] == req_keys)]['predicted'].iloc[0]
                
                # merged_result = merged_result_json.get(required_keys, '')
                
                merged_result = get_value_case_insensitive(merged_result_json, required_keys,req_keys, img_name+'_'+req_keys)
                
                executed_file[img_name+'_'+req_keys] = str(merged_result)
                
                mistral_key_value = mistral_sample_result_json.get(req_keys, '') + mistral_sample_result_json.get(req_keys.replace('_', ' '), '')
                
                # req_keys.replace('_', ' ').replace('-', ' ')
                
                actual_val_df = df[(df['File_Name'] == img_name) & (df['label_name'] == req_keys)]#['actual'].iloc[0]
                if not actual_val_df.empty:
                    actual_val = actual_val_df['actual'].iloc[0]
                    actual_val = str(remove_special_chars_check(actual_val))
                    
                    if not pd.isna(actual_val):
                        # if both_results or only_gpt_flag:
                        df.loc[(df['File_Name'] == img_name) & (df['label_name'] == req_keys), 'gpt_result'] = str(merged_result)
                        # if both_results or only_mistral_flag:
                        df.loc[(df['File_Name'] == img_name) & (df['label_name'] == req_keys), 'mistral_result'] = str(mistral_key_value)
                      
                        df_gpt.loc[(df_gpt['File_Name'] == img_name) & (df_gpt['label_name'] == req_keys), 'gpt_result'] = str(merged_result)
                        # if both_results or only_mistral_flag:
                        df_mistral.loc[(df_mistral['File_Name'] == img_name) & (df_mistral['label_name'] == req_keys), 'mistral_result'] = str(mistral_key_value)
                           
                        # if only_mistral_flag:
                        if mistral_key_value:
                            df.loc[(df['File_Name'] == img_name) & (df['label_name'] == req_keys), 'filter_prediction'] = str(mistral_key_value)
                            df_mistral.loc[(df_mistral['File_Name'] == img_name) & (df_mistral['label_name'] == req_keys), 'filter_prediction'] = str(mistral_key_value)
                        else:
                            df.loc[(df['File_Name'] == img_name) & (df['label_name'] == req_keys), 'filter_prediction'] = str(merged_result)
                        df_gpt.loc[(df_gpt['File_Name'] == img_name) & (df_gpt['label_name'] == req_keys), 'filter_prediction'] = str(merged_result)
                            
                        # if verify_prediction(actual_val, merged_result = merged_result, mistral_key_value = mistral_key_value, mistral_flag = only_mistral_flag, gpt_flag = only_gpt_flag, both_results = both_results):
                        
                        if actual_val.lower() in str(merged_result).lower() or actual_val.lower() in str(mistral_key_value).lower():
                            # if actual_val.lower() in str(mistral_key_value).lower():
                            # if actual_val.lower() in str(merged_result).lower():
                            df.loc[(df['File_Name'] == img_name) & (df['label_name'] == req_keys), 'Accuracy'] = 100
                            df.loc[(df['File_Name'] == img_name) & (df['label_name'] == req_keys), 'Match/No_Match'] = 1 
                            
                        if actual_val.lower() in str(mistral_key_value).lower():  
                            df_mistral.loc[(df_mistral['File_Name'] == img_name) & (df_mistral['label_name'] == req_keys), 'Accuracy'] = 100
                            df_mistral.loc[(df_mistral['File_Name'] == img_name) & (df_mistral['label_name'] == req_keys), 'Match/No_Match'] = 1 
                            
                        if actual_val.lower() in str(merged_result).lower(): 
                            df_gpt.loc[(df_gpt['File_Name'] == img_name) & (df_gpt['label_name'] == req_keys), 'Accuracy'] = 100
                            df_gpt.loc[(df_gpt['File_Name'] == img_name) & (df_gpt['label_name'] == req_keys), 'Match/No_Match'] = 1 
        # except Exception as e:
        #     print('ERROR OCCURING IN THE EXECUTION ')
        #     print(e)
        #     with open('executed_file1.json', 'w') as json_file:
        #         json.dump(executed_file, json_file, indent=4)  
                            
    df.to_csv(os.path.join(root_folder, 'filtered_file_combined.csv'), index=False)
    df_mistral.to_csv(os.path.join(root_folder, 'filtered_file_mistral.csv'), index=False)
    df_gpt.to_csv(os.path.join(root_folder, 'filtered_file_gpt.csv'), index=False)
    
    # for data_frame, sheet_type in zip([df, df_mistral, df_gpt], ['combine', 'mistral', 'gpt']):
    #     out_put_df = final_report('', os.path.join(root_folder, reports_gen_path), document_class_com, 'label_wise', sheet_type, pandas_data=data_frame)
    
    
        # List of DataFrames and sheet types
    data_frames = [df, df_mistral, df_gpt]
    sheet_types = ['combine', 'mistral', 'gpt']

    for data_frame, sheet_type in zip(data_frames, sheet_types):
        out_put_df = final_report('', os.path.join(root_folder, reports_gen_path), document_class_com, 'label_wise', sheet_type, pandas_data=data_frame)
        filtered_df = out_put_df[out_put_df['Label_Name'].isin(low_match_labels)]
        filtered_df = filtered_df[['Label_Name', 'Label_Count', 'Labels_Detected', 'Complete_Match_Count', 'Complete_Match_Percentage']]
        filtered_dfs.append(filtered_df)

    # Combine all filtered DataFrames
    combined_df = pd.concat(filtered_dfs, ignore_index=True)

    # Save combined DataFrame to CSV
    combined_csv_path = os.path.join(root_folder, 'combined_low_match_labels.csv')
    combined_df.to_csv(combined_csv_path, index=False)

    print(f"Combined CSV file saved to {combined_csv_path}")
    
    with open(os.path.join(root_folder, f'{document_class_com}_executed_file1.json'), 'w') as json_file:
        json.dump(executed_file, json_file, indent=4)  

    with open(os.path.join(root_folder, f'{document_class_com}_missing_ocr.json'), 'w') as json_file:
        json.dump(missing_ocr, json_file, indent=4)  
