import glob
import os
import json
import pandas as pd
from fuzzywuzzy import fuzz
import re

from post_processing_master_gt import generic_page_no
# from post_processing_pred import pp_cash_drawn_rules

from typing import List
from configparser import ConfigParser
from validation_utility import final_overall_analysis, final_report, stp_report

from config.prod_mapping import product_code_map, document_code_map 
from datetime import datetime

""" This script used to generate accuracy generation of the dataset and 
used documents like PL, CS, COO, BOL"""


import re
from fuzzywuzzy import fuzz
from geonamescache import GeonamesCache
from difflib import get_close_matches
import pycountry


def pp_cash_drawn_rules(x):
    """
    just take 522 or 600 from the string
    :param x:
    :return:
    """
    x = str(x)
    pattern: str = "522|600"
    match1 = re.search(pattern, x)
    return match1[0] if match1 else ""

def remove_symbols(text):
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = cleaned_text.strip()  # Strip spaces from the start and end of the text
    return cleaned_text

def extract_country_from_text(text):
    text = remove_symbols(text)
    gc = GeonamesCache()
    countries = gc.get_countries_by_names()
    country_names = [country["name"] for country in countries.values()]
    country_names += [country.alpha_3 for country in pycountry.countries]
    country_names += [country.alpha_2 for country in pycountry.countries]
    country_names.append("UAE")
    matching_countries = []
    for country in country_names:
        if fuzz.partial_ratio(country.lower(), text.lower()) >= 60:
            matching_countries.append(country.lower() if len(country.lower())>2 else "")

            
    words = re.findall(r'\b\w+\b', text)
    address_list = []
    for i in range(len(words)):
        for j in range(i + 1, len(words) + 1):
            # Combine consecutive words
            combined_words = ' '.join(words[i:j])
            if combined_words.lower() in matching_countries:
                #print('combined_words', combined_words)     
                address_list.append(combined_words)
    return address_list
def remove_start_end_spl_char(actual_text: str, pred_text:str):
	print(f'actual text : {actual_text}')
	print(f'pred text : {pred_text}')

	actual_text= actual_text.split(' ')
	pred_text= pred_text.split(' ')
	print(actual_text)
	print(pred_text)
	if len(actual_text)== len(pred_text):
		for actual, pred in zip(actual_text, pred_text):
			print('length of actual and predicted is same+++++++++')
			print(actual)
			print(pred)
			
			actual_indics= actual_text.index(actual)
			print(actual_indics)
			pred_indices= pred_text.index(pred)
			print(pred_indices)
			if actual_indics==0:
				a_text= "".join(ch for ch in actual if ch.isalnum() or ch=='.')
				actual_text[actual_indics]= a_text
			if actual_indics==len(actual_text)-1:
				a_text= "".join(ch for ch in actual if ch.isalnum())
				actual_text[actual_indics]= a_text

			if pred_indices==0:
				p_text="".join(ch for ch in pred if ch.isalnum() or ch== '.')
				pred_text[pred_indices]= p_text
			if pred_indices== len(pred_text)-1:
				p_text="".join(ch for ch in pred if ch.isalnum())
				pred_text[pred_indices]= p_text

	else:
		for word in pred_text:
			word_indices=  pred_text.index(word)
			if word_indices==0 :
				a_text= "".join(ch for ch in word if ch.isalnum() or ch=='.')
				pred_text[word_indices]= a_text
			elif word_indices== len(pred_text)-1:
				a_text= "".join(ch for ch in word if ch.isalnum())
				pred_text[word_indices]= a_text
			else:
				pass
		for word in actual_text:
			word_indices=  actual_text.index(word)
			if word_indices==0 :
				a_text= "".join(ch for ch in word if ch.isalnum() or ch=='.')
				actual_text[word_indices]= a_text
			elif word_indices== len(actual_text)-1:
				a_text= "".join(ch for ch in word if ch.isalnum())
				actual_text[word_indices]= a_text
			else:
				pass
		

	actual_text= " ".join(actual_text)
	pred_text= " ".join(pred_text)
	return actual_text,pred_text

def remove_spl_char_multiple_pred(text: str):
	text= text.split(' ')
	for word in text:
		word_indices= text.index(word)
		if word_indices==0 or word_indices==len(text)-1:
			a_text= "".join(ch for ch in word if ch.isalnum())
			text[word_indices]= a_text
	actual_text= " ".join(text)
	print(f'actual text: {actual_text}')
	return actual_text
	




def remove_alphabets(text):
	text = re.sub(r'[a-zA-Z]', '', text)
	return text

def preprocess_incoterm(text : str, incoterm_list: List):
  term = text[0]
  text =text.split(' ')
  for word in text:
    if word.upper() in incoterm_list:
      term= word
    else:
      pass
    if term  is not None:
      return term

	






# def remove_spl_char(actual_text: str, pred_text:str):
#     print(f'actual text : {actual_text}')
#     print(f'pred text : {pred_text}')

#     actual_text= actual_text.split('')
#     pred_text= pred_text.split('')

#     for actual, pred in (actual_text, pred_text):
#         actual_indics= actual_text.index(actual)
#         pred_indices= pred_text.index(pred)

#         a_text="".join(ch for ch in actual if ch.isalnum())
#         p_text="".join(ch for ch in pred if ch.isalnum())

#         actual_text[actual_indics]= a_text
#         pred_text[pred_indices]= p_text


        
        


def append_values(new_row, actual_value, predicted, to_do):
	print("value1 is", actual_value)
	print("list of predicted values are", predicted)
	print("to_do is", to_do)
	l2 = len(predicted)
	for j in range(l2):
		# print(" j is", str(j))
		intersection = get_iou_new(actual_value[1], predicted[j][1])
		if intersection > 0.25:
			# print("match found")
			print(f'entered after intersection {actual[i][0]}')
			exit()
			new_row.append(actual[i][0])
			new_row.append(predicted_label[j][0])
			accuracy = fuzz.ratio(str(new_row[2]).lower(), str(new_row[3]).lower())
			new_row.append(accuracy)
			if accuracy == 100:
				new_row.append(1)
			else:
				new_row.append(0)
			new_row.append(predicted_label[j][2])
			new_row.append(actual_value[1])
			data.append(new_row)
			try:
				to_do.remove(j)
			except:
				pass
			# print("going to  break")
			break
	else:
		# print("else executed")
		# print(new_row)
		new_row.append(actual_value[0])
		new_row.append("")
		new_row.append(0)
		new_row.append(0)
		new_row.append(0)
		new_row.append(actual_value[1])
		data.append(new_row)
	# print("new_to_do is",to_do)
	return to_do





def filter_prediction(new_row, actual_value, predicted, to_do, key, flag01):
	l2 = len(predicted)
	print(f'The actual value insider filter: {actual_value[0].lower()}')
	print(f'the prediction value: {predicted[0][0]}')
	print(f' to-do value: {to_do}')
	flag= True
	for j in range(l2):
		if actual_value[0].lower()== predicted[j][0].lower() and flag==True:
			print('entered into filter +++++++++++++++++===')
			flag01.append(1) 
			flag= False                                   # consider one prediction 
			new_row.append(actual_value[0])
			new_row.append(predicted[j][0])
			accuracy = fuzz.ratio(str(new_row[2]).lower(), str(new_row[3]).lower())
			new_row.append(accuracy)
			if accuracy == 100:
				new_row.append(1)
			new_row.append(predicted[j][2])
			new_row.append(actual_value[1])
			data.append(new_row)
			print(data)
			try:
				to_do.remove(j)
			except:
				pass
			# print("going to  break")
			break
	else:
		if key not in ["drawee_bank_address", "drawer_bank_address", "drawer_bank_bottom_address","drawee_address","document_enclosed", "consignee_address", "consignor_address", "coo_issuer_address", "description_of_goods", "marks_and_no_of_packages"]:
			print("else executed")
			# exit('+++++++++++++')
			# print(new_row)
			new_row.append(actual_value[0])
			new_row.append("")
			new_row.append(0)
			new_row.append(0)
			new_row.append(0)
			new_row.append(actual_value[1])
			data.append(new_row)
	# print("new_to_do is",to_do)



	return to_do



# get intersection over union of two bounding boxes
def get_iou_new(bb1, bb2):
	try:
		assert bb1[0] < bb1[2]
		assert bb1[1] < bb1[3]
		assert bb2[0] < bb2[2]
		assert bb2[1] < bb2[3]
	except:
		return 0

	# determine the coordinates of the intersection rectangle
	x_left = max(bb1[0], bb2[0])
	y_top = max(bb1[1], bb2[1])
	x_right = min(bb1[2], bb2[2])
	y_bottom = min(bb1[3], bb2[3])

	if x_right < x_left or y_bottom < y_top:
		return 0.0

	# The intersection of two axis-aligned bounding boxes is always an
	# axis-aligned bounding box
	intersection_area = (x_right - x_left) * (y_bottom - y_top)

	# compute the area of both AABBs
	bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
	bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
	assert iou >= 0.0
	assert iou <= 1.0
	return iou
def remove_special_chars(text):
    # Define the pattern for special characters
    pattern = r'^[\W_]+|[\W_]+$'
    # Remove special characters from start and end of text
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

def filter_address(x):
	x = str(x)
	x = x.strip()
	new_x = re.sub(r'\s+', ' ', x)
	print("newx")
	print(new_x)
	# remove the special characters like comma, semicolon etc
	new_x = "".join([x for x in new_x if x.isalnum() or x in [" "]])
	print(new_x)
	return new_x


def fuzzy_float_comparison(float1, float2, tolerance=1e-6):
	absolute_difference = abs(float1 - float2)
	print(f'absolute difference: {absolute_difference}')
	if absolute_difference <= tolerance:
		return 100  # Return 100 for a perfect match within the tolerance
	else:
		# Calculate a similarity score based on the relative difference
		similarity = 100 - (absolute_difference / max(abs(float1), abs(float2))) * 100
		return similarity

def remove_by_prefix(input_string):
    # Split the input string by space and remove the first element ('by')
    words = input_string.split(' ')
    if words[0].lower() == 'by':
        words = words[1:]
    return ' '.join(words)

def remove_extension(filename):
    return os.path.splitext(filename)[0]

def create_dataframe(folder_path):
    files = os.listdir(folder_path)
    filenames_without_extension = [remove_extension(file) for file in files]
    counts = [filenames_without_extension.count(name) for name in filenames_without_extension]

    df = pd.DataFrame({"image_name": filenames_without_extension, "count": counts})
    return df


def remove_suffix(filename):
    return filename.replace("_s_1", "").replace("_s_2", "").replace("_s_3", "").replace("_s_4", "").replace("_s_5", "").replace("_s_6", "").replace("_s_7", "").replace("_s_8", "").replace("_s_9", "").replace("_s_10", "").replace("_s_11", "")
    
def create_data_frame(folder_path):
    files = os.listdir(folder_path)
    filenames_without_extension = [remove_extension(file) for file in files]
    filenames_without_suffix = [remove_suffix(name) for name in filenames_without_extension]

    df = pd.DataFrame({"image_name": filenames_without_suffix})
    df['count'] = df.groupby('image_name')['image_name'].transform('size')
    df = df.drop_duplicates().reset_index(drop=True)

    return df


if __name__ == '__main__':
    
    product_config = ConfigParser()
    debug_mode = True
    # relative path => passed in validation
    product_config.read("/home/ntlpt19/Downloads/Evaluation_Data/updated_code/src/main/extraction/config/config.ini")
    prod_code = product_code_map[product_config["Product"]["code"]]
    doc_code = product_config["Product"]["document_code"]
    if '[' in doc_code:
        doc_elements = doc_code[1:-1].split(', ')
        # Convert elements to a Python list
        doc_code_list = [element.strip() for element in doc_elements]
    print(doc_code)
    #best_keys_list = ast.literal_eval(configur[f'{ground_truth}_BEST_KEYS']['keys'])
    # data folder path
    product_wise_folder = ConfigParser()
    product_wise_folder.read("/home/ntlpt19/Downloads/Evaluation_Data/updated_code/src/main/extraction/config/prod.ini")


    

    for doc_code_ in doc_code_list:
        print(doc_code_)
        doc_code = document_code_map[doc_code_]
        folder_path = product_wise_folder[prod_code][doc_code]

        occurrences: dict = {}
        data: list = []

        # Global statements
        # Step1: Reading configurations from ini files
        ###########################################################################
        ###########################################################################
        ###########################################################################
        # parser = ConfigParser()

        # conf_folder_path: str = "/media/tarun/D1/Trade-Finance/src/main/extraction/config"
        # config_file_name: str = "config.ini"

        # if os.path.exists(f"{conf_folder_path}/{config_file_name}"):
        # 	parser.read(f"{conf_folder_path}/{config_file_name}")

        # debug_mode = str(parser["PARAMS"]["debug_mode"])
        # log_min_level = str(parser["LOG"]["LOG_FILTER_LEVELMIN"])

        # # Step 2: Setting up logger
        # # setting the log folder and file
        # tu.set_basic_config_for_logging(folder_path=conf_folder_path, filename="accuracy_generation_file")
        # # setting the logger object and log level
        # logger = tu.get_logger_object_and_setting_the_loglevel(log_level=log_min_level)
        # folder_path: str = str(parser['PATHS']['root_folder'])
        ####################################################################################
        ####################################################################################
        
        result_path = os.path.join(folder_path, "Results_Images")
        data_path = os.path.join(folder_path, "New_Master_Data_Merged")

        data_files: list = os.listdir(data_path)
        result_files: list = os.listdir(result_path)
        incoterm_list: list = []

        with open("incoterm_list.txt") as fp:
            for line in fp:
                incoterm_list.append(line.strip())
        #print(incoterm_list)
        # exit('++++++++++')

        # check how many pngs are there in those two folders
        #print("Number of images in results", len(glob.glob(result_path + "/*.png")))
        #print("Number of images in master data", len(glob.glob(data_path + "/*.png")))

        # exit("+++++++++++")

        # validating the number of files coming from both these folders
        #print(f"number of files coming from the Master data folder: {len(data_files)}")
        #print(f"number of files coming from the Result cs validated folder: {len(result_files)}")

        # exit("++++++++++++++")

        new = []
        new_result = []

        # FINDING LIST OF DATAFILES AND RESULT FILES
        num_labels_files_traversed: int = 0
        for file in data_files:
            if str(file)[-10:] == "labels.txt":
                #print(f"file number is {num_labels_files_traversed}")
                num_labels_files_traversed = num_labels_files_traversed + 1
                # print(str(file)[-10:])
                new.append(file)
                new_result.append(file[:-11] + ".txt")

        # exit("++++++++++++++++")
        data_files = new
        result_files = new_result
        # print(result_files)
        #print(len(data_files))
        #print(len(result_files))
        # exit("+++++++++")
        # #print(data_files)
        # cou = 0
        # for file in data_files:
        #     if file[0:-11] in test_img_list:
        #         print('yes')
        #         print(file[0:-11])
        #         cou+=1 
        # print(len(test_img_list))
        # print(cou)
        
        # creating a dictionary containing number of occurrences of all our fields in our dataset.
        for count, file in enumerate(data_files):
            print("count is:", count)
            with open(os.path.join(data_path, file), "r") as f:
                labels = json.load(f)
                try:
                    labels = list(labels)
                except:
                    continue
            for label in labels:
                if label in occurrences:
                    occurrences[label] += 1
                else:
                    occurrences[label] = 1
        column_names = []
        names = list(occurrences.keys())
        print(f'Number of classes used : {len(names)}')

        #print(occurrences)
        #exit("+++++++++++++++++")
        column_names.append("File_Name")
        column_names.append("label_name")
        column_names.append("actual")
        column_names.append("predicted")
        column_names.append("Accuracy")
        column_names.append("Match/No_Match")
        column_names.append("model_confidence")
        column_names.append("bbox")
        #print(column_names)
        #print(len(column_names))
        # name = str(result_files[i])[0:-4]
        for file, predicted_files in zip(data_files, result_files):
            # if file[0:-11] in test_img_list:
            #print('######################$$$$$$$$$$$$$$$$$$$$$$$$$$$$############################################################################')
            #print("file name is:", file)
            #print("resulted filename is", predicted_files)
            # continue
            # finding number of characters and type of document.
            with open(os.path.join(data_path, file), "r") as f:
                labels = json.load(f)
            try:
                print(os.path.join(result_path, file[0:-11] + "1.txt"))
                with open(os.path.join(result_path, file[0:-11] + "1.txt"), "r") as f2:
                    predicted = json.load(f2)
            except:
                # print("some problem opening file")
                try:
                    #print(f'''printing the path: {result_path, file[0:-11] + "_s_11.txt"}''')
                    with open(os.path.join(result_path, file[0:-11] + "_s_11.txt"), "r") as f2:
                        predicted = json.load(f2)
                # print("opened")
                except:
                    print("still not opened")
                    continue
            print(f'acutal labels: {labels}')
            print(f'number of keys : {len(list(labels.keys()))}')
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++=')
            print(f'predicted labels: {predicted}')
            print(f'predicted labels: {len(predicted)}')
            #if file[0:-11] + "_s_1.png" in test_list or file[0:-11] + "_s_2.png" in test_list or file[0:-11] + ".png" in test_list:
            
            #exit('+++++++++++==')

            if predicted == {} and labels == {}:
                print("*******")
                #print(file)
                continue
            for key in list(occurrences.keys()):
                if key in predicted:
                # if key in ['net_weight', 'gross_weight']:
                    if key == 'net_weight':
                        net_weight_list = []
                        len1 = len(predicted[key])
                        if len1>1:
                            for i in range(len1):
                                net_weight_list.append(predicted[key][i][0])
                            gross_weig = min(net_weight_list)
                            if 'gross_weight' not in predicted.keys():
                                predicted['gross_weight'] = []
                                predicted['gross_weight'].append([gross_weig,predicted[key][0][1], predicted[key][0][2]])
                            else:
                                predicted['gross_weight'].append([gross_weig,predicted[key][0][1], predicted[key][0][2]])
                            #print(predicted)
            for key in list(occurrences.keys()):
                if key not in ["csh_bill_currency", "doc_settlement_instructions","drawee_bank_country", "drawee_country", "drawer_country","drawer_bank_country", 'consignee_country']:
                    # print("key is", key)
                    row = []
                    row.append(file[0:-11] + ".png")
                    if key in labels and key in predicted:
                        if len(labels[key]) == 1 and len(predicted[key]) == 1:            
                            print(key)
                            row.append(key)
                            row.append(str(labels[key][0][0]))
                            row.append(str(predicted[key][0][0]))
                            # print(f'row data: {row}')
                            # print(str(row[2]).lower(), str(row[3]).lower())
                            #remove special chars in starting and ending of the string
                            row[2], row[3] = remove_start_end_spl_char(row[2], row[3])
                            #print(f'after remove spl chars: {row}')
                            # exit()
                            # post-processing for
                            if key in ['pre_carriage_by']:
                                row[2] = remove_by_prefix(row[2])
                                row[3] = remove_by_prefix(row[3]) 
                            if key in ['net_weight', 'gross_weight','total_quantity_of_goods']:
                                print('entered into  remove alphabets++++++++++++++++')
                                #print(row[2])
                                #print(row[3])
                                row[2] = remove_alphabets(row[2])
                                row[3] = remove_alphabets(row[3])
                                #print(row[2])
                                #print(row[3])
                            # processing for page no
                            if key in ["incoterm"]:
                                #pass
                                row[2] = preprocess_incoterm(row[2], incoterm_list)
                                row[3] = preprocess_incoterm(row[3], incoterm_list)
                            
                            # post-processing for address
                            if key in ["drawee_bank_address", "drawer_bank_address", "drawer_bank_bottom_address",
                                        "drawee_address", "consignee_address", "consignor_address", "coo_issuer_address"]:
                                row[2] = filter_address(row[2])
                                row[3] = filter_address(row[3])
                                #print('#####################################################################################################')
                                print('row[2]', row[2])
                                print('row[3]', row[3])
                                #exit()
                            # processing for page no
                            if key in ["page_no"]:
                                row[2] = generic_page_no(row[2])
                                row[3] = generic_page_no(row[3])

                            # processing for cash drawn under rules
                            if key in ["csh_drawn_under_rules"]:
                                row[2] = pp_cash_drawn_rules(row[2])
                                row[3] = pp_cash_drawn_rules(row[3])

                            #print(f'the row[2] value: {row[2]}')
                            #print(f'the row[3] value: {row[3]}')
                            #print(f'type of row[2]: {type(row[2])}')
                            #print(f'type of row[3]: {type(row[3])}')
                            # accuracy = fuzz.ratio(str(row[2]).lower(), str(row[3]).lower())
                            # print(f'accuracy of {key}: {accuracy}')
                            if key == 'gross_weight' or key== "net_weight" or key== "total_quantity_of_goods":
                                try:
                                    accuracy = fuzz.ratio(float(str(row[2])), float(str(row[3])))
                                    #print(f'type of row[2]: {type(int(row[2]))}')
                                    #print(f'type of row[3]: {type(int(row[3]))}')
                                    #print(f'accuracy of {key}: {accuracy}')
                                except Exception as e:
                                    try:
                                        accuracy = fuzzy_float_comparison(float(str(row[2])), float(str(row[3])))
                                        #print(f'Executed second accuracy {key}: {accuracy}')
                                    except Exception as e:
                                        #print(f'the row[2] value: {row[2]}')
                                        #print(f'the row[3] value: {row[3]}')
                                        accuracy = fuzz.ratio(str(row[2]).lower(), str(row[3]).lower())
                                        #print(f'Executed third accuracy {key}: {accuracy}')

                                
                            if key in ["drawee_bank_address", "drawer_bank_address", "drawer_bank_bottom_address",
                                        "drawee_address","page_no","csh_drawn_under_rules", "doc_charge_instructions",
                                        "doc_delivery_instruction","csh_bill_currency","csh_presentation_date","csh_due_date", "consignee_address", "consignor_address", "coo_issuer_address"]:
                                #print("&&&&&&&&&&&&&&&")
                                accuracy = 100 if str(row[3]).__contains__(row[2]) else fuzz.ratio(str(row[2]).lower(), str(row[3]).lower())
                                #print("accuracy:", accuracy)

                            else:
                                accuracy = fuzz.ratio(str(row[2]).lower(), str(row[3]).lower())
                            row.append(accuracy)
                            if accuracy == 100:
                                row.append(1)
                            else:
                                row.append(0)
                            # print(row)
                            row.append(predicted[key][0][2])
                            row.append(labels[key][0][1])
                            data.append(row)
                            # if key == 'consignee_country':
                            # 	key='consignee_addres'
                            if key in ["drawee_bank_address", "drawer_bank_address","drawee_address","nostro_bank_address", "drawer_address", "consignee_address",'consignee_addres','shipper_address','notify_party_address', "consignor_address", "coo_issuer_address"]:

                                act_val_address = labels[key][0][0]
                                pre_val_address = predicted[key][0][0]
                                act_val_address, pre_val_address = remove_start_end_spl_char(act_val_address, pre_val_address)
                                #pre_val_address = remove_start_end_spl_char(pre_val_address)
                                
                                act_val_address = filter_address(act_val_address)
                                pre_val_address = filter_address(pre_val_address)
                                
                                act_val_address = remove_symbols(act_val_address)
                                pre_val_address = remove_symbols(pre_val_address)
                                print('act_val_address', act_val_address)
                                #exit()
                                actual_country = extract_country_from_text(act_val_address)
                                pred_country = extract_country_from_text(pre_val_address)
                                
                                print('pred_country', pred_country,actual_country)
                                #exit()      
                                if len(actual_country)>0 or len(pred_country)>0:
                                    row = []
                                    row.append(file[0:-11] + ".png")
                                    row.append(key+"_country")  	
            
                                    actual_country = " ".join(actual_country)
                                    pred_country = " ".join(pred_country) 		
                
                                    #if len(actual_country)>0
                                    row.append(actual_country)
                                    row.append(pred_country)
                                    accuracy = fuzz.ratio(actual_country.lower(), pred_country.lower())
                                    row.append(accuracy)
                                    if accuracy == 100:
                                        row.append(1)
                                    else:
                                        row.append(0)   
                                    print(row)  
                                    data.append(row)
                            # print(data)
                            # exit('+++++++')
                        else:
                            # exit()
                            row.append(key)
                            actual = labels[key]
                            predicted_label = predicted[key]
                            print(f'second condition: {actual}')
                            print(f'second condition: {predicted_label}')
                            print(row)
                            l1 = len(actual)
                            l2 = len(predicted_label)
                            if key in ['net_weight', 'gross_weight','total_quantity_of_goods']:
                                for i in range(l2):
                                    predicted_label[i][0]=remove_alphabets(predicted_label[i][0])
                            if key in ["incoterm"]:
                                for i in range(l2):
                                    #print(f'incoterm predicted: {predicted_label[i][0]}')
                                    predicted_label[i][0]=preprocess_incoterm(predicted_label[i][0], incoterm_list)

                            if key in ["drawee_bank_address", "drawer_bank_address", "drawer_bank_bottom_address",
                                        "drawee_address", "consignee_address","consignor_address", "coo_issuer_address"]:
                                print('address block ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                                print(actual)
                                print(predicted_label)
                                # exit('++++++++++++')
                                for i in range(l2):
                                    predicted_label[i][0] = filter_address(predicted_label[i][0])
                                for i in  range(l1):
                                    actual[i][0] = filter_address(actual[i][0])
                                print(actual)
                                print(predicted_label)
                                # exit('+++++++++++++++++++')
                            if key in ["page_no"]:
                                for i in range(l2):
                                    predicted_label[i][0] = generic_page_no(predicted_label[i][0])
                                for i in  range(l1):
                                    actual[i][0] = filter_address(actual[i][0])

                            # processing for cash drawn under rules
                            if key in ["csh_drawn_under_rules"]:
                                for i in range(l2):
                                    predicted_label[i][0] = pp_cash_drawn_rules(predicted_label[i][0])
                                for i in  range(l1):
                                    actual[i][0] = filter_address(actual[i][0])
                            for i in range(l2):
                                    print(row)
                                    print(predicted_label)
                                    print(f'The value is: {predicted_label[i][0]}')
                                    # exit('+++++++++++++==')
                                    predicted_label[i][0]=remove_spl_char_multiple_pred(predicted_label[i][0])
                            print(f'second condition after remove alphabets: {predicted_label}')
                            to_do = [*range(0, l2, 1)]
                            print("starting to_do are", to_do)
                            if key not in ["drawee_bank_address", "drawer_address", "drawer_bank_address", "drawer_bank_bottom_address","drawee_address","document_enclosed", "consignee_address","consignor_address", "coo_issuer_address", "description_of_goods", "marks_and_no_of_packages","notify_party_name", 'dimension', 'to_place', 'means_of_transport', 'issue_date']:
                                flag01 = []							
                                for i in range(l1):
                                    new_row = row.copy()
                                    actual_value = actual[i]
                                    actual[i][0]=remove_spl_char_multiple_pred(actual_value[0])
                                    print(f'the actual value is: {actual_value}')
                                    # exit('+++++++==')
                                    
                                    to_do = filter_prediction(new_row, actual_value, predicted_label, to_do, key, flag01)
                                print(to_do)
                            if key in ['to_place', 'means_of_transport', 'issue_date']:
                                act_val = ''
                                pre_val = ''
                                print(actual)
                                for i in range(len(actual)):
                                    act_val += actual[i][0]+", "
                                for j in range(len(predicted_label)):
                                    pre_val += predicted_label[j][0]+", "
                                    
                                new_row1 = row.copy()
                                new_row1.append(act_val)
                                new_row1.append(pre_val)
                                        
                                accuracy = fuzz.ratio(str(new_row1[2]).lower(), str(new_row1[3]).lower())
                                new_row1.append(accuracy)
                                if accuracy == 100:
                                    new_row1.append(1)
                                else:
                                    new_row1.append(0)
                                new_row1.append("predicted[j][2]")
                                new_row1.append("actual_value[1]")
                                data.append(new_row1)                            
                            if key in ["drawee_bank_address", "drawer_bank_address","drawee_address","drawer_address", "document_enclosed", "consignee_address","consignor_address", "coo_issuer_address", "description_of_goods", "marks_and_no_of_packages", 'notify_party_name','dimension']:
                                act_val = ''
                                pre_val = ''
                                print(actual)
                                for i in range(len(actual)):
                                    act_val += actual[i][0]+" "
                                for j in range(len(predicted_label)):
                                    pre_val += predicted_label[j][0]+" "
                                    
                                new_row1 = row.copy()
                                new_row1.append(act_val)
                                new_row1.append(pre_val)
                                        
                                accuracy = fuzz.ratio(str(new_row1[2]).lower(), str(new_row1[3]).lower())
                                new_row1.append(accuracy)
                                if accuracy == 100:
                                    new_row1.append(1)
                                else:
                                    new_row1.append(0)
                                new_row1.append("predicted[j][2]")
                                new_row1.append("actual_value[1]")
                                data.append(new_row1)

                                del new_row1
                            if key in ["drawee_bank_address", "drawer_bank_address","drawee_address","nostro_bank_address", "drawer_address", "consignee_address", "consignor_address", "coo_issuer_address"]:
                                actual = labels[key]
                                predicted_label = predicted[key]
                                act_val_address = ''
                                pre_val_address = ''
                                print(actual)
                                for i in range(len(actual)):
                                    act_val_address += actual[i][0]+" "
                                for j in range(len(predicted_label)):
                                    pre_val_address += predicted_label[j][0]+" "
                                
                                act_val_address, pre_val_address = remove_start_end_spl_char(act_val_address, pre_val_address)
                                #pre_val_address = remove_start_end_spl_char(pre_val_address)
                                
                                act_val_address = filter_address(act_val_address)
                                pre_val_address = filter_address(pre_val_address)
                                
                                act_val_address = remove_symbols(act_val_address)
                                pre_val_address = remove_symbols(pre_val_address)
                                print('act_val_address', act_val_address)
                                #exit()
                                actual_country = extract_country_from_text(act_val_address)
                                pred_country = extract_country_from_text(pre_val_address)
                                print('pred_country', pred_country,actual_country)
                                #exit()      
                                #if actual_country and pred_country:
                                if len(actual_country)>0 or len(pred_country)>0:
                                    row = []
                                    row.append(file[0:-11] + ".png")
                                    row.append(key+"_country")  	
            
                                    actual_country = " ".join(actual_country)
                                    pred_country = " ".join(pred_country) 		
                
                                    #if len(actual_country)>0
                                    row.append(actual_country)
                                    row.append(pred_country)
                                    accuracy = fuzz.ratio(actual_country.lower(), pred_country.lower())
                                    row.append(accuracy)
                                    if accuracy == 100:
                                        row.append(1)
                                    else:
                                        row.append(0)   
                                    print(row)  
                                    data.append(row)
                    elif key in labels and key not in predicted:
                        # exit()
                        print(f'key is : {key}')
                        print('entered into third condition')
                        print(row)
                        # exit('+++++++++++++===')
                        row.append(key)
                        if len(labels[key]) == 1:
                            new_row = row.copy()
                            new_row.append(str(labels[key][0][0]))
                            new_row.append("")
                            new_row.append(0)
                            new_row.append(0)
                            new_row.append(0)
                            new_row.append(labels[key][0][1])
                            data.append(new_row)
                        else:
                            for val in labels[key]:
                                new_row = row.copy()
                                new_row.append(val[0])
                                new_row.append("")
                                new_row.append(0)
                                new_row.append(0)
                                new_row.append(0)
                                new_row.append(val[1])
                                data.append(new_row)
                    # print(row)
                    elif key not in labels and key in predicted:
                        # exit()
                        row.append(key)
                        row.append("")
                        if len(predicted[key]) == 1:
                            row.append(str(predicted[key][0][0]))
                            row.append(0)
                            row.append(0)
                            row.append(predicted[key][0][2])
                            row.append(predicted[key][0][1])
                            data.append(row)
                        else:
                            predicted_label = predicted[key]
                            pre_val = '' 
                            for j in range(len(predicted_label)):
                                pre_val += predicted_label[j][0]+" "
                            #for val in predicted[key]:
                            new_row = row.copy()
                            new_row.append(pre_val)
                            new_row.append(0)
                            new_row.append(0)
                            new_row.append('')
                            new_row.append('')
                            data.append(new_row)
                        
                    
                    else:
                        continue
        #print(row)
        # exit('++++++++++++++++++')

        df = pd.DataFrame(data, columns=column_names)
        #print(df)
        # exit('+++++++++++==')

        #print(df["Match/No_Match"].sum())
        #print(df["Match/No_Match"])
        # exit()
        # post processing fuzzy match percentage
        df.loc[df["Accuracy"].apply(float) >= 90, "Match/No_Match"] = 1
        #print(df["Match/No_Match"].sum())
        # exit("++++++++++++")
        res_path= os.path.join(folder_path, 'result_path')
        if not os.path.exists(res_path):
            os.mkdir(res_path)
        df.to_csv(f'{res_path}/PL_analysis_pre_valid_june7_after_fuzzy_match_post-processing_latest.csv')
        
        accuracy_generation_file = os.path.join(folder_path, 'result_path', 'PL_analysis_pre_valid_june7_after_fuzzy_match_post-processing_latest.csv')
        
        
        file_path_csv1 = os.path.join(folder_path, 'result_path', f"{doc_code_}_{datetime.now().date()}_{datetime.now().hour}",
                                accuracy_generation_file)
        file_paths_csv2 = final_report(file_path_csv1, folder_path, doc_code_)

        # txt file generation
        stp_report(file_path_csv1, folder_path, doc_code_)

        final_overall_analysis(file_paths_csv2, file_path_csv1, folder_path, doc_code_)