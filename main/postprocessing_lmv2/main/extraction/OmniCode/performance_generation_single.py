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

import glob
import os
import json
import pandas as pd
from fuzzywuzzy import fuzz
import re
import dateparser
from datetime import datetime

from src.main.extraction.post_processing_master_gt import generic_page_no
from src.main.extraction.post_processing_pred import pp_cash_drawn_rules

from typing import List
from configparser import ConfigParser
from src.main.extraction.config.prod_mapping import product_code_map, document_code_map


""" This script used to generate accuracy generation of the dataset and 
used documents like PL, CS, COO, BOL"""


import re
from fuzzywuzzy import fuzz
from geonamescache import GeonamesCache
from difflib import get_close_matches
import pycountry

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
	
def keep_specific_characters(text):
    text = text.split(' ')
    final_text = ''
    for word in text:
        a_text = "".join(ch for ch in word if ch.isalnum() or ch in ',.')
        final_text = final_text + a_text + ' '
    return final_text.strip()

def preprocess_incoterm(text : str, incoterm_list: List):
	pattern = r'\b(?:' + '|'.join(re.escape(term) for term in incoterm_list) + r')\b'

	# Use re.findall to find all matching Incoterms in the text
	found_incoterms = re.findall(pattern, text, re.IGNORECASE)

	# If Incoterms are found, join them with spaces; otherwise, return the complete text
	result = " ".join(found_incoterms) if found_incoterms else text

	return result
    

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

def extract_currency_and_amount(input_string):
    curr = ''
    amt = ''
    # input_string = re.sub(r'\s+', '', input_string)
    # input_string = re.sub(r'(?<! ) +| +(?= )', '', input_string)
    input_string = re.sub(r'\s+', ' ', input_string)

    characters_to_remove = "'·!'|:()/-%;'*"     #('')
    translation_table = str.maketrans('', '', characters_to_remove)
    input_string = input_string.translate(translation_table) 
    list_of_input_string = input_string.split(' ')
    for j in list_of_input_string:
        flag = None
        for i in j:
            try:
                if int(i):
                    amt+=i
                    flag = 'amt'
                if i=='0':
                    amt+='0'
            except:
                if  i=='.' or i==',':
                    if flag == 'amt':
                        amt+=i
                    elif flag == 'curr':
                        curr+=i 
                    
                else:   
                    curr+=i 
                    flag = 'curr'
        if len(list_of_input_string)>1:
            curr = curr+' '
    return curr, amt



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
			try:
				new_row.append(predicted[j][2])
				new_row.append(actual_value[1])
			except:
				pass
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
import re
from datetime import datetime
# def convert_to_common_format(date_str):

def convert_half_months_to_full_months(date_str):
    # Define a list of month abbreviations and their corresponding full month names
    month_mappings = {
        'jan': 'January',
        'feb': 'February',
        'mar': 'March',
        'apr': 'April',
        'may': 'May',
        'jun': 'June',
        'jul': 'July',
        'aug': 'August',
        'sep': 'September',
        'oct': 'October',
        'nov': 'November',
        'dec': 'December'
    }
    
    cleaned_date_str = date_str.replace('.', ' ').replace(',', ' ')
    cleaned_date_str = re.sub(r'\s+', ' ', cleaned_date_str)  # Remove multiple spaces
    
    # Define a regular expression pattern to match half-month representations
    pattern = r'(\b(?:' + '|'.join(month_mappings.keys()) + r')\b)'

    # Use re.sub to find and replace half-month abbreviations with full month names
    def replace_month(match):
        return month_mappings[match.group(0)]

    # Perform the replacement
    converted_date_str = re.sub(pattern, replace_month, cleaned_date_str.lower())  # Convert to lowercase for case-insensitive matching

    return converted_date_str

def clean_date(original_string):
	# pattern = r'[^a-zA-Z0-9,./-]'
	pattern = r'[^a-zA-Z0-9,./\s-]'
	# Use re.sub() to replace the matched special characters with an empty string
	cleaned_string = re.sub(pattern, '', original_string)
	return cleaned_string



def convert_to_common_format_use_regex(date_str):
	try:
		date_str = convert_half_months_to_full_months(date_str)
		# Define regular expression patterns for various date formats
		patterns = [
			r'(\d{1,2}(?:st|nd|rd|th) [A-Za-z]+ , \d{4})',  # Modified pattern
			r'(\d{1,2}/\d{1,2}/\d{4})',
			r'(\d{1,2}\.\d{1,2}\.\d{4})',
			r'(\d{1,2} [A-Za-z]+ , \d{4})',
			r'(\d{1,2}[A-Za-z]+\d{4})',
			r'([A-Za-z]+\.\d{1,2},\d{4})',
			r'(\d{1,2}-[A-Za-z]+-\d{4})',
			r'(date+\.\d{2}\.\d{2}\.\d{4})',
			r'(DATE\.\d{2}\.\d{2}.\d{4})',
			r'([A-Za-z]+\.\d{2}\.\d{2}\.\d{4})'
			r'(\d{1,2} [A-Za-z]+ \d{4})',
			r'(\d{1,2}-\d{2}-\d{4})',
			r'(\d{1,2}\.\d{2}\.\d{2})',
			r'(\d{1,2}[A-Za-z]+\d{2,4})',  # Matches "02JUN2012" format
			r'(\d{2} [A-Za-z]+ , \d{4})',  # Additional pattern
			r'(\d{2} - [A-Za-z]+ - \d{2})',  # Additional pattern
			r'([A-Za-z]+ \. \d{1,2} , \d{4})',  # Additional pattern
			r'(\d{2} - [A-Za-z]+ - \d{4} \d{4})',  # Additional pattern
			r'([A-Za-z]+ \. \d{1,2} , \d{4})',  # Additional pattern March. 28 , 
			r'([A-Za-z]+. \d{1,2} , \d{4})',
			r'([A-Za-z]+ \. - \d{4})',  # Additional pattern
			r'(\d{2} - [A-Za-z]+ - \d{2})',  # Additional pattern 
			r'(\d{2} - [A-Za-z]+ - \d{4}) (?:\d{4})',
			r'([A-Za-z]+ \. \d{2} , \d{4})',  # Additional pattern "mar . 12 , 2013" ([A-Za-z]+ \. \d{1,2} , \d{4}) mar . 12 , 2013
			r'([A-Za-z]+ \. , \d{2} , \d{4})',
			r'(dt\.\d{2} / \d{2} / \d{4})',  # Additional pattern "aug . , 16 , 2013"
			r'(\'[A-Za-z]+ \. \d{2} , \d{4}\')',  # Additional pattern
			#r'([A-Za-z] \d{1,2} \d{4})',
			r'([A-Za-z]+ \d{1,2} \d{4})',
		
		]																																																																																															

		# Try to match date formats using regular expressions
		for pattern in patterns:
			match = re.search(pattern, date_str)
			if match:
				matched_date_str = match.group(1)
				break
		else:
			return ""
		format_strings = [
			'%B %d %Y',
			'%d %B , %Y',  # Corresponding format for the modified pattern
			'%d/%m/%Y',
			'%d.%m.%Y',
			'%d %B , %Y',
			'%d%b%Y',
			'%b.%d,%Y',
			'%d-%b-%Y',
			'date.%d.%m.%Y',
			'DATE.%d.%m.%Y',
			'%d %B %Y',
			'%d-%m-%Y',
			'%d.%m.%y',  # Format for "13.12.12" (assuming 2-digit year)
			'%d%b%Y',  # Format for "02JUN2012"
			'%d %B , %Y',  # Corresponding format for additional pattern
			'%d - %B - %y',  # Corresponding format for additional pattern
			'%B . %d , %Y',  # Corresponding format for additional pattern
			'%d - %B - %Y %Y',  # Corresponding format for additional pattern March. 28 , 2013
			'%B . %d , %Y',  # Corresponding format for additional pattern ([A-Za-z]+ \. \d{1,2} , \d{4}) mar . 12 , 2013
			'%B . %d , %Y',
			'%B . %d , %Y',
			'%B . , %d , %Y', 
			'%B . - %Y',  # Corresponding format for additional pattern
			'%d - %B - %y',  # Corresponding format for additional pattern
			'%B . - %y , %Y',  # Corresponding format for additional pattern
			'dt.%d / %m / %Y',  # Corresponding format for additional pattern
			'\'%B . %d , %Y\'',  # Corresponding format for additional pattern
			'%B  %d  %Y'
			
		]
		# Attempt to parse the date using each format
		for format_string in format_strings:
			try:
				date = datetime.strptime(matched_date_str, format_string)
				if date.year < 1000:
					continue
				return date.strftime('%d-%m-%Y')
			except ValueError:
				continue
				
			#     return date.strftime('%d-%m-%y')
			# except ValueError:
			#     continue
		return ""
	except:
		return ""



def convert_to_common_format(date_str):
    date_str = clean_date(date_str)
    pred_date = date_str
    filter_date = ''
    if len(pred_date)==6:
        filter_date = '-'.join([pred_date[i:i+2] for i in range(0, len(pred_date), 2)])
    if len(pred_date)==8:
        try:
            filter_date = datetime.strptime(pred_date, '%d%m%Y').strftime('%d-%m-%Y')
        except:
            filter_date = ''
    if filter_date=='':
        try:
            filter_date = dateparser.parse(pred_date).strftime("%d-%m-%Y")
        except:
            filter_date = ''
        if filter_date=='':
            filter_date = convert_to_common_format_use_regex(pred_date)
            if filter_date=='':
                filter_date = pred_date
    return filter_date
	



def doc_delivery_pp(input_string):
	# input_string = "RELEASE DOCUMENTS Against PAYMENT"

	# Define a regex pattern to match either "AGAINST PAYMENT" or "AGAINST ACCEPTANCE" (case-insensitive)
	pattern = r'(?i)AGAINST (PAYMENT|ACCEPTANCE)'

	# Use re.search with the IGNORECASE flag to find the pattern in the input string
	match = re.search(pattern, input_string)

	# Check if a match is found
	if match:
		# Extract the matched text
		extracted_text = match.group()
		print(extracted_text)
	else:
		# If no match is found, return the entire input string
		extracted_text = input_string
		print("Pattern not found in the input string. Returning the entire string:")
		print(extracted_text)
	return extracted_text

def doc_chages_pp(input_string):
	target_words = ["buyer's", "drawee", "drawer", "DRAWEES", "DRAWEE'S"]

	# Create a regular expression pattern to match the target words
	pattern = r'\b(?:' + '|'.join(re.escape(word) for word in target_words) + r')\b'

	# Find all matches in the input string
	matches = re.findall(pattern, input_string, re.IGNORECASE)

	# Check if any matches were found
	if matches:
		# If matches were found, print the segregated words
		final_string = 'from '+matches[0]
		print(final_string)
		
	else:
		# If no matches were found, return the entire input string
		print(input_string)
		final_string = input_string
	return final_string

def remove_text_after_phrases(text):
    cleaned_text = re.sub(r'(Ph\.|Tel\.|Zip|Phone)\s*.*', '', text, flags=re.IGNORECASE)	
    cleaned_text = cleaned_text.strip()
    return cleaned_text

def remove_words(text,words_to_remove_list):
    pattern = r'\b(?:{})\b'.format('|'.join(map(re.escape, words_to_remove_list)))
    cleaned_text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text)
    cleaned_text = cleaned_text.replace('&', '')
    cleaned_text = cleaned_text.strip()
    return cleaned_text

def get_incoterms():
    with open("src/main/extraction/incoterm_list.txt") as fp:
        for line in fp:
            incoterm_list.append(line.strip())
    print('**********')
    print(incoterm_list)
    return incoterm_list

if __name__ == '__main__':
    
    # Model_list : list = ["Lmv2-base", "Lmv2-large", "Lmv3-base" , "Lmv3-large", "Geolaoyout"]
    Model_list : list = ["Lmv2-base", "Lmv3-base" ,"Geolaoyout"]
    
    
	# count = 0
    occurrences = {}
    data = []
    configur = ConfigParser()
    configur.read('src/main/extraction/traini_valid_utility.ini')
    
    
    # product config
    product_config = ConfigParser()
    product_config.read("src/main/extraction/config/config.ini")

    prod_code = product_code_map[product_config["Product"]["code"]]
    doc_code = document_code_map[product_config["Product"]["document_code"]]

    # data folder path
    product_wise_folder = ConfigParser()
    product_wise_folder.read("src/main/extraction/config/prod.ini")
    folder_path = product_wise_folder[prod_code][doc_code]

    print("==================Trade Finance Solutions===================")
    print("Product Code: {product_code}")
    print("Documenry Code: {doc_code}")
    print(f"folder_path: {folder_path}")
    
    

    # folder_path = str(configur['PATHS']['folder_path'])
    # folder_path: str = "/home/ntlpt19/Downloads/Trade_finance_imp_stage_2/COO_ROOT"

    for model in Model_list:
    
        result_path = os.path.join(folder_path, "Results_testing_validated_{model}")
        data_path = os.path.join(folder_path, "New_Master_Data_Merged_{model}")

        # import json
        data_files = os.listdir(data_path)
        result_files = os.listdir(result_path)
        
        incoterm_list = get_incoterms()
    
        new : List = []
        new_result : List = []

        # FINDING LIST OF DATAFILES AND RESULT FILES
        num_labels_files_traversed: int = 0
        for file in data_files:
            if str(file)[-10:] == "labels.txt":
                #print(f"file number is {num_labels_files_traversed}")
                num_labels_files_traversed = num_labels_files_traversed + 1
                # print(str(file)[-10:])
                new.append(file)
                new_result.append(file[:-11] + ".txt")

        data_files = new
        result_files = new_result

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
        
        names = list(occurrences.keys())
        print(f'Number of classes used : {len(names)}')

        column_names: list = []
        column_names.append("File_Name")
        column_names.append("label_name")
        column_names.append("actual")
        column_names.append("predicted")
        column_names.append("Accuracy")
        column_names.append("Match/No_Match")
        column_names.append("model_confidence")
        column_names.append("bbox")

        # name = str(result_files[i])[0:-4]
        for file, predicted_files in zip(data_files, result_files):
            #print("file name is:", file)
            #print("resulted filename is", predicted_files)
            # continue
            # finding number of characters and type of document.
            with open(os.path.join(data_path, file), "r") as f:
                labels = json.load(f)
            try:
                print(os.path.join(result_path, file[0:-11] + "1.txt"))    #1.txt ==>actual
                with open(os.path.join(result_path, file[0:-11] + "1.txt"), "r") as f2:  #1.txt ==>actual
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

            if predicted == {} and labels == {}:
                print("*******")
                #print(file)
                continue
            print(list(occurrences.keys()))
            actual_list = list(labels)
            pred_list = list(predicted)
            print(actual_list,'*****',pred_list)
            actual_amt = []
            actual_curr = []
            
            if 'currency_amount' in actual_list:
                for i in range(len(labels['currency_amount'])):
                    act_currency, act_amount = extract_currency_and_amount(str(labels['currency_amount'][i][0]))
                    if len(act_amount)>0 and 'csh_bill_amount' in actual_list:
                        labels['csh_bill_amount'].append([act_amount, labels['currency_amount'][i][1]])
                        # labels['csh_bill_amount'].append(labels['currency_amount'][i][1])
                    else:
                        if len(act_amount)>0:
                            labels['csh_bill_amount']=[[act_amount, labels['currency_amount'][i][1]]]


                    if len(act_currency)>0 and 'csh_bill_currency' in actual_list:
                        labels['csh_bill_currency'].append([act_currency, labels['currency_amount'][i][1]])
                        # labels['csh_bill_amount'].append(labels['currency_amount'][i][1])
                    else:
                        if len(act_currency)>0:
                            labels['csh_bill_currency']=[[act_currency, labels['currency_amount'][i][1]]]

            if 'currency_amount' in pred_list:
                for i in range(len(predicted['currency_amount'])):
                    act_currency, act_amount = extract_currency_and_amount(str(predicted['currency_amount'][i][0]))
                    if len(act_amount)>0 and 'csh_bill_amount' in pred_list:
                        predicted['csh_bill_amount'].append([act_amount, predicted['currency_amount'][i][1], predicted['currency_amount'][i][2]])
                        # labels['csh_bill_amount'].append(labels['currency_amount'][i][1])
                    else:
                        if len(act_amount)>0:
                            predicted['csh_bill_amount']=[[act_amount, predicted['currency_amount'][i][1], predicted['currency_amount'][i][2]]]


                    if len(act_currency)>0 and 'csh_bill_currency' in pred_list:
                        predicted['csh_bill_currency'].append([act_currency, predicted['currency_amount'][i][1], predicted['currency_amount'][i][2]])
                        # labels['csh_bill_amount'].append(labels['currency_amount'][i][1])
                    else:
                        if len(act_currency)>0:
                            predicted['csh_bill_currency']=[[act_currency, predicted['currency_amount'][i][1], predicted['currency_amount'][i][2]]]

            if 'csh_bill_currency' in actual_list:
                for i in range(len(labels['csh_bill_currency'])):
                    act_currency, act_amount = extract_currency_and_amount(str(labels['csh_bill_currency'][i][0]))
                    if len(act_amount)>0 and 'csh_bill_amount' in actual_list:
                        labels['csh_bill_amount'].append([act_amount, labels['csh_bill_currency'][i][1]])
                        # labels['csh_bill_amount'].append(labels['currency_amount'][i][1])
                    else:
                        if len(act_amount)>0:
                            labels['csh_bill_amount']=[[act_amount, labels['csh_bill_currency'][i][1]]]


                    if len(act_currency)>0: #and 'csh_bill_currency' in actual_list:
                        labels['csh_bill_currency'][i][0] = act_currency   


            if 'csh_bill_currency' in pred_list:
                for i in range(len(predicted['csh_bill_currency'])):
                    act_currency, act_amount = extract_currency_and_amount(str(predicted['csh_bill_currency'][i][0]))
                    if len(act_amount)>0 and 'csh_bill_amount' in pred_list:
                        predicted['csh_bill_amount'].append([act_amount, predicted['csh_bill_currency'][i][1], predicted['csh_bill_currency'][i][2]])
                        # labels['csh_bill_amount'].append(labels['currency_amount'][i][1])
                    else:
                        if len(act_amount)>0:
                            predicted['csh_bill_amount']=[[act_amount, predicted['csh_bill_currency'][i][1], predicted['csh_bill_currency'][i][2]]]

                    if len(act_currency)>0: #and 'csh_bill_currency' in actual_list:
                        predicted['csh_bill_currency'][i][0] = act_currency   


            if 'csh_bill_amount' in actual_list:
                for i in range(len(labels['csh_bill_amount'])):
                    act_currency, act_amount = extract_currency_and_amount(str(labels['csh_bill_amount'][i][0]))
                    if len(act_currency)>0 and 'csh_bill_currency' in actual_list:
                        labels['csh_bill_currency'].append([act_currency, labels['csh_bill_amount'][i][1]])
                        # labels['csh_bill_amount'].append(labels['currency_amount'][i][1])
                    else:
                        if len(act_currency)>0:
                            labels['csh_bill_currency']=[[act_currency, labels['csh_bill_amount'][i][1]]]


                    if len(act_amount)>0: #and 'csh_bill_currency' in actual_list:
                        labels['csh_bill_amount'][i][0] = act_amount   


            if 'csh_bill_amount' in pred_list:
                for i in range(len(predicted['csh_bill_amount'])):
                    act_currency, act_amount = extract_currency_and_amount(str(predicted['csh_bill_amount'][i][0]))
                    if len(act_currency)>0 and 'csh_bill_currency' in pred_list:
                        predicted['csh_bill_currency'].append([act_currency, predicted['csh_bill_amount'][i][1], predicted['csh_bill_amount'][i][2]])
                        # labels['csh_bill_amount'].append(labels['currency_amount'][i][1])
                    else:
                        if len(act_currency)>0:
                            predicted['csh_bill_currency']=[[act_currency, predicted['csh_bill_amount'][i][1], predicted['csh_bill_amount'][i][2]]]


                    if len(act_amount)>0: #and 'csh_bill_currency' in actual_list:
                        predicted['csh_bill_amount'][i][0] = act_amount     
            
                print("*****************")
                print(file)
                print(predicted)


            for key in list(occurrences.keys()):
                if key not in ['currency_amount', "doc_settlement_instructions","drawee_bank_country", "drawee_country", "drawer_country","drawer_bank_country", "document_enclosed", 'consignee_country', "signed_stamp", "certificate_stamped", "declaration_by", "signature"]:
                    # print("key is", key)
                    row = []
                    row.append(file[0:-11] + ".png")
                    if key in labels and key in predicted:
                        if len(labels[key]) == 1 and len(predicted[key]) == 1:               
                            print(key)
                            row.append(key)
                            row.append(str(labels[key][0][0]))
                            row.append(str(predicted[key][0][0]))
            
                            if key in ['doc_delivery_instruction']:
                                row[3] = doc_delivery_pp(row[3])
                                row[2] = doc_delivery_pp(row[2])
                            if key in ['doc_charge_instructions']:
                                row[3] = doc_chages_pp(row[3])
                                row[2] = doc_chages_pp(row[2])

                            if key =="pre_carriage_by"or key =="mode_of_transport" or key == 'means_of_transport':										
                                row[2] = remove_words(row[2],['BY',"EXPORT", 'freight'])
                                row[3] = remove_words(row[3],['BY',"EXPORT", 'freight'])
                            if key in ["awb_date","bill_of_lading_date","bill_of_lading_issue_date","csh_presentation_date","date_of_invoice","end_date","expiry_date","indicator_date","invoice_date","invoice_due_date","issue_date","lc_date","sail_on_or_about_to_date","shipped_onboard_date","start_date","tenor_indicator_date","transaction_date"]: #actual
                            #if key in ['abc']:
                                row[2] = convert_to_common_format(row[2])
                                row[3] = convert_to_common_format(row[3])
                            if key not in ["awb_date","bill_of_lading_date","bill_of_lading_issue_date","csh_presentation_date","date_of_invoice","end_date","expiry_date","indicator_date","invoice_date","invoice_due_date","issue_date","lc_date","sail_on_or_about_to_date","shipped_onboard_date","start_date","tenor_indicator_date","transaction_date"]:
                                row[2], row[3] = remove_start_end_spl_char(row[2], row[3])
                            if key in ['pre_carriage_by', 'mode_of_transport']:
                                row[2] = remove_by_prefix(row[2])
                                row[3] = remove_by_prefix(row[3]) 
                            if key in ['net_weight', 'gross_weight','total_quantity_of_goods']:
                                print('entered into  remove alphabets++++++++++++++++')
                                row[2] = keep_specific_characters(row[2])
                                row[3] = keep_specific_characters(row[3])
                            # processing for page no
                            if key in ["incoterm"]:
                                #pass
                                row[2] = preprocess_incoterm(row[2], incoterm_list)
                                row[3] = preprocess_incoterm(row[3], incoterm_list)
                            
                            # post-processing for address
                            if key in ["drawee_bank_address", "drawer_bank_address", "drawer_bank_bottom_address",
                                        "drawee_address", "consignee_address", "consignor_address", "coo_issuer_address", 'nostro_bank_address', 'consignor_address', 'address_of_assured', 'drawee_address', 'insurance_issuer_address', 'remitter_address', 'beneficiary_address', 'coo_issuer_address', 'notify_party_address', 'drawer_bank_address', 'consignee_address', 'drawer_address', 'drawee_bank_address', 'insurance_issuer_address_bottom', 'drawer_bank_bottom_address', 'shipper_address', 'claim_payable_by_address']:
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


                            if key == 'gross_weight' or key== "net_weight" or key== "total_quantity_of_goods":
                                try:
                                    accuracy = fuzz.ratio(float(str(row[2])), float(str(row[3])))

                                except Exception as e:
                                    try:
                                        accuracy = fuzzy_float_comparison(float(str(row[2])), float(str(row[3])))
                                        #print(f'Executed second accuracy {key}: {accuracy}')
                                    except Exception as e:
                                        accuracy = fuzz.ratio(str(row[2]).lower(), str(row[3]).lower())
                                        #print(f'Executed third accuracy {key}: {accuracy}')

                                
                            if key in ["drawee_bank_address", "drawer_bank_address", "drawer_bank_bottom_address",
                                        "drawee_address","page_no","csh_drawn_under_rules", "doc_charge_instructions",
                                        "doc_delivery_instruction","csh_bill_currency","csh_presentation_date","csh_due_date", "consignee_address", "consignor_address", "coo_issuer_address", 'nostro_bank_address', 'consignor_address', 'address_of_assured', 'drawee_address', 'insurance_issuer_address', 'remitter_address', 'beneficiary_address', 'coo_issuer_address', 'notify_party_address', 'drawer_bank_address', 'consignee_address', 'drawer_address', 'drawee_bank_address', 'insurance_issuer_address_bottom', 'drawer_bank_bottom_address', 'shipper_address', 'claim_payable_by_address']:
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
                            try:
                                row.append(predicted[key][0][2])
                                row.append(labels[key][0][1])
                            except:
                                pass
                            data.append(row)
                            if key in ["drawee_bank_address", "drawer_bank_address","drawee_address","nostro_bank_address", "drawer_address", "consignee_address",'consignee_addres','shipper_address','notify_party_address', "consignor_address", "coo_issuer_address", 'nostro_bank_address', 'consignor_address', 'address_of_assured', 'drawee_address', 'insurance_issuer_address', 'remitter_address', 'beneficiary_address', 'coo_issuer_address', 'notify_party_address', 'drawer_bank_address', 'consignee_address', 'drawer_address', 'drawee_bank_address', 'insurance_issuer_address_bottom', 'drawer_bank_bottom_address', 'shipper_address', 'claim_payable_by_address']:

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
                            if key in ['doc_delivery_instruction']:
                                for i in range(l2):
                                    predicted_label[i][0] = doc_delivery_pp(predicted_label[i][0])
                                for j in range(l1):
                                    actual[j][0] = doc_delivery_pp(actual[j][0])
                            if key in ['doc_charge_instructions']:
                                for i in range(l2):
                                    predicted_label[i][0] = doc_chages_pp(predicted_label[i][0])
                                for j in range(l1):
                                    actual[j][0] = doc_chages_pp(actual[j][0])							
                            if key in ['net_weight', 'gross_weight','total_quantity_of_goods']:
                                for i in range(l2):
                                    predicted_label[i][0]=keep_specific_characters(predicted_label[i][0])
                                for j in range(l1):
                                    actual[j][0] = keep_specific_characters(actual[j][0])	
                            if key in ["incoterm"]:
                                for i in range(l2):
                                    #print(f'incoterm predicted: {predicted_label[i][0]}')
                                    predicted_label[i][0]=preprocess_incoterm(predicted_label[i][0], incoterm_list)
                                for j in range(l1):
                                    actual[j][0] = preprocess_incoterm(actual[j][0], incoterm_list)	
                            if key in ["drawee_bank_address", "drawer_bank_address", "drawer_bank_bottom_address",
                                        "drawee_address", "consignee_address","consignor_address", "coo_issuer_address", 'nostro_bank_address', 'consignor_address', 'address_of_assured', 'drawee_address', 'insurance_issuer_address', 'remitter_address', 'beneficiary_address', 'coo_issuer_address', 'notify_party_address', 'drawer_bank_address', 'consignee_address', 'drawer_address', 'drawee_bank_address', 'insurance_issuer_address_bottom', 'drawer_bank_bottom_address', 'shipper_address', 'claim_payable_by_address']:
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
                                    actual[i][0] = generic_page_no(actual[i][0])  #need to check this

                            # processing for cash drawn under rules
                            if key in ["csh_drawn_under_rules"]:
                                for i in range(l2):
                                    predicted_label[i][0] = pp_cash_drawn_rules(predicted_label[i][0])
                                for i in  range(l1):
                                    actual[i][0] = filter_address(actual[i][0])
                                    actual[i][0] = pp_cash_drawn_rules(actual[i][0])  #need to check this
                            if key in ["awb_date","bill_of_lading_date","bill_of_lading_issue_date","csh_presentation_date","date_of_invoice","end_date","expiry_date","indicator_date","invoice_date","invoice_due_date","issue_date","lc_date","sail_on_or_about_to_date","shipped_onboard_date","start_date","tenor_indicator_date","transaction_date"]:#, 'invoice_date']:
                            #if key in ['abc']:
                                pred_list = []
                                actual_list = []
                                for i in range(l2):
                                    predicted_label[i][0] = convert_to_common_format(predicted_label[i][0])
                                for i in range(l1):
                                    actual[i][0] = convert_to_common_format(actual[i][0])
                                    
                                for i in range(l2):
                                    pred_list.append(predicted_label[i][0])
                                for j in range(l1):
                                    actual_list.append(actual[j][0])
                                row.append(actual_list)
                                row.append(pred_list)
                                accuracy = fuzz.ratio(str(row[2]).lower(), str(row[3]).lower())
                                row.append(accuracy)
                                if accuracy == 100:
                                    row.append(1)
                                else:
                                    row.append(0)
                                row.append("predicted[j][2]")
                                row.append("actual_value[1]")
                                data.append(row) 
                            if key not in ["awb_date","bill_of_lading_date","bill_of_lading_issue_date","csh_presentation_date","date_of_invoice","end_date","expiry_date","indicator_date","invoice_date","invoice_due_date","issue_date","lc_date","sail_on_or_about_to_date","shipped_onboard_date","start_date","tenor_indicator_date","transaction_date", "awb_date","bill_of_lading_date","bill_of_lading_issue_date","csh_presentation_date","date_of_invoice","end_date","expiry_date","indicator_date","invoice_date","invoice_due_date","issue_date","lc_date","sail_on_or_about_to_date","shipped_onboard_date","start_date","tenor_indicator_date","transaction_date"]:
                                for i in range(l2):
                                        print(row)
                                        print(predicted_label)
                                        print(f'The value is: {predicted_label[i][0]}')
                                        # exit('+++++++++++++==')
                                        predicted_label[i][0]=remove_spl_char_multiple_pred(predicted_label[i][0])
                                print(f'second condition after remove alphabets: {predicted_label}')
                                to_do = [*range(0, l2, 1)]
                            print("starting to_do are", to_do)
                            if key not in ["drawee_bank_address", "drawer_bank_address","drawee_address","drawer_address", "document_enclosed", "consignee_address","consignor_address", "coo_issuer_address", "description_of_goods", "marks_and_no_of_packages", 'notify_party_name','dimension', 'consignee_addres', 'carrier_country', 'carrier_name', 'agent_country','agent_name','nostro_bank_address', 'consignor_address', 'address_of_assured', 'drawee_address', 'insurance_issuer_address', 'remitter_address', 'beneficiary_address', 'coo_issuer_address', 'notify_party_address', 'drawer_bank_address', 'consignee_address', 'drawer_address', 'drawee_bank_address', 'insurance_issuer_address_bottom', 'drawer_bank_bottom_address', 'shipper_address', 'claim_payable_by_address', 'to_place', 'means_of_transport', "awb_date","bill_of_lading_date","bill_of_lading_issue_date","csh_presentation_date","date_of_invoice","end_date","expiry_date","indicator_date","invoice_date","invoice_due_date","issue_date","lc_date","sail_on_or_about_to_date","shipped_onboard_date","start_date","tenor_indicator_date","transaction_date", "awb_date","bill_of_lading_date","bill_of_lading_issue_date","csh_presentation_date","date_of_invoice","end_date","expiry_date","indicator_date","invoice_date","invoice_due_date","issue_date","lc_date","sail_on_or_about_to_date","shipped_onboard_date","start_date","tenor_indicator_date","transaction_date","drawee_bank_address", 'consignee_addres', 'carrier_country', 'carrier_name', 'agent_country','agent_name', "drawer_address", "drawer_bank_address", "drawer_bank_bottom_address","drawee_address","document_enclosed", "consignee_address","consignor_address", "coo_issuer_address", "description_of_goods", "marks_and_no_of_packages","notify_party_name", 'dimension', 'to_place', 'means_of_transport', 'issue_date', 'date_of_invoice', 'lc_date', 'bill_of_lading_date','invoice_date','issue_date','nostro_bank_address', 'consignor_address', 'address_of_assured', 'drawee_address', 'insurance_issuer_address', 'remitter_address', 'beneficiary_address', 'coo_issuer_address', 'notify_party_address', 'drawer_bank_address', 'consignee_address', 'drawer_address', 'drawee_bank_address', 'insurance_issuer_address_bottom', 'drawer_bank_bottom_address', 'shipper_address', 'claim_payable_by_address']:
                                flag01 = []							
                                for i in range(l1):
                                    new_row = row.copy()
                                    actual_value = actual[i]
                                    actual[i][0]=remove_spl_char_multiple_pred(actual_value[0])   # ******change*************
                                    print(f'the actual value is: {actual_value}')
                                    
                                    to_do = filter_prediction(new_row, actual_value, predicted_label, to_do, key, flag01)
                                print(to_do)
                            if key in ['to_place', 'means_of_transport']:
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
                            if key in ["drawee_bank_address", "drawer_bank_address","drawee_address","drawer_address", "document_enclosed", "consignee_address","consignor_address", "coo_issuer_address", "description_of_goods", "marks_and_no_of_packages", 'notify_party_name','dimension', 'consignee_addres', 'carrier_country', 'carrier_name', 'agent_country','agent_name','nostro_bank_address', 'consignor_address', 'address_of_assured', 'drawee_address', 'insurance_issuer_address', 'remitter_address', 'beneficiary_address', 'coo_issuer_address', 'notify_party_address', 'drawer_bank_address', 'consignee_address', 'drawer_address', 'drawee_bank_address', 'insurance_issuer_address_bottom', 'drawer_bank_bottom_address', 'shipper_address', 'claim_payable_by_address']:
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
                            if key in ["drawee_bank_address", 'agent_country', 'carrier_country', "drawer_bank_address","drawee_address","nostro_bank_address", "drawer_address", "consignee_address", "consignor_address", "coo_issuer_address", 'consignee_addres', 'nostro_bank_address', 'consignor_address', 'address_of_assured', 'drawee_address', 'insurance_issuer_address', 'remitter_address', 'beneficiary_address', 'coo_issuer_address', 'notify_party_address', 'drawer_bank_address', 'consignee_address', 'drawer_address', 'drawee_bank_address', 'insurance_issuer_address_bottom', 'drawer_bank_bottom_address', 'shipper_address', 'claim_payable_by_address']:
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
                            if key in ["awb_date","bill_of_lading_date","bill_of_lading_issue_date","csh_presentation_date","date_of_invoice","end_date","expiry_date","indicator_date","invoice_date","invoice_due_date","issue_date","lc_date","sail_on_or_about_to_date","shipped_onboard_date","start_date","tenor_indicator_date","transaction_date"]:
                                fil_date = convert_to_common_format(str(labels[key][0][0]))
                                new_row.append(fil_date)
                            else:
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
                            if key in ["awb_date","bill_of_lading_date","bill_of_lading_issue_date","csh_presentation_date","date_of_invoice","end_date","expiry_date","indicator_date","invoice_date","invoice_due_date","issue_date","lc_date","sail_on_or_about_to_date","shipped_onboard_date","start_date","tenor_indicator_date","transaction_date"]:
                                print(str(predicted[key][0][0]))
                                fil_date = convert_to_common_format(str(predicted[key][0][0]))
                                row.append(fil_date)
                                if row[0]=='Covering_Schedule_231_page_1.png' and key == 'csh_presentation_date':
                                    print('>>>>>>>>>',fil_date)
                                    # exit("???????KKKKKKKKKK")
                            else:
                                row.append(str(predicted[key][0][0]))                        
                            # row.append(str(predicted[key][0][0]))
                            row.append(0)
                            row.append(0)
                            try:
                                row.append(predicted[key][0][2])
                                row.append(predicted[key][0][1])
                            except:
                                pass
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

        max_length = 8
        filtered_data = [sublist for sublist in data if len(sublist) <= max_length]
        df = pd.DataFrame(filtered_data, columns=column_names)

        df.loc[df["Accuracy"].apply(float) >= 90, "Match/No_Match"] = 1

        res_path= os.path.join(folder_path, 'result_path', f"{doc_code}_{datetime.now().date()}_{datetime.now().hour}")

        if not os.path.exists(res_path):
            os.makedirs(res_path)

        df.to_csv(f'{res_path}/{doc_code}_analysis_pre_valid_{datetime.now()}_after_fuzzy_match_post-processing_latest_{model}.csv')
        