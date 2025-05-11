import glob
import os
import json
import pandas as pd
from fuzzywuzzy import fuzz
import re
from difflib import get_close_matches
from post_processing_master_gt import generic_page_no
# from post_processing_pred import pp_cash_drawn_rules
from dateutil.parser import parse
# from test_country import extract_country_from_text
from typing import List
from configparser import ConfigParser
from validation_utility import final_overall_analysis, final_report, stp_report


"""This accuracy generation script used in Insurance certificate document"""

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

# import locationtagger

# def loc_tagger(sampl_text):
#     place_entity = locationtagger.find_locations(text = sampl_text)
#     return place_entity.countries

def remove_start_end_spl_char(actual_text: str, pred_text:str):
	# print(f'actual text : {actual_text}')
	# print(f'pred text : {pred_text}')

	actual_text= actual_text.split(' ')
	pred_text= pred_text.split(' ')
	# print(actual_text)
	# print(pred_text)
	if len(actual_text)== len(pred_text):
		for actual, pred in zip(actual_text, pred_text):
			# print('length of actual and predicted is same+++++++++')
			# print(actual)
			# print(pred)
			
			actual_indics= actual_text.index(actual)
			# print(actual_indics)
			pred_indices= pred_text.index(pred)
			# print(pred_indices)
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
	# print(f'actual text: {actual_text}')
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
	# print("value1 is", actual_value)
	# print("list of predicted values are", predicted)
	# print("to_do is", to_do)
	l2 = len(predicted)
	for j in range(l2):
		# print(" j is", str(j))
		intersection = get_iou_new(actual_value[1], predicted[j][1])
		if intersection > 0.25:
			# print("match found")
			# print(f'entered after intersection {actual[i][0]}')
			# exit()
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

def strip_extra_spaces(text):
    return re.sub(r'\s+',' ', text)

def extract_dates_from_string(text):
    dates = []
    text = strip_extra_spaces(text)
    words = text.split()
    # print(words)
    #exit()
    for word in words:
        try:
            #date = parse(word, fuzzy=True)
            dates.append(word)
        except ValueError:
            if len(word) == 8 and word.isdigit():
                dates.append(word)
    return dates
def eval_date_predictions(actual, predicted):
    
    actual_value = actual
    # print(actual_value)
    # print(predicted)
    #exit()
    predicted_value_list = extract_dates_from_string(predicted)
    # print(predicted_value_list)
    #exit()
    best_match = predicted
    max_similarity = 60

    for predicted_value in predicted_value_list:
        if predicted_value == actual_value:
            return predicted_value
        else:
            similarity = fuzz.ratio(str(predicted_value), str(actual_value[1]))
            # print(similarity)
            #exit()
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = predicted_value
    # print(best_match)     
    #exit()       
    return best_match

def validate_date(date_str):
    try:
		
        parsed_date = parse(date_str, fuzzy=False)
        return True,date_str
    except ValueError:
        x=" ".join(str(i) for i in extract_dates_from_string(date_str))
		

        return False, x



def filter_prediction(new_row, actual_value, predicted, to_do):
	l2 = len(predicted)
	actual_value_lower = actual_value[0].lower()
	max_confidence = 0
	max_index = None

	for j in range(l2):
		if actual_value_lower == predicted[j][0].lower():
			new_row.append(actual_value[0])
			new_row.append(predicted[j][0])
			accuracy = fuzz.ratio(new_row[2].lower(), new_row[3].lower())
			new_row.append(accuracy)
			if accuracy == 100:
				new_row.append(1)
			else:
				new_row.append(0)  
			try:
				new_row.append(predicted[j][2])		
				new_row.append(actual_value[1])
			except:
				pass
			data.append(new_row)
			try:
				to_do.remove(j)
			except:
				pass
			return to_do
		try:
			if predicted[j][2] > max_confidence:
				max_confidence = predicted[j][2]
				max_index = j
		except:
			pass
	if max_index is not None:
		new_row.append(actual_value[0])
		new_row.append(predicted[max_index][0])
		accuracy = fuzz.ratio(new_row[2].lower(), new_row[3].lower())
		new_row.append(accuracy)
		if accuracy >= 90:
			new_row.append(1)
		else:
			new_row.append(0)
		new_row.append(predicted[max_index][2])
		new_row.append(actual_value[1])
		data.append(new_row)
		try:
			to_do.remove(max_index)
		except:
			pass
	else:
		new_row.append(actual_value[0])
		new_row.append("")
		new_row.append(0)
		new_row.append(0)
		new_row.append(0)
		new_row.append(actual_value[1])
		data.append(new_row)

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


def filter_address(x):
	x = str(x)
	x = x.strip()
	new_x = re.sub(r'\s+', ' ', x)
	# print("newx")
	# print(new_x)
	# remove the special characters like comma, semicolon etc
	new_x = "".join([x for x in new_x if x.isalnum() or x in [" "]])
	# print(new_x)
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
	
def remove_label_name(text,regex):
    # Remove using regular expression    
    match = re.search(regex, text)
    if match:
        text = re.sub(regex, "", text)
        return text.strip()
    return text.strip()

def remove_special_chars(text):
    # Define the pattern for special characters
    pattern = r'^[\W_]+|[\W_]+$'
    # Remove special characters from start and end of text
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

def remove_words(text,words_to_remove_list):
    pattern = r'\b(?:{})\b'.format('|'.join(map(re.escape, words_to_remove_list)))
    cleaned_text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text)
    cleaned_text = cleaned_text.replace('&', '')
    cleaned_text = cleaned_text.strip()
    return cleaned_text

def remove_text_after_phrases(text):
    cleaned_text = re.sub(r'(Ph\.|Tel\.|Zip|Phone)\s*.*', '', text, flags=re.IGNORECASE)
    cleaned_text = cleaned_text.strip()
    return cleaned_text



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

def mapping_currency_amt(amt_lis):
	for i in range (len(amt_lis)):
		abc, amt_lis[i] = extract_currency_and_amount(amt_lis[i])
		amt_lis[i] = amt_lis[i].replace(',', '')
		# amt_lis[i] = amt_lis[i].replace('.', '')
	# print('*************')
	# print(amt_lis)
	amt_lis = [item for item in amt_lis if item != '' or item!=None ]
	try:
		if len(amt_lis)==2:
			amt_lis = [float(s) for s in amt_lis]
			# print(type(amt_lis[0]))
			# print(amt_lis[0])
			smaller_value = min(amt_lis)
			for i in range(len(amt_lis)):
				if amt_lis[i] == smaller_value:
					amt_lis[i] = '~'+str(amt_lis[i])
		# print(amt_lis)
	except:
		pass
	final_amts = ''
	if len(amt_lis)>0:
		for i in amt_lis:
			final_amts = final_amts+str(i)
			final_amts = final_amts+" "
	return final_amts

def wrapping_up(key, pre_act_dict, flag):
    sum_insured_currency_lis = []
    # print(pre_act_dict[key])
    for i in range(len(pre_act_dict[key])):
        if type(pre_act_dict[key][i][0]) is list:
            # print('???????????????', labels['sum_insured_currency'][i][0])
            for j in pre_act_dict[key][i][0]:
                # print('jjjjjjjjjjjjjjjjjjjjjjjjjjjjjjj', j)
                sum_insured_currency_lis.append(j)
                bbox = pre_act_dict[key][i][1]
                if flag =='pred':
                    confi = pre_act_dict[key][i][2]
        else:
            sum_insured_currency_lis.append(pre_act_dict[key][i][0])
            bbox = pre_act_dict[key][i][1]
            if flag =='pred':
                confi = pre_act_dict[key][i][2]
    if flag == 'actual':
        pre_act_dict[key]=[sum_insured_currency_lis, bbox]
    elif flag == 'pred':
        pre_act_dict[key]=[sum_insured_currency_lis, bbox, confi]
    return pre_act_dict

def clean_date(original_string):
	# pattern = r'[^a-zA-Z0-9,./-]'
	pattern = r'[^a-zA-Z0-9,./\s-]'
	# Use re.sub() to replace the matched special characters with an empty string
	cleaned_string = re.sub(pattern, '', original_string)
	return cleaned_string

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

import dateparser

def convert_to_common_format(date_str):
	try:
		updated_date = dateparser.parse(date_str).strftime("%d-%m-%Y")
	except:
		updated_date = ''
	return updated_date

def convert_to_common_format_(date_str):
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
def remove_spaces(s):
    # Removes spaces from a string
    return "".join(char for char in s if char != " ")

def remove_spl_char_spaces(text: str):
    #removing the double spaces in beween the words, also removes the special chars in each words, also remove the space at the start and end
    # Remove special characters within words and leading/trailing spaces
    words = text.split()
    cleaned_words = []

    for word in words:
        cleaned_word = "".join(ch for ch in word if ch.isalnum())
        if cleaned_word:
            cleaned_words.append(cleaned_word)

    cleaned_text = " ".join(cleaned_words)
    return cleaned_text


def check_and_modify_labels(json_data, bottom_value,label):
    if label in json_data and bottom_value in json_data:
        json_data[bottom_value][0][0] = "~ "+json_data[bottom_value][0][0]
        
        #joined_value = "~ ".join([json_data[label][0][0], json_data["insurance_issuer_address_bottom"][0][0]])
        joined_value = json_data[label]+json_data[bottom_value]
        json_data[label] = joined_value
        del json_data[bottom_value]
    else:
        if bottom_value in json_data:
            json_data[label] = json_data.pop(bottom_value)

    return json_data


def  merge_top_bottom_keys(key_name_changes, my_dict):
    for old_key, new_key in key_name_changes.items():
        if old_key in my_dict:
            # Get the value associated with the old key
            value = my_dict[old_key]

            # Delete the old key-value pair
            del my_dict[old_key]

            # Add the new key-value pair
            my_dict[new_key] = value
    return my_dict


def fuzzy_compare_ignore_spaces(str1, str2):
    #removing the double spaces in beween the words, also removes the special chars in each words, also remove the space at the start and end
    # Remove spaces from both strings
    str1_without_spaces = remove_spl_char_spaces(str1)
    str2_without_spaces = remove_spl_char_spaces(str2)
    print(str1_without_spaces)
    print(str2_without_spaces)

    # Calculate the Jaccard index between the modified strings
    similarity = fuzz.token_set_ratio(str1_without_spaces, str2_without_spaces)

    # You can adjust the threshold based on your requirement
    threshold = 80  # For example, consider strings with similarity 80 or higher as similar

    return similarity >= threshold

# import nltk
# nltk.download('averaged_perceptron_tagger')

from datetime import datetime
if __name__ == '__main__':
    start_time = datetime.now()

    # Get the file name of the executed Python script
    file_name = os.path.basename(__file__)
    # count = 0
    occurrences = {}
    data = []
    doc_code_ = 'ic'
    #folder_path: str = "/home/ntlpt19/Downloads/Trade_finance_imp_stage_2/CS_NEW_ROOT/CS_EVAL"
    configur = ConfigParser()
    configur.read('/home/ntlpt19/Downloads/Evaluation_Data/updated_code/src/main/extraction/inference_utility.ini')
    folder_path = str(configur['PATHS']['folder_path'])
    result_path = os.path.join(folder_path, "Results_Images")
    data_path = os.path.join(folder_path, "New_Master_Data_Merged")

    # import json
    data_files = os.listdir(data_path)
    result_files = os.listdir(result_path)
    incoterm_list=[]

    with open("incoterm_list.txt") as fp:
        for line in fp:
            incoterm_list.append(line.strip())
    # print(incoterm_list)
    # exit('++++++++++')

    # check how many pngs are there in those two folders
    # print("Number of images in results", len(glob.glob(result_path + "/*.png")))
    # print("Number of images in master data", len(glob.glob(data_path + "/*.png")))

    # exit("+++++++++++")

    # validating the number of files coming from both these folders
    # print(f"number of files coming from the Master data folder: {len(data_files)}")
    # print(f"number of files coming from the Result cs validated folder: {len(result_files)}")

    # exit("++++++++++++++")

    new = []
    new_result = []

    # FINDING LIST OF DATAFILES AND RESULT FILES
    num_labels_files_traversed: int = 0
    for file in data_files:
        if str(file)[-10:] == "labels.txt":
            # print(f"file number is {num_labels_files_traversed}")
            num_labels_files_traversed = num_labels_files_traversed + 1
            # print(str(file)[-10:])
            new.append(file)
            new_result.append(file[:-11] + ".txt")

    # exit("++++++++++++++++")
    data_files = new
    result_files = new_result
    # print(result_files)
    # print(len(data_files))
    # print(len(result_files))
    # exit("+++++++++")

    # creating a dictionary containing number of occurrences of all our fields in our dataset.
    for count, file in enumerate(data_files):
        # print("count is:", count)
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

    # print(f'Number of classes used : {len(names)}')
    # print(names)
    # exit("+++++++++++++++++")
    column_names.append("File_Name")
    column_names.append("label_name")
    column_names.append("actual")
    column_names.append("predicted")
    column_names.append("Accuracy")
    column_names.append("Match/No_Match")
    column_names.append("model_confidence")
    column_names.append("bbox")
    # print(column_names)
    # print(len(column_names))
    # name = str(result_files[i])[0:-4]
    for file, predicted_files in zip(data_files, result_files):
        # print("file name is:", file)
        # print("resulted filename is", predicted_files)
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
                print(f'''printing the path: {result_path, file[0:-11] + "_s_11.txt"}''')
                with open(os.path.join(result_path, file[0:-11] + "_s_11.txt"), "r") as f2:
                    predicted = json.load(f2)
            # print("opened")
            except:
                print("still not opened")
                continue

        # print(f'acutal labels: {labels}')
        # print(f'number of keys : {len(list(labels.keys()))}')
        # print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++=')
        # print(f'predicted labels: {predicted}')
        # print(f'predicted labels: {len(predicted)}')

        # exit('+++++++++++==')
        top_value = ["drawer_bank_address",'drawer_bank_name','drawer_bank_bic', 'insurance_issuer_address', 'insurance_issuer_name']
        bottom_value = ["drawer_bank_bottom_address",'drawer_bank_bottom_name','drawer_bank_bottom_bic', 'insurance_issuer_address_bottom', 'insurance_issuer_name_bottom']
        if len(top_value)== len(bottom_value):
            for i in range(len(top_value)):
                labels = check_and_modify_labels(labels,bottom_value[i], top_value[i])
                predicted = check_and_modify_labels(predicted,bottom_value[i], top_value[i])
        else:
            print('The len of top and bottom values should be same')
        if predicted == {} and labels == {}:
            print("*******")
            # print(file)
            continue
        actual_list = list(labels)
        pred_list = list(predicted)

        if 'sum_insured_amount' in actual_list:
                # exit('???????????????????????????????????')
            index_to_delete = []
            for i in range(len(labels['sum_insured_amount'])):
                amount_list = list()
                sum_insured = labels['sum_insured_amount'][i][0].split()
                for j in sum_insured:
                    act_currency, act_amount = extract_currency_and_amount(str(j))
                    if len(act_currency)>0 and 'sum_insured_currency' in actual_list:
                        labels['sum_insured_currency'].append([act_currency, labels['sum_insured_amount'][i][1]])
                        # labels['csh_bill_amount'].append(labels['currency_amount'][i][1])
                    else:
                        if len(act_currency)>0:
                            labels['sum_insured_currency']=[[act_currency, labels['sum_insured_amount'][i][1]]]
                    if len(act_amount)>0: #and 'csh_bill_currency' in actual_list:
                        amount_list.append(act_amount)
                if len(amount_list)>0:
                    labels['sum_insured_amount'][i][0] = amount_list
                else:
                    index_to_delete.append(i)
            if len(index_to_delete)>0:
                for d in index_to_delete:
                    del labels['sum_insured_amount'][d]
                if len(labels['sum_insured_amount'])==0:
                    del labels['sum_insured_amount']
                    # del labels['sum_insured_amount'][i]
                #else => needed to delete the i th element in the sum_insured_amount , if required.
        # if 'sum_insured_amount' in actual_list:

        if 'sum_insured_amount' in pred_list:
            index_to_delete = []
            for i in range(len(predicted['sum_insured_amount'])):
                amount_list = []
                sum_insured = predicted['sum_insured_amount'][i][0].split()
                for j in sum_insured:
                    act_currency, act_amount = extract_currency_and_amount(str(j))
                    if len(act_currency)>0 and 'sum_insured_currency' in pred_list:
                        predicted['sum_insured_currency'].append([act_currency, predicted['sum_insured_amount'][i][1], predicted['sum_insured_amount'][i][2]])
                        # labels['csh_bill_amount'].append(labels['currency_amount'][i][1])
                    else:
                        if len(act_currency)>0:
                            predicted['sum_insured_currency']=[[act_currency, predicted['sum_insured_amount'][i][1], predicted['sum_insured_amount'][i][2]]]
        
                    
                    if len(act_amount)>0: #and 'csh_bill_currency' in actual_list:
                        amount_list.append(act_amount)
                if len(amount_list)>0:
                    predicted['sum_insured_amount'][i][0] = amount_list
                else:
                    index_to_delete.append(i)
            if len(index_to_delete)>0:
                for d in index_to_delete:
                    del predicted['sum_insured_amount'][d]
                if len(predicted['sum_insured_amount'])==0:
                    del predicted['sum_insured_amount']


        if 'sum_insured_currency' in actual_list:
            index_to_delete = []
            for i in range(len(labels['sum_insured_currency'])):
                if file[0:-11]=='Insurance_Certificate_34_page_0':
                    print(labels['sum_insured_currency'])
                    # exit()
                currency_list = []
                sum_insured = labels['sum_insured_currency'][i][0].split()
                if file[0:-11]=='Insurance_Certificate_34_page_0':
                    print(labels['sum_insured_currency'])
                    print(sum_insured)
                    # exit()
                for j in sum_insured:
                    act_currency, act_amount = extract_currency_and_amount(str(j))
                    if len(act_amount)>0 and 'sum_insured_amount' in actual_list:
                        labels['sum_insured_amount'].append([act_amount, labels['sum_insured_currency'][i][1]])
                        # labels['csh_bill_amount'].append(labels['currency_amount'][i][1])
                    else:
                        if len(act_amount)>0:
                            labels['sum_insured_amount']=[[act_amount, labels['sum_insured_currency'][i][1]]]
        
                    
                    if len(act_currency)>0: #and 'csh_bill_currency' in actual_list:
                        currency_list.append(act_currency)
                        
                                                                                                                                                                                                                                                                                                           
                if len(currency_list)>0:
                    labels['sum_insured_currency'][i][0] = currency_list

                else:
                    index_to_delete.append(i)
            if len(index_to_delete)>0:
                for d in index_to_delete:
                    del labels['sum_insured_currency'][d]
                if len(labels['sum_insured_currency'])==0:
                    del labels['sum_insured_currency']
                    
        if 'sum_insured_currency' in pred_list:
            index_to_delete = []
            for i in range(len(predicted['sum_insured_currency'])):
                currency_list = []
                sum_insured = predicted['sum_insured_currency'][i][0].split()
                for j in sum_insured:
                    act_currency, act_amount = extract_currency_and_amount(str(j))
                    if len(act_amount)>0 and 'sum_insured_amount' in pred_list:
                        predicted['sum_insured_amount'].append([act_amount, predicted['sum_insured_currency'][i][1], predicted['sum_insured_currency'][i][2]])
                        # labels['csh_bill_amount'].append(labels['currency_amount'][i][1])
                    else:
                        if len(act_amount)>0:
                            predicted['sum_insured_amount']=[[act_amount, predicted['sum_insured_currency'][i][1], predicted['sum_insured_currency'][i][2]]]
        
                    
                    if len(act_currency)>0: #and 'csh_bill_currency' in actual_list:
                        currency_list.append(act_currency)
                    
                if len(currency_list)>0:
                    predicted['sum_insured_currency'][i][0] = currency_list
                else:
                    index_to_delete.append(i)
            if len(index_to_delete)>0:
                for d in index_to_delete:
                    del predicted['sum_insured_currency'][d]
                if len(predicted['sum_insured_currency'])==0:
                    del predicted['sum_insured_currency']
                #else => needed to delete the i th element in the sum_insured_currency if there is nothing found in currency , but it is present in the prediction 
                # eg: 'sum_insured_currency' : 2,340 => it will be filterd as sum_insured_amount and sum_insured_currency will be empty , 
                # in this case we need to delete i th element in sum_insured_currency
                #else => needed to delete the i th element in the sum_insured_amount , if required.(wrong)
                #sample:
                '''Insurance_Certificate_1_page_0_labels.txt
                actual_currency #######################################
                [[['USD'], [392, 1287, 763, 1342]]]
                predicted_curreny #######################################
                [['( 3 )', [390, 1287, 416, 1308], 66.34], [['usd'], [718, 1319, 759, 1336], 96.96]]
                '( 3 )'==> got filtered and appended to the amount  and remain nothing in currency, as we are not deleting the i th element it remain as it is
                predicted_amount ####################################
                [[['314600', '4840'], [660, 1291, 736, 1336], 49.89], ['3', [390, 1287, 416, 1308]]]'''
        if 'sum_insured_amount' in list(labels):
            print(file)
            print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxx', labels['sum_insured_amount'])
            
            labels = wrapping_up('sum_insured_amount', labels, 'actual')
            print('actual #######################################')
            print('actual #######################################')
            print('actual #######################################')
            print('yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy', labels['sum_insured_amount'])
           
            
        if 'sum_insured_currency' in list(labels):
            print(file)
            labels = wrapping_up('sum_insured_currency', labels, 'actual')
            print('actual #######################################')
            print('actual #######################################')
            print('actual #######################################')
            print('yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy',labels['sum_insured_currency'])   
            # print(labels['sum_insured_amount'])
            
            #[[['060.00'], [1310, 692, 1432, 714]], ['49', [1269, 690, 1309, 714]]]
        if 'sum_insured_amount' in list(predicted):
            print(file)
            predicted = wrapping_up('sum_insured_amount', predicted, 'pred')
            print('actual #######################################')
            print('actual #######################################')
            print('actual #######################################')
            print('yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy', predicted['sum_insured_amount'])
            
            
        if 'sum_insured_currency' in list(predicted):
            print(file)
            predicted = wrapping_up('sum_insured_currency', predicted, 'pred')
            print('actual #######################################')
            print('actual #######################################')
            print('actual #######################################')
            print('yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy',predicted['sum_insured_currency']) 
            
        for key in list(occurrences.keys()):
            if key not in ["certificate_Stamped", "signature"]:
                # print("key is", key)
                row = []
                row.append(file[0:-11] + ".png")
                if key in labels and key in predicted:
                    actual = labels[key]
                    predicted_label = predicted[key]
                    if key in ['sum_insured_currency', 'sum_insured_amount']:
                        act_val = ''
                        pre_val = ''
                        act_val = str(actual[0])
                        pre_val = str(predicted_label[0])
                        print(act_val)
                        # exit()
                        print(actual)
                        # for i in range(len(actual)):
                        #     act_val += str(actual[i][0])+", "
                        # for j in range(len(predicted_label)):
                        #     pre_val += str(predicted_label[j][0])+", "
                        row.append(key)    
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
                    
                    if key not in ['sum_insured_currency', 'sum_insured_amount']:
                        if len(labels[key]) == 1 and len(predicted[key]) == 1:
                            if key not in ['']:
                                row.append(key)
                                row.append(str(labels[key][0][0]))
                                row.append(str(predicted[key][0][0]))
                                # print(f'row data: {row}')
                                # print(str(row[2]).lower(), str(row[3]).lower())
                                #remove special chars in starting and ending of the string
                                if key not in ['invoice_date', 'csh_due_date', 'transaction_date', 'lc_date', 'csh_presentation_date', 'expiry_date', 'bill_of_lading_issue_date', 'sail_on_or_about_to_date', 'date_of_invoice', 'tenor_indicator_date', 'shipped_onboard_date', 'issue_date', 'awb_date', 'bill_of_lading_date']:
                                    row[2]=remove_special_chars(row[2])
                                    row[3]=remove_special_chars(row[3])
                                    row[2], row[3] = remove_start_end_spl_char(row[2], row[3])
                                print(f'after remove spl chars: {row}')
                                # exit()
                                
                                if key in ['lc_date','end_date','start_date','issue_date','sail_on_or_about_to_date','expiry_date', 'invoice_date', 'csh_due_date', 'transaction_date', 'lc_date', 'csh_presentation_date', 'expiry_date', 'bill_of_lading_issue_date', 'sail_on_or_about_to_date', 'date_of_invoice', 'tenor_indicator_date', 'shipped_onboard_date', 'issue_date', 'awb_date', 'bill_of_lading_date']:
                                    """row[2]=row[2].strip()
                                    row[3]=row[3].strip()"""
                                    # if row[2].strip().lower()!=row[3].strip().lower():	
                                    #     #row[2]=validate_date(row[2])[1]
                                    #     row[3]=eval_date_predictions(row[2], row[3])
                                    print(row[2], row[3])
                                    row[2] = clean_date(row[2])
                                    row[3] = clean_date(row[3])
                                    # row[2] = convert_to_common_format(row[2])
                                    pred_date = row[3]
                                    filter_date = convert_to_common_format(row[3])
                                    if filter_date!="":
                                        row[3] = filter_date
                                    elif len(pred_date)==6:
                                        formatted_date_str = '-'.join([pred_date[i:i+2] for i in range(0, len(pred_date), 2)])
                                        row[3] = formatted_date_str
                                    elif filter_date=='':
                                        filter2 = convert_to_common_format_(pred_date)
                                        if filter2!='':
                                            row[3] = filter2
                                        else:
                                            row[3] = pred_date
                                    else:
                                        row[3] = pred_date

                                    act_date = row[2]
                                    filter_date = convert_to_common_format(row[2])
                                    if filter_date!="":
                                        row[2] = filter_date
                                    elif len(act_date)==6:
                                        formatted_date_str = '-'.join([act_date[i:i+2] for i in range(0, len(act_date), 2)])
                                        row[2] = formatted_date_str
                                    elif filter_date=='':
                                        filter2 = convert_to_common_format_(act_date)
                                        if filter2!='':
                                            row[2] = filter2
                                        else:
                                            row[2] = act_date
                                    else:
                                        row[2] = act_date
                                # post-processing for 
                                if key in ['net_weight', 'gross_weight','total_quantity_of_goods']:
                                    print('entered into  remove alphabets++++++++++++++++')
                                    print(row[2])
                                    print(row[3])
                                    row[2] = remove_alphabets(row[2])
                                    row[3] = remove_alphabets(row[3])
                                    print(row[2])
                                    print(row[3])
                                # processing country name using lookup
                                # if key in ['carrier_country',"country_of_final_destination","country_of_origin_of_goods", 'agent_country',"country_of_origin_origin_of_goods"]:
                                # 	row[2] = extract_country_from_text(row[2])
                                # 	row[3] = extract_country_from_text(row[3])
                                #removing port of loading and port of discharge text out of data
                                if key in ['port_of_discharge', 'port_of_loading']:
                                    if key =='port_of_discharge':
                                        regex="([PDo])?(o)?(r)?(t)?(\s)?(o)?(f)?(\s)?([DO0])?(i)?(s)?(c)?(h)?(a)?(r)?(g)?(e)?"
                                        row[2] = remove_label_name(row[2],regex)
                                        row[3] = remove_label_name(row[3],regex)
                                    elif key =='port_of_loading':
                                        regex=r"([PDo])?(o)?(r)?(t)?(\s)?(o)?(f)?(\s)?([IL1T\|])?(o)?(a)?(d)?(i)?(n)?(g)?"
                                        row[2] = remove_label_name(row[2],regex)
                                        row[3] = remove_label_name(row[3],regex)
                                if key in ["page_no","original_Number","to_place","lc_ref_no","lc_ref_number","consignee_name"]:
                                    regex=[r"^(?i)To\s*:",r"^(?i)(?:LC\s*NO\.?\s*|NO\.|no\.)","(?i)(page|no\.|no|pago\s*no\.|pago|Page)",r"(?i)NOTIFY\s*:\s*"]
                                    if key =='page_no' or key=="original_Number":
                                        row[2] = generic_page_no(remove_special_chars(remove_label_name(row[2],regex[2])))
                                        row[3] = generic_page_no(remove_special_chars(remove_label_name(row[3],regex[2])))
                                    elif key =='consignee_name':
                                        row[2] = remove_label_name(row[2],regex[3])
                                        row[3] = remove_label_name(row[3],regex[3])
                                    elif key =='to_place':
                                        row[2] = remove_label_name(row[2],regex[0])
                                        row[3] = remove_label_name(row[3],regex[0])
                                    elif key =='lc_ref_no'or "lc_ref_number":
                                        row[2] = remove_label_name(row[2],regex[1])
                                        row[3] = remove_label_name(row[3],regex[1])
                                if key in ["pre_carriage_by","consignor_name","declaration_by","declaration","mode_of_transport","from_place"]:
                                    if key =="consignor_name":										
                                        row[2] = remove_words(row[2],['EXPORTER'])
                                        row[3] = remove_words(row[3],['EXPORTER'])
                                    if key =="from_place":										
                                        row[2] = remove_words(row[2],['FROM'])
                                        row[3] = remove_words(row[3],['FROM'])
                                    if key =="pre_carriage_by"or key =="mode_of_transport":										
                                        row[2] = remove_words(row[2],['BY',"EXPORT"])
                                        row[3] = remove_words(row[3],['BY',"EXPORT"])				
                                    if key =="declaration_by" or key=="declaration":
                                        row[2] = remove_words(row[2],["REMOVE", "FOR", "NAME", "OF", "THE", "AUTHORISED", "SIGNATORY", "SEAL"])
                                        row[3] = remove_words(row[3],["REMOVE", "FOR", "NAME", "OF", "THE", "AUTHORISED", "SIGNATORY", "SEAL"])
                                if key in ['drawer_bank_address',
                'consignee_address',
                'insurance_issuer_address',
                'beneficiary_address',
                'drawer_bank_bottom_address',
                'coo_issuer_address',
                'shipper_address',
                'address_of_assured',
                'drawer_address',
                'consignor_address',
                'notify_party_address',
                'claim_payable_by_address',
                'drawee_address',
                'drawee_bank_address',
                'remitter_address',
                'claim_payable_by_address',
                'nostro_bank_address'
            ]:
                                    if key == 'insurance_issuer_address':
                                        for i in range(2, 4):
                                            values = row[i].split('~')
                                            for j in range(len(values)):
                                                values[j] = remove_text_after_phrases(filter_address(values[j]))
                                            row[i] = '~'.join(values)
                                    else:
                                        row[2] = remove_text_after_phrases(filter_address(row[2]))
                                        row[3] = remove_text_after_phrases(filter_address(row[3]))
                                
                                    

                                # processing for page no
                                if key in ["incoterm"]:
                                    row[2] = preprocess_incoterm(row[2], incoterm_list)
                                    row[3] = preprocess_incoterm(row[3], incoterm_list)
                                print(f'the row[2] value: {row[2]}')
                                print(f'the row[3] value: {row[3]}')
                                print(f'type of row[2]: {type(row[2])}')
                                print(f'type of row[3]: {type(row[3])}')
                                # accuracy = fuzz.ratio(str(row[2]).lower(), str(row[3]).lower())
                                # print(f'accuracy of {key}: {accuracy}')
                                if key == 'gross_weight' or key== "net_weight" or key== "total_quantity_of_goods":
                                    try:
                                        accuracy = fuzz.ratio(float(str(row[2])), float(str(row[3])))
                                        print(f'type of row[2]: {type(int(row[2]))}')
                                        print(f'type of row[3]: {type(int(row[3]))}')
                                        print(f'accuracy of {key}: {accuracy}')
                                    except Exception as e:
                                        try:
                                            accuracy = fuzzy_float_comparison(float(str(row[2])), float(str(row[3])))
                                            print(f'Executed second accuracy {key}: {accuracy}')
                                        except Exception as e:
                                            print(f'the row[2] value: {row[2]}')
                                            print(f'the row[3] value: {row[3]}')
                                            accuracy = fuzz.ratio(str(row[2]).lower(), str(row[3]).lower())
                                            print(f'Executed third accuracy {key}: {accuracy}')

                                else:
  
                                    if fuzzy_compare_ignore_spaces(row[2], row[3]):
                                        accuracy = 100
                                    else:
                                        row_2= remove_spl_char_spaces(row[2])
                                        row_3=  remove_spl_char_spaces(row[3])
                                        accuracy = fuzz.ratio(str(row_2).lower(), str(row_3).lower())
                                if key not in ['lc_date','end_date','start_date','issue_date','sail_on_or_about_to_date','expiry_date', 'invoice_date', 'csh_due_date']:
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
                                # print(data)
                                # exit('+++++++')
                                if key in ['claim_payable_by_address', "address_of_assured", "claim_payable_by_address","insurance_issuer_address"]:

                                    act_val_address = labels[key][0][0]
                                    pre_val_address = predicted[key][0][0]
                                    act_val_address, pre_val_address = remove_start_end_spl_char(act_val_address, pre_val_address)
                                    #pre_val_address = remove_start_end_spl_char(pre_val_address)
                                    
                                    act_val_address = filter_address(act_val_address)
                                    pre_val_address = filter_address(pre_val_address)
                                    print('act_val_address', act_val_address)
                                    #exit()
                                    actual_country = extract_country_from_text(act_val_address)
                                    pred_country = extract_country_from_text(pre_val_address)
                                    # if len(actual_country)==0:
                                    #     actual_country = loc_tagger(act_val_address)
                                    # if len(pred_country)==0:
                                    #     pred_country = loc_tagger(pre_val_address)
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
                            if key not in ['sum_insured_amount', 'sum_insured_currency']:
                                row.append(key)
                                actual = labels[key]
                                predicted_label = predicted[key]
                                actual_value=str(labels[key][0][0])
                                print(f'second condition: {actual}')
                                print(f'second condition: {predicted_label}')
                                print(row)
                                l1 = len(actual)
                                l2 = len(predicted_label)
                                                    
                                if key in ["country_of_final_destination","country_of_origin_of_goods", 'agent_country',"country_of_origin_origin_of_goods"]:
                                    for i in range(l2):
                                        predicted_label[i][0]=extract_country_from_text(predicted_label[i][0])
                                    
                                if key in ['lc_date','end_date','start_date','issue_date','sail_on_or_about_to_date','expiry_date', 'invoice_date', 'csh_due_date','invoice_date', 'csh_due_date', 'transaction_date', 'lc_date', 'csh_presentation_date', 'expiry_date', 'bill_of_lading_issue_date', 'sail_on_or_about_to_date', 'date_of_invoice', 'tenor_indicator_date', 'shipped_onboard_date', 'issue_date', 'awb_date', 'bill_of_lading_date']:
                                    pred_list = []
                                    actual_list = []
                                    for i in range(l2):
                                        pred_date = predicted_label[i][0]
                                        pred_date = clean_date(pred_date)
                                        filter_date = convert_to_common_format(predicted_label[i][0])
                                        if filter_date!="":
                                            predicted_label[i][0] = filter_date
                                        elif len(pred_date)==6:
                                            formatted_date_str = '-'.join([pred_date[i:i+2] for i in range(0, len(pred_date), 2)])			
                                            predicted_label[i][0] = formatted_date_str
                                        elif filter_date=='':
                                            filter2 = convert_to_common_format_(pred_date)
                                            if filter2!='':
                                                predicted_label[i][0] = filter2
                                            else:
                                                predicted_label[i][0] = pred_date
                                        else:
                                            predicted_label[i][0] = pred_date
                                    for i in range(l1):
                                        act_date = actual[i][0]
                                        # act_date = clean_date(act_date)
                                        filter_date = convert_to_common_format(actual[i][0])
                                        if filter_date!="":
                                            actual[i][0] = filter_date
                                        elif len(act_date)==6:
                                            formatted_date_str = '-'.join([act_date[i:i+2] for i in range(0, len(act_date), 2)])			
                                            actual[i][0] = formatted_date_str
                                        elif filter_date=='':
                                            filter2 = convert_to_common_format_(act_date)
                                            if filter2!='':
                                                actual[i][0] = filter2
                                            else:
                                                actual[i][0] = act_date
                                        else:
                                            actual[i][0] = act_date         
                
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

                                    # for i in range(l2):
                                    #     if actual_value.strip().lower()!=predicted_label[i][0].strip().lower():
                                    #         predicted_label[i][0]=eval_date_predictions(actual_value, predicted_label[i][0])
                                if key in ['net_weight', 'gross_weight','total_quantity_of_goods']:
                                    for i in range(l2):
                                        predicted_label[i][0]=remove_alphabets(predicted_label[i][0])
                                if key in ['net_weight', 'gross_weight','total_quantity_of_goods']:
                                    for i in range(l2):
                                        predicted_label[i][0]=remove_alphabets(predicted_label[i][0])
                                if key in ["incoterm"]:
                                    for i in range(l2):
                                        print(f'incoterm predicted: {predicted_label[i][0]}')
                                        predicted_label[i][0]=preprocess_incoterm(predicted_label[i][0], incoterm_list)
                                if key in ['port_of_discharge', 'port_of_loading']:
                                    regex=["([PDo])?(o)?(r)?(t)?(\s)?(o)?(f)?(\s)?([DO0])?(i)?(s)?(c)?(h)?(a)?(r)?(g)?(e)?",r"([PDo])?(o)?(r)?(t)?(\s)?(o)?(f)?(\s)?([IL1T\|])?(o)?(a)?(d)?(i)?(n)?(g)?"]
                                    if key =='port_of_discharge':
                                        for i in range(l2):							
                                            predicted_label[i][0] = remove_label_name(predicted_label[i][0],regex[0])
                                    elif key =='port_of_loading':
                                        for i in range(l2):							
                                            predicted_label[i][0] = remove_label_name(predicted_label[i][0],regex[1])
                                if key in ["page_no","original_Number","to_place","lc_ref_no","lc_ref_number","consignee_name"]:
                                    regex=[r"^(?i)To\s*:",r"^(?i)(?:LC\s*NO\.?\s*|NO\.|no\.)","(?i)(page|no\.|no|pago\s*no\.|pago|Page)",r"(?i)NOTIFY\s*:\s*"]
                                    if key =='page_no'or key=="original_Number":
                                        for i in range(l2):		
                                            actual_value=	generic_page_no(remove_special_chars(remove_label_name(actual_value,regex[2])))				
                                            predicted_label[i][0] = generic_page_no(remove_special_chars(remove_label_name(predicted_label[i][0],regex[2])))
                                    elif key =='consignee_name':
                                        for i in range(l2):							
                                            predicted_label[i][0] = remove_label_name(predicted_label[i][0],regex[3])
                                    elif key =='to_place':
                                        for i in range(l2):							
                                            predicted_label[i][0] = remove_label_name(predicted_label[i][0],regex[0])
                                    elif key =='lc_ref_no' or "lc_ref_number":
                                        for i in range(l2):							
                                            predicted_label[i][0] = remove_label_name(predicted_label[i][0],regex[1])
                                if key in ["pre_carriage_by","consignor_name","declaration_by","declaration","mode_of_transport","from_place"]:
                                    if key =="consignor_name":
                                        for i in range(l2):							
                                                predicted_label[i][0] = remove_words(predicted_label[i][0],['EXPORTER'])
                                    if key =="from_place":
                                        for i in range(l2):							
                                                predicted_label[i][0] = remove_words(predicted_label[i][0],['FROM'])
                                    if key =="pre_carriage_by" or key=="mode_of_transport":
                                        for i in range(l2):							
                                                predicted_label[i][0] = remove_words(predicted_label[i][0],['BY','EXPORT'])
                                    if key =="declaration_by" or key=="declaration":
                                        for i in range(l2):							
                                                predicted_label[i][0] = remove_words(predicted_label[i][0],["REMOVE", "FOR", "NAME", "OF", "THE", "AUTHORISED", "SIGNATORY", "SEAL"])
                                #handling address for prediction
                                if key in [
                'drawer_bank_address',
                'consignee_address',
                'insurance_issuer_address',
                'beneficiary_address',
                'drawer_bank_bottom_address',
                'coo_issuer_address',
                'shipper_address',
                'address_of_assured',
                'drawer_address',
                'consignor_address',
                'notify_party_address',
                'claim_payable_by_address',
                'drawee_address',
                'drawee_bank_address',
                'remitter_address',
                'nostro_bank_address'
            ]:
                                    for i in range(l2):		
                                                if key == 'insurance_issuer_address':
                                                    
                                                    values = predicted_label[i][0].split('~')
                                                    for j in range(len(values)):
                                                        values[j] = remove_text_after_phrases(filter_address(values[j]))
                                                    predicted_label[i][0] = '~'.join(values)					
                                                else:
                                                    predicted_label[i][0] = remove_text_after_phrases(filter_address(predicted_label[i][0]))
                                    

                                for i in range(l2):
                                        print(row)
                                        print(predicted_label)
                                        print(f'The value is: {predicted_label[i][0]}')
                                        # exit('+++++++++++++==')
                                        print(key)
                                        if key not in ['lc_date','end_date','start_date','issue_date','sail_on_or_about_to_date','expiry_date', 'invoice_date', 'csh_due_date','invoice_date', 'csh_due_date', 'transaction_date', 'lc_date', 'csh_presentation_date', 'expiry_date', 'bill_of_lading_issue_date', 'sail_on_or_about_to_date', 'date_of_invoice', 'tenor_indicator_date', 'shipped_onboard_date', 'issue_date', 'awb_date', 'bill_of_lading_date']:	
                                            predicted_label[i][0]=remove_spl_char_multiple_pred(predicted_label[i][0])
                                print(f'second condition after remove alphabets: {predicted_label}')
                                to_do = [*range(0, l2, 1)]
                                print("starting to_do are", to_do)
                                if row[0]=='Insurance_Certificate_1_page_0.png' and row[1]=='sum_insured_amount':
                                    print(row)
                                    #exit() 
                                for i in range(l1):
                                    new_row = row.copy()
                                    actual_value = actual[i]
                                    if key not in ['lc_date','end_date','start_date','issue_date','sail_on_or_about_to_date','expiry_date', 'invoice_date', 'csh_due_date','invoice_date', 'csh_due_date', 'transaction_date', 'lc_date', 'csh_presentation_date', 'expiry_date', 'bill_of_lading_issue_date', 'sail_on_or_about_to_date', 'date_of_invoice', 'tenor_indicator_date', 'shipped_onboard_date', 'issue_date', 'awb_date', 'bill_of_lading_date']:
                                        actual_value[0]=remove_special_chars(actual_value[0]).strip()
                                    #handling address for actual
                                    if key in ['drawer_bank_address',
                'consignee_address',
                'insurance_issuer_address',
                'beneficiary_address',
                'drawer_bank_bottom_address',
                'coo_issuer_address',
                'shipper_address',
                'address_of_assured',
                'drawer_address',
                'consignor_address',
                'notify_party_address',
                'drawee_address',
                'drawee_bank_address',
                'remitter_address',
                'nostro_bank_address'
            ]:
                                        if key =='insurance_issuer_address':
                                            values = actual_value[0].split('~')
                                            for j in range(len(values)):
                                                values[j] = remove_text_after_phrases(filter_address(values[j]))
                                            actual_value[0] = '~'.join(values)
                        
                                        else:

                                            actual_value[0] = remove_text_after_phrases(filter_address(actual_value[0]))
                                    
                                    if key in ["country_of_final_destination","country_of_origin_of_goods","country_of_origin_origin_of_goods"]:
                                                                            
                                        actual_value[0]=extract_country_from_text(actual_value[0])
                                    if key in ["page_no","original_Number","to_place","lc_ref_no","lc_ref_number","consignee_name"]:
                                        regex=[r"^(?i)To\s*:",r"^(?i)(?:LC\s*NO\.?\s*|NO\.|no\.)","(?i)(page|no\.|no|pago\s*no\.|pago|Page)",r"(?i)NOTIFY\s*:\s*"]
                                        if key =='page_no'or key=="original_Number":  
                                            actual_value[0]=generic_page_no(remove_special_chars(actual_value[0]))
                                    if key not in ['lc_date','end_date','start_date','issue_date','sail_on_or_about_to_date','expiry_date', 'invoice_date', 'csh_due_date','invoice_date', 'csh_due_date', 'transaction_date', 'lc_date', 'csh_presentation_date', 'expiry_date', 'bill_of_lading_issue_date', 'sail_on_or_about_to_date', 'date_of_invoice', 'tenor_indicator_date', 'shipped_onboard_date', 'issue_date', 'awb_date', 'bill_of_lading_date']:	
                                        actual[i][0]=remove_spl_char_multiple_pred(actual_value[0])
                                    print(f'the actual value is: {actual_value}')
                                    # exit('+++++++==')
                                    
                                    ########################filter prediction
                                    
                                    if key not in ['declaration_by', 'claim_payable_by_address', 'lc_date','end_date','start_date','issue_date','sail_on_or_about_to_date','expiry_date', 'invoice_date', 'csh_due_date','sum_insured_amount', 'sum_insured_currency','insurance_issuer_address']:
                                        to_do = filter_prediction(new_row, actual_value, predicted_label, to_do)
                                ##handiling address actual + predicted
                                if key in ['declaration_by', 'insurance_issuer_address','address_of_assured','claim_payable_by_address']:
                                    act_val = ''
                                    pre_val = ''
                                    print(actual)
                                    for i in range(len(actual)):
                                        actual[i][0] = remove_text_after_phrases(actual[i][0])
                                        actual[i][0] = remove_special_chars(actual[i][0])
                                        actual[i][0] = filter_address(actual[i][0])
                                        act_val += actual[i][0]+"~"
                                    for j in range(len(predicted_label)):
                                        predicted_label[j][0] = remove_text_after_phrases(predicted_label[j][0])
                                        predicted_label[j][0] = remove_special_chars(predicted_label[j][0])
                                        predicted_label[j][0] = filter_address(predicted_label[j][0])
                                        pre_val += predicted_label[j][0]+"~"
                                        
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
                
                                print(to_do)
                                if key in ["address_of_assured", "claim_payable_by_address","insurance_issuer_address", 'claim_payable_by_address']:
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
                                    
                                    print('act_val_address', act_val_address)
                                    #exit()
                                    actual_country = extract_country_from_text(act_val_address)
                                    pred_country = extract_country_from_text(pre_val_address)
                                    # if len(actual_country)==0:
                                    #     actual_country = loc_tagger(act_val_address)
                                    # if len(pred_country)==0:
                                    #     pred_country = loc_tagger(pre_val_address)
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
                                # exit('+++++++++++++')
                                # # print("remaining to_do are", to_do)
                                # # print(predicted_label)
                                # for i in to_do:
                                # 	new_row = row.copy()
                                # 	new_row.append("")
                                # 	new_row.append(predicted_label[i][0])
                                # 	new_row.append(0)
                                # 	new_row.append(0)
                                # 	new_row.append(predicted_label[i][2])
                                # 	new_row.append(predicted_label[i][1])
                                # 	data.append(new_row)
                elif key in labels and key not in predicted:
                    if key not in ['sum_insured_amount', 'sum_insured_currency']:
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
                    if key not in ['sum_insured_amount', 'sum_insured_currency']:
                        # exit()
                        row.append(key)
                        row.append("")
                        if len(predicted[key]) == 1:
                            row.append(str(predicted[key][0][0]))
                            row.append(0)
                            row.append(0)
                            try:
                                row.append(predicted[key][0][2])
                                row.append(predicted[key][0][1])
                            except:
                                pass
                            data.append(row) 
                        else:
                            for val in predicted[key]:
                                new_row = row.copy()
                                new_row.append(val[0])
                                new_row.append(0)
                                new_row.append(0)
                                try:
                                    new_row.append(val[2])
                                    new_row.append(val[1])
                                except:
                                    pass
                                data.append(new_row)
                    else:
                        continue
        print(row)
    # exit('++++++++++++++++++')

    df = pd.DataFrame(data, columns=column_names)
    #df.columns = column_names
    print(df)
    # exit('+++++++++++==')

    print(df["Match/No_Match"].sum())
    print(df["Match/No_Match"])
    # exit()
    # post processing fuzzy match percentage
    try:
        df.loc[df["Accuracy"].apply(float) > 90, "Match/No_Match"] = 1
        # address
        df.loc[((df["label_name"].isin([
        'insurance_issuer_address',
        'address_of_assured',
        'claim_payable_by_address',
        "from_place", "to_place"])) & (df["Accuracy"].apply(int) > 70)), "Match/No_Match"] = 1
        print(df["Match/No_Match"].value_counts())

        df.loc[(df["label_name"].isin(["sum_insured_currency_buyer", "sum_insured_currency_buyer"]) & (df["predicted"].str.contains("rs|usd|inr|eur", regex=True))), "Match/No_Match"] = 1
        print(df["Match/No_Match"].value_counts())

        df.loc[(df["label_name"].isin(["mode_of_transport"]) & (
            df["predicted"].str.contains("rail|sea|road|air", regex=True))), "Match/No_Match"] = 1
        print(df["Match/No_Match"].value_counts())

        df.loc[(df["label_name"].isin(["insurance_issuer_name", "name_of_assured", "claim payable_by_name"]) & (
                df["Accuracy"].apply(int) > 70)), "Match/No_Match"] = 1
        print(df["Match/No_Match"].value_counts())

        df.loc[(df["label_name"].isin(["policy_or_certificate_no"]) & (
                df["Accuracy"].apply(int) > 70)), "Match/No_Match"] = 1
        print(df["Match/No_Match"].value_counts())

        df.loc[(df["label_name"].isin(["original_Number"]) & (df["Match/No_Match"] == 0) & (
            df["predicted"].apply(lambda x: str(x).lower().replace("page", "").strip())) == df[
                    "actual"].str.lower()), "Match/No_Match"] = 1
        print(df["Match/No_Match"].value_counts())



        print("claim payable_by_name")

        aa = df.loc[(df["label_name"].isin(["claim payable_by_name"]) & (
                df["Accuracy"].apply(int) > 75) & (df["Match/No_Match"] == 0)), :]
        print(aa.shape)

        df.loc[(df["label_name"].isin(["claim payable_by_name"]) & (
                df["Accuracy"].apply(int) > 75)), "Match/No_Match"] = 1
        print(df["Match/No_Match"].value_counts())

        """print("claim_payable_by_address")
        df.loc[(df["label_name"].isin(["claim_payable_by_address"]) & (
                df["Accuracy"].apply(int) > 70)), "Match/No_Match"] = 1
        print(df["Match/No_Match"].value_counts())"""

        print("conditions_of_coverage")
        df.loc[(df["label_name"].isin(["conditions_of_coverage"]) & (
                df["Accuracy"].apply(int) > 50)), "Match/No_Match"] = 1
        print(df["Match/No_Match"].value_counts())

        print("original_Number")
        df.loc[(df["label_name"].isin(["original_Number","page_no"]) & (
                df["predicted"].apply(lambda x: re.sub(r'page|Page', "", str(x).strip()).strip()) == df[
            "actual"].str.lower())), "Match/No_Match"] = 1
        print(df["Match/No_Match"].value_counts())

        print("sum_insured_amount")
        df.loc[(df["label_name"].isin(["sum_insured_amount"]) & (
                df["Accuracy"].apply(int) > 70)), "Match/No_Match"] = 1
        print(df["Match/No_Match"].value_counts())

        print("vessel_or_flight_name")
        df.loc[(df["label_name"].isin(["vessel_or_flight_name"]) & (
                df["Accuracy"].apply(int) > 75)), "Match/No_Match"] = 1
        print(df["Match/No_Match"].value_counts())

        #for visual objects
        print(df['label_name'])
        print(df["model_confidence"])
        print(df["Accuracy"])
    except:
        pass
    try:
        df.loc[(df["label_name"].isin(["declaration_by"])) & 
            (df["model_confidence"].apply(float) > 75), "Match/No_Match"] = 1
    except:
        pass

    print(df["Match/No_Match"].value_counts())

    print(df["Match/No_Match"].sum())
    # exit("++++++++++++")
    res_path= os.path.join(folder_path, 'result_path')
    if not os.path.exists(res_path):
        os.mkdir(res_path)
    df.to_csv(f'{res_path}/PL_analysis_pre_valid_june7_after_fuzzy_match_post-processing_latest.csv')
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print("Total execution time:", elapsed_time)

    # Write the execution details to timestamp.txt
    with open("timestamp.txt", "a") as f:
        f.write("File Name: {}\n".format(file_name))
        f.write("Total Execution Time: {}\n".format(elapsed_time))
        
    accuracy_generation_file = os.path.join(folder_path, 'result_path', 'PL_analysis_pre_valid_june7_after_fuzzy_match_post-processing_latest.csv')
    file_path_csv1 = os.path.join(folder_path, 'result_path', f"{doc_code_}_{datetime.now().date()}_{datetime.now().hour}",
                            accuracy_generation_file)
    file_paths_csv2 = final_report(file_path_csv1, folder_path, doc_code_)

    # txt file generation
    stp_report(file_path_csv1, folder_path, doc_code_)

    final_overall_analysis(file_paths_csv2, file_path_csv1, folder_path, doc_code_)