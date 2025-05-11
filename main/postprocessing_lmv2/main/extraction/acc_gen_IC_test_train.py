import glob
import os
import json
import pandas as pd
from fuzzywuzzy import fuzz
import re
from difflib import get_close_matches
from post_processing_master_gt import generic_page_no
from post_processing_pred import pp_cash_drawn_rules
from dateutil.parser import parse
from test_country import extract_country_from_text
from typing import List
from configparser import ConfigParser


"""This accuracy generation script used in Insurance certificate document"""

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

def extract_country_from_address(text):
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

def strip_extra_spaces(text):
    return re.sub(r'\s+',' ', text)

def extract_dates_from_string(text):
    dates = []
    text = strip_extra_spaces(text)
    words = text.split()
    print(words)
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
    print(actual_value)
    print(predicted)
    #exit()
    predicted_value_list = extract_dates_from_string(predicted)
    print(predicted_value_list)
    #exit()
    best_match = predicted
    max_similarity = 60

    for predicted_value in predicted_value_list:
        if predicted_value == actual_value:
            return predicted_value
        else:
            similarity = fuzz.ratio(str(predicted_value), str(actual_value[1]))
            print(similarity)
            #exit()
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = predicted_value
    print(best_match)     
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
            new_row.append(predicted[j][2])		
            new_row.append(actual_value[1])
            data.append(new_row)
            try:
                to_do.remove(j)
            except:
                pass
            return to_do

        if predicted[j][2] > max_confidence:
            max_confidence = predicted[j][2]
            max_index = j

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

from datetime import datetime

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
    start_time = datetime.now()

    # Get the file name of the executed Python script
    file_name = os.path.basename(__file__)
    # count = 0
    occurrences = {}
    data = []

    #folder_path: str = "/home/ntlpt19/Downloads/Trade_finance_imp_stage_2/CS_NEW_ROOT/CS_EVAL"
    configur = ConfigParser()
    configur.read('traini_valid_utility.ini')
    folder_path = str(configur['PATHS']['folder_path'])
    result_path = os.path.join(folder_path, "Results_CS_validated")
    data_path = os.path.join(folder_path, "New_Master_Data_Merged")
    fols_path = os.path.join(folder_path,'train')
    df = create_data_frame(fols_path)
    # import json
    data_files = os.listdir(data_path)
    result_files = os.listdir(result_path)
    incoterm_list=[]
    test_img_list=[]

    for i in df['image_name']:
        test_img_list.append(i)
    with open("incoterm_list.txt") as fp:
        for line in fp:
            incoterm_list.append(line.strip())
    print(incoterm_list)
    # exit('++++++++++')

    # check how many pngs are there in those two folders
    print("Number of images in results", len(glob.glob(result_path + "/*.png")))
    print("Number of images in master data", len(glob.glob(data_path + "/*.png")))

    # exit("+++++++++++")

    # validating the number of files coming from both these folders
    print(f"number of files coming from the Master data folder: {len(data_files)}")
    print(f"number of files coming from the Result cs validated folder: {len(result_files)}")

    # exit("++++++++++++++")

    new = []
    new_result = []

    # FINDING LIST OF DATAFILES AND RESULT FILES
    num_labels_files_traversed: int = 0
    for file in data_files:
        if str(file)[-10:] == "labels.txt":
            print(f"file number is {num_labels_files_traversed}")
            num_labels_files_traversed = num_labels_files_traversed + 1
            # print(str(file)[-10:])
            new.append(file)
            new_result.append(file[:-11] + ".txt")

    # exit("++++++++++++++++")
    data_files = new
    result_files = new_result
    # print(result_files)
    print(len(data_files))
    print(len(result_files))
    # exit("+++++++++")

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
    print(column_names)
    print(len(column_names))
    # name = str(result_files[i])[0:-4]
    for file, predicted_files in zip(data_files, result_files):
        if file[0:-11] in test_img_list:
            print("file name is:", file)
            print("resulted filename is", predicted_files)
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

            print(f'acutal labels: {labels}')
            print(f'number of keys : {len(list(labels.keys()))}')
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++=')
            print(f'predicted labels: {predicted}')
            print(f'predicted labels: {len(predicted)}')

            # exit('+++++++++++==')

            if predicted == {} and labels == {}:
                print("*******")
                print(file)
                continue

            for key in list(occurrences.keys()):
                if key not in ["certificate_Stamped", "signature"]:
                    # print("key is", key)
                    row = []
                    row.append(file[0:-11] + ".png")
                    if key in labels and key in predicted:
                        if len(labels[key]) == 1 and len(predicted[key]) == 1:
                            if key in ['sum_insured_amount']:
                                row.append(key)
                                row.append(str(labels[key][0][0]))
                                #row.append(str(predicted[key][0][0])) 
                                amt = str(predicted[key][0][0])
                                amt_lis = [amt.split(' ')]
                                
                                if row[2] in amt_lis:
                                    for i in amt_lis:
                                        if i==row[2]:
                                            row.append(i)      
                                            accuracy = fuzz.ratio(str(row[2]).lower(), str(i).lower())
                                            row.append(accuracy)
                                            if accuracy == 100:
                                                row.append(1)
                                            else:
                                                row.append(0)   
                                    data.append(row)        
                                else:
                                    row.append(amt)
                                    accuracy = fuzz.ratio(str(row[2]).lower(), str(row[3]).lower())
                                    row.append(accuracy)
                                    if accuracy == 100:
                                        row.append(1)
                                    else:
                                        row.append(0)    
                                    data.append(row)  
                                print('row********', row)
                                print('************', amt_lis)
                                if row[0]=='Insurance_Certificate_1_page_0.png' and row[1]=='sum_insured_amount':
                                    print('row')
                                    #exit()                               
                            if key not in ['sum_insured_amount']:
                                row.append(key)
                                row.append(str(labels[key][0][0]))
                                row.append(str(predicted[key][0][0]))
                                # print(f'row data: {row}')
                                # print(str(row[2]).lower(), str(row[3]).lower())
                                #remove special chars in starting and ending of the string
                                row[2]=remove_special_chars(row[2])
                                row[3]=remove_special_chars(row[3])
                                if key not in ['invoice_date', 'csh_due_date', 'transaction_date', 'lc_date', 'csh_presentation_date', 'expiry_date', 'bill_of_lading_issue_date', 'sail_on_or_about_to_date', 'date_of_invoice', 'tenor_indicator_date', 'shipped_onboard_date', 'issue_date', 'awb_date', 'bill_of_lading_date']:
                                    row[2], row[3] = remove_start_end_spl_char(row[2], row[3])
                                print(f'after remove spl chars: {row}')
                                # exit()
                                
                                if key in ['invoice_date', 'csh_due_date', 'transaction_date', 'lc_date', 'csh_presentation_date', 'expiry_date', 'bill_of_lading_issue_date', 'sail_on_or_about_to_date', 'date_of_invoice', 'tenor_indicator_date', 'shipped_onboard_date', 'issue_date', 'awb_date', 'bill_of_lading_date']:
                                    """row[2]=row[2].strip()
                                    row[3]=row[3].strip()"""
                                    if row[2].strip().lower()!=row[3].strip().lower():	
                                        #row[2]=validate_date(row[2])[1]
                                        row[3]=eval_date_predictions(row[2], row[3])
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
                                # print(data)
                                # exit('+++++++')
                                if key in ["address_of_assured", "claim_payable_by_address","insurance_issuer_address"]:

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
                            actual_value=str(labels[key][0][0])
                            print(f'second condition: {actual}')
                            print(f'second condition: {predicted_label}')
                            print(row)
                            l1 = len(actual)
                            l2 = len(predicted_label)
                            if key in ['sum_insured_amount']:
                                lis = []
                                for i in range(l2):
                                    lis.append(predicted_label[i][0])
                                flag_amt = True    
                                for j in range(l1):
                                    for i in lis:
                                        if actual_value[j][0] == i:
                                            #if actual_value_lower == predicted[j][0].lower():
                                            flag_amt = False
                                            row.append(actual_value[j][0])
                                            row.append(i)
                                            accuracy = fuzz.ratio(row[2].lower(), row[3].lower())
                                            row.append(accuracy)
                                            if accuracy == 100:
                                                row.append(1)
                                            else:
                                                row.append(0)                
                                            #new_row.append(predicted[j][2])		
                                            #new_row.append(actual_va   
                                #if  flag_amt == True:
                                    
                                    
                                                
                                        
                                
                            if key in ['carrier_country',"country_of_final_destination","country_of_origin_of_goods", 'agent_country',"country_of_origin_origin_of_goods"]:
                                for i in range(l2):
                                    predicted_label[i][0]=extract_country_from_text(predicted_label[i][0])
                                
                            if key in ['invoice_date', 'csh_due_date', 'transaction_date', 'lc_date', 'csh_presentation_date', 'expiry_date', 'bill_of_lading_issue_date', 'sail_on_or_about_to_date', 'date_of_invoice', 'tenor_indicator_date', 'shipped_onboard_date', 'issue_date', 'awb_date', 'bill_of_lading_date']:
                                for i in range(l2):
                                    if actual_value.strip().lower()!=predicted_label[i][0].strip().lower():
                                        predicted_label[i][0]=eval_date_predictions(actual_value, predicted_label[i][0])
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
                                    if key not in ['invoice_date', 'csh_due_date', 'transaction_date', 'lc_date', 'csh_presentation_date', 'expiry_date', 'bill_of_lading_issue_date', 'sail_on_or_about_to_date', 'date_of_invoice', 'tenor_indicator_date', 'shipped_onboard_date', 'issue_date', 'awb_date', 'bill_of_lading_date']:	
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
                                actual_value[0]=remove_special_chars(actual_value[0]).strip()
                                
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
            'nostro_bank_address'
        ]:
                                    if key =='insurance_issuer_address':
                                        values = actual_value[0].split('~')
                                        for j in range(len(values)):
                                            values[j] = remove_text_after_phrases(filter_address(values[j]))
                                        actual_value[0] = '~'.join(values)
                    
                                    else:

                                        actual_value[0] = remove_text_after_phrases(filter_address(actual_value[0]))
                                
                                if key in ['carrier_country',"country_of_final_destination","country_of_origin_of_goods", 'agent_country',"country_of_origin_origin_of_goods"]:
                                                                        
                                    actual_value[0]=extract_country_from_text(actual_value[0])
                                if key in ['invoice_date', 'csh_due_date', 'transaction_date', 'lc_date', 'csh_presentation_date', 'expiry_date', 'bill_of_lading_issue_date', 'sail_on_or_about_to_date', 'date_of_invoice', 'tenor_indicator_date', 'shipped_onboard_date', 'issue_date', 'awb_date', 'bill_of_lading_date']:
                                    pass
                                    #actual_value[0]=validate_date(actual_value[0])[1]
                                if key in ["page_no","original_Number","to_place","lc_ref_no","lc_ref_number","consignee_name"]:
                                    regex=[r"^(?i)To\s*:",r"^(?i)(?:LC\s*NO\.?\s*|NO\.|no\.)","(?i)(page|no\.|no|pago\s*no\.|pago|Page)",r"(?i)NOTIFY\s*:\s*"]
                                    if key =='page_no'or key=="original_Number":
                                            
                                        actual_value[0]=	generic_page_no(remove_special_chars(actual_value[0]))
                                if key not in ['invoice_date', 'csh_due_date', 'transaction_date', 'lc_date', 'csh_presentation_date', 'expiry_date', 'bill_of_lading_issue_date', 'sail_on_or_about_to_date', 'date_of_invoice', 'tenor_indicator_date', 'shipped_onboard_date', 'issue_date', 'awb_date', 'bill_of_lading_date']:	
                                    actual[i][0]=remove_spl_char_multiple_pred(actual_value[0])
                                print(f'the actual value is: {actual_value}')
                                # exit('+++++++==')
                                if key not in ['sum_insured_amount', 'sum_insured_currency','insurance_issuer_address']:
                                    to_do = filter_prediction(new_row, actual_value, predicted_label, to_do)
                            if key in ['insurance_issuer_address']:
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
                                    
                            if key in ['sum_insured_amount', 'sum_insured_currency']:
                                #row.append(key)
                                for i in range(l1):
                                    new_row = row.copy()
                                    actual_value = actual[i]
                                    actual_value[0]=remove_special_chars(actual_value[0]).strip()
                                    l2 = len(predicted_label)
                                    actual_value_lower = actual_value[0].lower()
                                    flag_amt = True
                                    flag_amt3=True
                                    for j in range(l2):
                                        prid_amt = [str(predicted_label[j][0]).split(' ')]
                                        prid_amt = prid_amt[0]
                                        print('*******', prid_amt)
                                        flag_amt2=True
                                        if actual_value_lower in prid_amt and flag_amt2==True:
                                            flag_amt2=False
                                            
                                            for i in prid_amt:
                                                if i==actual_value_lower and flag_amt3==True:
                                                    flag_amt3 = False
                                                    flag_amt = False
                                                    new_row.append(actual_value_lower)
                                                    new_row.append(i)      
                                                    accuracy = fuzz.ratio(actual_value_lower.lower(), str(i).lower())
                                                    new_row.append(accuracy)
                                                    if accuracy == 100:
                                                        new_row.append(1)
                                                    else:
                                                        new_row.append(0) 
                                                            
                                                    data.append(new_row)  
                                            if len(new_row)>8:
                                                print(predicted_label)
                                                print('actual', actual)
                                                print(new_row)
                                                exit()  
                                    else:
                                        if flag_amt==True:
                                            new_row.append(actual_value[0])
                                            new_row.append("")
                                            new_row.append(0)
                                            new_row.append(0)
                                            new_row.append(0)
                                            new_row.append(actual_value[1])
                                            data.append(new_row)       
            
            
                            print(to_do)
                            if key in ["address_of_assured", "claim_payable_by_address","insurance_issuer_address"]:
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
                            for val in predicted[key]:
                                new_row = row.copy()
                                new_row.append(val[0])
                                new_row.append(0)
                                new_row.append(0)
                                new_row.append(val[2])
                                new_row.append(val[1])
                                data.append(new_row)
                    else:
                        continue
        #print(row)
    # exit('++++++++++++++++++')

    df = pd.DataFrame(data, columns=column_names)
    #df.columns = column_names
    print(df)
    # exit('+++++++++++==')

    print(df["Match/No_Match"].sum())
    print(df["Match/No_Match"])
    # exit()
    # post processing fuzzy match percentage
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
