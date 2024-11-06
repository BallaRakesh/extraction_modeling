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

import pandas as pd
import numpy as np
import glob
from typing import List
import json
from bisect import bisect_left
import re
import os
from configparser import ConfigParser
import logging
import csv
from datetime import datetime
# from src.main.extraction.config.prod_mapping import product_code_map, document_code_map

configur = ConfigParser()
product_config = ConfigParser()

# configur.read('src/main/extraction/traini_valid_utility.ini')
# gv_key = configur['OCR']['gv_key']

# # product config
# product_config.read("src/main/extraction/config/config.ini")

# prod_code = product_code_map[product_config["Product"]["code"]]
# doc_code = document_code_map[product_config["Product"]["document_code"]]

# # data folder path
# product_wise_folder = ConfigParser()
# product_wise_folder.read("src/main/extraction/config/prod.ini")
# folder_path = product_wise_folder[prod_code][doc_code]

# print("==================Trade Finance Solutions===================")
# print("Product Code: {product_code}")
# print("Documenry Code: {doc_code}")
# print(f"folder_path: {folder_path}")

match_name = "Match/No_Match"

try:
	with open(os.path.join('src/main/extraction/codes-all.csv'), 'r') as file:
		my_reader = csv.reader(file, delimiter=',')
		currency_list = [row[2] for row in my_reader if len(row) > 2]
except Exception as e:
	print(e)
	print("The file unable to read pls check the file path")

# correcting the currency list
# currency_list.remove("AlphabeticCode")
# currency_list.remove("")
# print(currency_list)


def set_basic_config_for_logging(filename: str = None):
	"""
	Set the basic config for logging python program.   
	:return: None   
	"""
	# Create and configure logger
	log_file_path = os.path.join(folder_path, f"{filename}.log")
	logging.basicConfig(filename=log_file_path, format='%(asctime)s %(message)s',
						filemode='w')


def get_logger_object_and_setting_the_loglevel():
	"""    get the logger object and set the loglevel for the logger object
	:return: Logger Object    
	"""
	# Creating an object
	logger_object = logging.getLogger()
	# Setting the threshold of logger to DEBUG
	logger_object.setLevel(logging.DEBUG)
	return logger_object


def area(coordinates):
	l = coordinates[2] - coordinates[0] + 1
	h = coordinates[3] - coordinates[1] + 1
	return l * h


# finds the minimum distance between a given bounding box
def minimum_distance(bb1, bb2):
	# bb1 points
	assert bb1[0] < bb1[2]
	assert bb1[1] < bb1[3]
	assert bb2[0] < bb2[2]
	assert bb2[1] < bb2[3]
	min_distance = 9999999999
	p_11 = np.array((bb1[0], bb1[1]))
	p_12 = np.array((bb1[0], bb1[3]))
	p_13 = np.array((bb1[2], bb1[3]))
	p_14 = np.array((bb1[2], bb1[1]))
	all_points_bb1 = [p_11, p_12, p_13, p_14]
	# bb2 points
	p_21 = np.array((bb2[0], bb2[1]))
	p_22 = np.array((bb2[0], bb2[3]))
	p_23 = np.array((bb2[2], bb2[3]))
	p_24 = np.array((bb2[2], bb2[1]))
	all_points_bb2 = [p_21, p_22, p_23, p_24]
	for point1 in all_points_bb1:
		for point2 in all_points_bb2:
			dist = abs(np.linalg.norm(point1 - point2))
			if dist < min_distance:
				min_distance = dist
	return min_distance


# finds and merges the modeloutput for text result, OCR confidence and Model confidence
def model_output_sum(key, box, data):
	all_values = data[key]
	text_result = ""
	for value in all_values:
		try:
			inter_percent = get_intersection_percentage(box, value[1])
		except Exception as e:
			print(str(e))
			continue
		if inter_percent > 0.4:
			if text_result == "":
				text_result += value[0]
			else:
				text_result += " " + value[0]
	return text_result


def merge_val(all_values, data, i, length, eps, key):
	while i in range(length - 1):
		bb1 = all_values[i][1]
		bb2 = all_values[i + 1][1]
		min_dist = minimum_distance(bb1, bb2)
		try:
			IOU = get_iou_new(bb1, bb2)
			inter_percentage = get_intersection_percentage(bb1, bb2)
		except Exception:
			i = i + 1
			continue
		if min_dist <= eps or IOU > 0 or inter_percentage > 0:
			x_left = min(bb1[0], bb2[0])
			y_top = min(bb1[1], bb2[1])
			x_right = max(bb1[2], bb2[2])
			y_bottom = max(bb1[3], bb2[3])
			box = [x_left, y_top, x_right, y_bottom]
			text = model_output_sum(key, box, data)
			new_value = [text, box]
			all_values.remove(all_values[i])
			all_values.remove(all_values[i])
			all_values.insert(i, new_value)
			length = len(all_values)
			if length == 1:
				break
		else:
			i = i + 1


# merge surrounding boxes
def merge_surrounding(data):
	for key in list(data.keys()):
		eps = 100
		all_values = data[key]
		length = len(all_values)
		if length > 1:
			i = 0
			merge_val(all_values, data, i, length, eps, key)
		else:
			continue


def get_intersection_percentage(bb1, bb2):
	"""
	Finds the percentage of intersection  with a smaller box. (what percernt of smaller box is in larger box)
	"""
	assert bb1[0] <= bb1[2]
	assert bb1[1] <= bb1[3]
	assert bb2[0] <= bb2[2]
	assert bb2[1] <= bb2[3]

	# determine the coordinates of the intersection rectangle
	x_left = max(bb1[0], bb2[0])
	y_top = max(bb1[1], bb2[1])
	x_right = min(bb1[2], bb2[2])
	y_bottom = min(bb1[3], bb2[3])

	if x_right < x_left or y_bottom < y_top:
		return 0.0

	# The intersection of two axis-aligned bounding boxes is always an
	# axis-aligned bounding box
	intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)

	# compute the area of both AABBs
	# bb1_area = (bb1[2] - bb1[0] + 1) * (bb1[3] - bb1[1] + 1)
	bb2_area = (bb2[2] - bb2[0] + 1) * (bb2[3] - bb2[1] + 1)
	# min_area = min(bb1_area,bb2_area)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	intersection_percent = intersection_area / bb2_area
	assert intersection_percent >= 0.0
	assert intersection_percent <= 1.0
	return intersection_percent


# finds intersection over union of two bounding boxes
def get_iou_new(bb1, bb2):
	"""
	Calculate the Intersection over Union (IoU) of two bounding boxes.

	Parameters
	----------
	bb1 : dict
		Keys: {0, '2', 1, '3'}
		The (x1, 1) position is at the top left corner,
		the (2, 3) position is at the bottom right corner
	bb2 : dict
		Keys: {0, '2', 1, '3'}
		The (x, y) position is at the top left corner,
		the (2, 3) position is at the bottom right corner

	Returns
	-------
	float
		in [0, 1]
	"""
	assert bb1[0] <= bb1[2]
	assert bb1[1] <= bb1[3]
	assert bb2[0] <= bb2[2]
	assert bb2[1] <= bb2[3]

	# determine the coordinates of the intersection rectangle
	x_left = max(bb1[0], bb2[0])
	y_top = max(bb1[1], bb2[1])
	x_right = min(bb1[2], bb2[2])
	y_bottom = min(bb1[3], bb2[3])

	if x_right < x_left or y_bottom < y_top:
		return 0.0

	# The intersection of two axis-aligned bounding boxes is always an
	# axis-aligned bounding box
	intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)

	# compute the area of both AABBs
	bb1_area = (bb1[2] - bb1[0] + 1) * (bb1[3] - bb1[1] + 1)
	bb2_area = (bb2[2] - bb2[0] + 1) * (bb2[3] - bb2[1] + 1)

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
	assert iou >= 0.0
	assert iou <= 1.0
	return iou

try:
	with open("/New_Volume/Rakesh/Trade_Finance_Training/trade-finance-mvp/training/lmv2code/src/main/extraction/countries.txt", "r") as f:
		countries_name: List = f.readlines()
		countries_name = list(map(lambda x: x.strip(), countries_name))
	# countries list
	countries_list: List = countries_name
	print(countries_list)

	f.close()
except:
    pass



# post_processing_master_gt
def generic_date(x):
	"""
	1) should not start and end with any special characters
	2) should have specific length
	:return:
	"""
	start = 0
	end = -1
	# remove all special character until you get any number
	# this we need to do from starting as well as beginning
	while x[start].isalnum():
		x = "".join(x[start])
		start += 1

	while x[end].isalnum():
		x = "".join(x[:end])
		end -= 1
	return x


def generic_page_no(x):
	"""
	remove if any alphabetic characters are there in this
	# 1) if of is present in ground truth , split on of and get the first element and strip that string.
	:param x:
	:return:
	"""
	x = str(x)
	x = x.lower()
	if "of" not in x:
		return x.strip()
	list_page_nos: List = x.split("of")

	# take the first element of list_page_nos
	first_element = list_page_nos[0] if list_page_nos else x

	return first_element.strip()


def generic_address(x):
	"""
	remove the country from the address
	:param x:
	:return:
	"""
	address, coord = x
	address = address.lower()
	for country in countries_list:
		if address.__contains__(country):
			address = address.replace(country, "")

	return [address, coord]


def generic_bic(x):
	"""
	1) should have some fixed length string - may be 12 digit or something
	2) does not contain any special character
	:return:
	"""
	pass


def binary_search(a, x, lo=0, hi=None):
	if hi is None:
		hi = len(a)
	pos = bisect_left(a, x, lo, hi)
	return pos if pos != hi and a[pos].__contains__(x) else -1


def pp_amount(x):
	"""
	Input should be the merged currency field. The idea is if we are getting the scattered token for currency amount,
	in prediction, need to merge those and then post-processing is applied on it.
	you need to take the currency from the combined currency and
	Solution: 1) may be you can remove the stop words such as comma, we cannot remove the dot because it can change the
	value of the amount.
	2) other that we need to remove the currency portion from this and take the amount separately from this.

	:param x:
	:return:
	"""
	string_after_removing_stop_words: str = ''.join(e for e in x if e.isalnum())
	# now take out currency from the string
	# assuming we should have a lookup for the currency
	# i think we can have currency list in one text file

	# binary search for matching string
	# sort the currency list
	index_of_matching_currency = binary_search(currency_list, string_after_removing_stop_words)

	string_after_currency_removal: str = string_after_removing_stop_words.replace(
		currency_list[index_of_matching_currency], ""
	)

	return string_after_currency_removal


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


def final_overall_analysis(path1, path2, folder_path, doc_code):
	# varaiable definition
	report_path = path1
	analysis_report_path = path2
	accuracy_lookup = pd.read_csv(report_path)
	analysis_report = pd.read_csv(analysis_report_path)
	data = []
	label_names = []

	# final_Report_Summary_Best_Fields
	for index, row in accuracy_lookup.iterrows():
		if row["Complete_Match_Percentage"] >= 80 and row["Fuzzy_Match_Percentage"] >= 85 and row[
			"Label_Name"] != "OVERALL":  # change these metrics to create buckets.
			label_names.append(str(row["Label_Name"]))
			data.append(list(row)[1:])

	if not os.path.exists(os.path.join(folder_path, 'result_path', f"{doc_code}_{datetime.now().date()}_{datetime.now().hour}")):
		os.makedirs(os.path.join(folder_path, 'result_path', f"{doc_code}_{datetime.now().date()}_{datetime.now().hour}"))


	save_path_1 = os.path.join(folder_path, 'result_path',f"{doc_code}_{datetime.now().date()}_{datetime.now().hour}", 
							f"{doc_code}_Final_Report_Summary_Best_Fields_after_fuzzy_match_change_{str(datetime.now())}.csv")

	df2 = pd.DataFrame(data, columns=list(accuracy_lookup.columns)[1:])
	df2.to_csv(save_path_1)

	new_data = [
		list(row)[1:]
		for index, row in analysis_report.iterrows()
		if str(row["label_name"]) in label_names
	]
	save_path_2 = os.path.join(folder_path, 'result_path', f"{doc_code}_{datetime.now().date()}_{datetime.now().hour}", \
							f"{doc_code}_Overall_Analysis_Best_Fields_after_fuzzy_match_change_{str(datetime.now())}.csv")

	df2 = pd.DataFrame(new_data, columns=list(analysis_report.columns)[1:])
	df2.to_csv(save_path_2)


def final_report(csv01, folder_path, doc_code):
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
		not_detected = name_df["predicted"].isnull().sum()
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

	if not os.path.exists(os.path.join(folder_path, 'result_path', f"{doc_code}_{datetime.now().date()}_{datetime.now().hour}")):
		os.makedirs(os.path.join(folder_path, 'result_path', f"{doc_code}_{datetime.now().date()}_{datetime.now().hour}"))


	name = f"final_report_{doc_code}_pre_{str(datetime.now())}_after_fuzzy_match_change.csv"

	file_path_csv2 = os.path.join(folder_path, 'result_path', f"{doc_code}_{datetime.now().date()}_{datetime.now().hour}", name)
	report.to_csv(file_path_csv2)

	# txt file generation
	name = f"final_report_{datetime.now()}" + ".txt"
	name_path = os.path.join(folder_path, 'result_path',f"{doc_code}_{datetime.now().date()}_{datetime.now().hour}",  name)
	text_report = {
		row["Label_Name"]: {
			"Label_Count": row["Label_Count"],
			"Detection_Accuracy": row["Detection_Accuracy"],
			"Fuzzy_Match_Percentage": row["Fuzzy_Match_Percentage"],
			"Complete_Match_Count_Detected": row["Complete_Match_Count_Detected"],
			"Complete_Match_Count": row["Complete_Match_Count"],
			"Complete_Match_Percentage": row["Complete_Match_Percentage"],
		}
		for index, row in report.iterrows()
	}
	with open(name_path, "w") as f:
		json.dump(text_report, f)
	f.close()

	return file_path_csv2


def stp_report(csv02, folder_path, doc_code):
	df = pd.read_csv(csv02)
	g = df.groupby("File_Name")
	stp_data = []
	num_files = 0
	match_files = 0
	for name, name_df in g:
		num_files += 1
		match_value = sum(list(name_df[match_name]))
		num_labels = len(list(name_df[match_name]))
		if match_value == num_labels:
			match_files += 1
			flag = 1
		else:
			flag = 0
		row = [str(name), match_value, num_labels, flag]
		stp_data.append(row)
	print("stp is", (match_files / num_files) * 100)

	df2 = pd.DataFrame(stp_data, columns=["File_Name", "Labels_Matched", "Labels_Present", "STP_Match"])
	file_path_csv3 = os.path.join(folder_path, 'result_path', 
                               f"{doc_code}_{datetime.now().date()}_{datetime.now().hour}", 
                               f"STP_Summary_{datetime.now()}.csv")
	df2.to_csv(file_path_csv3)


'''if __name__ == "__main__":
	configur = ConfigParser()
	configur.read('config.ini')

	gv_key = configur['OCR']['gv_key']
	folder_path = str(configur['PATHS']['folder_path'])
	variable_to_post_process_with_corresponding_functions: dict = {"drawee_bank_address": generic_address,
																"drawer_bank_address": generic_address,
																"drawer_address": generic_address,
																"drawee_address": generic_address
																}

	files: List = glob.glob(f"{folder_path}/*_labels.txt")

	print(files)
	for file in files:
		print(file)
		complete_file_path: str = os.path.join(folder_path, file)
		try:
			# reading the text file as json
			with open(os.path.join(folder_path, f"{file}"), "r") as f:
				master_data_json = json.load(f)
			f.close()
		except Exception:
			print("error")
			continue

		for key, func_name in variable_to_post_process_with_corresponding_functions.items():
			print("key is", key)
			if key not in master_data_json.keys():
				continue
			# fetch the internal_data from the master internal_data json for the corresponding key
			value_corresponding_for_each_key: List[List] = master_data_json[key]
			print(value_corresponding_for_each_key)
			# now you need to iterate over all values in this list
			# apply post-processing
			value_corresponding_for_each_key = list(map(func_name,
														value_corresponding_for_each_key))
			master_data_json[key] = value_corresponding_for_each_key
			print("After")
			print(master_data_json[key])

		print(master_data_json)
		with open(os.path.join(folder_path, f"{file}"), "w") as f:
			json.dump(master_data_json, f, sort_keys=True, indent=4,
					ensure_ascii=False)
		f.close()

	variable_to_post_process_with_corresponding_functions: dict = {"currency_amount": pp_amount}
	folder_path: str = ""
	files: List = os.listdir(folder_path)
	for file in files:
		complete_file_path: str = os.path.join(folder_path, file)

		# reading the text file as json
		with open(os.path.join(folder_path, f"{file}_text.txt"), "r") as f:
			master_data_json = json.load(f)
		f.close()
		for key, func_name in variable_to_post_process_with_corresponding_functions.items():
			# fetch the internal_data from the master internal_data json for the corresponding key
			value_corresponding_for_each_key: List[List] = master_data_json[key]

			# now you need to iterate over all values in this list
			# apply post-processing
			value_corresponding_for_each_key = list(map(func_name,
														value_corresponding_for_each_key))
			master_data_json[key] = value_corresponding_for_each_key'''
