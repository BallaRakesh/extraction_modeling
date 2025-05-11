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
from typing import List
import os
import json
from configparser import ConfigParser

"""
Sample master internal_data json:

{"drawer_bank_name": [["HSBC BANK MALAYSIA BERHAD", [260, 311, 686, 337]]], 
"drawer_bank_address": [["2 LEBOH AMPANG 50100 KUALA LUMPUR", [256, 346, 569, 407]], 
["KUALA LUMPUR MAIN", [260, 278, 557, 304]], 
["MALAYSIA", [257, 415, 405, 442]]], "page_no": [["1", [1255, 213, 1287, 239]]], "drawee_bank_name": [["ICICI BANK LTD", [257, 548, 508, 577]]], "drawee_bank_address": [["REGIONAL TRADE SVCS UNIT - NEW DEHI 9A PHELPS BUILDING , CONNAUGHT PLACE", [259, 582, 860, 646]], ["NEW DELHI 110001", [261, 654, 531, 681]], ["INDIA", [542, 652, 638, 682]]], "csh_presentation_date": [["19FEB2016", [1197, 685, 1356, 715]]], "drawer_name": [["INTERPRINT DECOR MALAYSIA SDN BHD", [444, 754, 1003, 780]]], "drawee_name": [["GREENLAM INDUSTRIES LIMITED", [444, 787, 904, 814]]], "drawee_address": [["DISTT SOLAN , HIMACHAL PRADES ,", [442, 821, 930, 854]], ["INDIA", [942, 820, 1038, 853]]], "csh_bill_currency": [["EUR", [630, 1093, 683, 1123]]], "csh_bill_amount": [["42,026.57", [840, 1093, 1002, 1126]]], "usance_tenor": [["180", [443, 924, 496, 950]]], "tenor_indicator": [["DAYS", [527, 925, 605, 950]]], "tenor_indicator_type": [["FROM", [615, 922, 685, 951]]], "tenor_indicator_date": [["INVOICE DATE", [697, 920, 903, 951]]], "csh_ref_no": [["OBCKLH770358", [958, 614, 1169, 648]]], "csh_drawn_under_rules": [["522", [1023, 1532, 1086, 1565]]], "doc_charge_instructions": [["COLLECT YOUR CHARGES AND EXPENSES FROM DRAWEE", [278, 1767, 1036, 1797]]], "doc_delivery_instruction": [["* ACCEPTANCE / PAYMENT MAY BE DEFERRED PENDING ARRIVAL OF CARRYING VESSEL .", [261, 1702, 1173, 1759]]], 
"doc_settlement_instructions": [["HSBC BANK PLC ,", [1074, 2039, 1312, 2068]]]}
"""

# GLOBAL statements
######################################################################################
parser = ConfigParser()

conf_folder_path: str = "src/main/extraction/"
config_file_name: str = "train_valid_utility.ini"

if os.path.exists(f"{conf_folder_path}/{config_file_name}"):
	parser.read(f"{conf_folder_path}/{config_file_name}")
try:
	with open("src/main/extraction/countries.txt", "r") as f:
		countries_name: List = f.readlines()
		countries_name = list(map(lambda x: x.strip(), countries_name))
	# countries list
	countries_list: List = countries_name
	print(countries_list)
except:
    pass


######################################################################################

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
	# Need to be implemented
	pass


if __name__ == "__main__":
	variable_to_post_process_with_corresponding_functions: dict = {"drawee_bank_address": generic_address,
	                                                               "drawer_bank_address": generic_address,
	                                                               "drawer_address": generic_address,
	                                                               "drawee_address": generic_address
	                                                               }
	folder_path = str(parser['PATHS']['root_folder_path'])
	files: List = glob.glob(f"{folder_path}/*_labels.txt")

	for file in files:
		print(file)
		complete_file_path: str = os.path.join(folder_path, file)
		try:
			# reading the text file as json
			with open(os.path.join(folder_path, f"{file}"), "r") as f:
				master_data_json = json.load(f)
		except:
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
