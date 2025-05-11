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

from fuzzywuzzy import fuzz
import re
from src.main.extraction.validation_utility import final_overall_analysis, final_report, stp_report
import os
from configparser import ConfigParser
import psutil
from datetime import datetime
from src.main.extraction.validation_utility import get_logger_object_and_setting_the_loglevel, set_basic_config_for_logging
from src.main.extraction.config.prod_mapping import product_code_map, document_code_map
import glob
# logger = get_logger_object_and_setting_the_loglevel()
# process_memory = psutil.Process()


def append_values(new_row, actual_value, predicted, to_do):
	l2 = len(predicted)
	for j in range(l2):
		intersection = get_iou_new(actual_value[1], predicted[j][1])
		if intersection > 0.25:
			new_row.append(actual_value[i][0])
			new_row.append(predicted[j][0])
			accuracy = fuzz.ratio(str(new_row[2]).lower(), str(new_row[3]).lower())
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
			except Exception:
				pass
			break
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
	except Exception:
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
	new_x = "".join([x for x in new_x if x.isalnum() or x in {" "}])
	print(new_x)
	return new_x



if __name__ == '__main__':
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

	gv_key = configur['OCR']['gv_key']

	set_basic_config_for_logging(filename="validation")
	logger = get_logger_object_and_setting_the_loglevel()
	t_start = datetime.now()

	folder_name_path = glob.glob(f"{folder_path}/result_path/*")
	folder_name_path.sort(reverse=True)
	folder_name_path = folder_name_path[0]
	print(folder_name_path)
	# exit("+++++++++++++")
 
	accuracy_generation_file = glob.glob(f"{folder_name_path}/*.csv")
	accuracy_generation_file.sort(reverse=True)
	print(f"Number of folders: {len(accuracy_generation_file)}")
	# assert len(accuracy_generation_file) == 1	

	accuracy_generation_file = accuracy_generation_file[0]

	file_path_csv1 = os.path.join(folder_path, 'result_path', f"{doc_code}_{datetime.now().date()}_{datetime.now().hour}",
                               accuracy_generation_file)
	file_paths_csv2 = final_report(file_path_csv1, folder_path)

	# txt file generation
	stp_report(file_path_csv1, folder_path)

	final_overall_analysis(file_paths_csv2, file_path_csv1, folder_path)

	t_end = datetime.now()
	logger.info(f"Time Taken validation process is :{str(t_end - t_start)}")
