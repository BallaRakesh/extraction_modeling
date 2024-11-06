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
import copy
from datetime import datetime
from functools import cmp_to_key
from multiprocessing import Pool
from typing import List
import psutil
from funcy import chunks

"""
Review internal_data
Aug 10 , 2023 => by Tarun Sharma
"""

from configparser import ConfigParser
import training_utility as tu
import os
import cv2
import json
import shutil
import random
import pickle

# Global statements
# Step1: Reading configurations from ini files
parser = ConfigParser()

conf_folder_path: str = "/media/tarun/D1/Trade-Finance/src/main/extraction/config"
config_file_name: str = "config.ini"

if os.path.exists(f"{conf_folder_path}/{config_file_name}"):
	parser.read(f"{conf_folder_path}/{config_file_name}")

image_type: str = str(parser["Data"]["imgtype"])
ocr_file_suffix_to_save = "_text.txt"

"""
Todo: 
Items|status
1) print all the folder names | Not Done
2) add the folder validations | Not Done
3) need to add home and mount base both dynamically |  Not Done
4) checking intersection in Labels and images folder | Not Done
5) getting some important information before running the code
    a) free memory b) current cpu utilization b) gpu availability d) num of cores
    e) num of threads f) parent directory g)free hard disk space in the system
6) file_size check before the file
7) reading zip file extracting here within the code
8) dumping in h5py format in python
9) tensorboard monitoring
10) save metrics
11) auto stopping code
"""


def train_test_split(split_file, flag, img_files, ocr_file_suffix_to_save, image_data_path):
	if debug_mode:
		print("In the train-test split code function")
	logger.info("In the train-test split code function")
	annotation_validation: bool = parser['PARAMS']['annotation_validation']
	annotation_data: list = []
	logger.info("is annotation_data is instance of list? %s", isinstance(annotation_data, list))
	for file in split_file:
		if file in img_files:
			if os.path.exists(os.path.join(ocr_path, file + ocr_file_suffix_to_save)):
				try:
					with open(os.path.join(ocr_path, file + ocr_file_suffix_to_save), "r") as f:
						ocr_data = json.load(f)['word_coordinates']
				except Exception as _:
					if debug_mode:
						exit("Error opening ocr word coordinates file!")
				finally:
					f.close()

				"""
				[{"word": "\u012e", "left": 157, "top": 1542, "width": 10, "height": 16, "x1": 157, 
				"y1": 1542, "x2": 167, "y2": 1558}, {"word": "HSBC", "left": 218, "top": 105, "width": 302, 
				"height": 93, "x1": 218, "y1": 105, "x2": 520, "y2": 198}, {"word": "KUALA", "left": 265, 
				"top": 284, "width": 82, "height": 17, "x1": 265, "y1": 284, "x2": 347, "y2": 301}]
				"""

				# give each token a value "O"
				for t in ocr_data:
					t["label"] = "O"

			if os.path.exists(os.path.join(images_path, file + ".png")):
				try:
					image = cv2.imread(os.path.join(images_path, file + ".png"))
				except Exception as e:
					logger.info(f"Error in reading an image file => Name of the image is: {file}")
					if not debug_mode:
						continue
					else:
						exit("Error in reading a file! Please fix the issue")

				if image is not None:
					h, w, _ = image.shape

				try:
					with open(os.path.join(labels_path, file + ".txt"), "r") as f:
						label = (f.read())
						label = label.split("\n")
						labelled_data = []
						logger.info("is labelled_data is instance of list? %s", isinstance(labelled_data, list))
				except Exception as e:
					if debug_mode:
						logger.info(f"Error opening a label file named as {file}")
				finally:
					f.close()

				debug_image_token_bbox = copy.deepcopy(image)
				debug_image_label_bbox = copy.deepcopy(image)

				for l in label:
					l = l.split()
					if (len(l) > int(parser['PARAMS']['nill_val'])) and (
							int(l[0]) != int(parser['PARAMS']['label_length'])):
						# if int(l[0]) != int(configur['PARAMS']['label_length']):
						l_class = dict_mapping[int(l[0])]
						x_center = float(l[1]) * w
						y_center = float(l[2]) * h
						width = float(l[3]) * w
						height = int(float(l[4]) * h)
						d_value = int(parser['PARAMS']['div_value'])
						x0 = int(x_center - (width / d_value))
						x1 = int(x_center + (width / d_value))
						y0 = int(y_center - (height / d_value))
						y1 = int(y_center + (height / d_value))

						if debug_mode and annotation_validation:
							debug_image_label_bbox = cv2.rectangle(debug_image_label_bbox, (x0, y0), (x1, y1),
							                                       (0, 255, 0), 4)

						labelled_data.append({
							"label": l_class,
							"x1": x0,
							"y1": y0,
							"x2": x1,
							"y2": y1
						})

				try:
					if not os.path.exists(f"{master_path}/debug/annotation_validation/label_bbox"):
						os.makedirs(f"{master_path}/debug/annotation_validation/label_bbox")
					cv2.imwrite(filename=f"{master_path}/debug/annotation_validation/label_bbox/{file}_label_bbox.png",
					            img=debug_image_label_bbox)
				except Exception as _:
					logger.info("Error writing a file for annotation validation")

				if len(labelled_data) > int(parser['PARAMS']['nill_val']):
					for data in labelled_data:
						for t in ocr_data:
							intersection_2 = tu.get_intersection_percentage(data, t)
							if intersection_2 >= float(parser['PARAMS']['percent_val']):
								t['label'] = data['label']
					for t in ocr_data:
						blue_col = int(parser['PARAMS']['color_val'])
						thick_val = int(parser['PARAMS']['thick_val2'])
						nul_val = int(parser['PARAMS']['nill_val'])

						if t['label'] == "O":
							debug_image_token_bbox = cv2.rectangle(debug_image_token_bbox, (int(t['x1']), int(t['y1'])),
							                                       (int(t['x2']), int(t['y2'])),
							                                       (blue_col, nul_val, nul_val), thick_val)
						else:
							t["label"] = "S-" + t["label"]
							debug_image_token_bbox = cv2.rectangle(debug_image_token_bbox, (int(t['x1']), int(t['y1'])),
							                                       (int(t['x2']), int(t['y2'])),
							                                       (nul_val, nul_val, blue_col), thick_val)

					try:
						if not os.path.exists(f"{master_path}/debug/annotation_validation/token_bbox"):
							os.makedirs(f"{master_path}/debug/annotation_validation/token_bbox")
						cv2.imwrite(
							filename=f"{master_path}/debug/annotation_validation/token_bbox/{file}_token_bbox.png",
							img=debug_image_token_bbox)
					except Exception as _:
						logger.info("Error writing a file for annotation validation")

					if len(ocr_data) <= thresh:
						shutil.copy(os.path.join(images_path, file + ".png"),
						            os.path.join(image_data_path, file + ".png"))
						final_labelled_data = {
							"filename": file + ".png",
							"words": [],
							"bbox": [],
							"labels": []
						}
						logger.info("is final_labelled_data is instance of dict? %s",
						            isinstance(final_labelled_data, dict))
						for l_d in ocr_data:
							final_labelled_data["words"].append(l_d['word'])
							final_labelled_data["bbox"].append(
								tu.normalize([l_d['x1'], l_d['y1'], l_d['x2'], l_d['y2']], w, h))
							final_labelled_data["labels"].append(l_d['label'])
						annotation_data.append(final_labelled_data)
					else:
						final_labelled_data = {
							"filename": file + ".png",
							"words": [],
							"bbox": [],
							"labels": []
						}
						for i, l_d in enumerate(ocr_data):
							if (i + 1) % thresh == int(parser['PARAMS']['nill_val']):
								final_labelled_data["filename"] = file + "_s_" + str(int((i + 1) / thresh)) + ".png"
								shutil.copy(os.path.join(images_path, file + ".png"), os.path.join(image_data_path,
								                                                                   file + "_s_" + str(
									                                                                   int((
											                                                                       i + 1) / thresh)) + ".png"))
								annotation_data.append(final_labelled_data)
								final_labelled_data = {
									"filename": file + "_s_" + str(int(len(ocr_data) / thresh) + 1) + ".png",
									"words": [],
									"bbox": [],
									"labels": []
								}
							final_labelled_data["words"].append(l_d['word'])
							final_labelled_data["bbox"].append(
								tu.normalize([l_d['x1'], l_d['y1'], l_d['x2'], l_d['y2']], w, h))
							final_labelled_data["labels"].append(l_d['label'])
						if len(final_labelled_data['words']) > int(parser['PARAMS']['nill_val']):
							shutil.copy(os.path.join(images_path, file + ".png"), os.path.join(image_data_path,
							                                                                   file + "_s_" + str(
								                                                                   int(len(
									                                                                   ocr_data) / thresh) + 1) + ".png"))
							annotation_data.append(final_labelled_data)

			else:
				try:
					os.remove(os.path.join(labels_path, file + ".txt"))
				except Exception:
					pass
				try:
					os.remove(os.path.join(images_path, file + ".png"))
				except Exception:
					pass

	seed_val = int(parser['PARAMS']['random_seed_val'])
	# ratio_val = float(parser['PARAMS']['train_div_ratio']) # patch tarun
	random.seed(seed_val)
	if flag == 'train':
		random.shuffle(annotation_data)
		train_samples = annotation_data  # [:-int(ratio_val * len(annotation_data))]
		logger.info(f"length of the training samples: {len(train_samples)}")
		train_samples = sorted(train_samples, key=lambda x: x['filename'])

		words_train, boxes_train, labels_train = tu.data_convert(dataset=train_samples)

		# creating train folder
		if not os.path.exists(os.path.join(root_folder_path, 'train')):
			os.makedirs(os.path.join(root_folder_path, 'train'))

		# dumping pickle file
		with open(os.path.join(root_folder_path, 'train.pkl'), 'wb') as t:
			pickle.dump([words_train, labels_train, boxes_train], t)
		t.close()

		# copying the images into train folder
		for t in train_samples:
			shutil.copy(os.path.join(image_data_path, t['filename']),
			            os.path.join(root_folder_path, "train", t['filename']))

	# similar for test
	if flag == 'test':
		random.shuffle(annotation_data)
		test_samples = annotation_data  # [-int(ratio_val * len(annotation_data)):]
		len(test_samples)
		test_samples = sorted(test_samples, key=lambda x: x['filename'])
		words_test, boxes_test, labels_test = tu.data_convert(test_samples)
		if not os.path.exists(os.path.join(root_folder_path, 'test')):
			os.mkdir(os.path.join(root_folder_path, 'test'))
		with open(os.path.join(root_folder_path, 'test.pkl'), 'wb') as t:
			pickle.dump([words_test, labels_test, boxes_test], t)
		t.close()
		for t in test_samples:
			shutil.copy(os.path.join(image_data_path, t['filename']),
			            os.path.join(root_folder_path, "test", t['filename']))


def read_image(images_path, image_type, file):
	try:
		image = cv2.imread(os.path.join(images_path, f"{file}.{image_type}"))
		return image
	except Exception as e:
		print(f"error loading an image named : {file} at path : {images_path}")
		return None


def get_ocr(ocr_engine_mode, file):
	if ocr_engine_mode == "Vision":
		try:
			word_coordinates, all_text = tu.get_ocr_vision_api(image_path=
			                                                   os.path.join(images_path, file + ".png"))
		except:
			raise ConnectionError("Problem in calling the google vision API")

	elif ocr_engine_mode == "Tesseract":
		word_coordinates, all_text = tu.get_ocr_tesserct(image_path=
		                                                 os.path.join(images_path, file + ".png"),
		                                                 tessdata_dir=tessdata_dir)
	return word_coordinates, all_text


def dump_ocr_data(word_coordinates, ocr_file_suffix_to_save, file):
	try:
		with open(os.path.join(ocr_path, file + ocr_file_suffix_to_save), "w") as f:
			json.dump({"word_coordinates": word_coordinates}, f)
	except IOError as e:
		if debug_mode:
			exit(f"Error in writing ocr data in path {ocr_path} in master folder path: {master_path}")
	finally:
		f.close()


def creating_master_folder(word_coordinates, all_text, file):
	# copying  original image and its labels to master folder
	# copying original image to master folder
	shutil.copy(src=
	            os.path.join(images_path, f"{file}.{image_type}"),
	            dst=os.path.join(f"{master_path}/data", f"{file}.{image_type}"),
	            )

	# copying labels into master folder
	shutil.copy(
		src=os.path.join(labels_path, f"{file}.txt"),
		dst=os.path.join(f"{master_path}/data", f"{file}_LabelImg.txt"),
	)
	# copy ocr coordinates into master folder
	try:
		with open(os.path.join(f"{master_path}/data", f"{file}_text.txt"), "w") as f:
			json.dump({"word_coordinates": word_coordinates}, f)
	except IOError as e:
		if debug_mode:
			logger.exception("Error opening a file which supposed to store all text")
			exit(f"Error in writing ocr data in path {ocr_path} in master folder path: {master_path}")
	finally:
		f.close()

	# copy ocr all text into master folder
	try:
		with open(os.path.join(f"{master_path}/data", f"{file}_all_text.txt"), "w") as f:
			json.dump({"all_text": all_text}, f)
	except Exception as e:
		if debug_mode:
			logger.exception("Error opening a file which supposed to store all text")
	finally:
		f.close()



def label_data_preparation(label, w, h, image):
	labelled_data = []
	for l in label:
		l = l.split()
		if len(l) > 0 and int(l[0]) < int(parser['PARAMS']['label_length']):  # patch tarun
			l_class = dict_mapping[int(l[0])]
			d_value = int(parser['PARAMS']['div_value'])
			x_center = float(l[1]) * w
			y_center = float(l[2]) * h
			width = int(float(l[3]) * w)
			height = int(float(l[4]) * h)
			x0 = int(x_center - (width / d_value))
			x1 = int(x_center + (width / d_value))
			y0 = int(y_center - (height / d_value))
			y1 = int(y_center + (height / d_value))
			color_value = int(parser['PARAMS']['color_val'])
			thickness = int(parser['PARAMS']['thick_val'])
			nill_val = int(parser['PARAMS']['nill_val'])

			# plotting an image
			cv2.rectangle(image, (x0, y0), (x1, y1), (nill_val, color_value, nill_val), thickness)

			labelled_data.append({
				"label": l_class,
				"x1": x0,
				"y1": y0,
				"x2": x1,
				"y2": y1
			})
			return labelled_data


def data_creation(names_of_labelled_files, names_of_img_files):
	logger.info("is labelled_files is a instance of list? %s", isinstance(names_of_labelled_files, list))
	annotation_data = []
	logger.info("is annotation_data is instance of list? %s", isinstance(annotation_data, list))
	thresh = int(parser['PARAMS']['thresh_value'])  # 300
	logger.info("is threshold value is instance of int? %s", isinstance(thresh, int))

	if debug_mode:
		print(f"labelled files are {names_of_labelled_files}")
		print(f"image files are {names_of_img_files}")

	logger.info(f"labelled files are {names_of_labelled_files}")
	logger.info(f"image files are {names_of_img_files}")

	for file in names_of_labelled_files:
		if file in names_of_img_files:
			if debug_mode:
				print(f"Name of the file whose label and image both are present: {file}")
			logger.info(f"Name of the file whose label and image both are present: {file}")

			if os.path.exists(os.path.join(images_path, f"{file}.{image_type}")):
				if debug_mode:
					print(f"image file exists with the name {file}.{image_type}")
				logger.info(f"image file exists with the name {file}.{image_type}")

				# reading of an image
				image = read_image(images_path=images_path, image_type=image_type, file=file)

				# OCR extraction => input => image path
				# step 4: OCR
				word_coordinates, all_text = get_ocr(ocr_engine_mode=ocr_engine_mode, file=file)

				# dumping of the ocr data to the disk
				dump_ocr_data(word_coordinates, ocr_file_suffix_to_save, file)

				# Content Check
				# First Check: Content length is 0
				if len(word_coordinates) == 0:
					print(f"file name which do not have enough text: {file}")
					if debug_mode:
						if not os.path.exists(f"{master_path}/debug"):
							os.makedirs(f"{master_path}/debug")
						with open(f"{master_path}/debug/content_length_zero_files.txt", "w") as file:
							file.writelines(f"{file}")
						logger.info(f"Not enough text for file {file}")
				else:
					# master data creation
					creating_master_folder(word_coordinates, all_text, file)

					if image is not None:
						h, w, _ = image.shape
					else:
						continue

					# opening the annotation file per file
					try:
						with open(os.path.join(labels_path, f"{file}.txt"), "r") as f:
							label = (f.read())
					except IOError as e:
						if debug_mode:
							logger.info(f"error in reading the label file : {file}")
					finally:
						f.close()

					label = label.split("\n")
					labelled_data = []
					logger.info("is labelled_data is instance of list? %s", isinstance(labelled_data, list))

					"""
					example annotation
					0 x1 y1 x2 y2
					"""
					# Step 5 : Labelled Data Preparation
					labelled_data = label_data_preparation(label, w, h, image)

					if labelled_data:
						dataset = dict()
						logger.info("is dataset is instance of dict? %s", isinstance(dataset, dict))
						# iterating over all annotated boxes
						for data in labelled_data:
							overlapping_boxes: list = []
							logger.info("is overlapping_boxes is instance of list? %s",
							            isinstance(overlapping_boxes, list))
							labelled_text: str = ""
							logger.info("is labelled_text is instance of str? %s", isinstance(labelled_text, str))
							# iterating over all word coordinates
							for t in word_coordinates:
								try:
									if tu.get_intersection_percentage(data, t) >= float(
											parser['PARAMS']['percent_val']):  # 0.40
										t['label'] = data['label']
										overlapping_boxes.append(t)
								except Exception as e:
									print(t)
									print(e)
							#  sorting of the overlapping boxes
							overlapping_boxes = sorted(overlapping_boxes, key=cmp_to_key(tu.contour_sort))
							nil_value = int(parser['PARAMS']['nill_val'])

							# for t in overlapping_boxes:
							#     if len(labelled_text) == nil_value:
							#         labelled_text = t['word']
							#     else:
							#         labelled_text += " " + t['word']

							# patch tarun
							labelled_text = " ".join(t['word'] for t in overlapping_boxes)

							if len(labelled_text.strip()) == nil_value:
								print(f"{file} - {str(data)} - {len(overlapping_boxes)}")
							else:
								if data['label'] not in list(dataset.keys()):
									dataset[data['label']] = []
								dataset[data['label']].append(
									[labelled_text, [data['x1'], data['y1'], data['x2'], data['y2']]])
						if not dataset:
							print(f"{file} - blank")

						# remove garbage from ocr output
						tu.remove_garbage(dataset)

						with open(os.path.join(f"{master_path}/data", f"{file}_labels.txt"), "w") as f:
							json.dump(dataset, f)
						f.close()

			else:
				print("No")

	zoom = int(parser['PARAMS']['thresh_value']) / int(parser['PARAMS']['zoom_val'])  # 300 / 72
	image_data_path = os.path.join(root_folder_path, "Images_Data")

	# patch tarun
	# if not os.path.exists(ocr_path):
	#     os.mkdir(ocr_path)
	if not os.path.exists(image_data_path):
		os.mkdir(image_data_path)

	random.shuffle(labelled_files)
	return labelled_files


if __name__ == "__main__":
	ocr_engine_mode: str = parser['OCR']['OCR_ENGINE']

	if ocr_engine_mode == "Vision":
		gv_key = parser['OCR']['gv_key']  # google vision key
	elif ocr_engine_mode == "Tesseract":
		tessdata_dir = parser['OCR']['TESSDATA_DIR']  # tessdata dir for tesseract

	root_folder_path = str(parser['PATHS']['root_folder'])
	debug_mode = str(parser["PARAMS"]["debug_mode"])
	log_min_level = str(parser["LOG"]["LOG_FILTER_LEVELMIN"])

	# Step 2: Setting up logger
	# setting the log folder and file
	tu.set_basic_config_for_logging(folder_path=conf_folder_path, filename="data_preparation")
	# setting the logger object and log level
	logger = tu.get_logger_object_and_setting_the_loglevel(log_level=log_min_level)

	images_path: str = os.path.join(root_folder_path, "Images")  # already present
	labels_path: str = os.path.join(root_folder_path, "Labels")  # already present
	master_path: str = os.path.join(root_folder_path, "Master_Data")  # will be created
	ocr_path: str = os.path.join(root_folder_path, "OCR")  # will be created

	if debug_mode:
		print(f'folder path: {root_folder_path}')
		print(f'conf folder path: {conf_folder_path} and filename : {config_file_name}')
		print(f'images path: {images_path}')
		print(f'labels path: {labels_path}')
		print(f'master path: {master_path}')
		print(f'ocr path: {ocr_path}')

	logger.info(f'folder path: {root_folder_path}')
	logger.info(f'conf folder path: {conf_folder_path} and filename : {config_file_name}')
	logger.info(f'images path: {images_path}')
	logger.info(f'labels path: {labels_path}')
	logger.info(f'master path: {master_path}')
	logger.info(f'ocr path: {ocr_path}')

	# raising the FileNotFoundError if you must present directory is not there
	if not os.path.exists(images_path):
		raise FileNotFoundError("Image Directory does not exist")
	if not os.path.exists(labels_path):
		raise FileNotFoundError("Labels Directory does not exist")

	if not os.path.exists(master_path):
		if debug_mode:
			print("Master Directory does not exist! Creating on the Fly")
		logger.info("Master Directory does not exist! Creating on the Fly")
		os.makedirs(master_path)

	if not os.path.exists(f"{master_path}/data"):
		if debug_mode:
			print("Master data Directory => master_data/data does not exist! Creating on the Fly")
		os.makedirs(f"{master_path}/data")

	if not os.path.exists(ocr_path):
		if debug_mode:
			print("OCR Path does not exist! Creating on the Fly")
		logger.info("OCR Path does not exist! Creating on the Fly")
		os.makedirs(ocr_path)

	# Initial configuration and usages
	"""
	Represents an OS process with the given PID.
	If PID is omitted current process PID (os.getpid()) is used.
	Raise NoSuchProcess if PID does not exist.

	#  ============================================================
	# | FIELD  | DESCRIPTION                         | AKA  | TOP  |
	#  ============================================================
	# | rss    | resident set size                   |      | RES  |
	# | vms    | total program size                  | size | VIRT |
	# | shared | shared pages (from shared mappings) |      | SHR  |
	# | text   | text ('code')                       | trs  | CODE |
	# | lib    | library (unused in Linux 2.6)       | lrs  |      |
	# | data   | data + stack                        | drs  | DATA |
	# | dirty  | dirty pages (unused in Linux 2.6)   | dt   |      |
	#  ============================================================

	"""
	process = psutil.Process()
	start_time = datetime.now()
	cpu_utilization_start = psutil.cpu_percent()
	before_memory_rss = process.memory_info().rss
	before_memory_vms = process.memory_info().vms
	before_memory_shared = process.memory_info().shared
	before_memory_text = process.memory_info().text
	before_memory_lib = process.memory_info().lib
	before_memory_data = process.memory_info().data
	before_memory_dirty = process.memory_info().dirty

	if debug_mode:
		print("=============Initial configuration=================")
		print(f"process memory rss is {before_memory_rss}")
		print(f"process memory vms is {before_memory_vms}")
		print(f"process memory shared is {before_memory_shared}")
		print(f"process memory text is {before_memory_text}")
		print(f"process memory lib is {before_memory_lib}")
		print(f"process memory data is {before_memory_data}")
		print(f"process memory dirty is {before_memory_dirty}")
		print(f"start time is: {start_time}")
		print(f"cpu utilization: {cpu_utilization_start}")

	logger.info("=============Initial configuration=================")
	logger.info(f"process memory rss is {before_memory_rss}")
	logger.info(f"process memory vms is {before_memory_vms}")
	logger.info(f"process memory shared is {before_memory_shared}")
	logger.info(f"process memory text is {before_memory_text}")
	logger.info(f"process memory lib is {before_memory_lib}")
	logger.info(f"process memory data is {before_memory_data}")
	logger.info(f"process memory dirty is {before_memory_dirty}")
	logger.info(f"start time is: {start_time}")
	logger.info(f"cpu utilization: {cpu_utilization_start}")

	# folder creation
	if not os.path.exists(master_path):
		os.makedirs(master_path)
	if not os.path.exists(ocr_path):
		os.makedirs(ocr_path)

	# step 3: class and label mapping
	try:
		with open(os.path.join(root_folder_path, "label.txt"), "r") as file:
			class_names: List = file.readlines()
			class_names = list(map(lambda x: x.strip(), class_names))
			logger.info("is class_names is a instance of list? %s", isinstance(class_names, list))
			dict_mapping = dict(enumerate(class_names))
			logger.info("is dict_mapping is a instance of dict? %s", isinstance(dict_mapping, dict))
			logger.info(f"dict mapping is {dict_mapping}")
	except IOError:
		if debug_mode:
			exit("Error opening a label.txt file")
		logger.info("Error opening a label.txt file")

	finally:
		file.close()

	# iterating over labelled files
	labelled_files: List = os.listdir(path=labels_path)
	names_of_labelled_files: List = [x.split(".txt")[0] for x in labelled_files]

	img_files: list = os.listdir(images_path)
	names_of_img_files = [x.split(f".{image_type}")[0] for x in img_files]

	# # function 80
	with Pool(processes=5) as pool:
		result = pool.starmap(
			func=data_creation,
			iterable=[(inner_chunk, names_of_img_files)
			          for inner_chunk in chunks(100, names_of_labelled_files)])

	# labelled_files = data_creation(names_of_labelled_files, names_of_img_files)

	complete_labelled_files = []
	merge_result = [complete_labelled_files + result for result in labelled_files]

	thresh = int(parser['PARAMS']['thresh_value'])  # 300
	seed_val = int(parser['PARAMS']['random_seed_val'])
	ratio_val = float(parser['PARAMS']['train_div_ratio'])
	logger.info(f"Threshold value is {thresh}")
	logger.info(f"seed value is {seed_val}")
	logger.info(f"ratio value is {ratio_val}")

	train_samples_imgs = merge_result[:-int(ratio_val * len(merge_result))]
	logger.info(f"Length of the train data is {len(train_samples_imgs)}")
	print(len(train_samples_imgs))
	test_samples_imgs = merge_result[-int(ratio_val * len(merge_result)):]
	logger.info(f"Length of the test data is {len(test_samples_imgs)}")
	print(len(test_samples_imgs))

	train_test_split(train_samples_imgs, 'train', img_files=names_of_img_files,
	                 ocr_file_suffix_to_save=ocr_file_suffix_to_save)
	train_test_split(test_samples_imgs, 'test', img_files=names_of_img_files,
	                 ocr_file_suffix_to_save=ocr_file_suffix_to_save)

	end_time = datetime.now()
	cpu_utilization_end = psutil.cpu_percent()
	diff = end_time - start_time
	after_memory_rss = process.memory_info().rss
	after_memory_vms = process.memory_info().vms
	after_memory_shared = process.memory_info().shared
	after_memory_text = process.memory_info().text
	after_memory_lib = process.memory_info().lib
	after_memory_data = process.memory_info().data
	after_memory_dirty = process.memory_info().dirty
	cpu_utt = cpu_utilization_end - cpu_utilization_start
	memory_consumption_rss = after_memory_rss - before_memory_rss
	memory_consumption_vms = after_memory_vms - before_memory_vms
	memory_consumption_shared = after_memory_shared - before_memory_shared
	memory_consumption_text = after_memory_text - before_memory_text
	memory_consumption_lib = after_memory_lib - before_memory_lib
	memory_consumption_data = after_memory_data - before_memory_data
	memory_consumption_dirty = after_memory_dirty - before_memory_dirty
	logger.info("total time taken for internal_data preparation:" + str(diff))
	logger.info("cpu_utilization %:" + str(cpu_utt))
	logger.info("memory_consumption in bytes rss:" + str(memory_consumption_rss))
	logger.info("memory_consumption in bytes:" + str(memory_consumption_vms))
	logger.info("memory_consumption in bytes:" + str(memory_consumption_shared))
	logger.info("memory_consumption in bytes:" + str(memory_consumption_text))
	logger.info("memory_consumption in bytes:" + str(memory_consumption_lib))
	logger.info("memory_consumption in bytes:" + str(memory_consumption_data))
	logger.info("memory_consumption in bytes:" + str(memory_consumption_dirty))
	logger.info(('RAM memory % used:', psutil.virtual_memory()[2]))
