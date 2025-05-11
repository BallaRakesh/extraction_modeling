# importing outside packages
from google.cloud import vision
import numpy as np
import logging
import psutil
from base64 import b64encode
import os
from datetime import datetime
import warnings
import pytesseract
from configparser import ConfigParser

from seqeval.metrics import (
	classification_report,
	f1_score,
	precision_score,
	recall_score)

# removing the warnings
warnings.filterwarnings("ignore")

# config snippet
configur = ConfigParser()
configur.read('training/lmv2_code/src/main/Augmentations/traini_valid_utility.ini')

gv_key = configur['OCR']['gv_key']
folder_path = str(configur['PATHS']['folder_path'])

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


set_basic_config_for_logging(filename="training")
logger = get_logger_object_and_setting_the_loglevel()
process_memory = psutil.Process()

def data_convert(dataset):
	words = []
	boxes = []
	labels = []

	for datas in (dataset):
		words.append(datas['words'])
		boxes.append(datas['bbox'])
		labels.append(datas['labels'])
	return words, boxes, labels


def get_intersection_percentage(bb1, bb2):
	"""
    Finds the percentage of intersection  with a smaller box. (what percernt of smaller box is in larger box)
    """
	assert bb1['x1'] < bb1['x2']
	assert bb1['y1'] < bb1['y2']
	assert bb2['x1'] < bb2['x2']
	assert bb2['y1'] < bb2['y2']

	# determine the coordinates of the intersection rectangle
	x_left = max(bb1['x1'], bb2['x1'])
	y_top = max(bb1['y1'], bb2['y1'])
	x_right = min(bb1['x2'], bb2['x2'])
	y_bottom = min(bb1['y2'], bb2['y2'])

	if x_right < x_left or y_bottom < y_top:
		return 0.0

	# The intersection of two axis-aligned bounding boxes is always an
	# axis-aligned bounding box
	intersection_area = (x_right - x_left) * (y_bottom - y_top)

	# compute the area of both AABBs
	bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
	# min_area = min(bb1_area,bb2_area)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	intersection_percent = intersection_area / bb2_area
	assert intersection_percent >= float(configur['PARAMS']['intersect_per2'])
	assert intersection_percent <= float(configur['PARAMS']['intersect_per'])
	return intersection_percent


def normalize(points: list, width: int, height: int) -> list:
	x0, y0, x2, y2 = [int(p) for p in points]
	val = int(configur['PARAMS']['norm_val'])
	zero_val = int(configur['PARAMS']['nill_val'])
	x0 = int(val * (x0 / width))
	x2 = int(val * (x2 / width))
	y0 = int(val * (y0 / height))
	y2 = int(val * (y2 / height))
	if x0 > val:
		x0 = val
	if x0 < zero_val:
		x0 = zero_val
	if x2 > val:
		x2 = val
	if x2 < zero_val:
		x2 = zero_val
	if y0 > val:
		y0 = val
	if y0 < zero_val:
		y0 = zero_val
	if y2 > val:
		y2 = val
	if y2 < zero_val:
		y2 = zero_val
	return [x0, y0, x2, y2]


def get_iou(bb1, bb2):
	assert bb1['x1'] < bb1['x2']
	assert bb1['y1'] < bb1['y2']
	assert bb2['x1'] < bb2['x2']
	assert bb2['y1'] < bb2['y2']
	# determine the coordinates of the intersection rectangle
	x_left = max(bb1['x1'], bb2['x1'])
	y_top = max(bb1['y1'], bb2['y1'])
	x_right = min(bb1['x2'], bb2['x2'])
	y_bottom = min(bb1['y2'], bb2['y2'])

	if x_right < x_left or y_bottom < y_top:
		return 0.0
	# The intersection of two axis-aligned bounding boxes is always an
	# axis-aligned bounding box
	intersection_area = (x_right - x_left) * (y_bottom - y_top)
	# compute the area of both AABBs
	bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
	bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
	assert iou >= 0.0
	assert iou <= 1.0
	return iou


def contour_sort(a, b):
	if abs(a['y1'] - b['y1']) <= 15:
		return a['x1'] - b['x1']

	return a['y1'] - b['y1']


def remove_garbage(dataset):
	to_remove = ["\u00da", "\u00c6", "\u00c4", "\u00b4", "\u00c5", "Á"]
	for key in dataset.keys():
		values = dataset[key]
		for value in values:
			string = value[0]
			new_string = ""
			for char in string:
				if char not in to_remove:
					new_string += char
			new_string = new_string.strip()
			value[0] = new_string


def get_ocr_vision_api(image_path):
	os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(gv_key)
	with open(image_path, 'rb') as f:
		ctxt = b64encode(f.read()).decode()
	f.close()
	client = vision.ImageAnnotatorClient()
	image = vision.Image(content=ctxt)
	response = client.text_detection(image=image)

	word_coordinates = []
	all_text = ""
	logger.info("is word_coordinates is a instance of list? %s", isinstance(word_coordinates, list))
	logger.info("is all_text is a instance of str? %s", isinstance(all_text, str))
	for i, text in enumerate(response.text_annotations):
		if i != 0:
			x1 = min([v.x for v in text.bounding_poly.vertices])
			x2 = max([v.x for v in text.bounding_poly.vertices])
			y1 = min([v.y for v in text.bounding_poly.vertices])
			y2 = max([v.y for v in text.bounding_poly.vertices])
			if x2 - x1 == 0:
				x2 += 1
			if y2 - y1 == 0:
				y2 += 1
			word_coordinates.append({
				"word": text.description,
				"left": x1,
				"top": y1,
				"width": x2 - x1,
				"height": y2 - y1,
				"x1": x1,
				"y1": y1,
				"x2": x2,
				"y2": y2
			})
		else:
			all_text = text.description

	return word_coordinates, all_text


def results_test(preds, out_label_ids, labels):
	preds = np.argmax(preds, axis=2)
	label_val = int(configur['PARAMS']['out_label_ids'])
	label_map = {i: label for i, label in enumerate(labels)}

	out_label_list = [[] for _ in range(out_label_ids.shape[0])]
	preds_list = [[] for _ in range(out_label_ids.shape[0])]

	for i in range(out_label_ids.shape[0]):
		for j in range(out_label_ids.shape[1]):

			if out_label_ids[i, j] != -label_val:
				out_label_list[i].append(label_map[out_label_ids[i][j]])
				preds_list[i].append(label_map[preds[i][j]])

	results = {
		"precision": precision_score(out_label_list, preds_list),
		"recall": recall_score(out_label_list, preds_list),
		"f1": f1_score(out_label_list, preds_list),
	}
	return results, classification_report(out_label_list, preds_list)


def results_train(preds, out_label_ids, labels):
	preds = np.argmax(preds, axis=2)
	label_val = int(configur['PARAMS']['out_label_ids'])
	label_map = {i: label for i, label in enumerate(labels)}

	out_label_list = [[] for _ in range(out_label_ids.shape[0])]
	preds_list = [[] for _ in range(out_label_ids.shape[0])]

	for i in range(out_label_ids.shape[0]):
		for j in range(out_label_ids.shape[1]):
			if out_label_ids[i, j] != -label_val:
				out_label_list[i].append(label_map[out_label_ids[i][j]])
				preds_list[i].append(label_map[preds[i][j]])

	results = {
		"precision": precision_score(out_label_list, preds_list),
		"recall": recall_score(out_label_list, preds_list),
		"f1": f1_score(out_label_list, preds_list),
	}
	return results, classification_report(out_label_list, preds_list)


class OcrTextGeneration:
	def get_ocr_vision_api(self, image_path):
		os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(gv_key)
		with open(image_path, 'rb') as f:
			ctxt = b64encode(f.read()).decode()
		client = vision.ImageAnnotatorClient()
		image = vision.Image(content=ctxt)

		response = client.text_detection(image=image)

		word_coordinates = []
		all_text = ""
		for i, text in enumerate(response.text_annotations):
			if i != 0:
				x1 = min([v.x for v in text.bounding_poly.vertices])
				x2 = max([v.x for v in text.bounding_poly.vertices])
				y1 = min([v.y for v in text.bounding_poly.vertices])
				y2 = max([v.y for v in text.bounding_poly.vertices])
				if x2 - x1 == 0:
					x2 += 1
				if y2 - y1 == 0:
					y2 += 1
				word_coordinates.append({
					"word": text.description,
					"left": x1,
					"top": y1,
					"width": x2 - x1,
					"height": y2 - y1,
					"x1": x1,
					"y1": y1,
					"x2": x2,
					"y2": y2
				})
			else:
				all_text = text.description

		return word_coordinates, all_text

	def get_ocr_tesserract(self, img):
		t1 = datetime.now()
		print("called Image OCR...", end="")
		d = pytesseract.image_to_data(img)
		all_text = pytesseract.image_to_string(img)
		word_coordinates = []
		for i, b in enumerate(d.splitlines()):
			if i != 0:
				b = b.split()
				if len(b) == 12:
					word = b[11]
					x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
					word_coordinates.append({
						"word": word,
						"left": x,
						"top": y,
						"width": w,
						"height": h,
						"x1": x,
						"y1": y,
						"x2": x + w,
						"y2": y + h
					})
		t2 = datetime.now()
		print("OCR time", t2 - t1)
		return word_coordinates, all_text
