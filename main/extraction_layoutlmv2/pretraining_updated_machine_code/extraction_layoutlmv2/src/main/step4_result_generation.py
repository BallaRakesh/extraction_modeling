import fitz
import torchvision.transforms as transforms
import torch
from google.cloud import vision
from base64 import b64encode
from PIL import Image, ImageDraw, ImageFont
import os
from datetime import datetime
import cv2
import random
import warnings
from PIL import ImageSequence
import pytesseract
import gc
import configparser

folder_path: str = "/home/ntlpt19/Downloads/Evaluation_Data/FinalEvaluationEvalData/PL"

ocr_path = os.path.join(folder_path, "OCR")
path = os.path.join(folder_path, "Images")
label_path = os.path.join(folder_path, "classes.txt")
App_Filepath = os.path.dirname(os.path.abspath(__file__))
config = configparser.ConfigParser()
config.read(App_Filepath + '/config.ini')

os.environ["TOKENIZERS_PARALLELISM"] = "true"
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.cluster import DBSCAN
from functools import cmp_to_key
import json
import traceback
from nltk import ngrams
import re
from scipy.special import softmax
from fuzzywuzzy import fuzz
import psutil

# config = configparser.ConfigParser()
# config.read('config_.ini')
transform2 = transforms.ToPILImage()
transform = transforms.ToTensor()
result_path = os.path.join(folder_path, "Results_BOL_validated")
if not os.path.exists(result_path):
        os.mkdir(result_path)
    

zoom = 300 / 72
mat = fitz.Matrix(zoom, zoom)


# "****************************************************************************************************************************************"

def area(coordinates):
	l = coordinates[2] - coordinates[0]
	h = coordinates[3] - coordinates[1]
	return l * h


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
	assert bb1[0] < bb1[2]
	assert bb1[1] < bb1[3]
	assert bb2[0] < bb2[2]
	assert bb2[1] < bb2[3]

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


def minimum_distance(bb1, bb2):
	# bb1 points
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


def model_output_sum(key, box, model_output):
	all_values = model_output[key]
	all_values = sorted(all_values, key=cmp_to_key(contour_sort))
	all_text = ""
	for value in all_values:
		try:
			iou = get_iou_new(value[1], box)
		except:
			continue
		if iou > 0:
			if all_text == "":
				all_text = value[0]
			else:
				all_text = all_text + " " + value[0]
	return all_text


def merge_surrounding(data, model_output):
	new = data.copy()
	for key in list(data.keys()):
		print(key)
		bboxes = [x[1] for x in data[key]]
		eps = 100
		all_values = data[key]
		print(all_values)
		length = len(all_values)
		if length > 1:
			i = 0
			while i in range(length - 1):
				print(i)
				bb1 = all_values[i][1]
				bb2 = all_values[i + 1][1]
				confs = [all_values[i][2], all_values[i + 1][2]]
				# ocr_confs = [all_values[i][3],all_values[i+1][3]]
				min_dist = minimum_distance(bb1, bb2)
				try:
					IOU = get_iou_new(bb1, bb2)
				except:
					i = i + 1
					continue
				if min_dist <= eps or IOU > 0:
					print("merging: " + all_values[i][0] + " and " + all_values[i + 1][0])
					x_left = min(bb1[0], bb2[0])
					y_top = min(bb1[1], bb2[1])
					x_right = max(bb1[2], bb2[2])
					y_bottom = max(bb1[3], bb2[3])
					box = [x_left, y_top, x_right, y_bottom]
					text = model_output_sum(key, box, model_output)
					print("merged text is ", text)
					avg_confs = (confs[0] * area(bb1) + confs[1] * area(bb2)) / (area(bb1) + area(bb2))
					"""if "NA" in ocr_confs:
						avg_ocr_confs = "NA"
					else:
						avg_ocr_confs = ( ocr_confs[0]* area(bb1) + ocr_confs[1]*area(bb2) )/(area(bb1) + area(bb2))"""
					new_value = [text, box, avg_confs]
					print(new_value)
					all_values.remove(all_values[i])
					all_values.remove(all_values[i])
					all_values.insert(i, new_value)
					print(all_values)
					length = len(all_values)
					if length == 1:
						print("will break")
						break
				else:
					print("distance is very high")
					i = i + 1
		else:
			print("will continue")
			continue


# find most common label
def most_common(lst):
	return max(set(lst), key=lst.count)


# converts a tuple to string
def tuple_to_string(sen):
	str_test = sen
	word = ""
	for i, w in enumerate(str_test):
		if i == 0:
			word = word + w
		else:
			word = word + ' ' + w
	return word


# runs the lookup script for comparison with main model

# lookup function to find contries in 'bill to', 'ship to', 'remit to' fields
def lookup(
		text,
		n_words,
		match_threshold,
		file_path,
		result_set,
		key
):
	try:
		out_dict = {}
		out_list = []
		all_box = []
		message = ""

		file = open(file_path, "r")
		lines = file.readlines()

		txt_words = []
		for l in lines:
			line = l.split("\n")
			txt_words.append(line[0])

		if n_words > 5:
			status = "N words larger than 5, provide N words less than 5"
			out_dict["status"] = status
			return out_dict

		if match_threshold < 80:
			status = "Matching threshold value less than 80, provide Matching threshold greater than 80"
			out_dict["status"] = status
			return out_dict
		else:

			res = re.sub(r"[^\w\s]", "", text)

			gram_list = []

			# fourgrams = ngrams(res.split(), n)

			for j in range(n_words):
				gram_count = ngrams(res.split(), j + 1)

				# ourgrams = ngrams(res.split(), n)

				for gram in gram_count:
					sen = tuple_to_string(gram)
					gram_list.append(sen)

			for word in gram_list:
				for txt_char in txt_words:

					# print(word[0].lower())

					if fuzz.ratio(word.lower(), txt_char) > match_threshold:
						# if word.lower() in txt_words:
						# print(word, "----", txt_char, fuzz.ratio(word.lower(), txt_char))

						info_dict = {}
						info_dict["searched_string"] = word  # searched string is our data.
						info_dict["found_string"] = txt_char  # found string is present in countries.txt (lookup file)
						info_dict["string_match_value"] = fuzz.ratio(
							word.lower(), txt_char
						)
						out_list.append(info_dict)
			if len(out_list) != 0:
				for res in out_list:
					look_up = res['found_string']
					# print(found)
					original = res['searched_string']
					original_words = original.split()
					for val in result_set[key]:
						for word in original_words:
							if fuzz.ratio(word.lower(), val[0]) > match_threshold:
								all_box.append(val[1])
					x1 = min([x[0] for x in all_box])
					x2 = max([x[2] for x in all_box])
					y1 = min([x[1] for x in all_box])
					y2 = max([x[3] for x in all_box])
					box_result = [x1, y1, x2, y2]
					res['bbox'] = box_result

			print(out_list)
			# out_dict['response'] = out_list
			# out_dict['status'] = 'success'
			# print(out_dict)

			return out_list
	except Exception as e:
		print(traceback.format_exc())


# Normalizes all points after zooming
def normalize(points: list, width: int, height: int) -> list:
	x0, y0, x2, y2 = [int(p) for p in points]

	x0 = int(1000 * (x0 / width))
	x2 = int(1000 * (x2 / width))
	y0 = int(1000 * (y0 / height))
	y2 = int(1000 * (y2 / height))

	if x0 > 1000:
		x0 = 1000
	# print(">")
	if x0 < 0:
		x0 = 0
	# print("<")
	if x2 > 1000:
		x2 = 1000
	# print(">")
	if x2 < 0:
		x2 = 0
	# print("<")
	if y0 > 1000:
		y0 = 1000
	# print(">")
	if y0 < 0:
		y0 = 0
	# print("<")
	if y2 > 1000:
		y2 = 1000
	# print(">")
	if y2 < 0:
		y2 = 0
	# print("<")
	return [x0, y0, x2, y2]


# Unnormalize the file
def unnormalize_box(bbox, width, height):
	return [
		int(width * (bbox[0] / 1000)),
		int(height * (bbox[1] / 1000)),
		int(width * (bbox[2] / 1000)),
		int(height * (bbox[3] / 1000)),
	]


# used to order the field values inside a label. Refer Videos.
def contour_sort(a, b):
	if abs(a[1][1] - b[1][1]) <= 15:
		return a[1][0] - b[1][0]
	return a[1][1] - b[1][1]


# OCR Vision function
def get_ocr_vision_api(file):
	image = file
	os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "linen-creek-370205-4084e44af41e.json"
	ctxt = b64encode(image.read()).decode()
	client = vision.ImageAnnotatorClient()
	image = vision.Image(content=ctxt)

	response = client.text_detection(image=image)

	for res in response.text_annotations:
		print(res.confidence)

	word_coordinates = []
	all_text = ""

	for i, text in enumerate(response.text_annotations):
		if i != 0:
			# print('=' * 30)
			# print(text.description)
			vertices = [(v.x, v.y) for v in text.bounding_poly.vertices]
			x1 = min([v.x for v in text.bounding_poly.vertices])
			x2 = max([v.x for v in text.bounding_poly.vertices])
			y1 = min([v.y for v in text.bounding_poly.vertices])
			y2 = max([v.y for v in text.bounding_poly.vertices])
			# print('bounds: ' + str(vertices))
			if x2 - x1 == 0:
				x2 += 1
			if y2 - y1 == 0:
				y2 += 1
			""""left": x1,
				"top": y1,
				"width": x2 - x1,
				"height": y2 - y1,"""
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


# OCR tesseract function
def get_ocr_tesserract(img):
	t1 = datetime.now()
	print("called Image OCR...", end="")
	# img = cv2.imread(image)
	# hImg,wImg,_ = img.shape
	# img = cv2.imread(image)
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


# GETOCR FITZ
def get_text_fitz(page):
	print("called Text OCR...", end="")
	# t1 = datetime.now()
	word_coordinates = []
	all_text = ""
	# all_text += page.getText()
	text = page.getText("words")
	for t in text:
		word_coordinates.append({
			"word": t[4],
			"left": t[0] * zoom,
			"top": t[1] * zoom,
			"right": t[2] * zoom,
			"bottom": t[3] * zoom,
			"x1": t[0] * zoom,
			"y1": t[1] * zoom,
			"x2": t[2] * zoom,
			"y2": t[3] * zoom,
		})
	# t2=datetime.now()
	# print("OCRtime", t2-t1)
	return word_coordinates, all_text


# "****************************************************************************************************************************************"
# function called when image file is passed(png, jpeg, tif)
def image_result(file, path01, model, processor, device, folder_path):
	print("file name is:", file)
	t_total_start = datetime.now()
	print("Calling Image result")
	all_page_result = {}
	count = 0
	im = Image.open(os.path.join(path, file))
	for i, image in enumerate(ImageSequence.Iterator(im)):
		count += 1
		print("******* Page " + str(count) + "********")
		t_page_start = datetime.now()
		w, h = image.size
		temp = image.convert("L")
		image_data = np.asarray(temp)
		image = cv2.cvtColor(image_data, cv2.COLOR_GRAY2RGB)
		# plt.imshow(image)
		# plt.show()
		# plt.close()
		arr = transform(image)
		# arr = np.array(image)
		# print("Shape is", arr.shape)
		# print(arr)
		try:
			with open(os.path.join(ocr_path, file[:-4] + "_text.txt"), "r") as f:
				word_coordinates = json.load(f)['word_coordinates']
				print("word_coordinates are:")
				print(word_coordinates)
		except Exception as e:
			print(e)
			# exit("++++++++++++")
			try:
				word_coordinates, all_text = get_ocr_vision_api(image)
			except:
				word_coordinates, all_text = get_ocr_tesserract(image)
		if len(word_coordinates) == 0:
			print("Not enough text")
		words = []
		bboxes = []
		bounding_boxes = []
		for t in word_coordinates:
			if 'right' in list(t.keys()):
				t['x1'] = t['left']
				t['y1'] = t['top']
				t['x2'] = t['right']
				t['y2'] = t['bottom']
			words.append(t['word'])
			bounding_boxes.append([t['x1'], t['y1'], t['x2'], t['y2']])
			bboxes.append(normalize([t['x1'], t['y1'], t['x2'], t['y2']], w, h))
		encoded_inputs = processor(arr, words, boxes=bboxes, return_tensors="pt")
		input_id_chunks = list(encoded_inputs['input_ids'][0].split(510))
		token_type_id_chunks = list(encoded_inputs['token_type_ids'][0].split(510))
		mask_chunks = list(encoded_inputs['attention_mask'][0].split(510))
		bbox_chunks = list(encoded_inputs['bbox'][0].split(510))
		image_chunk = encoded_inputs['image'][0]
		image_chunks = list()
		# loop through each chunk
		for i in range(len(input_id_chunks)):
			image_chunks.append(image_chunk)
			# add CLS and SEP tokens to input IDs
			input_id_chunks[i] = torch.cat([
				torch.tensor([101]), input_id_chunks[i], torch.tensor([102])
			])
			token_type_id_chunks[i] = torch.cat([
				torch.tensor([0]), token_type_id_chunks[i], torch.tensor([0])
			])
			# add attention tokens to attention mask
			mask_chunks[i] = torch.cat([
				torch.tensor([1]), mask_chunks[i], torch.tensor([1])
			])
			bbox_chunks[i] = torch.cat([
				torch.tensor([[0, 0, 0, 0]]), bbox_chunks[i], torch.tensor([[0, 0, 0, 0]])
			])
			# get required padding length
			pad_len = 512 - input_id_chunks[i].shape[0]
			# check if tensor length satisfies required chunk size
			if pad_len > 0:
				# if padding length is more than 0, we must add padding
				input_id_chunks[i] = torch.cat([
					input_id_chunks[i], torch.Tensor([0] * pad_len)
				])
				token_type_id_chunks[i] = torch.cat([
					token_type_id_chunks[i], torch.Tensor([0] * pad_len)
				])
				mask_chunks[i] = torch.cat([
					mask_chunks[i], torch.Tensor([0] * pad_len)
				])

				bbox_chunks[i] = torch.cat([
					bbox_chunks[i], torch.Tensor([[0, 0, 0, 0]] * pad_len)
				])

		input_ids = torch.stack(input_id_chunks)
		attention_mask = torch.stack(mask_chunks)
		token_type_ids = torch.stack(token_type_id_chunks)
		bbox = torch.stack(bbox_chunks)
		images = torch.stack(image_chunks)
		input_dict = {
			'input_ids': input_ids.long().to(device),
			'attention_mask': attention_mask.float().to(device),
			'token_type_ids': token_type_ids.long().to(device),
			'bbox': bbox.long().to(device),
			'image': images.float().to(device)
		}
		outputs = model(**input_dict)
		print("Model Called")
		print('RAM memory % used:', psutil.virtual_memory()[2])
		# print(outputs)
		all_predictions = []
		all_boxes = []
		all_confidences = []
		all_text = []
		# this is the list of classes that will be given to us to be extracted.
		with open(label_path, "r") as f:
			labels = eval(f.read())  # labels = list(labels)

		# Creating two dictionaries labels2id and id2labels
		labels = [x.replace("S-", "") for x in labels]
		label2id = {label: idx for idx, label in enumerate(labels)}
		id2label = {idx: label for idx, label in enumerate(labels)}
		# print(label2id)
		# print(id2label)
		number_of_colors = 80
		color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]
		# color
		label2color = {}
		for i, l in enumerate(labels):
			label2color[l] = color[i]
		for i, output in enumerate(outputs.logits):
			print(i, output)
			new_img = transform2(arr)
			# ImageDraw.Draw.textsize = 12
			# font = ImageFont.load_default(size=16)
			font = ImageFont.truetype("arial.ttf", 20)
			predictions = output.argmax(-1).squeeze().tolist()
			confidences = softmax(output.cpu().detach().numpy(), axis=1)
			max_confidences = np.max(confidences, axis=1).reshape(confidences.shape[0], -1)
			all_confidences += [x[0] for x in max_confidences]
			token_boxes = bbox_chunks[i].squeeze().tolist()
			width, height = new_img.size
			true_predictions = [id2label[prediction] for prediction in predictions]
			all_predictions += true_predictions
			true_boxes = [unnormalize_box(box, width, height) for box in token_boxes]
			all_boxes += true_boxes
			for id in input_dict['input_ids'][i]:
				all_text.append(processor.tokenizer.decode(id))
		# print(all_text)
		# print("Output size", len(outputs))
		del outputs
		new_img = transform2(arr)
		draw = ImageDraw.Draw(new_img)
		# print("%20s - %30s - %12s - %30s" % ("Text", "Prediction", "Confidence", "Bounding Box"))
		curr_box = []
		results_pred = []
		results_conf = []
		results_bbox = []
		results_text = []
		temp_preds = []
		temp_confs = []
		temp_text = []
		sep_index = all_text.index('[SEP]')
		if len(all_text) > 512:
			if '[PAD]' in all_text:
				sep_index = all_text.index('[PAD]') - 2
			else:
				sep_index = len(all_text) - 3
		# print(sep_index)
		for i in range(len(all_text)):
			if all_text[i] not in ['[CLS]', '[SEP]', '[PAD]']:  # and all_predictions[i] != 'O':
				# print(i)
				if (curr_box != all_boxes[i] and len(temp_text) > 0) or (i == sep_index - 1 and len(temp_text) > 0):
					# print("1: ", all_text[i])
					if i == sep_index - 1:
						temp_text.append(all_text[i])
						temp_confs.append(all_confidences[i])
						temp_preds.append(all_predictions[i])
					text = ""
					pred = "O"
					preds = [x for x in temp_preds if x != 'O']
					conf = 0
					if len(preds) > 0:
						pred = most_common(preds)
					for j in range(len(temp_text)):
						text += temp_text[j].replace("##", "")
						if temp_preds[j] == pred:
							conf += temp_confs[j]
					conf = float(np.round(conf * 100 / len(temp_text), 2))

					results_text.append(text)
					results_conf.append(conf)
					results_pred.append(pred)
					results_bbox.append(curr_box)

					temp_text = []
					temp_confs = []
					temp_preds = []
					if i != sep_index - 1:
						temp_text.append(all_text[i])
						temp_confs.append(all_confidences[i])
						temp_preds.append(all_predictions[i])
						curr_box = all_boxes[i]
				elif curr_box == all_boxes[i]:
					# print("2: ", all_text[i])
					temp_text.append(all_text[i])
					temp_confs.append(all_confidences[i])
					temp_preds.append(all_predictions[i])
				elif len(temp_text) == 0:
					# print("3: ", all_text[i])
					temp_text.append(all_text[i])
					temp_confs.append(all_confidences[i])
					temp_preds.append(all_predictions[i])
					curr_box = all_boxes[i]

		# print(results_text)
		# print("%20s - %30s - %12s - %30s" % ("Text", "Prediction", "Confidence", "Bounding Box"))
		result_set = {}
		for i in range(len(results_pred)):
			# print("%20s - %30s - %12s - %30s" % (results_text[i], results_pred[i], str(results_conf[i]), str(results_bbox[i])))
			# draw.rectangle(results_bbox[i], outline=label2color[results_pred[i]])
			# draw.text((results_bbox[i][0] + 5, results_bbox[i][1] - 20), text=results_pred[i] + " - " + str(results_conf[i]), fill=label2color[results_pred[i]], font=font)
			if results_pred[i] != 'O':
				if results_pred[i] not in list(result_set.keys()):
					result_set[results_pred[i]] = []
				result_set[results_pred[i]].append([results_text[i], results_bbox[i], results_conf[i]])
		# print(result_set)
		model_output = result_set.copy()
		print("+++++++++++++++++++reached here+++++++++++++++++")
		# exit("+++++++++")
		with open(os.path.join(result_path, file[:-4] + str(count) + "model_output.txt"), "w") as f:
			json.dump(result_set, f)
		final_result_set = {}

		for k in list(result_set.keys()):
			try:
				alpha = float(config[k]['ALPHA'])
			except:
				alpha = float(config['Default']['ALPHA'])
			if len(result_set[k]) > 1:
				print("++++++++++++++entry in this block+++++++++++")
				texts = [x[0] for x in result_set[k]]
				bboxes = [x[1] for x in result_set[k]]
				confs = [x[2] for x in result_set[k]]
				avg_w = np.mean([abs(x[0] - x[2]) for x in bboxes])
				avg_h = np.mean([abs(x[1] - x[3]) for x in bboxes])
				eps = np.sqrt(avg_w ** 2 + avg_h ** 2) * alpha

				clustering = DBSCAN(eps=eps, min_samples=1).fit(bboxes)

				label_set = set(clustering.labels_)
				for l in label_set:
					selected = list(np.where(clustering.labels_ == l)[0])
					selected_texts = [x for i, x in enumerate(texts) if i in selected]
					selected_boxes = [x for i, x in enumerate(bboxes) if i in selected]
					selected_confs = [x for i, x in enumerate(confs) if i in selected]
					text_boxes = [[x, y] for x, y in zip(selected_texts, selected_boxes)]
					text_boxes = sorted(text_boxes, key=cmp_to_key(contour_sort))
					text_result = ""

					for tb in text_boxes:
						if text_result == "":
							text_result += tb[0]
						else:
							text_result += " " + tb[0]
					# print(text_result)
					x1 = min([x[0] for x in selected_boxes])
					x2 = max([x[2] for x in selected_boxes])
					y1 = min([x[1] for x in selected_boxes])
					y2 = max([x[3] for x in selected_boxes])
					box_result = [x1, y1, x2, y2]
					conf_result = float(np.round(np.mean(selected_confs), 2))
					# print(box_result)
					if k not in list(final_result_set.keys()):
						final_result_set[k] = []
					final_result_set[k].append([text_result, box_result, conf_result])
			else:
				if k not in list(final_result_set.keys()):
					final_result_set[k] = []
				final_result_set[k].append([result_set[k][0][0], result_set[k][0][1], result_set[k][0][2]])
			# draw.rectangle(result_set[k][0][1], outline=label2color[k], width=2)
			# draw.text((result_set[k][0][1][0] + 5, result_set[k][0][1][1] - 20),
			# text=k + " - " + str(result_set[k][0][2]), fill=label2color[k], font=font)
		merge_surrounding(final_result_set, model_output)
		print("+++++++++++reached here after merge surrounding++++++++++")
		for k in list(final_result_set.keys()):
			all_values = final_result_set[k]
			for value in all_values:
				draw.rectangle(value[1], outline=label2color[k], width=2)
				draw.text((value[1][0] + 5, value[1][1] - 20),
				          text=k + " - " + str(value[2]), fill=label2color[k], font=font)
		lookup_result = {}
		t_page_end = datetime.now()
		print("Time taken for page" + str(count) + ":", end=" ")
		print(t_page_end - t_page_start)
		print()
		all_page_result["Page Number " + str(count)] = final_result_set
		# print(all_page_result)
		for k in list(final_result_set.keys()):
			if k in ["applicant_country", "beneficiary_country"]:
				for val in final_result_set[k]:
					result_country = lookup(val[0], 4, 90, "countries.txt", result_set, k)
					result_company = lookup(val[0], 4, 90, "organization.txt", result_set, k)
					# print(val)
					# print(result)
					# replacing ocr result with correct result
					for res in result_company:
						found = res['found_string']
						searched = res['searched_string']
						new_val = val[0].replace(searched, found)
						val[0] = new_val
					# replacing ocr result with correct result
					for res in result_country:
						found = res['found_string']
						searched = res['searched_string']
						new_val = val[0].replace(searched, found)
						val[0] = new_val
					for res in result_country:
						if (str(k) + "-country") not in lookup_result:
							lookup_result[('LUT_' + str(k) + "-country")] = []
						lookup_result[('LUT_' + str(k) + "-country")].append(
							(res['found_string'], res['string_match_value'], res["bbox"]))
					for res in result_company:
						if (str(k) + "-organization") not in lookup_result:
							lookup_result[('LUT_' + str(k) + "-organization")] = []
						lookup_result[('LUT_' + str(k) + "-organization")].append(
							(res['found_string'], res['string_match_value'], res["bbox"]))
		# print(lookup_result)
		with open(os.path.join(result_path, file[:-4] + str(count) + "_lookup.txt"), "w") as f:
			json.dump(lookup_result, f)
		with open(os.path.join(result_path, file[:-4] + str(count) + ".txt"), "w") as f:
			json.dump(final_result_set, f)
		new_img.save(os.path.join(result_path, file[:-4] + str(count) + ".png"))

	# break'''
	# print("running")
	with open(os.path.join(result_path, file[:-4] + "all_page_result.txt"), "w") as f:
		json.dump(all_page_result, f)
	# shutil.rmtree(mainpath)
	# shutil.rmtree(dir_path)
	# os.remove(os.path.join(mainpath, name))
	print("*****Generated All Pages******")
	print('RAM memory % used:', psutil.virtual_memory()[2])
	t_total_end = datetime.now()
	print("Processing time taken for all pages" + str(count) + ":", end=" ")
	print(t_total_end - t_total_start)
	del image
	del new_img
	del encoded_inputs
	gc.collect()
	# print("image file", all_page_result)
	return all_page_result


# "******************************************************************************************************************************************"


def pdf_result(file, model, processor, device):
	t_total_start = datetime.now()
	print("Calling PDF result")
	# orig_path = os.path.join(path, "original")
	# file = os.listdir(orig_path)[0]
	all_page_result = {}
	count = 0
	doc = fitz.open(os.path.join(path, file))
	for page in doc:
		count += 1
		print("******* Page " + str(count) + "********")
		t_page_start = datetime.now()
		pix = page.get_pixmap(matrix=mat)
		# print("created pixmap")
		image = np.array(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
		img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
		# print("created image")
		h, w, _ = image.shape
		# print(h,w)
		word_coordinates = []
		# word_coordinates, all_text = fitz.getText(page)
		print("returned")
		if len(word_coordinates) <= 5:
			try:
				word_coordinates, all_text = get_ocr_vision_api(img)
			except Exception as e:
				print(e)
				word_coordinates, all_text = get_ocr_tesserract(img)
		if len(word_coordinates) == 0:
			print("Not enough text")
		# print("ocr is!", word_coordinates)
		words = []
		bboxes = []
		bounding_boxes = []
		for t in word_coordinates:
			if 'right' in list(t.keys()):
				t['x1'] = t['left']
				t['y1'] = t['top']
				t['x2'] = t['right']
				t['y2'] = t['bottom']
			words.append(t['word'])
			bounding_boxes.append([t['x1'], t['y1'], t['x2'], t['y2']])
			bboxes.append(normalize([t['x1'], t['y1'], t['x2'], t['y2']], w, h))
		encoded_inputs = processor(image, words, boxes=bboxes, return_tensors="pt")
		input_id_chunks = list(encoded_inputs['input_ids'][0].split(510))
		token_type_id_chunks = list(encoded_inputs['token_type_ids'][0].split(510))
		mask_chunks = list(encoded_inputs['attention_mask'][0].split(510))
		bbox_chunks = list(encoded_inputs['bbox'][0].split(510))
		image_chunk = encoded_inputs['image'][0]
		image_chunks = list()
		# loop through each chunk
		for i in range(len(input_id_chunks)):
			image_chunks.append(image_chunk)
			# add CLS and SEP tokens to input IDs
			input_id_chunks[i] = torch.cat([
				torch.tensor([101]), input_id_chunks[i], torch.tensor([102])
			])
			token_type_id_chunks[i] = torch.cat([
				torch.tensor([0]), token_type_id_chunks[i], torch.tensor([0])
			])
			# add attention tokens to attention mask
			mask_chunks[i] = torch.cat([
				torch.tensor([1]), mask_chunks[i], torch.tensor([1])
			])
			bbox_chunks[i] = torch.cat([
				torch.tensor([[0, 0, 0, 0]]), bbox_chunks[i], torch.tensor([[0, 0, 0, 0]])
			])
			# get required padding length
			pad_len = 512 - input_id_chunks[i].shape[0]
			# check if tensor length satisfies required chunk size
			if pad_len > 0:
				# if padding length is more than 0, we must add padding
				input_id_chunks[i] = torch.cat([
					input_id_chunks[i], torch.Tensor([0] * pad_len)
				])
				token_type_id_chunks[i] = torch.cat([
					token_type_id_chunks[i], torch.Tensor([0] * pad_len)
				])
				mask_chunks[i] = torch.cat([
					mask_chunks[i], torch.Tensor([0] * pad_len)
				])

				bbox_chunks[i] = torch.cat([
					bbox_chunks[i], torch.Tensor([[0, 0, 0, 0]] * pad_len)
				])
		input_ids = torch.stack(input_id_chunks)
		attention_mask = torch.stack(mask_chunks)
		token_type_ids = torch.stack(token_type_id_chunks)
		bbox = torch.stack(bbox_chunks)
		images = torch.stack(image_chunks)
		input_dict = {
			'input_ids': input_ids.long().to(device),
			'attention_mask': attention_mask.float().to(device),
			'token_type_ids': token_type_ids.long().to(device),
			'bbox': bbox.long().to(device),
			'image': images.float().to(device)
		}
		outputs = model(**input_dict)
		print("Model Called")
		print('RAM memory % used:', psutil.virtual_memory()[2])
		# print(outputs)
		all_predictions = []
		all_boxes = []
		all_confidences = []
		all_text = []
		# this is the list of classes that will be given to us to be extracted.
		with open(label_path, "r") as f:
			labels = eval(f.read())  # labels = list(labels)

		# Creating two dictionaries labels2id and id2labels
		labels = [x.replace("S-", "") for x in labels]
		label2id = {label: idx for idx, label in enumerate(labels)}
		id2label = {idx: label for idx, label in enumerate(labels)}
		# print(label2id)
		# print(id2label)
		number_of_colors = 80
		color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]
		color
		label2color = {}
		for i, l in enumerate(labels):
			label2color[l] = color[i]
		for i, output in enumerate(outputs.logits):
			# print(i)
			new_img = Image.fromarray(image.copy())
			# ImageDraw.Draw.textsize = 12
			# font = ImageFont.load_default(size=16)
			font = ImageFont.truetype("arial.ttf", 20)
			predictions = output.argmax(-1).squeeze().tolist()
			confidences = softmax(output.cpu().detach().numpy(), axis=1)
			max_confidences = np.max(confidences, axis=1).reshape(confidences.shape[0], -1)
			all_confidences += [x[0] for x in max_confidences]
			token_boxes = bbox_chunks[i].squeeze().tolist()
			width, height = new_img.size
			true_predictions = [id2label[prediction] for prediction in predictions]
			all_predictions += true_predictions
			true_boxes = [unnormalize_box(box, width, height) for box in token_boxes]
			all_boxes += true_boxes
			for id in input_dict['input_ids'][i]:
				all_text.append(processor.tokenizer.decode(id))
		# print(all_text)
		# print("Output size", len(outputs))
		del outputs
		new_img = Image.fromarray(image.copy())
		draw = ImageDraw.Draw(new_img)
		# print("%20s - %30s - %12s - %30s" % ("Text", "Prediction", "Confidence", "Bounding Box"))
		curr_box = []
		results_pred = []
		results_conf = []
		results_bbox = []
		results_text = []
		temp_preds = []
		temp_confs = []
		temp_text = []
		sep_index = all_text.index('[SEP]')
		if len(all_text) > 512:
			if '[PAD]' in all_text:
				sep_index = all_text.index('[PAD]') - 2
			else:
				sep_index = len(all_text) - 3
		# print(sep_index)
		for i in range(len(all_text)):
			if all_text[i] not in ['[CLS]', '[SEP]', '[PAD]']:  # and all_predictions[i] != 'O':
				# print(i)
				if (curr_box != all_boxes[i] and len(temp_text) > 0) or (i == sep_index - 1 and len(temp_text) > 0):
					# print("1: ", all_text[i])
					if i == sep_index - 1:
						temp_text.append(all_text[i])
						temp_confs.append(all_confidences[i])
						temp_preds.append(all_predictions[i])
					text = ""
					pred = "O"
					preds = [x for x in temp_preds if x != 'O']
					conf = 0
					if len(preds) > 0:
						pred = most_common(preds)
					for j in range(len(temp_text)):
						text += temp_text[j].replace("##", "")
						if temp_preds[j] == pred:
							conf += temp_confs[j]
					conf = float(np.round(conf * 100 / len(temp_text), 2))

					results_text.append(text)
					results_conf.append(conf)
					results_pred.append(pred)
					results_bbox.append(curr_box)

					temp_text = []
					temp_confs = []
					temp_preds = []
					if i != sep_index - 1:
						temp_text.append(all_text[i])
						temp_confs.append(all_confidences[i])
						temp_preds.append(all_predictions[i])
						curr_box = all_boxes[i]
				elif curr_box == all_boxes[i]:
					# print("2: ", all_text[i])
					temp_text.append(all_text[i])
					temp_confs.append(all_confidences[i])
					temp_preds.append(all_predictions[i])
				elif len(temp_text) == 0:
					# print("3: ", all_text[i])
					temp_text.append(all_text[i])
					temp_confs.append(all_confidences[i])
					temp_preds.append(all_predictions[i])
					curr_box = all_boxes[i]

		# print(results_text)
		# print("%20s - %30s - %12s - %30s" % ("Text", "Prediction", "Confidence", "Bounding Box"))
		result_set = {}
		for i in range(len(results_pred)):
			# print("%20s - %30s - %12s - %30s" % (results_text[i], results_pred[i], str(results_conf[i]), str(results_bbox[i])))
			# draw.rectangle(results_bbox[i], outline=label2color[results_pred[i]])
			# draw.text((results_bbox[i][0] + 5, results_bbox[i][1] - 20), text=results_pred[i] + " - " + str(results_conf[i]), fill=label2color[results_pred[i]], font=font)
			if results_pred[i] != 'O':
				if results_pred[i] not in list(result_set.keys()):
					result_set[results_pred[i]] = []
				result_set[results_pred[i]].append([results_text[i], results_bbox[i], results_conf[i]])
		model_output = result_set.copy()
		with open(os.path.join(result_path, file[:-4] + str(count) + "model_output.txt"), "w") as f:
			json.dump(result_set, f)
		final_result_set = {}
		with open(os.path.join(result_path, file[:-4] + str(count) + "model_output.txt"), "w") as f:
			json.dump(result_set, f)
		for k in list(result_set.keys()):
			try:
				alpha = float(config[k]['ALPHA'])
			except:
				alpha = float(config['Default']['ALPHA'])
			if len(result_set[k]) > 1:
				texts = [x[0] for x in result_set[k]]
				bboxes = [x[1] for x in result_set[k]]
				confs = [x[2] for x in result_set[k]]
				avg_w = np.mean([abs(x[0] - x[2]) for x in bboxes])
				avg_h = np.mean([abs(x[1] - x[3]) for x in bboxes])
				eps = np.sqrt(avg_w ** 2 + avg_h ** 2) * alpha
				print(k, eps)

				clustering = DBSCAN(eps=eps, min_samples=1).fit(bboxes)

				label_set = set(clustering.labels_)
				for l in label_set:
					selected = list(np.where(clustering.labels_ == l)[0])
					selected_texts = [x for i, x in enumerate(texts) if i in selected]
					selected_boxes = [x for i, x in enumerate(bboxes) if i in selected]
					selected_confs = [x for i, x in enumerate(confs) if i in selected]
					text_boxes = [[x, y] for x, y in zip(selected_texts, selected_boxes)]
					text_boxes = sorted(text_boxes, key=cmp_to_key(contour_sort))
					text_result = ""

					for tb in text_boxes:
						if text_result == "":
							text_result += tb[0]
						else:
							text_result += " " + tb[0]
					# print(text_result)
					x1 = min([x[0] for x in selected_boxes])
					x2 = max([x[2] for x in selected_boxes])
					y1 = min([x[1] for x in selected_boxes])
					y2 = max([x[3] for x in selected_boxes])
					box_result = [x1, y1, x2, y2]
					conf_result = float(np.round(np.mean(selected_confs), 2))
					if k not in list(final_result_set.keys()):
						final_result_set[k] = []
					final_result_set[k].append([text_result, box_result, conf_result])
			else:
				if k not in list(final_result_set.keys()):
					final_result_set[k] = []
				final_result_set[k].append([result_set[k][0][0], result_set[k][0][1], result_set[k][0][2]])
			# draw.rectangle(result_set[k][0][1], outline=label2color[k], width=2)
			# draw.text((result_set[k][0][1][0] + 5, result_set[k][0][1][1] - 20),
			# text=k + " - " + str(result_set[k][0][2]), fill=label2color[k], font=font)
		merge_surrounding(final_result_set, model_output)
		for k in list(final_result_set.keys()):
			all_values = final_result_set[k]
			for value in all_values:
				draw.rectangle(value[1], outline=label2color[k], width=2)
				draw.text((value[1][0] + 5, value[1][1] - 20),
				          text=k + " - " + str(value[2]), fill=label2color[k], font=font)
		lookup_result = {}
		print("generated page: ", str(count))
		print('RAM memory % used:', psutil.virtual_memory()[2])
		t_page_end = datetime.now()
		print("Time taken for page" + str(count) + ":", end=" ")
		print(t_page_end - t_page_start)
		print()
		all_page_result["Page Number " + str(count)] = final_result_set
		# print(all_page_result)
		for k in list(final_result_set.keys()):
			if k in ["applicant_country", "beneficiary_country"]:
				for val in final_result_set[k]:
					result_country = lookup(val[0], 4, 90, "countries.txt", result_set, k)
					result_company = lookup(val[0], 4, 90, "organization.txt", result_set, k)
					# print(val)
					# print(result)
					for res in result_company:
						found = res['found_string']
						# print("lookup string", found)
						searched = res['searched_string']
						# print("original string", searched)
						new_val = val[0].replace(searched, found)
						val[0] = new_val
					for res in result_country:
						found = res['found_string']
						# print(found)
						searched = res['searched_string']
						# print(searched)
						new_val = val[0].replace(searched, found)
						val[0] = new_val
					for res in result_country:
						if (str(k) + "-country") not in lookup_result:
							lookup_result[('LUT_' + str(k) + "-country")] = []
						lookup_result[('LUT_' + str(k) + "-country")].append(
							(res['found_string'], res['string_match_value'], res["bbox"]))
					for res in result_company:
						if (str(k) + "-organization") not in lookup_result:
							lookup_result[('LUT_' + str(k) + "-organization")] = []
						lookup_result[('LUT_' + str(k) + "-organization")].append(
							(res['found_string'], res['string_match_value'], res["bbox"]))
		# print(lookup_result)
		with open(os.path.join(result_path, file[:-4] + str(count) + "_lookup.txt"), "w") as f:
			json.dump(lookup_result, f)
		with open(os.path.join(result_path, file[:-4] + str(count) + ".txt"), "w") as f:
			json.dump(final_result_set, f)
		new_img.save(os.path.join(result_path, file[:-4] + str(count) + ".png"))

	# break'''
	# print("running")
	with open(os.path.join(result_path, file[:-4] + "all_page_result.txt"), "w") as f:
		json.dump(all_page_result, f)
	# shutil.rmtree(mainpath)
	# shutil.rmtree(dir_path)
	# os.remove(os.path.join(mainpath, name))
	print()
	print("******Generated All Pages******")
	print('RAM memory % used:', psutil.virtual_memory()[2])
	t_total_end = datetime.now()
	print("Processing time taken for all pages" + str(count) + ":", end=" ")
	print(t_total_end - t_total_start)
	del image
	del new_img
	del encoded_inputs
	gc.collect()
	# print("pdf", all_page_result)
	return all_page_result


