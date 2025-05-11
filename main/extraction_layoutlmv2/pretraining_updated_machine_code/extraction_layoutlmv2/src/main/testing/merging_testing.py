from inference_utility_testing import *




def dbscan(result_set):
    final_result_set = {}
    for k in list(result_set.keys()):
        # if k == 'beneficiary_address':
        if k not in single_text_labels:
            try:
                alpha = float(configur[k]['ALPHA'])
            except:
                alpha = float(configur['Default']['ALPHA'])
            if len(result_set[k]) > 1:
                print("++++++++++++++entry in this block+++++++++++")
                texts = [x[0] for x in result_set[k]]
                bboxes = [x[1] for x in result_set[k]]
                confs = [x[2] for x in result_set[k]]
                avg_w = np.mean([abs(x[0] - x[2]) for x in bboxes])
                avg_h = np.mean([abs(x[1] - x[3]) for x in bboxes])
                eps = np.sqrt(avg_w ** 2 + avg_h ** 2) * alpha
                # if eps<=0.0:
                # 	eps=0.1
                clustering = DBSCAN(eps=eps, min_samples=1).fit(bboxes)
                label_set = set(clustering.labels_)
                for l in label_set:
                    selected = list(np.where(clustering.labels_ == l)[0])
                    selected_texts = [x for i, x in enumerate(texts) if i in selected]
                    selected_boxes = [x for i, x in enumerate(bboxes) if i in selected]
                    selected_confs = [x for i, x in enumerate(confs) if i in selected]
                    text_boxes = [[x, y] for x, y in zip(selected_texts, selected_boxes)]
                    print('text_boxes >>>>>>>>>>>>>')
                    print(text_boxes)
                    text_boxes = sorted(text_boxes, key=cmp_to_key(contour_sort))
                    print("after contour sort")
                    print(text_boxes)
                    text_boxes = validate_contour_sort(text_boxes)
                    print(text_boxes)
                    text_result = ""
                    print(k)
                    print(text_boxes)
                    for tb in text_boxes:
                        if text_result == "":
                            text_result += tb[0]
                        else:
                            text_result += " " + tb[0]
                    print(text_result)
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
        else:
            if len(result_set[k]) > 1:
                print("++++++++++++++entry in this block+++++++++++")
                texts = [x[0] for x in result_set[k]]
                bboxes = [x[1] for x in result_set[k]]
                confs = [x[2] for x in result_set[k]]
                for i, value in enumerate(zip(texts, bboxes, confs)):
                    print(list(value))
                    if k not in list(final_result_set.keys()):
                        final_result_set[k] = []
                    final_result_set[k].append(list(value))
            else:
                if k not in list(final_result_set.keys()):
                    final_result_set[k] = []
                final_result_set[k].append([result_set[k][0][0], result_set[k][0][1], result_set[k][0][2]])

    return final_result_set



def merge_surrounding_latest(data, model_output, w, h):
	new = data.copy()
	print('entered into merging_surroundings ++++++++++++++++++++++++++++++++++++++')
	for key in list(data.keys()):
		print(key)
		if key in vertical_merge_labels:
			all_values = data[key]
			# all_values = vertical_horizontal_values(new_all_values)
			# if key=='drawee_address':
			#     print(all_values)
			#     exit("PPPPPPPPPPPP")
			bboxes = [x[1] for x in data[key]]
			eps_horizontal = 100  # Threshold for horizontal merging
			eps_vertical = 100  # Threshold for vertical merging
			######################
			# if w>h:
			#     eps_horizontal = round(h*17/100)#100
			#     eps_vertical = round(w*12/100) #100
			# else:
			#     eps_horizontal = round(h*12/100)#100
			#     eps_vertical = round(w*17/100) #100
			############################
			all_values = data[key]
			print(all_values)
			# for all_values in new_all_values:
			length = len(all_values)
			if length > 1:
				i = 0
				# if (bb1_height > bb1_width and bb2_height > bb2_width) or (bb1_height < bb1_width and bb2_height < bb2_width):
				while i in range(length - 1):
					print(i)
					bb1 = all_values[i][1]
					bb2 = all_values[i + 1][1]
					bb1_token = all_values[i][0]
					bb2_token = all_values[i + 1][0]
					confs = [all_values[i][2], all_values[i + 1][2]]
					min_dist_horizontal = minimum_distance(bb1, bb2)
					min_dist_vertical = minimum_distance_vertical(bb1, bb2)

					try:
						IOU_horizontal = get_iou_horizontal(bb1, bb2)
						IOU_vertical = get_iou_vertical(bb1, bb2)
						inter_percentage = get_intersection_percentage(bb1, bb2)
					except:
						i = i + 1
						continue
					bb1_x1, bb1_y1, bb1_x2, bb1_y2 = bb1
					bb2_x1, bb2_y1, bb2_x2, bb2_y2 = bb2
					bb1_width = bb1_x2 - bb1_x1
					bb1_height = bb1_y2 - bb1_y1
					bb2_width = bb2_x2 - bb2_x1
					bb2_height = bb2_y2 - bb2_y1
					# if len(bb1_token)<3:
					print('beore', bb2_width, bb2_height)
					print('bb2_token length', len(bb2_token))
					flag1 = True
					flag2 = True
					flag1 = special_chr_check(bb1_token, flag1)
					flag2 = special_chr_check(bb2_token, flag2)

					if len(bb1_token) == 1 or (len(bb1_token) < 3 and flag1 == False):
						temp = bb1_width
						bb1_width = bb1_height
						bb1_height = temp
					if len(bb2_token) == 1 or (len(bb2_token) < 3 and flag2 == False):
						temp = bb2_width
						bb2_width = bb2_height
						bb2_height = temp
					print('flag2', flag2)
					print('bb2_token', bb2_token)
					print('bb1_width', bb1_width, 'bb1_height', bb1_height)
					print('bb2_width', bb2_width, 'bb2_height', bb2_height)
					if (bb1_height >= bb1_width and bb2_height >= bb2_width) or (
							bb1_height <= bb1_width and bb2_height <= bb2_width):
						print('entered into first if')
						if (min_dist_horizontal <= eps_horizontal or IOU_horizontal > 0 or inter_percentage > 0) or (
								min_dist_vertical <= eps_vertical or IOU_vertical > 0 or inter_percentage):
							print('entered into second if')
							print("merging: " + all_values[i][0] + " and " + all_values[i + 1][0])
							x_left = min(bb1[0], bb2[0])
							y_top = min(bb1[1], bb2[1])
							x_right = max(bb1[2], bb2[2])
							y_bottom = max(bb1[3], bb2[3])
							box = [x_left, y_top, x_right, y_bottom]
							text = model_output_sum(key, box, model_output)
							print("merged text is ", text)
							avg_confs = (confs[0] * area(bb1) + confs[1] * area(bb2)) / (area(bb1) + area(bb2))
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
						i = i + 1
			else:
				print("will continue")
				continue

		else:
			print('Vertical merging not happening ++++++++++++++++++++++++++++++++++++')
			print(key)
			bboxes = [x[1] for x in data[key]]
			if w > h:
				v_eps = round(h * 1.5 / 100)  # 10round(number)
				h_eps = round(w * 5 / 100)  # 36
			else:
				v_eps = round(h * 1.1 / 100)  # 10round(number)
				h_eps = round(w * 5.8 / 100)  # 36
			all_values = data[key]
			print('all_values >>>>>>>>>>>', all_values)
			length = len(all_values)
			if length > 1:
				i = 0
				while i in range(length - 1):
					print(i)
					bb1 = all_values[i][1]
					bb2 = all_values[i + 1][1]
					confs = [all_values[i][2], all_values[i + 1][2]]
					# ocr_confs = [all_values[i][3],all_values[i+1][3]]
					# min_dist = minimum_distance(bb1, bb2)
					vertical_distance = check_vertical_distribution(bb1, bb2)
					# dist_bwt_words = (abs(bb1[2]-bb2[0])/w)*100
					hori_distance = abs(bb1[2] - bb2[0])
					print('min_dist========>', vertical_distance, 'bb1', bb1, 'bb2', bb2)
					print('hori_distance ==========>', hori_distance)
					try:
						IOU = get_iou_new(bb1, bb2)
						print('IOU>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>', IOU)
					except:
						i = i + 1
						continue
					if (vertical_distance <= v_eps and hori_distance < h_eps) or IOU > 0.1:
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
		# if key=="shipper_address":
		#     print('True')
		#     # print(all_values)
		#     exit('+++++++++++++++++=')
from PIL import Image
import os
img_path = '/home/ntlpt19/Downloads/Evaluation_Data/issues/sorting_issue/Packing_List(2012_08_25_13_28_24_0857)_789_page_11.png'
with Image.open(img_path) as img:
    w,h = img.size
    
with open('/home/ntlpt19/Downloads/Evaluation_Data/issues/sorting_issue/Packing_List(2012_08_25_13_28_24_0857)_789_page_11.txt', 'r') as exp:
    final_result_set = json.load(exp)
with open('/home/ntlpt19/Downloads/Evaluation_Data/issues/sorting_issue/Packing_List(2012_08_25_13_28_24_0857)_789_page_11model_output.txt', 'r') as exp:
    model_output = json.load(exp)

final_result_set = dbscan(model_output)

merge_surrounding_latest(final_result_set, model_output, w, h)
print('>>>>>>>>>>>>>>>>>>>..')
print('>>>>>>>>>>>>>>>>>>>..')
print('>>>>>>>>>>>>>>>>>>>..')
print('>>>>>>>>>>>>>>>>>>>..')
print(final_result_set)
import cv2
import numpy as np

def draw_coordinates_and_values(image_path, data_dict):
    # Read the image
    img = cv2.imread(image_path)

    # Loop through the dictionary
    for key, value_dict_list in data_dict.items():
        for value_dict in value_dict_list:
            value, coordinates, _ = value_dict

            # Convert coordinates to integers
            x_min, y_min, x_max, y_max = map(int, coordinates)

            # Draw a rectangle
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Write the value on top of the rectangle
            cv2.putText(img, key, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the image with drawn coordinates and values
    cv2.imwrite('output_img.png', img)
    
# draw_coordinates_and_values(img_path, final_result_set)
