from configparser import ConfigParser
import training_utility as tu
import os
from typing import List
import cv2
import json
import shutil
from functools import cmp_to_key
import random
import pickle
import logging
import psutil
from datetime import datetime
from training_utility import get_logger_object_and_setting_the_loglevel, set_basic_config_for_logging
configur = ConfigParser()
configur.read('traini_valid_utility.ini')

folder_path = str(configur['PATHS']['folder_path'])

print(f'folder path: {folder_path}')
# exit('+++====')


images_path = os.path.join(folder_path, "Images")
labels_path = os.path.join(folder_path, "Labels")
master_path = os.path.join(folder_path, "Master_Data")
ocr_path = os.path.join(folder_path, "OCR")

set_basic_config_for_logging(filename="data_preparation")
logger = get_logger_object_and_setting_the_loglevel()
process_memory = psutil.Process()
start_time = datetime.now()
cpu_utilization_start = psutil.cpu_percent()
before_memory = process_memory.memory_info().rss



# folder creation
if not os.path.exists(master_path):
    os.mkdir(master_path)
if not os.path.exists(ocr_path):
    os.mkdir(ocr_path)
with open(os.path.join(folder_path, "label.txt"), "r") as file:
    class_names: List = file.readlines()
    class_names = list(map(lambda x: x.strip(), class_names))
    logger.info("is class_names is a instance of list? %s",isinstance(class_names, list))
    dict_mapping = dict(enumerate(class_names))
    logger.info("is dict_mapping is a instance of dict? %s",isinstance(dict_mapping, dict))
file.close()

print(dict_mapping)

labelled_files = os.listdir(labels_path)
labelled_files = [x.split(".txt")[0] for x in labelled_files]
logger.info("is labelled_files is a instance of list? %s",isinstance(labelled_files, list))
# using intersection percentage
annotation_data = []
logger.info("is annotation_data is instance of list? %s", isinstance(annotation_data,list))
thresh = int(configur['PARAMS']['thresh_value'])#300
logger.info("is threshold value is instance of int? %s", isinstance(thresh, int))
text_file_nme = "_text.txt"
for file in labelled_files:
    print(file)
    if os.path.exists(os.path.join(images_path, f"{file}.png")):
        print('yes')
        image = cv2.imread(os.path.join(images_path, f"{file}.png"))
        word_coordinates, all_text = tu.get_ocr_vision_api(os.path.join(images_path, file + ".png"))
        with open(os.path.join(ocr_path, file + text_file_nme), "w") as f:
            json.dump({"word_coordinates": word_coordinates}, f)
        f.close()    
        if len(word_coordinates) == 0:
            print(file)
            print("Not enough text")
        else:
            shutil.copy(
                os.path.join(images_path, f"{file}.png"),
                os.path.join(master_path, f"{file}.png"),
            )
            shutil.copy(
                os.path.join(labels_path, f"{file}.txt"),
                os.path.join(master_path, f"{file}_LabelImg.txt"),
            )
            with open(os.path.join(master_path, f"{file}_text.txt"), "w") as f:
                json.dump({"word_coordinates": word_coordinates}, f)
            f.close()    
            try:
                with open(os.path.join(master_path, f"{file}_all_text.txt"), "w") as f:
                    json.dump({"all_text": all_text}, f)
                f.close()    
            except Exception:
                pass

            h, w, _ = image.shape
            with open(os.path.join(labels_path, f"{file}.txt"), "r") as f:
                label = (f.read())
            label = label.split("\n")
            labelled_data = []
            logger.info("is labelled_data is instance of list? %s", isinstance(labelled_data, list))
            f.close()
            for l in label:
                l = l.split()
                if len(l) > 0 and int(l[0]) != int(configur['PARAMS']['label_length']): #30
                    l_class = dict_mapping[int(l[0])]
                    d_value = int(configur['PARAMS']['div_value'])
                    x_center = float(l[1]) * w
                    y_center = float(l[2]) * h
                    width = float(l[3]) * w
                    height = int(float(l[4]) * h)
                    x0 = int(x_center - (width / d_value))
                    x1 = int(x_center + (width / d_value))
                    y0 = int(y_center - (height / d_value))
                    y1 = int(y_center + (height / d_value))
                    color_value = int(configur['PARAMS']['color_val'])
                    thickness = int(configur['PARAMS']['thick_val'])
                    nul_val = int(configur['PARAMS']['color_val'])
                    cv2.rectangle(image, (x0, y0), (x1, y1), (nul_val,color_value,nul_val), thickness)
                    labelled_data.append({
                        "label": l_class,
                        "x1": x0,
                        "y1": y0,
                        "x2": x1,
                        "y2": y1
                    })
            if labelled_data:
                dataset = {}
                logger.info("is dataset is instance of dict? %s", isinstance(dataset, dict))
                for data in labelled_data:
                    overlapping_boxes = []
                    logger.info("is overlapping_boxes is instance of list? %s", isinstance(overlapping_boxes, list))
                    labelled_text = ""
                    logger.info("is labelled_text is instance of str? %s", isinstance(labelled_text, str))
                    for t in word_coordinates:
                        try:
                            if tu.get_intersection_percentage(data, t) >= float(configur['PARAMS']['percent_val']): #0.40
                                t['label'] = data['label']
                                overlapping_boxes.append(t)
                        except Exception as e:
                            print(t)
                            print(e)
                    overlapping_boxes = sorted(overlapping_boxes, key=cmp_to_key(tu.contour_sort))
                    nil_value = int(configur['PARAMS']['nill_val'])
                    for t in overlapping_boxes:
                        if len(labelled_text) == nil_value:
                            labelled_text = t['word']
                        else:
                            labelled_text += " " + t['word']
                    if len(labelled_text.strip()) == nil_value:
                        print(f"{file} - {str(data)} - {len(overlapping_boxes)}")
                    else:
                        if data['label'] not in list(dataset.keys()):
                            dataset[data['label']] = []
                        dataset[data['label']].append(
                            [labelled_text, [data['x1'], data['y1'], data['x2'], data['y2']]])
                if not dataset:
                    print(f"{file} - blank")
                tu.remove_garbage(dataset)
                with open(os.path.join(master_path, f"{file}_labels.txt"), "w") as f:
                    json.dump(dataset, f)
                f.close()    

    else:
        print("No")









zoom = int(configur['PARAMS']['thresh_value'])/int(configur['PARAMS']['zoom_val'])#300 / 72


image_data_path = os.path.join(folder_path, "Images_Data")

if not os.path.exists(ocr_path):
    os.mkdir(ocr_path)
if not os.path.exists(image_data_path):
    os.mkdir(image_data_path)

annotation_data = []
logger.info("is annotation_data is instance of list? %s", isinstance(annotation_data, list))
thresh = int(configur['PARAMS']['thresh_value'])#300
for file in labelled_files:
    if os.path.exists(os.path.join(ocr_path, file + text_file_nme)):
        with open(os.path.join(ocr_path, file + text_file_nme), "r") as f:
            pdf_text = json.load(f)['word_coordinates']
        f.close()    
        for t in pdf_text:
            t["label"] = "O"
    if os.path.exists(os.path.join(images_path, file + ".png")):
        image = cv2.imread(os.path.join(images_path, file + ".png"))
        h, w, _ = image.shape
        with open(os.path.join(labels_path, file + ".txt"), "r") as f:
            label = (f.read())
        label = label.split("\n")
        labelled_data = []
        logger.info("is labelled_data is instance of list? %s", isinstance(labelled_data, list))
        f.close()
        for l in label:
            l = l.split()
            if (len(l) > int(configur['PARAMS']['nill_val'])) and (int(l[0]) != int(configur['PARAMS']['label_length'])):
                #if int(l[0]) != int(configur['PARAMS']['label_length']):
                l_class = dict_mapping[int(l[0])]
                x_center = float(l[1]) * w
                y_center = float(l[2]) * h
                width = float(l[3]) * w
                height = int(float(l[4]) * h)
                x0 = int(x_center - (width / d_value))
                x1 = int(x_center + (width / d_value))
                y0 = int(y_center - (height / d_value))
                y1 = int(y_center + (height / d_value))
                # cv2.rectangle(image, (x0, y0), (x1, y1), (0,255,0), 4)
                labelled_data.append({
                    "label": l_class,
                    "x1": x0,
                    "y1": y0,
                    "x2": x1,
                    "y2": y1
                })

        if len(labelled_data) > int(configur['PARAMS']['nill_val']):
            for data in labelled_data:
                for t in pdf_text:
                    intersection_2 = tu.get_intersection_percentage(data, t)
                    if intersection_2 >= float(configur['PARAMS']['percent_val']):
                        t['label'] = data['label']
            for t in pdf_text:
                if t['label'] == "O":
                    blue_col = int(configur['PARAMS']['color_val'])
                    thick_val = int(configur['PARAMS']['thick_val2'])
                    nul_val = int(configur['PARAMS']['nill_val'])
                    cv2.rectangle(image, (int(t['x1']), int(t['y1'])), (int(t['x2']), int(t['y2'])), (blue_col,nul_val,nul_val), thick_val)
                else:
                    t["label"] = "S-" + t["label"]

            if len(pdf_text) <= thresh:
                shutil.copy(os.path.join(images_path, file + ".png"), os.path.join(image_data_path, file + ".png"))
                final_labelled_data = {
                    "filename": file + ".png",
                    "words": [],
                    "bbox": [],
                    "labels": []
                }
                logger.info("is final_labelled_data is instance of dict? %s", isinstance(final_labelled_data, dict))
                for l_d in pdf_text:
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
                for i, l_d in enumerate(pdf_text):
                    if (i + 1) % thresh == int(configur['PARAMS']['nill_val']):
                        final_labelled_data["filename"] = file + "_s_" + str(int((i + 1) / thresh)) + ".png"
                        shutil.copy(os.path.join(images_path, file + ".png"), os.path.join(image_data_path,
                                                                                            file + "_s_" + str(int((
                                                                                                                            i + 1) / thresh)) + ".png"))
                        annotation_data.append(final_labelled_data)
                        final_labelled_data = {
                            "filename": file + "_s_" + str(int(len(pdf_text) / thresh) + 1) + ".png",
                            "words": [],
                            "bbox": [],
                            "labels": []
                        }
                    final_labelled_data["words"].append(l_d['word'])
                    final_labelled_data["bbox"].append(
                        tu.normalize([l_d['x1'], l_d['y1'], l_d['x2'], l_d['y2']], w, h))
                    final_labelled_data["labels"].append(l_d['label'])
                if len(final_labelled_data['words']) > int(configur['PARAMS']['nill_val']):
                    shutil.copy(os.path.join(images_path, file + ".png"), os.path.join(image_data_path,
                                                                                        file + "_s_" + str(int(len(
                                                                                            pdf_text) / thresh) + 1) + ".png"))
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

seed_val = int(configur['PARAMS']['random_seed_val'])
ratio_val = float(configur['PARAMS']['train_div_ratio'])
random.seed(seed_val)

random.shuffle(annotation_data)

train_samples = annotation_data[:-int(ratio_val * len(annotation_data))]
len(train_samples)

train_samples = sorted(train_samples, key=lambda x: x['filename'])

test_samples = annotation_data[-int(ratio_val * len(annotation_data)):]
len(test_samples)

test_samples = sorted(test_samples, key=lambda x: x['filename'])

words_train, boxes_train, labels_train = tu.data_convert(train_samples)
words_test, boxes_test, labels_test = tu.data_convert(test_samples)

if not os.path.exists(os.path.join(folder_path, 'train')):
    os.mkdir(os.path.join(folder_path, 'train'))

if not os.path.exists(os.path.join(folder_path, 'test')):
    os.mkdir(os.path.join(folder_path, 'test'))

with open(os.path.join(folder_path, 'train.pkl'), 'wb') as t:
    pickle.dump([words_train, labels_train, boxes_train], t)
t.close()    
with open(os.path.join(folder_path, 'test.pkl'), 'wb') as t:
    pickle.dump([words_test, labels_test, boxes_test], t)
t.close()    

for t in train_samples:
    shutil.copy(os.path.join(image_data_path, t['filename']), os.path.join(folder_path, "train", t['filename']))
for t in test_samples:
    shutil.copy(os.path.join(image_data_path, t['filename']), os.path.join(folder_path, "test", t['filename']))

end_time = datetime.now()
cpu_utilization_end = psutil.cpu_percent()
diff = end_time - start_time 
after_memory = process_memory.memory_info().rss
cpu_utt = cpu_utilization_end - cpu_utilization_start
memory_consumption = after_memory - before_memory
logger.info("total time taken for data preparation:" + str(diff))
logger.info("cpu_utilization %:"+str(cpu_utt))
logger.info("memory_consumption in bytes:"+str(memory_consumption))
logger.info(('RAM memory % used:', psutil.virtual_memory()[2]))







