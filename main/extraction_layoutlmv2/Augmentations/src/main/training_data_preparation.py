from google.cloud import vision
from base64 import b64encode
import pytesseract
from configparser import ConfigParser
import os
from typing import List
import cv2
import json
import shutil
import random
import pickle
import psutil
from datetime import datetime

from training.lmv2code.src.main.Augmentations.src.main.training_utility import get_logger_object_and_setting_the_loglevel, set_basic_config_for_logging
from training.lmv2code.src.main.Augmentations.src.main import training_utility as tu


zoom = 300 / 72


def contour_sort(a, b):
    if abs(a['y1'] - b['y1']) <= 15:
        return a['x1'] - b['x1']

    return a['y1'] - b['y1']


def remove_garbage(dataset):
    to_remove = ["\u00da", "\u00c6", "\u00c4", "\u00b4", "\u00c5", "Á", "\n", "|"]
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
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "spheric-time-383904-f1b421d86eef.json"
    with open(image_path, 'rb') as f:
        ctxt = b64encode(f.read()).decode()
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=ctxt)

    response = client.text_detection(image=image)

    word_coordinates = []
    all_text = ""

    for i, text in enumerate(response.text_annotations):
        if i != 0:
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


def get_ocr_tesseract(img_path, labels_list):
    image = cv2.imread(img_path, 0)
    # thresh = 255 - cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    dataset = {}
    for item in labels_list:
        label = item['label']
        x1 = item['x1']
        y1 = item['y1']
        x2 = item['x2']
        y2 = item['y2']
        ROI = image[y1:y2, x1:x2]
        labelled_text = pytesseract.image_to_string(ROI, lang='eng', config='--psm 6')
        # dataset[label] = []
        # dataset[label].append([labelled_text, [x1,y1, x2, y2]])
        if label in list(dataset.keys()):
            dataset[label].append([labelled_text, [x1, y1, x2, y2]])
        else:
            dataset[label] = []
            dataset[label].append([labelled_text, [x1, y1, x2, y2]])
    return dataset


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
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
    # min_area = min(bb1_area,bb2_area)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    if bb1_area > bb2_area:
        intersection_percent = intersection_area / bb2_area
    else:
        intersection_percent = intersection_area / bb1_area
        if intersection_percent < 0.5:
            intersection_percent = 1  # if ocr bounding box is big then we need to consider the entire token if the intersection is less than o.5 also

    # print("The intersection percentage  of {text}, and {label},= {inter}".format(text = bb1['label'], label = bb2['word'], inter= intersection_percent))
    # exit('++++++++++++++==')
    assert intersection_percent >= 0.0
    assert intersection_percent <= 1.0
    return intersection_percent


def train_test_split(split_file, flag):
    annotation_data = []
    logger.info("is annotation_data is instance of list? %s", isinstance(annotation_data, list))
    for file in split_file:
        if file in (img_files and labelled_files):
            print(file)
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
                    if (len(l) > int(configur['PARAMS'][
                                         'nill_val'])):
                        l_class = dict_mapping[int(l[0])]
                        labelled_data.append({
                            "label": l_class,
                            "x1": int(l[1]),
                            "y1": int(l[2]),
                            "x2": int(l[3]),
                            "y2": int(l[4])
                        })

                if len(labelled_data) > int(configur['PARAMS']['nill_val']):
                    for data in labelled_data:
                        print(data)
                        for t in pdf_text:
                            intersection_2 = get_intersection_percentage(data, t)
                            if intersection_2 >= float(configur['PARAMS']['percent_val']):
                                t['label'] = data['label']
                                # print(data['label'])
                                # if data['label']=='nostro_bank_name':
                                #     exit('*****************exited')
                    for t in pdf_text:
                        if t['label'] == "O":
                            blue_col = int(configur['PARAMS']['color_val'])
                            thick_val = int(configur['PARAMS']['thick_val2'])
                            nul_val = int(configur['PARAMS']['nill_val'])
                            cv2.rectangle(image, (int(t['x1']), int(t['y1'])), (int(t['x2']), int(t['y2'])),
                                          (blue_col, nul_val, nul_val), thick_val)
                        else:
                            t["label"] = "S-" + t["label"]

                    if len(pdf_text) <= thresh:
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
                        for l_d in pdf_text:
                            final_labelled_data["words"].append(l_d['word'])
                            final_labelled_data["bbox"].append(
                                tu.normalize([l_d['x1'], l_d['y1'], l_d['x2'], l_d['y2']], w, h))
                            final_labelled_data["labels"].append(l_d['label'])
                        annotation_data.append(final_labelled_data)
                        import pandas as pd
                        df = pd.DataFrame(final_labelled_data)
                        df.to_csv(f'{SEGREGATION_path}/{file}.csv')
                    else:
                        import pandas as pd
                        concat_final_label = []
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
                                                                                                   file + "_s_" + str(
                                                                                                       int((
                                                                                                                   i + 1) / thresh)) + ".png"))
                                annotation_data.append(final_labelled_data)
                                concat_final_label.append(pd.DataFrame(final_labelled_data))
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
                            # print(final_labelled_data["labels"])
                        if len(final_labelled_data['words']) > int(configur['PARAMS']['nill_val']):
                            shutil.copy(os.path.join(images_path, file + ".png"), os.path.join(image_data_path,
                                                                                               file + "_s_" + str(
                                                                                                   int(len(
                                                                                                       pdf_text) / thresh) + 1) + ".png"))
                            annotation_data.append(final_labelled_data)
                            concat_final_label.append(pd.DataFrame(final_labelled_data))
                        result = pd.concat(concat_final_label, axis=1)
                        result.to_csv(f'{SEGREGATION_path}/{file}.csv')
                        # print(final_labelled_data["labels"])
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

    # exit()
    if flag == 'train':
        random.shuffle(annotation_data)
        train_samples = annotation_data  # [:-int(ratio_val * len(annotation_data))]
        len(train_samples)
        train_samples = sorted(train_samples, key=lambda x: x['filename'])
        words_train, boxes_train, labels_train = tu.data_convert(train_samples)
        if not os.path.exists(os.path.join(folder_path, 'train')):
            os.mkdir(os.path.join(folder_path, 'train'))
        with open(os.path.join(folder_path, 'train.pkl'), 'wb') as t:
            pickle.dump([words_train, labels_train, boxes_train], t)
        t.close()
        for t in train_samples:
            shutil.copy(os.path.join(image_data_path, t['filename']), os.path.join(folder_path, "train", t['filename']))

    if flag == 'test':
        random.shuffle(annotation_data)
        test_samples = annotation_data  # [-int(ratio_val * len(annotation_data)):]
        len(test_samples)
        test_samples = sorted(test_samples, key=lambda x: x['filename'])
        words_test, boxes_test, labels_test = tu.data_convert(test_samples)
        if not os.path.exists(os.path.join(folder_path, 'test')):
            os.mkdir(os.path.join(folder_path, 'test'))
        with open(os.path.join(folder_path, 'test.pkl'), 'wb') as t:
            pickle.dump([words_test, labels_test, boxes_test], t)
        t.close()
        for t in test_samples:
            shutil.copy(os.path.join(image_data_path, t['filename']), os.path.join(folder_path, "test", t['filename']))



    for file in img_files:
        annotate_data = []
        word_cordinate_data = []
        IOU_value = []
        if os.path.exists(os.path.join(labels_path, file + ".txt")):
            print('yes')
            print(f'file name: {file}')
            image = cv2.imread(os.path.join(images_path, file + ".png"))
            word_coordinates = []
            word_coordinates, all_text = get_ocr_vision_api(os.path.join(images_path, file + ".png"))

            if len(word_coordinates) == 0:
                print(file)
                print("Not enough text")
            else:
                shutil.copy(os.path.join(images_path, file + ".png"), os.path.join(master_path, file + ".png"))
                shutil.copy(os.path.join(labels_path, file + ".txt"), os.path.join(master_path, file + "_LabelImg.txt"))

                with open(os.path.join(master_path, file + "_text.txt"), "w") as f:
                    json.dump({"word_coordinates": word_coordinates}, f)
                with open(os.path.join(ocr_path, file + "_text.txt"), "w") as f:
                    json.dump({"word_coordinates": word_coordinates}, f)
                try:
                    with open(os.path.join(master_path, file + "_all_text.txt"), "w") as f:
                        json.dump({"all_text": all_text}, f)
                except:
                    pass

                h, w, _ = image.shape
                with open(os.path.join(labels_path, file + ".txt"), "r") as f:
                    label = (f.read())
                label = label.split("\n")
                labelled_data = []
                for l in label:
                    l = l.split()
                    if len(l) > 0:
                        l_class = classes[int(l[0])]
                        labelled_data.append({
                            "label": l_class,
                            "x1": int(l[1]),
                            "y1": int(l[2]),
                            "x2": int(l[3]),
                            "y2": int(l[4])
                        })
                if len(labelled_data) > 0:
                    dataset = {}
                    for data in labelled_data:
                        overlapping_boxes = []
                        labelled_text = ""
                        for t in word_coordinates:
                            try:
                                annotate_data.append(data)
                                word_cordinate_data.append(t)
                                IOU_value.append(get_intersection_percentage(data, t))
                                if get_intersection_percentage(data, t) >= 0.40:
                                    t['label'] = data['label']
                                    overlapping_boxes.append(t)
                            except Exception as e:
                                print(t)
                                print(e)
                        for t in overlapping_boxes:
                            if len(labelled_text) == 0:
                                labelled_text = t['word']
                            else:
                                labelled_text += " " + t['word']
                        if len(labelled_text.strip()) == 0:
                            print(file + " - " + str(data) + " - " + str(len(overlapping_boxes)))
                        else:
                            if data['label'] in list(dataset.keys()):
                                dataset[data['label']].append(
                                    [labelled_text, [data['x1'], data['y1'], data['x2'], data['y2']]])
                            else:
                                dataset[data['label']] = []
                                dataset[data['label']].append(
                                    [labelled_text, [data['x1'], data['y1'], data['x2'], data['y2']]])
                    if dataset == {}:
                        print(file + " - blank")
                    remove_garbage(dataset)
                    with open(os.path.join(master_path, file + "_labels.txt"), "w") as f:
                        json.dump(dataset, f)
                    with open(os.path.join(master_labels_path, file + "_labels.txt"), "w") as f:
                        json.dump(dataset, f)
                    print(dataset)
                    csv_data = {'annotate_data': annotate_data, 'word_cordinate_data': word_cordinate_data,
                                'IOU_value': IOU_value}
                    import pandas as pd

                    df = pd.DataFrame(csv_data)
                    df.to_csv(f'{iou_path}/{file}.csv')
        else:
            print("No")

set_basic_config_for_logging(filename="data_preparation")
logger = get_logger_object_and_setting_the_loglevel()
process_memory = psutil.Process()
start_time = datetime.now()
cpu_utilization_start = psutil.cpu_percent()
before_memory = process_memory.memory_info().rss

configur = ConfigParser()
folder_path = "/home/tarun/Downloads/BG_LC_Cancellation"
images_path = os.path.join(folder_path, "Images")
labels_path = os.path.join(folder_path, "Labels")
master_path = os.path.join(folder_path, "Master_Data")
master_labels_path = os.path.join(folder_path, 'Master_Labels')
ocr_path = os.path.join(folder_path, "OCR")
iou_path = os.path.join(folder_path, 'Iou_check')
SEGREGATION_path = os.path.join(folder_path, 'SEGREGATION_Check')
if not os.path.exists(master_path):
    os.mkdir(master_path)
if not os.path.exists(ocr_path):
    os.mkdir(ocr_path)
if not os.path.exists(master_labels_path):
    os.mkdir(master_labels_path)
if not os.path.exists(iou_path):
    os.mkdir(iou_path)
if not os.path.exists(SEGREGATION_path):
    os.mkdir(SEGREGATION_path)

with open(os.path.join(folder_path, "label.txt"), "r") as f:
    classes = (f.read())
    classes = classes.split("\n")
labelled_files = os.listdir(labels_path)
labelled_files = [x.split(".txt")[0] for x in labelled_files]
img_files = os.listdir(images_path)
img_files = [x.split(".png")[0] for x in img_files]
annotation_data = []
thresh = 300  # 300  #need to change to 250
count = 1

# folder creation
if not os.path.exists(master_path):
    os.mkdir(master_path)
if not os.path.exists(ocr_path):
    os.mkdir(ocr_path)
with open(os.path.join(folder_path, "label.txt"), "r") as file:
    class_names: List = file.readlines()
    class_names = list(map(lambda x: x.strip(), class_names))
    logger.info("is class_names is a instance of list? %s", isinstance(class_names, list))
    dict_mapping = dict(enumerate(class_names))
    logger.info("is dict_mapping is a instance of dict? %s", isinstance(dict_mapping, dict))
file.close()

print(dict_mapping)

labelled_files = os.listdir(labels_path)
labelled_files = [x.split(".txt")[0] for x in labelled_files]
img_files = os.listdir(images_path)
img_files = [x.split(".png")[0] for x in img_files]

logger.info("is labelled_files is a instance of list? %s", isinstance(labelled_files, list))
# using intersection percentage
annotation_data = []
logger.info("is annotation_data is instance of list? %s", isinstance(annotation_data, list))
thresh = int(configur['PARAMS']['thresh_value'])  # 300
logger.info("is threshold value is instance of int? %s", isinstance(thresh, int))
text_file_nme = "_text.txt"

zoom = int(configur['PARAMS']['thresh_value']) / int(configur['PARAMS']['zoom_val'])  # 300 / 72

image_data_path = os.path.join(folder_path, "Images_Data")

if not os.path.exists(ocr_path):
    os.mkdir(ocr_path)
if not os.path.exists(image_data_path):
    os.mkdir(image_data_path)

thresh = 300  # int(configur['PARAMS']['thresh_value'])  # 300
seed_val = int(configur['PARAMS']['random_seed_val'])
ratio_val = float(configur['PARAMS']['train_div_ratio'])
random.seed(seed_val)
random.shuffle(labelled_files)
train_samples_imgs = labelled_files[:-int(ratio_val * len(labelled_files))]
print(len(train_samples_imgs))
test_samples_imgs = labelled_files[-int(ratio_val * len(labelled_files)):]
print(len(test_samples_imgs))

os.path.join(folder_path, "label.txt")

train_test_split(train_samples_imgs, 'train')
train_test_split(test_samples_imgs, 'test')

end_time = datetime.now()
cpu_utilization_end = psutil.cpu_percent()
diff = end_time - start_time
after_memory = process_memory.memory_info().rss
cpu_utt = cpu_utilization_end - cpu_utilization_start
memory_consumption = after_memory - before_memory
logger.info("total time taken for data preparation:" + str(diff))
logger.info("cpu_utilization %:" + str(cpu_utt))
logger.info("memory_consumption in bytes:" + str(memory_consumption))
logger.info(('RAM memory % used:', psutil.virtual_memory()[2]))
