import json
import requests
import base64
import io
import pandas as pd
from PIL import Image
from tqdm import tqdm
import time
import os

LEFT = "left"
TOP = "top"
WIDTH = "width"
HEIGHT = "height"
X1 = "x1";
Y1 = "y1"
X2 = "x2";
Y2 = "y2"
WORD = "word"
LAYOUT = "layout"
PAGES = "pages"
TEXTS = "texts"
SMALL_TEXT = "text"
LINES = "lines"
WORDS = "words"
CONFIDENCE = "confidence"
POSITION = "position"

recognition_mode = "Thorough".capitalize()
language = "english".capitalize()
# https://abbyyocr.centralindia.cloudapp.azure.com/FineReaderServer14

ABBY_OCR_ENGINE_DETAILS = {
    "ABBYY_CONFIG_PATH": "/datadrive2/IDP/abbyy_config.json",
    "ABBY_API": "http://74.225.164.73:8080/FineReaderServer14/api",
    "ABBY_PROCESSING_ENDPOINT": "workflows/Default%20Workflow/input/ticket",
    "ABBYY_STATUS_ENDPOINT": "jobs/%7B{}%7D",
    "ABBY_TEXT_RESULT_ENDPOINT": "result/outputDocuments/1/files/0",
    "ABBY_JSON_RESULT_ENDPOINT": "result/outputDocuments/2/files/0",
    "ABBY_HEADER": {
        "Content-Type": "application/json"
    },
    "SSL_VERIFICATION": "FALSE"
}


def get_config(image, file_name):
    image_buffer = io.BytesIO()
    image.save(image_buffer, format="PNG")
    image_buffer.seek(0)
    base64_image = base64.b64encode(image_buffer.getvalue()).decode('utf-8')
    try:
        with open(ABBY_OCR_ENGINE_DETAILS["ABBYY_CONFIG_PATH"], "r") as config:
            config_file = json.load(config)
    except Exception:
        with open("./abby_config.json", "r") as config:
            config_file = json.load(config)

    config_file["InputFiles"][0]["FileData"]["FileContents"] = base64_image
    config_file["RecognitionParams"]["Languages"] = [language]
    config_file["InputFiles"][0]["FileData"]["FileName"] = file_name.split("/")[-1]
    config_file["RecognitionParams"]["recognitionQuality"] = "{}{}".format("RQS_", recognition_mode)
    return config_file


def get_word_cordinates(json_result):
    word_coordinates = []
    for page in json_result[LAYOUT][PAGES]:
        for text_block in page[TEXTS]:
            for line in text_block[LINES]:
                for word_data in line[WORDS]:
                    kl = list(word_data.keys())
                    if SMALL_TEXT not in kl or CONFIDENCE not in kl or POSITION not in kl:
                        print("Word Skipped :", word_data)
                        continue
                    word = word_data[SMALL_TEXT]
                    confidence = word_data[CONFIDENCE]
                    position = word_data[POSITION]

                    width = position['r'] - position['l']
                    height = position['b'] - position['t']

                    word_info = {
                        WORD: word,
                        LEFT: position['l'],
                        TOP: position['t'],
                        WIDTH: width,
                        HEIGHT: height,
                        X1: position['l'],
                        Y1: position['t'],
                        X2: position['r'],
                        Y2: position['b'],
                        "confidence": confidence
                    }

                    word_coordinates.append(word_info)

    return word_coordinates


def new_get_word_cordinates(json_result):
    word_coordinates = []
    for page in json_result[LAYOUT][PAGES]:
        for text_block in page[TEXTS]:
            for line in text_block[LINES]:
                for word_data in line[WORDS]:
                    kl = list(word_data.keys())
                    if SMALL_TEXT not in kl or CONFIDENCE not in kl or POSITION not in kl:
                        continue
                    word = word_data[SMALL_TEXT]
                    confidence = word_data[CONFIDENCE]
                    position = word_data[POSITION]

                    width = position['r'] - position['l']
                    height = position['b'] - position['t']
                    x1 = position['l'];
                    y1 = position['t']
                    x2 = position['r'];
                    y2 = position['b']
                    vertices = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

                    word_info = {
                        WORD: word,
                        "text": word,
                        "confidence": confidence,
                        "vertices": vertices,
                        LEFT: position['l'],
                        TOP: position['t'],
                        WIDTH: width,
                        HEIGHT: height,
                        X1: position['l'],
                        Y1: position['t'],
                        X2: position['r'],
                        Y2: position['b']
                    }
                    word_coordinates.append(word_info)

        for table_block in page["tables"]:
            for cell in table_block["cells"]:
                for line in cell[LINES]:
                    for word_data in line[WORDS]:
                        word = word_data[SMALL_TEXT]
                        confidence = word_data[CONFIDENCE]
                        position = word_data[POSITION]
                        width = position['r'] - position['l']
                        height = position['b'] - position['t']
                        x1 = position['l'];
                        y1 = position['t']
                        x2 = position['r'];
                        y2 = position['b']
                        vertices = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                        word_info = {
                            WORD: word,
                            "text": word,
                            "confidence": confidence,
                            "vertices": vertices,
                            LEFT: position['l'],
                            TOP: position['t'],
                            WIDTH: width,
                            HEIGHT: height,
                            X1: position['l'],
                            Y1: position['t'],
                            X2: position['r'],
                            Y2: position['b']
                        }
                        word_coordinates.append(word_info)
    return word_coordinates


def generate_ocr_string_and_word_coordinates(image, file_name):
    config_file = get_config(image, file_name)

    main_response = requests.post(
        "{}/{}".format(ABBY_OCR_ENGINE_DETAILS["ABBY_API"],
                       ABBY_OCR_ENGINE_DETAILS["ABBY_PROCESSING_ENDPOINT"]),
        headers=ABBY_OCR_ENGINE_DETAILS["ABBY_HEADER"],
        data=json.dumps(config_file), verify=False)
    try:
        job_id = main_response.text.replace("{", "").replace("}", "").replace('"', '')
    except Exception as e:
        raise RuntimeError(f"Exception is ===> {e}\nResponse is ===> {main_response.text}")

    api_status_and_result_endpoint = "{}/{}".format(
        ABBY_OCR_ENGINE_DETAILS["ABBY_API"],
        ABBY_OCR_ENGINE_DETAILS["ABBYY_STATUS_ENDPOINT"].format(job_id))
    while True:
        status_response = requests.get(api_status_and_result_endpoint, verify=False)
        state_info = status_response.json()
        if state_info["State"] == "JS_Complete":
            print("  Job complete")
            break
        elif state_info["State"] == "JS_NoSuchJob":
            raise KeyError("Job \"%s\" not found" % (job_id))
        else:
            print("  Job state is %s, %d%% complete" % (
                state_info["State"], state_info["Progress"]))

    text_result = requests.get("{}/{}".format(api_status_and_result_endpoint,
                                              ABBY_OCR_ENGINE_DETAILS["ABBY_TEXT_RESULT_ENDPOINT"]), verify=False)
    all_text = text_result.text

    json_result = requests.get(
        "{}/{}".format(api_status_and_result_endpoint, ABBY_OCR_ENGINE_DETAILS["ABBY_JSON_RESULT_ENDPOINT"]),
        verify=False)
    json_result = json_result.json()

    word_coordinates = new_get_word_cordinates(json_result)

    return all_text, word_coordinates


def perform_ocr_operation(image, image_name):
    file_name = image_name.rsplit("/", 1)[1]

    # print(f"************************************************************************************************************************************************")
    # print(f"######## Performing OCR on {file_name}  #########\n")

    all_text, word_coordinates = generate_ocr_string_and_word_coordinates(image, file_name)
    return all_text, word_coordinates

    print("\n\nGot All text :", all_text)
    print("\n\nGot Word Co-ordinates :", word_coordinates)

    print(
        "************************************************************************************************************************************************\n")


if __name__ == "__main__":
    
    '''image_path = "/datadrive/geo_data/grasim_test_samples/Images/544522_Invoice_page_0.png"
    output_folder = "/home/ntlpt58/tf/BNI_abby_ocr"
    base_filename = "2"
    img = Image.open(image_path)
    all_text, word_coordinates = perform_ocr_operation(img, image_path)
    print(word_coordinates)
    exit('OK')
    text_file_path = os.path.join(output_folder, f"{base_filename}_all_text.txt")
    with open(text_file_path, 'w', encoding='utf-8') as text_file:
        text_file.write(all_text)
    print(f"Text saved to {text_file_path}")

    # Save word coordinates to a .json file
    json_file_path = os.path.join(output_folder, f"{base_filename}_word_coordinates.json")
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(word_coordinates, json_file, ensure_ascii=False, indent=4)
    print(f"Word coordinates saved to {json_file_path}")
    print(all_text)
    print(word_coordinates)
    exit()'''
    folder_path = "/datadrive/geo_data/grasim_test_samples"
    image_path = os.path.join(folder_path, 'Images')
    ocr_all_text_folder = os.path.join(folder_path,'OCR')
    if not os.path.exists(ocr_all_text_folder):
        os.makedirs(ocr_all_text_folder)
    start_time = time.time()
    for imgs in tqdm(os.listdir(image_path)):
        text_file_path = os.path.join(ocr_all_text_folder, f"{imgs[:-4]}_textAndCoordinates.txt")
        if not os.path.exists(text_file_path):
            print('Generating for Image:', imgs)
            img = Image.open(os.path.join(image_path, imgs))
            all_text, word_coordinates = perform_ocr_operation(img, os.path.join(image_path, imgs))
            all_text_file_path = os.path.join(ocr_all_text_folder, f"{imgs[:-4]}_all_text.txt")
            with open(all_text_file_path, 'w') as text_file:
                text_file.write(all_text)
            text_file.close()
            with open(text_file_path, 'w') as text_file:
                text_file.write(str(word_coordinates))
            text_file.close()
            time_taken = time.time() - start_time
            log_message = f"TIME TAKEN FOR OCR GENERATION {time_taken:.2f} seconds."
            print(log_message)
            # exit('OK')
        else:
            print(f"{imgs} text file already exists.")
    time_taken = time.time() - start_time
    log_message = f"TIME TAKEN FOR OCR GENERATION {time_taken:.2f} seconds."
    print(log_message)
    with open("ocr_generation_log.txt", "a") as log_file:  # 'a' mode appends to the file
        log_file.write(log_message)