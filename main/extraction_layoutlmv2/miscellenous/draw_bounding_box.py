#/home/ntlpt-42/Documents/mani_projects/document_processing_retraining/demo/final_data
#/home/ntlpt-42/Documents/mani_projects/IDP/covering_schedule/test_data/code/draw_bounding_box.py

import cv2
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import shutil
import json
from functools import cmp_to_key
from google.cloud import vision
from base64 import b64encode
import time




path= "/home/ntlpt-42/Documents/mani_projects/IDP/IDE/Error_analysis_STP_generation/stp_verification_on_100_samples/certificate_of_origin"
image_name= "Packing_List_38_page_2.png"



# # Load the image
# img = cv2.imread(f'{path}/{image_name}', 0)

# # Define the bounding box coordinates
# x, y, w, h = (564, 635, 385,83)

# # Draw the bounding box
# cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 4)

# # Show the image with the bounding box
# cv2.imshow("Bounding Box", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


def draw_bounding_box(img_path, labels_list):
    image = cv2.imread(img_path)
    # thresh = 255 - cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    for item in labels_list:
        label= item['label']
        x1= item['x1']
        y1= item['y1']
        x2= item['x2']
        y2= item['y2']
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
        image,
        label,
        (int(x1), int(y1)),
        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
        fontScale = 0.6,
        color = (255, 0, 0),
        thickness=2
    )

    return image



# def draw_word_bounding_box():
#     pass













if __name__=="__main__":
    #folder_path = "Data Generation"
    # folder_path = "Credit Note"
    # folder_path = "Credit Note updated"
    #folder_path = "Invoices combined"
    folder_path = "/media/ntlpt19/5250315B5031474F/finance_data_modeling/Classification/benchmark_images/table_data/set_data/train_master_data"
    #pdf_path = os.path.join(folder_path, "1_page_pdfs")
    images_path = os.path.join(folder_path, "images")
    labels_path = os.path.join(folder_path, "labels")
    #master_path = os.path.join(folder_path, "Master_Data")
    #ocr_path = os.path.join(folder_path, "OCR")
    bounding_box_path= os.path.join(folder_path,"bounding_box")
    #save_labels = os.path.join(folder_path, "labels_40_updated")
    if not os.path.exists(bounding_box_path):
        os.mkdir(bounding_box_path)


    with open(os.path.join(folder_path, "classes.txt"), "r") as f:
        classes = (f.read())
        classes = classes.split("\n")
    labelled_files = os.listdir(labels_path)
    labelled_files = [x.split(".txt")[0] for x in labelled_files]

    

# vision_api_key_path = "linen-creek-370205-4084e44af41e.json"


    #using intersection percentage
    annotation_data = []
    thresh = 300
    for file in labelled_files:
        #file = "CreditNote-ITALIAN-IM-000000001897385-AP_page_1"
        #file = "CreditNote-SPANISH-IM-000000002302232-AP_page_1"
        #file = "CreditNote-SPANISH-IM-000000002302232-AP_page_2"
        #if os.path.exists(os.path.join(pdf_path, file + ".pdf")) and os.path.exists(os.path.join(images_path, file + ".png")) or :
        if os.path.exists(os.path.join(images_path, file + ".png")):
            print('yes')
            print(file)
            # master_files = os.listdir(master_path)
            #doc = fitz.open(os.path.join(pdf_path, file + ".pdf"))
            image = cv2.imread(os.path.join(images_path, file + ".png"))
            # word_coordinates = []
            # word_coordinates, all_text = get_ocr_tesseract(os.path.join(images_path, file + ".png"))

            # # with open(os.path.join(ocr_path, file + "_text.txt"), "w") as f:
            # #     json.dump({"word_coordinates": word_coordinates}, f)
            # if len(word_coordinates) == 0:
            #     print(file)
            #     print("Not enough text")
            # else:
            #     shutil.copy(os.path.join(images_path, file + ".png"), os.path.join(master_path, file + ".png"))
            #     #shutil.copy(os.path.join(pdf_path, file + ".pdf"), os.path.join(master_path, file + ".pdf"))
            #     shutil.copy(os.path.join(labels_path, file + ".txt"), os.path.join(master_path, file + "_LabelImg.txt"))
            #     with open(os.path.join(master_path, file + "_text.txt"), "w") as f:
            #         json.dump({"word_coordinates": word_coordinates}, f)
            #     with open(os.path.join(ocr_path, file + "_text.txt"), "w") as f:
            #         json.dump({"word_coordinates": word_coordinates}, f)
            #     try:
            #         with open(os.path.join(master_path, file + "_all_text.txt"), "w") as f:
            #             json.dump({"all_text": all_text}, f)
            #     except:
            #         pass
                
            h, w, _ = image.shape
            with open(os.path.join(labels_path, file + ".txt"), "r") as f:
                label = (f.read())
            label = label.split("\n")
            labelled_data = []
            for l in label:
                l = l.split()
                if len(l) > 0:
                    l_class = classes[int(l[0])]
                    x_center = float(l[1]) * w
                    y_center = float(l[2]) * h
                    width = float(l[3]) * w
                    height = int(float(l[4]) * h)
                    x0 = int(x_center - (width/2))
                    x1 = int(x_center + (width/2))
                    y0 = int(y_center - (height / 2))
                    y1 = int(y_center + (height / 2))
                    cv2.rectangle(image, (x0, y0), (x1, y1), (0,255,0), 2)
                    labelled_data.append({
                        "label": l_class,
                        "x1": x0,
                        "y1": y0,
                        "x2": x1,
                        "y2": y1
                    })
            print(labelled_data)
            img_path= os.path.join(images_path, file + ".png")
            processed_img= draw_bounding_box(img_path,labelled_data)
            bounding_box= os.path.join(bounding_box_path, file+ ".png")
            cv2.imwrite(bounding_box, processed_img)



