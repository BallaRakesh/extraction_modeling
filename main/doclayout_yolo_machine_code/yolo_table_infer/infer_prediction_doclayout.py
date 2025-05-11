import cv2
from doclayout_yolo import YOLOv10
import os
import json
# from abby_ocr import perform_ocr_operation
from PIL import Image
# from google_vision_ocr import GV_Ocr
import matplotlib.pyplot as plt
# from huggingface_hub import hf_hub_download


# Load the pre-trained model
model_file = "/home/data_science/rakesh/financial_data_table/doclayout_yolo/DocLayout-YOLO/public_dataset/D4LA/yolov10m-doclayout_/home/data_science/rakesh/financial_data_table/table_data/doclayout_only_tf_epoch100_imgsz1024_bs4_pretrain_unknown/weights/best.pt"
# image_path = "/home/ng6309/datascience/anand/grasim/398274_Invoice_page_0.png"
# Perform prediction
# filepath = hf_hub_download(repo_id="juliozhao/DocLayout-YOLO-DocStructBench", filename="doclayout_yolo_docstructbench_imgsz1024.pt")
# doclayout_model = YOLOv10(filepath)

def plot_bboxes(predictions, image_path, output_path="output_current.jpg"):
    """
    Plots the bounding boxes of the first 4 predictions on the image and saves it.
    
    :param predictions: List of predictions, each in the format [x1, y1, x2, y2, _, _]
    :param image_path: Path to the input image
    :param output_path: Path to save the output image with bounding boxes
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to read the image.")
        return
    
    # Convert color from BGR to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Define colors for bounding boxes
    color = (255, 0, 0)  # Red color for bounding box
    thickness = 2  # Thickness of bounding box lines
    
    # Draw the first 4 bounding boxes
    for i in range(min(4, len(predictions))):
        x1, y1, x2, y2, _, _ = predictions[i]
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, thickness)
    
    # Save the output image
    cv2.imwrite(output_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    
    # Display the image with bounding boxes
    plt.figure(figsize=(10, 6))
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.show()
    
    print(f"Image saved at: {output_path}")

def is_inside_bbox(word_bbox, pred_box):
    """
    Check if the OCR word bounding box is inside the predicted bbox.
    word_bbox: [x1, y1, x2, y2]
    pred_bbox: [x_min, y_min, x_max, y_max]
    """
    pred_bbox = [j for j in pred_box]
    return (
        word_bbox[0] >= pred_bbox[0]
        and word_bbox[1] >= pred_bbox[1]
        and word_bbox[2] <= pred_bbox[2]
        and word_bbox[3] <= pred_bbox[3]
    )

def extract_text_from_bboxes(pred_bboxes, word_coordinates):
    """
    Extract text from words inside predicted bounding boxes.
    pred_bboxes: List of YOLO bboxes, each in the format [x_min, y_min, x_max, y_max]
    word_coordinates: List of OCR word data with their coordinates
    """
    extracted_texts = []
    for bbox in pred_bboxes:
        bbox_text = []
        for word in word_coordinates:
            word_bbox = [word['x1'], word['y1'], word['x2'], word['y2']]
            if is_inside_bbox(word_bbox, bbox):
                bbox_text.append(word['word'])
        extracted_texts.append(" ".join(bbox_text))
    return extracted_texts

import ast

def get_prediction(dir, model_file, image_for_prediction, output_folder_path, ocr_path, output_folder_results):
    ## For Prediction ##
    model = YOLOv10(model_file)
    det_res = model.predict(
    image_for_prediction,   # Image to predict
    imgsz=1024,        # Prediction image size
    conf=0.2,          # Confidence threshold
    device="cpu"    # Device to use (e.g., 'cuda:0' or 'cpu')
    )
    # result = {}
    image_name = image_for_prediction.rsplit("/",1)[1].rsplit(".",1)[0]
    # if dir in ['BOE', 'PO', 'PI', 'CI', 'BOL', 'COO', 'IC']:
    #     ocr_path = os.path.join(ocr_path, f"{image_name}_text.txt")
    # else:
    ocr_path = os.path.join(ocr_path, f"{image_name}.json")
    res_path = os.path.join(output_folder_results, f"{image_name}.json")
    # result[image_name] = {}
    predict = []
    for i, result in enumerate(det_res):
        idx2label = result.names
        # print(idx2label)
        # print(">>>>>>>>>>>>>>>>>>>>>..", type(result))
        boxes = result.boxes.data       # Boxes object for bbox outputs
        classes = result.boxes.cls      # Classes object for bbox outputs
        confidence = result.boxes.conf  # Confidence object for bbox outputs
        # for box in boxes:
        #     pred.append([j for j in box][:4])
        # print(pred)
        # exit()
        # print(image_name)
        # print(idx2label)
        print('???????????????????????')
        print('???????????????????????')
        print('result', result)
        print('boxes', boxes)
        print("boxes are ===>", boxes)
        print("classes are ===>", classes)
        print("confidence are ===>", confidence)
        pred = []
        for bbox in boxes:
            coords = []
            for t in bbox[:4]:
                coords.append(int(t.item()))
            for k in bbox[4:]:
                coords.append(k.item())
            pred.append(coords)
        # pred = [[int(t.item()) for t in bbox][:4] for bbox in boxes]
        print(pred)
        predict.extend(pred)
        print(predict)
        print("predictions are ===>", pred)
        print("classes are ===>", classes)
        # for bbox in pred:
        #     x_min, y_min, x_max, y_max = map(int, bbox)
        #     cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Blue box

        # cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print([float(j) for j in confidence])
        # exit()
        # print(f"{i} :: {boxes} ")
        # print(f"{i} :: {[idx2label[i.detach().cpu().item()] for i in classes]} ")
        # print(f"{i} :: {confidence} ")
        labels = [idx2label[i.detach().cpu().item()] for i in classes]
        im_array = result.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        # im.show()  # show image
        # print(os.path.join(output_folder_path, f'{image_for_prediction.rsplit("/",1)[1].rsplit(".",1)[0]}_output_results.jpg'))
        im.save(os.path.join(output_folder_path, f'{image_for_prediction.rsplit("/",1)[1].rsplit(".",1)[0]}_output_results.png'))
        # with open(ocr_path, 'r') as f:
        #     ocr_data = json.load(f)
        # if os.path.exists(ocr_path):
        #     try:
        #         # with open(ocr_path, "r", encoding="utf-8") as file:
        #         #     content = file.read().strip()
        #         with open(ocr_path, "r", encoding="utf-8") as file:
        #             ocr_data = json.load(file)
        #         word_coordinates = ocr_data["word_coordinates"]
        #     except:
        #         pass
        #         # try:
        #         #     with open(ocr_path, "r", encoding="utf-8") as file:
        #         #         content = file.read().strip()
        #         #     data = eval(content)
        #         #     word_coordinates = data["word_coordinates"]
        #         # except:
        #         #     print("ocr path is +++++++++>", ocr_path)
        #         #     img = Image.open(image_for_prediction)
        #         #     all_text, word_coordinates = perform_ocr_operation(img, image_for_prediction)
        #         #     ocr_info = {"all_text":all_text, "word_coordinates":word_coordinates}
        #         #     with open(ocr_path, "w", encoding="utf-8") as file:
        #         #         json.dump(ocr_info, file, indent=4)
        # else:
            # try:
        print("ocr path is +++++++++>", ocr_path)
        img = Image.open(image_for_prediction)
        print("image for prediction is +++++++++>", image_for_prediction)
        # OBJ = GV_Ocr()
        base_filename = os.path.splitext(os.path.basename(image_for_prediction))[0]
        ocr_file_path = os.path.join(master_ocr, f"{base_filename}_coordinates.txt")
        predictions = []
        
        
        
        if need_ocr:
            with open(ocr_file_path, "r", encoding="utf-8") as f:
                extracted_text = f.read()
                extracted_text = ast.literal_eval(extracted_text)
            f.close()
            # print("extracted text is +++++++++>", extracted_text)
            print("extracted text is +++++++++>", type(extracted_text))
            # ocr_info = OBJ.perform_ocr(image_for_prediction)
            # with open(ocr_path, "w", encoding="utf-8") as file:
            #     json.dump(ocr_info, file, indent=4)
                # except:
                #     word_coordinates = {}
                #     ocr_info = {"all_text":"", "word_coordinates":{}}
                #     with open(ocr_path, "w", encoding="utf-8") as file:
                #         json.dump(ocr_info, file, indent=4)
            # try:
            texts_inside_bboxes = extract_text_from_bboxes(pred, extracted_text)
            predictions = []
            for i, text in enumerate(texts_inside_bboxes):
                output = {}
                output["label"] = labels[i]
                output["text"] = text
                output["bbox"] = [int(j) for j in boxes[i]]
                predictions.append(output)
            # except:
            #     predictions = {}
            # # print("image_name", image_name)
            # print("predict is ===>", predict)
            # plot_bboxes(predict, image_for_prediction)
            # print("predictions are ===>", predictions)
            # exit()
            
            
         # Save the prediction to the file

    fina_result = {"prediction": predict}
    with open(res_path, 'w') as f:
        json.dump(fina_result, f, indent=4)
            
    return predictions, confidence


##################################### Seperate folder path, way #################################################
input_folder = "/home/data_science/rakesh/financial_data_table/table_data/doclayout_only_tf"
base_output_folder = "/home/data_science/rakesh/financial_data_table/table_data/doclayout_only_tf/table_test_output"
master_ocr = ""
need_ocr = False
for dir in os.listdir(input_folder):
    if dir in ["images"]:
        output_folder = f"{base_output_folder}/{dir}"
        image_folder = f"{input_folder}/{dir}"   
        output_folder_path = f"{output_folder}/{dir}_annotations"
        output_folder_json = f"{output_folder}/{dir}_result_ocr_json"
        output_folder_results = f"{output_folder}/{dir}_result_json"
        ocr_path = os.path.join(os.path.join(input_folder, dir), "OCR")
        os.makedirs(output_folder_path, exist_ok=True)
        os.makedirs(output_folder_json, exist_ok=True)
        os.makedirs(ocr_path, exist_ok=True)
        os.makedirs(output_folder_results, exist_ok=True)
        # test_images_list = []
        # with open("/home/ng6309/datascience/anand/doclayout_new_data_with_tf/test.txt","r") as f:
        #     for line in f:
        #         img = line.split("/")[-1].strip()
        #         test_images_list.append(img)
                
        # complete_image_folder = os.path.join(image_folder, "Images")
        for root, d, files in os.walk(image_folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                    # if file not in test_images_list:
                    #     continue
                    print('file %%%%%%%%%%%%%%>>', file)
                    result = {}
                    full_path = os.path.join(root, file)
                    directory,base_name = os.path.split(full_path)
                    base_name_without_extenstion,_ = os.path.splitext(base_name)
                    output_file_path = os.path.join(output_folder_json,base_name_without_extenstion+".json")
                    prediction,confidence = get_prediction(dir, model_file, full_path, output_folder_path, ocr_path, output_folder_results)
                    result["image_name"] = file
                    result["prediction"] = prediction
                    result["confidence"] = [float(j) for j in confidence]
                    # with open(os.path.join(output_folder_json, f"{file.rsplit('.',1)[0]}.json"), "w") as f:
                    #     f.write(str(result))
                    print(result)
                    with open(output_file_path, 'w') as output_file:
                        json.dump(result, output_file, indent=4)
    else:
        continue
                    # with open(os.path.join(ocr_path,base_name_without_extenstion+"_ocr.json"), 'w') as output_file:
                    #     json.dump(ocr_info, output_file, indent=4)
            
    # Annotate and save the result
    # output_folder_path = "/home/ng6309/datascience/anand/doclayout_yolo_results"
    # os.makedirs(output_folder_path, exist_ok=True)
    # image_name = image_path.split("/")[-1]
    # complete_path = f'{output_folder_path}/{image_name}'
    # annotated_frame = det_res[0].plot(pil=True, line_width=5, font_size=20)
    # cv2.imwrite(complete_path, annotated_frame)
 