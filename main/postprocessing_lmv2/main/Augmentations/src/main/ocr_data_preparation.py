import cv2
import os
import json


def rotate90Deg(bndbox, img_width):  # just passing width of image is enough for 90 degree rotation.
    x_min, y_min, x_max, y_max = bndbox
    new_xmin = y_min
    new_ymin = img_width - x_max
    new_xmax = y_max
    new_ymax = img_width - x_min
    return [new_xmin, new_ymin, new_xmax, new_ymax]


def rotate_90Deg(bndbox, image_width):
    """
    image_width: Width of the image after clockwise rotation of 90 degrees
    """
    x_min, y_min, x_max, y_max = bndbox
    new_xmin = image_width - y_max  # Reflection about center X-line
    new_ymin = x_min
    new_xmax = image_width - y_min  # Reflection about center X-line
    new_ymax = x_max
    return [new_xmin, new_ymin, new_xmax, new_ymax]


if __name__ == "__main__":

    ocr_path = '/media/tarun/D3/TradeFinance/final_delivery/lc_cancellation_test/OCR'
    # ocr_path_text = '/home/ntlpt19/Downloads/MERGED_DATA/AIR_WAY/ROOT_AIR_WAY/OCR/Bill_Of_Landing_234_page_0_text.txt'

    # ocr_path_01 = '/home/ntlpt19/Downloads/MERGED_DATA/AIR_WAY/ROOT_AIR_WAY/OCR/Bill_Of_Landing_234_page_0_text_right' \
    #               '.txt '
    # img_path = '/home/ntlpt19/Downloads/MERGED_DATA/AIR_WAY/ROOT_AIR_WAY/Images/Bill_Of_Landing_234_page_0.png'
    images_path = '/media/tarun/D3/TradeFinance/final_delivery/lc_cancellation_test/Images'

    img_files = os.listdir(images_path)
    img_files = [x.split(".png")[0] for x in img_files]

    labelled_files = os.listdir(ocr_path)
    labelled_files = [x.split(".txt")[0] for x in labelled_files]
    for file in labelled_files:
        print("file: ", file)
        if file.split("_data.txt")[0] in img_files:
            img_path = os.path.join(images_path, file.split("_data.txt")[0] + '.png')
            print(f"image path: {img_path}")
            # exit("++++++++++")
            image = cv2.imread(img_path)
            h, w, _ = image.shape
            perfect_ocr_path = os.path.join(ocr_path, file + '.txt')
            print(f"ocr path: {perfect_ocr_path}")
            exit("++++++++++++++")
            with open(perfect_ocr_path, "r") as f:
                # file_path = os.path.join(data_path, file)
                labels = json.load(f)
                print(labels['word_coordinates'])
            lis = []
            for i in labels['word_coordinates']:
                new_x1, new_y1, new_x2, new_y2 = rotate_90Deg([i['x1'], i['y1'], i['x2'], i['y2']], h)
                lis.append({'word': i['word'], 'left': new_x1, 'top': new_y1, 'width': new_x2 - new_x1,
                            'height': new_y2 - new_y1, 'x1': new_x1, 'y1': new_y1, 'x2': new_x2, 'y2': new_y2})
            modified_data = {'word_coordinates': lis}
            Update_ocr_path = os.path.join(ocr_path, file[:-5] + 'r' + '_text.txt')
            with open(Update_ocr_path, "w") as modified_file:
                json.dump(modified_data, modified_file)

            lis = []
            for i in labels['word_coordinates']:
                new_x1, new_y1, new_x2, new_y2 = rotate90Deg([i['x1'], i['y1'], i['x2'], i['y2']], w)
                lis.append({'word': i['word'], 'left': new_x1, 'top': new_y1, 'width': new_x2 - new_x1,
                            'height': new_y2 - new_y1, 'x1': new_x1, 'y1': new_y1, 'x2': new_x2, 'y2': new_y2})
            modified_data = {'word_coordinates': lis}
            Update_ocr_path = os.path.join(ocr_path, file[:-5] + 'l' + '_text.txt')
            with open(Update_ocr_path, "w") as modified_file:
                json.dump(modified_data, modified_file)

            lis = []
            for i in labels['word_coordinates']:
                new_x1, new_y1, new_x2, new_y2 = rotate90Deg([i['x1'], i['y1'], i['x2'], i['y2']], w)
                new_x1, new_y1, new_x2, new_y2 = rotate90Deg([new_x1, new_y1, new_x2, new_y2], h)
                lis.append({'word': i['word'], 'left': new_x1, 'top': new_y1, 'width': new_x2 - new_x1,
                            'height': new_y2 - new_y1, 'x1': new_x1, 'y1': new_y1, 'x2': new_x2, 'y2': new_y2})
            modified_data = {'word_coordinates': lis}
            Update_ocr_path = os.path.join(ocr_path, file[:-5] + 'i' + '_text.txt')
            with open(Update_ocr_path, "w") as modified_file:
                json.dump(modified_data, modified_file)
