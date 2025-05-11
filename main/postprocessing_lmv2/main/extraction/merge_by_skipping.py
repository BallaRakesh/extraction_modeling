from inference_utility_testing_fix_merging import  merge_surrounding, merge_by_skipping_running
import cv2
import random
import warnings
import cv2
import numpy as np
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
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

            # Write the value on top of the rectangle
            cv2.putText(img, key, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Save the image with drawn coordinates and values
    cv2.imwrite('pdf35-10_image.jpg', img)




def draw_bbox_image(image, data_dict):
    # Loop through the dictionary and draw the bounding boxes and labels
    for key, value in data_dict.items():
        label, bbox, confidence = value
        x1, y1, x2, y2 = bbox
        
        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        
        # Put the label near the bounding box
        cv2.putText(image, f"{label} ({confidence:.2f})", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Save or display the image
    cv2.imwrite('pdf35-10_image.jpg', image)

import cv2
import numpy as np

def vertical_merging(bb1, bb2):
    # Load your image
    image = cv2.imread('/home/ntlpt19/Downloads/Evaluation_Data/updated_code/pp_repo/Report_Generation_for_lmv2/main/extraction/pdf35-10_image.jpg')

    # Bounding box coordinates for bb1 and bb2
    x1_bb1, y1_bb1, x2_bb1, y2_bb1 = bb1
    x1_bb2, y1_bb2, x2_bb2, y2_bb2 = bb2

    # Calculate midpoints on bb1 for top and bottom
    mid_top_bb1 = (((y1_bb1 + y2_bb1) / 2 + y1_bb1) / 2)
    mid_bottom_bb1 = (((y1_bb1 + y2_bb1) / 2 + y2_bb1) / 2)

    # Points for drawing
    points = {
        "y1_bb2": (x1_bb2, y1_bb2),
        "y2_bb1": (x1_bb1, y2_bb1),
        "y1_bb1": (x1_bb1, y1_bb1),
        "y2_bb2": (x1_bb2, y2_bb2),
        "mid_top_bb1": (x2_bb1, mid_top_bb1),
        "mid_bottom_bb1": (x2_bb1, mid_bottom_bb1)
    }

    # Draw the points on the image
    for label, point in points.items():
        point = (int(point[0]), int(point[1]))
        cv2.circle(image, point, radius=5, color=(0, 0, 255), thickness=-1)
        cv2.putText(image, label, point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Draw lines to indicate the vertical distances being calculated
    distances = {
        "abs(y1_bb2 - y2_bb1)": (points["y1_bb2"], points["y2_bb1"]),
        "abs(y1_bb1 - y2_bb2)": (points["y1_bb1"], points["y2_bb2"]),
        "abs(y1_bb1 - y1_bb2)": (points["y1_bb1"], points["y1_bb2"]),
        "abs(y2_bb1 - y2_bb2)": (points["y2_bb1"], points["y2_bb2"]),
        "abs(y1_bb2 - mid_top_bb1)": (points["y1_bb2"], points["mid_top_bb1"]),
        "abs(y2_bb2 - mid_bottom_bb1)": (points["y2_bb2"], points["mid_bottom_bb1"])
    }

    for label, (point1, point2) in distances.items():
        cv2.line(image, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1])), color=(0, 255, 0), thickness=2)
        mid_point = ((int(point1[0]) + int(point2[0])) // 2, (int(point1[1]) + int(point2[1])) // 2)
        cv2.putText(image, label, mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Optionally, save the image
    cv2.imwrite('vertical_distance_visualization.jpg', image)



def minimum_distance(bb1, bb2):
    # bb1 points
    min_distance = 9999999999
    p_11 = np.array((bb1[0], bb1[1]))
    #(x2, (y1 + y2) / 2)
    # p_12 = np.array((bb1[0], bb1[3]))
    r_m_point = np.array((bb1[1], (bb1[3]+bb1[1])/2))
    print('r_m_point', r_m_point)
    p_13 = np.array((bb1[2], bb1[3]))
    p_14 = np.array((bb1[2], bb1[1]))
    # all_points_bb1 = [p_11, p_12, p_13, p_14]
    all_points_bb1 = [p_11, p_13, p_14]#, r_m_point]
    # bb2 points
    p_21 = np.array((bb2[0], bb2[1]))
    p_22 = np.array((bb2[0], bb2[3]))
    p_23 = np.array((bb2[2], bb2[3]))
    # p_24 = np.array((bb2[2], bb2[1]))
    # all_points_bb2 = [p_21, p_22, p_23, p_24]
    all_points_bb2 = [p_21, p_22, p_23]
    for point1 in all_points_bb1:
        for point2 in all_points_bb2:
            dist = abs(np.linalg.norm(point1 - point2))
            if dist < min_distance:
                min_distance = dist
                
    image = cv2.imread('/home/ntlpt19/Downloads/Evaluation_Data/updated_code/pp_repo/Report_Generation_for_lmv2/main/extraction/pdf35-10_image.jpg')       
    # Convert points to integers for drawing
    # points = [p_11, r_m_point, p_13, p_14, p_21, p_22, p_23]
    points = [p_11, p_13, p_14, p_21, p_22, p_23]
    points = [(int(p[0]), int(p[1])) for p in points]

    # Draw each point on the image
    for point in points:
        cv2.circle(image, point, radius=5, color=(0, 0, 255), thickness=-1)

    # Optionally, you can label the points if needed
    for i, point in enumerate(points):
        cv2.putText(image, f'p_{i+11}', point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 4)

    # Save the result if you want
    cv2.imwrite('image_with_horizontal_points.jpg', image)
    return min_distance

bb1 = [337, 358, 647, 376]
bb2 = [139, 467, 460, 566]
data = {"diclaration_by": [['500 telangana , india', [337, 358, 647, 376], 72.71438958513396], ['hamparan perak .', [466, 466, 688, 486], 92.73923263720252], ['dusun i pauh , kecamatan kabupaten deli serdang , sumatera utara 20374 indonesia', [139, 467, 460, 566], 90.73422564330926]]}

# draw_coordinates_and_values('/home/ntlpt19/TF_testing_EXT/dummy_responces/imgs/pdf35-10.png', data)
# print(minimum_distance(bb1, bb2))

vertical_merging(bb1, bb2)

# # Initial data in dictionary format
# data_dict = {1: ['500 , telangana', [161, 172, 271, 180], 65.17869158878504], 3: ['hamparan', [224, 224, 287, 233], 95.34], 4: [', india', [271, 172, 309, 180], 66.35], 5: ['perak .', [290, 224, 330, 233], 84.01], 0: ['dusun i pauh , kecamatan kabupaten deli serdang , sumatera utara - 20374 indonesia', [66, 224, 221, 271], 88.36]}
# idx_ = 0
# inx_ele_skip = [1]
# req_val = inx_ele_skip[idx_]
# while idx_ < len(inx_ele_skip):
#     req_val = req_val + 1
#     if req_val in data_dict:
#         inx_ele_skip[idx_] = req_val
#         idx_ = idx_ + 1
#     else:
#         pass
# # Extracting the first two elements from the values of the dictionary

# # first_two_elements = [value[:2] for value in ]
# # print(first_two_elements)
# data_dict = dict(sorted(data_dict.items()))


exit('..............')


data = {'shipper_name': [['taiko', [65, 134, 100, 142], 96.87], ['chandernagar', [103, 134, 197, 142], 96.33], ['chemicals pvt.ltd', [200, 134, 312, 142], 96.88]], 'shipper_address': [['plot no . 3-16-724 , road no.3 , sardar patel colony , secundrabad - 015', [65, 147, 226, 180], 81.15], ['trimulgherry', [203, 159, 292, 168], 87.18], [',', [292, 159, 296, 168], 69.78]], 'consignee_address': [['500 ,', [161, 172, 200, 180], 56.72], ['telangana', [203, 172, 271, 180], 70.03], [', india', [271, 172, 309, 180], 66.35], ['dusun i pauh , kecamatan kabupaten deli serdang , sumatera utara - 20374 indonesia', [66, 224, 221, 271], 88.36], ['hamparan', [224, 224, 287, 233], 95.34], ['perak .', [290, 224, 330, 233], 84.01]], 'consignee_name': [['pt bumi', [66, 211, 110, 220], 92.09], ['karyatama', [113, 211, 185, 219], 85.73], ['raharja', [190, 211, 240, 219], 80.42]], 'notify_party_name': [['same as', [66, 336, 113, 345], 96.66], ['consignee', [116, 336, 179, 344], 95.96]], 'vessel_name': [['m.v. jutha', [66, 372, 128, 380], 95.31], ['dhammaraksa', [131, 372, 221, 380], 93.13]], 'port_of_discharge': [['ciwandan', [66, 408, 127, 418], 93.63], ['port ,', [130, 408, 161, 417], 88.0], ['indonesia', [165, 408, 226, 417], 81.59]], 'goods_description': [['bentonite lumps', [140, 461, 241, 470], 94.2]], 'port_of_loading': [['kakinada', [238, 372, 298, 379], 90.53], [',', [297, 372, 300, 379], 87.97], ['india', [304, 372, 336, 379], 86.36], [',', [301, 408, 304, 416], 24.47]], 'bill_of_lading_number': [['jd / kkd -', [434, 118, 480, 127], 88.25], ['cwn', [480, 118, 506, 127], 92.57], ['/ 001', [507, 117, 526, 127], 93.11]], 'bol_original_or_copy': [['copy', [359, 275, 397, 290], 75.05]], 'place_of_receipt': [['jakarta', [252, 408, 302, 417], 23.58], ['indonesia', [308, 408, 369, 416], 21.82]], 'gross_weight': [['8000', [398, 461, 419, 469], 83.49], ['m', [422, 461, 431, 469], 83.5], ['/', [432, 461, 437, 469], 81.1], ['tons', [435, 461, 458, 469], 76.35]], 'freight_collect_at': [['charter party', [204, 496, 293, 505], 49.96]], 'bol_original_number': [['3 (', [234, 648, 245, 655], 61.36], ['three', [246, 648, 275, 655], 79.11], [')', [278, 648, 282, 655], 56.76]], 'place_of_issue': [['kakinada port', [318, 604, 405, 611], 93.61]], 'bill_of_lading_issue_date': [['22/04/2017', [451, 604, 497, 611], 89.79]], 'signed_By_agent': [['rojjanaporn baxico', [408, 624, 495, 673], 15.5]]}
model_output = {'shipper_name': [['taiko', [65, 134, 100, 142], 96.87], ['chandernagar', [103, 134, 197, 142], 96.33], ['chemicals', [200, 134, 264, 142], 96.93], ['pvt.ltd', [267, 134, 312, 142], 96.82]], 'shipper_address': [['plot', [65, 147, 94, 155], 89.92], ['no', [96, 147, 112, 155], 86.72], ['.', [112, 147, 115, 155], 89.31], ['3-16-724', [118, 147, 159, 155], 90.0], [',', [159, 147, 162, 155], 85.73], ['road', [165, 147, 196, 155], 88.78], ['no.3', [199, 147, 224, 155], 85.5], [',', [223, 147, 226, 155], 61.96], ['sardar', [66, 159, 111, 168], 87.83], ['patel', [114, 159, 149, 168], 86.66], ['colony', [152, 159, 198, 168], 82.67], [',', [197, 159, 201, 168], 83.64], ['trimulgherry', [203, 159, 292, 168], 87.18], [',', [292, 159, 296, 168], 69.78], ['secundrabad', [66, 172, 149, 180], 79.62], ['-', [151, 172, 158, 180], 57.67], ['015', [180, 172, 197, 180], 61.31]], 'consignee_address': [['500', [161, 172, 177, 180], 47.64], [',', [197, 172, 200, 180], 65.81], ['telangana', [203, 172, 271, 180], 70.03], [',', [271, 172, 274, 180], 61.72], ['india', [277, 172, 309, 180], 70.98], ['dusun', [66, 224, 104, 233], 94.14], ['i', [107, 224, 110, 233], 95.03], ['pauh', [114, 224, 144, 233], 95.04], [',', [144, 224, 147, 233], 95.42], ['kecamatan', [150, 224, 221, 233], 95.4], ['hamparan', [224, 224, 287, 233], 95.34], ['perak', [290, 224, 327, 233], 95.05], ['.', [326, 224, 330, 233], 72.97], ['kabupaten', [66, 237, 133, 245], 93.47], ['deli', [137, 237, 161, 245], 94.22], ['serdang', [165, 237, 217, 245], 93.97], [',', [217, 237, 220, 245], 81.9], ['sumatera', [66, 249, 126, 259], 84.15], ['utara', [131, 250, 168, 258], 80.59], ['-', [170, 250, 175, 258], 81.59], ['20374', [178, 249, 205, 258], 78.68], ['indonesia', [66, 263, 127, 271], 73.42]], 'consignee_name': [['pt', [66, 211, 77, 220], 92.19], ['bumi', [81, 211, 110, 219], 91.99], ['karyatama', [113, 211, 185, 219], 85.73], ['raharja', [190, 211, 240, 219], 80.42]], 'notify_party_name': [['same', [66, 336, 96, 345], 96.66], ['as', [99, 336, 113, 344], 96.67], ['consignee', [116, 336, 179, 344], 95.96]], 'vessel_name': [['m.v.', [66, 372, 90, 380], 95.26], ['jutha', [92, 372, 128, 380], 95.36], ['dhammaraksa', [131, 372, 221, 380], 93.13]], 'port_of_discharge': [['ciwandan', [66, 408, 127, 418], 93.63], ['port', [130, 408, 159, 417], 90.76], [',', [158, 408, 161, 417], 85.25], ['indonesia', [165, 408, 226, 417], 81.59]], 'goods_description': [['bentonite', [140, 461, 200, 470], 94.86], ['lumps', [204, 461, 241, 469], 93.55]], 'port_of_loading': [['kakinada', [238, 372, 298, 379], 90.53], [',', [297, 372, 300, 379], 87.97], ['india', [304, 372, 336, 379], 86.36], [',', [301, 408, 304, 416], 24.47]], 'bill_of_lading_number': [['jd', [434, 118, 448, 127], 79.67], ['/', [447, 118, 451, 127], 89.16], ['kkd', [451, 118, 476, 127], 91.7], ['-', [476, 118, 480, 127], 92.46], ['cwn', [480, 118, 506, 127], 92.57], ['/', [507, 118, 511, 127], 92.59], ['001', [510, 117, 526, 127], 93.63]], 'bol_original_or_copy': [['copy', [359, 275, 397, 290], 75.05]], 'place_of_receipt': [['jakarta', [252, 408, 302, 417], 23.58], ['indonesia', [308, 408, 369, 416], 21.82]], 'gross_weight': [['8000', [398, 461, 419, 469], 83.49], ['m', [422, 461, 431, 469], 83.5], ['/', [432, 461, 437, 469], 81.1], ['tons', [435, 461, 458, 469], 76.35]], 'freight_collect_at': [['charter', [204, 496, 255, 505], 49.56], ['party', [258, 496, 293, 504], 50.36]], 'bol_original_number': [['3', [234, 648, 239, 655], 57.36], ['(', [241, 648, 245, 655], 65.37], ['three', [246, 648, 275, 655], 79.11], [')', [278, 648, 282, 655], 56.76]], 'place_of_issue': [['kakinada', [318, 604, 375, 611], 93.64], ['port', [377, 604, 405, 611], 93.58]], 'bill_of_lading_issue_date': [['22/04/2017', [451, 604, 497, 611], 89.79]], 'signed_By_agent': [['rojjanaporn', [420, 624, 495, 635], 19.96], ['baxico', [408, 656, 470, 673], 11.05]]}
data = {"diclaration_by": [["for", [34, 481, 48, 499], 79.02], ["sanvijay", [33, 438, 48, 479], 74.07], ["& rolling", [32, 390, 47, 435], 47.46], ["ltd. .", [32, 335, 45, 361], 59.56], ["for : sanvijay roll . & engg . ltd.", [308, 818, 470, 828], 79.93198828125], ["for", [485, 475, 502, 494], 84.04], ["sanvijay", [485, 431, 501, 473], 84.09], ["rolling", [485, 394, 501, 429], 83.82], ["ltd. engg . &", [484, 327, 501, 390], 83.21], ["for gammon india limised", [381, 903, 547, 913], 83.51201329378158]], "signature": [["signatory jayap engg authorsed", [32, 330, 79, 438], 62.52], ["signatory", [367, 863, 414, 874], 48.82], ["gry signatory vimy authorised", [506, 322, 533, 434], 66.43], ["danh", [430, 911, 492, 936], 68.29], ["che thathi signatories", [502, 896, 574, 955], 55.76]], "drawee_bank_name": [["oriental bank of commerce", [197, 270, 366, 277], 76.42842317073172], ["oriental bank of commerce", [111, 833, 269, 843], 76.3848344370861]], "drawee_bank_address": [["prabhadevi marg prabhadevi br.aman mumbai chambers - 400 veer sawarkar", [111, 283, 380, 302], 84.72747603833865], ["prabhadevi br.aman chambers veer sawarkar marg prabhadevi mumbai - 400 025", [110, 849, 278, 887], 84.98]], "lc_ref_no": [["no.1143950011513", [111, 308, 189, 316], 55.02]], "lc_date": [["11.10.2013", [233, 308, 274, 316], 58.23]], "drawee_name": [["m / s gammon india ltd .", [300, 308, 432, 316], 82.76896285179232], ["m / s gammon", [355, 500, 422, 509], 82.77730158730158], ["india ltd .", [110, 513, 163, 521], 82.56888888888888], ["a / c m / s gammon india ltd .", [110, 908, 249, 917], 82.74855753617321]], "drawer_name": [["cardhaman urban co - op bank ltd", [239, 26, 355, 36], 82.78052631578946]], "drawer_address": [["73 - c sewasadan chok central avenue , kagpur - 440318", [240, 41, 351, 65], 89.1274107142857]], "stamp": [["turban coce", [266, 110, 315, 124], 71.28], ["ward nagpur", [259, 147, 304, 168], 71.26]], "tenore_details": [["180 ( one hundred eighty ) days from the date of bill of exchange", [110, 446, 405, 455], 83.68279635305409]], "drawer_bank_address": [["negpu", [223, 473, 273, 485], 82.15]], "boe_currency": [["jaci 1991", [124, 471, 155, 486], 15.824], ["rs", [152, 487, 163, 496], 60.89]], "boe_amount": [["\u0432\u0435\u0448\u044c", [162, 471, 196, 486], 5.64], ["74,19,283-000", [165, 487, 223, 496], 45.53], ["7419283", [309, 418, 340, 427], 64.4], ["7419283", [363, 788, 401, 795], 62.72]], "amount_in_words": [["hundred eighty three only seventy four lac nineteen thousand two", [111, 487, 427, 509], 90.3813556033255]], "drawee_address": [["t & d business g - 55 , midc , industrial area butibori , nagpur", [111, 923, 281, 964], 87.3106027429983]], "bill_exchange_date": [["27-10-2013", [307, 359, 410, 372], 27.82]], "invoice_due_date": [["25-04-2014", [307, 389, 414, 403], 23.47]]}
model_output = {"diclaration_by": [["for", [34, 481, 48, 499], 79.02], ["sanvijay", [33, 438, 48, 479], 74.07], ["rolling", [32, 399, 47, 435], 58.65], ["&", [33, 390, 46, 398], 36.27], [".", [32, 356, 45, 361], 53.42], ["ltd.", [32, 335, 45, 353], 65.7], ["for", [308, 818, 327, 828], 83.65], [":", [326, 818, 330, 828], 83.64], ["sanvijay", [332, 818, 374, 828], 83.74], ["roll", [376, 818, 398, 828], 82.39], [".", [396, 818, 400, 828], 77.9], ["&", [404, 818, 412, 828], 73.89], ["engg", [416, 818, 444, 828], 69.35], [".", [444, 818, 447, 828], 79.71], ["ltd.", [450, 818, 470, 828], 80.47], ["for", [485, 475, 502, 494], 84.04], ["sanvijay", [485, 431, 501, 473], 84.09], ["rolling", [485, 394, 501, 429], 83.82], ["&", [485, 382, 501, 390], 83.36], ["engg", [484, 353, 500, 380], 82.97], [".", [484, 350, 500, 354], 83.41], ["ltd.", [484, 327, 500, 347], 83.1], ["for", [381, 903, 399, 913], 83.97], ["gammon", [403, 903, 458, 913], 83.97], ["india", [462, 903, 496, 913], 83.66], ["limised", [498, 903, 547, 913], 82.66]], "signature": [["engg", [32, 360, 45, 385], 47.98], ["jayap", [46, 338, 63, 396], 66.89], ["authorsed", [62, 383, 78, 438], 67.55], ["signatory", [63, 330, 79, 382], 67.67], ["signatory", [367, 863, 414, 874], 48.82], ["vimy", [506, 358, 518, 399], 69.25], ["gry", [506, 322, 517, 353], 69.23], ["authorised", [522, 380, 533, 434], 61.44], ["signatory", [522, 328, 532, 376], 65.79], ["danh", [430, 911, 492, 936], 68.29], ["che", [507, 896, 562, 948], 68.47], ["thathi", [529, 911, 574, 943], 68.89], ["signatories", [502, 944, 549, 955], 29.92]], "drawee_bank_name": [["oriental", [197, 270, 251, 277], 76.39], ["bank", [254, 270, 284, 277], 76.4], ["of", [286, 270, 300, 277], 76.43], ["commerce", [305, 270, 366, 277], 76.47], ["oriental", [111, 834, 160, 843], 76.39], ["bank", [163, 834, 190, 842], 76.36], ["of", [195, 834, 206, 842], 76.36], ["commerce", [213, 833, 269, 842], 76.41]], "drawee_bank_address": [["prabhadevi", [111, 283, 179, 290], 84.63], ["br.aman", [182, 283, 229, 290], 84.54], ["chambers", [233, 283, 290, 290], 84.52], ["veer", [294, 283, 321, 290], 84.58], ["sawarkar", [324, 283, 380, 290], 84.6], ["marg", [111, 295, 141, 302], 84.78], ["prabhadevi", [144, 295, 212, 302], 84.77], ["mumbai", [215, 295, 259, 302], 84.86], ["-", [262, 295, 265, 302], 84.98], ["400", [267, 295, 282, 302], 85.01], ["prabhadevi", [111, 849, 173, 859], 84.66], ["br.aman", [178, 850, 222, 858], 84.64], ["chambers", [226, 849, 278, 857], 84.76], ["veer", [110, 865, 136, 873], 84.95], ["sawarkar", [138, 865, 192, 872], 84.95], ["marg", [199, 865, 228, 872], 85.08], ["prabhadevi", [110, 878, 174, 887], 85.06], ["mumbai", [180, 878, 222, 886], 85.12], ["-", [226, 878, 229, 886], 85.22], ["400", [231, 878, 247, 886], 85.17], ["025", [250, 877, 265, 886], 85.17]], "lc_ref_no": [["no.1143950011513", [111, 308, 189, 316], 55.02]], "lc_date": [["11.10.2013", [233, 308, 274, 316], 58.23]], "drawee_name": [["m", [300, 308, 309, 316], 82.78], ["/", [310, 308, 315, 316], 82.8], ["s", [315, 308, 320, 316], 82.81], ["gammon", [324, 308, 372, 316], 82.82], ["india", [374, 308, 405, 316], 82.8], ["ltd", [408, 308, 429, 316], 82.76], [".", [428, 308, 432, 316], 82.42], ["m", [355, 500, 363, 509], 82.75], ["/", [363, 500, 367, 509], 82.77], ["s", [368, 500, 372, 509], 82.78], ["gammon", [376, 500, 422, 509], 82.78], ["india", [110, 513, 138, 521], 82.74], ["ltd", [141, 513, 160, 520], 82.68], [".", [161, 513, 163, 520], 81.97], ["a", [110, 910, 117, 917], 82.74], ["/", [116, 909, 120, 916], 82.74], ["c", [120, 909, 123, 916], 82.79], ["m", [127, 909, 134, 916], 82.82], ["/", [136, 909, 139, 916], 82.82], ["s", [139, 909, 143, 916], 82.83], ["gammon", [147, 909, 193, 916], 82.83], ["india", [198, 908, 225, 916], 82.79], ["ltd", [228, 908, 247, 915], 82.71], [".", [247, 908, 249, 915], 81.98]], "drawer_name": [["cardhaman", [239, 26, 279, 36], 82.8], ["urban", [281, 26, 302, 36], 82.84], ["co", [304, 26, 313, 36], 82.83], ["-", [310, 26, 314, 36], 82.8], ["op", [314, 26, 323, 36], 82.81], ["bank", [324, 26, 343, 36], 82.77], ["ltd", [343, 26, 355, 36], 82.59]], "drawer_address": [["73", [254, 41, 264, 50], 89.04], ["-", [262, 41, 266, 50], 89.17], ["c", [264, 41, 269, 50], 89.21], ["sewasadan", [272, 41, 310, 50], 89.16], ["chok", [313, 41, 336, 50], 89.06], ["central", [240, 57, 267, 65], 89.06], ["avenue", [268, 57, 294, 65], 89.16], [",", [291, 57, 295, 65], 89.15], ["kagpur", [297, 57, 322, 65], 89.17], ["-", [325, 57, 328, 65], 89.14], ["440318", [328, 57, 351, 65], 89.05]], "stamp": [["turban", [266, 110, 292, 124], 71.27], ["coce", [296, 110, 315, 124], 71.29], ["ward", [259, 147, 277, 166], 71.26], ["nagpur", [276, 157, 304, 168], 71.27]], "tenore_details": [["180", [110, 446, 124, 455], 84.14], ["(", [127, 446, 131, 455], 84.05], ["one", [133, 446, 149, 455], 83.95], ["hundred", [152, 446, 187, 455], 83.52], ["eighty", [191, 446, 216, 455], 83.28], [")", [217, 446, 219, 455], 83.8], ["days", [223, 446, 238, 455], 84.09], ["from", [243, 446, 260, 455], 84.08], ["the", [264, 446, 276, 455], 84.06], ["date", [279, 446, 294, 455], 83.96], ["of", [298, 446, 306, 455], 83.89], ["bill", [309, 446, 329, 455], 83.81], ["of", [333, 446, 345, 455], 83.58], ["exchange", [349, 446, 405, 455], 83.08]], "drawer_bank_address": [["negpu", [223, 473, 273, 485], 82.15]], "boe_currency": [["jaci", [124, 471, 154, 486], 17.88], ["1991", [125, 473, 155, 483], 12.74], ["rs", [152, 487, 163, 496], 60.89]], "boe_amount": [["\u0432\u0435\u0448\u044c", [162, 471, 196, 486], 5.64], ["74,19,283-000", [165, 487, 223, 496], 45.53], ["7419283", [309, 418, 340, 427], 64.4], ["7419283", [363, 788, 401, 795], 62.72]], "amount_in_words": [["seventy", [258, 487, 288, 496], 90.29], ["four", [292, 487, 309, 496], 90.42], ["lac", [313, 487, 323, 496], 90.4], ["nineteen", [326, 487, 360, 496], 90.42], ["thousand", [366, 487, 406, 496], 90.42], ["two", [409, 487, 427, 496], 90.41], ["hundred", [111, 500, 145, 509], 90.41], ["eighty", [148, 500, 174, 509], 90.4], ["three", [177, 500, 199, 509], 90.39], ["only", [203, 500, 219, 509], 90.23]], "drawee_address": [["t", [111, 924, 117, 932], 87.33], ["&", [119, 923, 125, 931], 87.37], ["d", [129, 923, 135, 931], 87.35], ["business", [139, 923, 186, 931], 87.27], ["g", [111, 940, 117, 949], 87.36], ["-", [117, 939, 120, 948], 87.36], ["55", [120, 939, 130, 948], 87.36], [",", [131, 939, 133, 948], 87.31], ["midc", [136, 939, 162, 948], 87.36], [",", [162, 939, 164, 948], 87.26], ["industrial", [165, 939, 203, 948], 87.32], ["area", [207, 939, 225, 948], 87.3], [",", [228, 938, 228, 947], 87.23], ["butibori", [232, 938, 279, 947], 87.32], [",", [279, 938, 281, 947], 87.05], ["nagpur", [111, 955, 150, 964], 87.27]], "bill_exchange_date": [["27-10-2013", [307, 359, 410, 372], 27.82]], "invoice_due_date": [["25-04-2014", [307, 389, 414, 403], 23.47]]}
w, h = 619, 874
updated_key_value = merge_by_skipping_running(data, model_output, w, h, 'diclaration_by', data['diclaration_by'])
# Load your image
image_path = '/home/ntlpt19/Downloads/Evaluation_Data/Evaluation_Data_updated/BOE/vertical_merging_issue/Bill_of_Exchange_266_0.png'
image = cv2.imread(image_path)
draw_bbox_image(image, updated_key_value)

print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
updated_key_value = [item for item in updated_key_value.values()]

print(updated_key_value)
exit('???????????????????')
merged_result = merge_surrounding(data, model_output, w, h)
draw_coordinates_and_values('/home/ntlpt19/Downloads/Evaluation_Data/Evaluation_Data_updated/BOL/Images/Bill_of_lading_1029_page_6.png', data)
print(data)