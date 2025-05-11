from inference_utility_testing_fix_merging import *

def update_inx_ele_skip(inx_ele_skip, addition_val):
    # Add addition_val to the elements specified by inx_ele_skip
    for index in range(len(inx_ele_skip)):
        inx_ele_skip[index] += addition_val 
    return inx_ele_skip

import copy
def merge_by_skipping(data, model_output, w, h, key, all_values):
    print("entered final validation", all_values)
    '''
    [["500 , telangana , india", [161, 172, 308, 180], 67.32535079575597], =>1
    ["perak .", [290, 222, 329, 232], 84.68], =>2
    ["dusun kabupaten i pauh utara deli , kecamatan sumatera serdang 20374 , hamparan indonesia -", [65, 223, 285, 270], 91.24132404181185]],
                                                => 3 (need to merge 2 and 3)
    '''
    all_values = sorted(all_values, key=lambda bbox: bbox[1][0])#, reverse=True)
    print("initial values: ", all_values)
    no_of_ele_skip = 1
    inx_ele_skip = [0]
    addtion_val = 0
    while len(all_values)>1 and len(inx_ele_skip)-1 < len(all_values)-2:
        all_values_zeros = [0] * len(all_values)
        org_all_values = copy.deepcopy(all_values)
        inx_ele_skip = update_inx_ele_skip(inx_ele_skip, addtion_val)
        #after updating
        all_values = after_skipping(all_values, inx_ele_skip)
        saving_idx = 0
        print(f"AFTER SKIPPING FOR THE INDEX >>> inx_ele_skip >> {inx_ele_skip} >>> from > {all_values}")
        # all_values = data[key]
        # for sort_idx in range(0,4):
        bboxes = [x[1] for x in all_values]
        vertical_alignment_bbox = calculate_orientation(bboxes)
        vertical_alignment_bbox = False
        print(key, ">>>>>>> vertical_alignment_bbox >>>>>>>>>", vertical_alignment_bbox)
        print(all_values)
        length = len(all_values)
        # eps_horizontal = 100  # Threshold for horizontal merging
        # eps_vertical = 50  # Threshold for vertical merging
        # eps_vertical2 = 100  # Threshold for vertical merging
        eps_horizontal = round(w*5.5/100)#100
        eps_vertical = round(h*2.5/100) #100
        eps_vertical2 = 100  # Threshold for vertical merging
        if length > 1:
            i = 0
            merging_flag = False
            while i in range(length - 1):
                # if i not in inx_ele_skip and i + 1 not in inx_ele_skip:
                print(f"CHECK inx_ele_skip FOR iTH >>{i}>>value{inx_ele_skip}")
                bb1 = all_values[i][1]
                bb2 = all_values[i + 1][1]
                bb1_token = all_values[i][0]
                bb2_token = all_values[i + 1][0]
                print(f"TOKENS GOING CHECKING FOR MERGING >> {bb1_token} >> and >> {bb2_token}")
                confs = [all_values[i][2], all_values[i + 1][2]]
                min_dist_horizontal = minimum_distance(bb1, bb2)
                min_dist_vertical = minimum_distance_vertical(bb1, bb2)
                vertical_flag = check_vertical_indetween(bb1, bb2)
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

                # if (bb1_height >= bb1_width and bb2_height >= bb2_width) or (
                # 		bb1_height <= bb1_width and bb2_height <= bb2_width):

                print('entered into first if')
                print(">>>>>>>>>>RRRRRRRRAAAAAAAAAKKKKKKKEEE", min_dist_horizontal)
                if key in master_keys or vertical_alignment_bbox:
                    merge_flag = (min_dist_horizontal <= eps_horizontal or IOU_horizontal > 0 or inter_percentage > 0) or (
                            min_dist_vertical <= eps_vertical2 or IOU_vertical > 0 or inter_percentage)
                else:
                    merge_flag = (min_dist_horizontal <= eps_horizontal or IOU_horizontal > 0 or inter_percentage > 0) and (
                            min_dist_vertical <= eps_vertical or IOU_vertical > 0 or inter_percentage)# or vertical_flag) 
                print('min_dist_vertical', min_dist_vertical)
                print("IOU_horizontal > 0 or inter_percentage", IOU_horizontal, inter_percentage)
                if merge_flag:
                    saving_idx = org_all_values.index(all_values[i + 1][1])
                    merge_flag = True
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
                    # else:
                    # 	print("distance is very high")
                    # 	i = i + 1

                else:
                    if merge_flag:
                        all_values_zeros[saving_idx] = new_value
                        merge_flag = False
                    else:
                        all_values_zeros[org_all_values.index(all_values[i])] = all_values[i]
                    i = i + 1
            if new_value not in all_values_zeros:#if all  values mergeed
                all_values_zeros[saving_idx] = new_value
            if all_values[i + 1] not in all_values_zeros: #if last two values not merged
                all_values_zeros[org_all_values.index(all_values[i + 1])] = all_values[i + 1]
            #readding the skipped values
            for idx_skip in inx_ele_skip:
                all_values_zeros[idx_skip] = org_all_values[idx_skip]
            #remove zeros and make the all_values as all_values_zeros
            if addtion_val >= len(all_values) :
                appending_ele = len(inx_ele_skip)
                new_inx_ele_skip = []
                for i in range(appending_ele+1):
                    new_inx_ele_skip.append(i)
                inx_ele_skip = new_inx_ele_skip
                addtion_val = 0
            else:
                addtion_val +=1
            
        else:
            break
        if len(all_values)-2 == len(inx_ele_skip):
            break
        print('testing:')
        print(all_values)
    return all_values



all_values = [['dusun kabupaten i pauh utara deli , kecamatan sumatera serdang 20374 , hamparan indonesia -', [65, 223, 285, 270], 91.24132404181185], ['500 , telangana , india', [161, 172, 308, 180], 67.32535079575597], ['perak .', [290, 222, 329, 232], 84.68]]
key = "consignee_address"
model_output = {'shipper_name': [['taiko', [66, 133, 99, 141], 96.85], ['chandernagar', [103, 133, 195, 141], 96.28], ['chemicals', [200, 133, 263, 141], 96.94], ['pvt.ltd', [267, 133, 311, 141], 96.83]], 'shipper_address': [['plot', [65, 147, 92, 154], 90.72], ['no', [96, 147, 111, 154], 88.63], ['.', [112, 147, 114, 154], 90.59], ['3-16-724', [118, 147, 157, 154], 91.03], [',', [159, 147, 160, 154], 87.3], ['road', [165, 147, 195, 154], 90.01], ['no.3', [199, 147, 223, 154], 87.93], [',', [223, 147, 226, 154], 62.92], ['sardar', [66, 158, 109, 168], 88.83], ['patel', [113, 158, 146, 167], 88.35], ['colony', [151, 158, 196, 167], 84.39], [',', [197, 158, 198, 167], 84.05], ['trimulgherry', [204, 158, 290, 167], 87.7], [',', [290, 158, 293, 167], 69.21], ['secundrabad', [66, 172, 148, 180], 81.78], ['-', [152, 172, 157, 180], 56.03], ['015', [180, 172, 196, 180], 58.72]], 'consignee_address': [['500', [161, 172, 176, 180], 51.85], [',', [197, 172, 198, 180], 68.68], ['telangana', [203, 172, 270, 180], 72.45], [',', [272, 172, 273, 180], 60.48], ['india', [278, 172, 308, 180], 69.39], ['dusun', [65, 223, 102, 233], 94.6], ['i', [106, 223, 109, 232], 95.33], ['pauh', [113, 223, 141, 232], 95.3], [',', [143, 223, 144, 232], 95.64], ['kecamatan', [149, 223, 218, 232], 95.7], ['hamparan', [223, 223, 285, 232], 95.64], ['perak', [290, 222, 324, 232], 95.45], ['.', [326, 222, 329, 231], 73.91], ['kabupaten', [66, 236, 131, 246], 94.58], ['deli', [136, 236, 159, 245], 94.85], ['serdang', [164, 235, 215, 245], 94.58], [',', [215, 235, 218, 244], 83.1], ['sumatera', [66, 250, 126, 258], 88.76], ['utara', [130, 250, 168, 258], 86.94], ['-', [171, 250, 174, 258], 87.02], ['20374', [178, 250, 204, 258], 84.64], ['indonesia', [66, 263, 124, 270], 81.55]], 'consignee_name': [['pt', [65, 210, 77, 219], 92.39], ['bumi', [81, 210, 109, 218], 92.19], ['karyatama', [113, 210, 183, 218], 84.31], ['raharja', [188, 210, 239, 218], 76.96]], 'notify_party_name': [['same', [66, 337, 96, 344], 96.5], ['as', [100, 337, 113, 344], 96.55], ['consignee', [117, 337, 179, 344], 95.97]], 'vessel_name': [['m.v.', [66, 370, 88, 380], 95.2], ['jutha', [91, 370, 126, 379], 95.1], ['dhammaraksa', [130, 370, 220, 379], 92.15]], 'port_of_discharge': [['ciwandan', [66, 407, 126, 417], 94.6], ['port', [130, 407, 158, 416], 93.58], [',', [159, 407, 160, 416], 91.95], ['indonesia', [165, 407, 225, 416], 90.53]], 'goods_description': [['bentonite', [140, 461, 199, 468], 94.79], ['lumps', [204, 461, 240, 467], 93.27]], 'port_of_loading': [['kakinada', [238, 371, 296, 379], 92.01], [',', [297, 371, 299, 378], 91.25], ['india', [303, 371, 333, 378], 90.69], ['jakarta', [252, 407, 301, 416], 42.4], ['indonesia', [308, 407, 367, 415], 42.37]], 'final_destination': [[',', [302, 407, 303, 415], 66.23]], 'freight_collect_at': [['charter', [204, 497, 254, 505], 54.9], ['party', [258, 496, 292, 505], 55.9]], 'bill_of_lading_number': [['jd', [434, 118, 446, 127], 75.17], ['/', [447, 118, 450, 127], 88.62], ['kkd', [450, 118, 474, 127], 89.69], ['-', [476, 118, 478, 127], 92.09], ['cwn', [480, 118, 506, 127], 92.26], ['/', [506, 118, 509, 127], 92.43], ['001', [510, 118, 526, 127], 93.62]], 'bol_original_or_copy': [['copy', [360, 275, 396, 289], 78.92]], 'bol_original_number': [['3', [234, 648, 238, 654], 51.62], ['(', [239, 647, 242, 653], 63.32], ['three', [245, 647, 273, 653], 79.06], [')', [277, 647, 280, 653], 52.89]], 'gross_weight': [['8000', [398, 461, 418, 468], 85.51], ['m', [422, 461, 429, 468], 85.76], ['/', [433, 461, 436, 468], 84.91], ['tons', [435, 461, 457, 468], 82.16]], 'place_of_issue': [['kakinada', [318, 603, 372, 611], 93.61], ['port', [376, 603, 403, 610], 93.57]], 'bill_of_lading_issue_date': [['22/04/2017', [451, 603, 496, 610], 89.74]], 'signed_By_agent': [['huwasan', [438, 682, 484, 706], 39.43]]}
merge_by_skipping({}, model_output,619, 874, key, all_values)