
def are_on_same_line(bbox1, bbox2, min_distance=0, tolerance=10):
    # Check if the vertical distance between the bottom of bbox1 and the top of bbox2 is within the tolerance
    # and if the overall distance is at least min_distance
    return (
        abs(bbox2[1] - bbox1[1]) <= tolerance
        and abs(bbox2[0] - bbox1[2]) >= min_distance
    )

def check_same_line_bbox(bbox_data, line_tolerance=10):
    all_bboxes_ = []
    line_wise_index = {}
    # Create sublists of OCR data in the same line with horizontal tolerance
    idx = 0
    for master_bbox in bbox_data:
        check_flag = False
        bbox = master_bbox
        if bbox not in all_bboxes_:
            all_bboxes_.append(bbox)
            line_wise_index[idx] = [bbox]
            check_flag = True

        if check_flag:
            prev_bbox = line_wise_index[idx][0]
            print("prev_bbox >>>>>>>>>>>", prev_bbox)
            print("line_wise_index[idx]>>>>>>>>>>", line_wise_index[idx])
            for single_bbox in bbox_data:
                print('single_bbox>>>>>>>>>>>>>', single_bbox)
                if are_on_same_line(prev_bbox, single_bbox, tolerance=line_tolerance):
                    print("?????????????????????????????????", single_bbox)
                    print(all_bboxes_)
                    if single_bbox not in all_bboxes_:
                        
                        line_wise_index[idx].append(single_bbox)
                        all_bboxes_.append(single_bbox)

            idx += 1

    return line_wise_index



abc = [['pefrey', [69, 87, 113, 96]], ['460', [70, 76, 83, 83]], ['5', [84, 76, 88, 83]],\
    ['v.f', [91, 76, 104, 83]], ['road', [106, 76, 130, 83]], ['mumbai', [70, 101, 105, 110]],\
        ['400', [105, 101, 119, 108]], ['mansion', [114, 87, 152, 94]], ['004', [120, 101, 133, 108]]]
bboxes = []
for i in abc:
    bboxes.append(i[-1])
print(bboxes)
# bboxes = [[105, 101, 119, 108], [120, 101, 133, 108], [70, 101, 105, 110]]
print(check_same_line_bbox(bboxes))
# final_text = ''
# for index, values in check_same_line_bbox(bboxes).items():
#     for val in values:
#         for j in abc:
#             if j[-1] == val:
#                 final_text += j[0]+" "
# print(final_text)