import os
import json
from ast import literal_eval

ocr_file = '/home/gpu1admin/rakesh/ingram_rakesh_data/ocr'
root_path = '/home/gpu1admin/rakesh/ingram_rakesh_data'
custom_path = os.path.join(root_path, "custom_data")
if not os.path.exists(custom_path):
	os.mkdir(custom_path)
all_words_path = os.path.join(custom_path, "all_words")
if not os.path.exists(all_words_path):
	os.mkdir(all_words_path)

# file_save_numb = 9
file_save_numb = 23
# Assuming 'filenames' contains the list of filenames
filtered_filenames = [filename for filename in os.listdir(ocr_file) if 'textAndCoordinates' in filename]

for j in filtered_filenames:
    print(j[0:-file_save_numb])
    
    # Open the file in read mode ('r')
    if j.endswith('.txt'):
        with open(os.path.join(ocr_file, j), 'r') as file:
            # Read the contents of the file
            # word_coordinates = eval(file.read())
            word_coordinates = literal_eval(file.read())
        print(word_coordinates)

    else:
        with open(os.path.join(ocr_file, j), "r") as f:
            word_coordinates = json.load(f)#['word_coordinates']
        f.close()
    cou = 1
    final = {}
    import json
    for i in word_coordinates:
        final[cou]={'text':i['word'], 'bbox':[i['x1'],i['y1'],i['x2'], i['y2']]}
        cou+=1
    # print(final)
    file_name = os.path.join(all_words_path, j[0:-file_save_numb]+'.json')
    with open(file_name, "w") as json_file:
        json.dump(final, json_file)