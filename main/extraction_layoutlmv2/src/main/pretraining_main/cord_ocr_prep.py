import json
import os

def gen_ocr(file_path):
    with open(file_path, 'r') as f:
        json_data = json.load(f)
    # Transform the data
    output = []
    for line in json_data["valid_line"]:
        for word in line["words"]:
            quad = word["quad"]
            
            x1, y1 = quad["x1"], quad["y1"]
            x2, y2 = quad["x2"], quad["y2"]
            x3, y3 = quad["x3"], quad["y3"]
            x4, y4 = quad["x4"], quad["y4"]
            
            # Calculate rectangular bounding box
            x_min = min(x1, x2, x3, x4)
            y_min = min(y1, y2, y3, y4)
            x_max = max(x1, x2, x3, x4)
            y_max = max(y1, y2, y3, y4)
            
            x1_, y1_, x2_, y2_ = x_min, y_min, x_max, y_max #quad["x1"], quad["y1"], quad["x4"], quad["y4"]
            
            # Compute bounding box
            left = x1_ #min(x1, quad["x4"])
            top = y1_ #min(y1, quad["y2"])
            width = abs(x2_ - x1_)
            height = abs(y2_ - y1_)
            
            # Append to the output
            output.append({
                "word": word["text"],
                "left": left,
                "top": top,
                "width": width,
                "height": height,
                "x1": x1_,
                "y1": y1_,
                "x2": x2_,
                "y2": y2_
            })



    # Process "repeating_symbol" key
    for symbol_group in json_data.get("repeating_symbol", []):
        for symbol in symbol_group:
            quad = symbol["quad"]
            
            # x1, y1, x2, y2 = quad["x1"], quad["y1"], quad["x2"], quad["y2"]

            x1, y1 = quad["x1"], quad["y1"]
            x2, y2 = quad["x2"], quad["y2"]
            x3, y3 = quad["x3"], quad["y3"]
            x4, y4 = quad["x4"], quad["y4"]
            
            # Calculate rectangular bounding box
            x_min = min(x1, x2, x3, x4)
            y_min = min(y1, y2, y3, y4)
            x_max = max(x1, x2, x3, x4)
            y_max = max(y1, y2, y3, y4)
            
            x1_, y1_, x2_, y2_ = x_min, y_min, x_max, y_max #quad["x1"], quad["y1"], quad["x4"], quad["y4"]
            
            # Compute bounding box
            left = x1_ #min(x1, quad["x4"])
            top = y1_ #min(y1, quad["y2"])
            width = abs(x2_ - x1_)
            height = abs(y2_ - y1_)
            
            # Append to the output
            output.append({
                "word": symbol["text"],
                "left": left,
                "top": top,
                "width": width,
                "height": height,
                "x1": x1_,
                "y1": y1_,
                "x2": x2_,
                "y2": y2_
            })


    # Print the result
    print(output)
    return output
        # Save to a file if needed
        # with open(os.path.join(ocr_folder, "output.json"), "w") as f:
        #     json.dump(output, f, indent=4)
        
if __name__ == "__main__":
    file_path = '/home/data_science/geo_testing/lmv2_code/local_cord_v2_dataset/validation/Labels'
    ocr_folder = '/home/data_science/geo_testing/lmv2_code/local_cord_v2_dataset/validation/OCR'
    for files in os.listdir(file_path):
        wc_cords = gen_ocr(os.path.join(file_path, files))
        files = files.replace('label', 'image')
        text_file_path = os.path.join(ocr_folder, f"{files[:-5]}_text.txt")
        print(text_file_path)
        # Save the extracted text to the text file
        with open(text_file_path, 'w') as text_file:
            text_file.write(str(wc_cords))