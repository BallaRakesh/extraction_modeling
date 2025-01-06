import os
import json

def process_txt_files(folder_path, save_path):
    # Ensure the folder path exists
    if not os.path.isdir(folder_path):
        print(f"The folder path '{folder_path}' does not exist.")
        return

    # Create an 'annotations' folder inside the given folder path
    annotations_folder = os.path.join(save_path, "annotations")
    os.makedirs(annotations_folder, exist_ok=True)

    # Iterate through all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)

            # Open and read the JSON data from the file
            with open(file_path, "r") as file:
                file_content = file.read()
                data = eval(file_content)
                file.close()
            # Transform the data into the desired format
            if "word_coordinates" in data:
                transformed_data = {"form": []}

                for idx, word_data in enumerate(data["word_coordinates"]):
                    box = [
                        word_data["x1"],
                        word_data["y1"],
                        word_data["x2"],
                        word_data["y2"],
                    ]
                    form_entry = {
                        "box": box,
                        "text": word_data["word"],
                        "label": "other",
                        "words": [{"box": box, "text": word_data["word"]}],
                        "linking": [],
                        "id": idx,
                    }
                    transformed_data["form"].append(form_entry)
                file_name = file_name.replace("_text", "")
                # Save the transformed data as a JSON file in the 'annotations' folder
                annotation_file_path = os.path.join(annotations_folder, f"{os.path.splitext(file_name)[0]}.json")
                with open(annotation_file_path, "w") as file:
                    json.dump(transformed_data, file, indent=2)

                print(f"Processed and saved: {annotation_file_path}")
            else:
                print(f"'word_coordinates' key not found in file: {file_name}")

# Input the folder path from the user
folder_path = '/media/ntlpt19/5250315B5031474F/geo_code_gpu/Geo_original_code/funsd_data/testing_funsd/OCR'
sv_path = '/media/ntlpt19/5250315B5031474F/geo_code_gpu/Geo_original_code/funsd_data/testing_funsd'
process_txt_files(folder_path, sv_path)
