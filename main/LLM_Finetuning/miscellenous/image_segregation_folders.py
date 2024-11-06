import os
import shutil
import re

def process_images(input_folder, save_folder):
    # Ensure the save folder exists
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # Regular expression to match the pattern "_s_" followed by a number and the file extension
    pattern = re.compile(r'_s_\d+(\.[a-zA-Z]+)$')
    
    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        # Check if the file is an image (you can add more extensions if needed)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # Use regex to remove the "_s_" part followed by a number
            print(filename)
            new_name = pattern.sub(r'\1', filename)
            print(new_name)
            # Define full paths for input and output
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(save_folder, new_name)
            
            # Check if the renamed image already exists in the save folder
            if not os.path.exists(output_path):
                # Copy the file to the save folder with the new name
                shutil.copy(input_path, output_path)
                print(f"Saved: {output_path}")
            else:
                print(f"File already exists: {output_path}")

# Example usage
input_folder = "/home/ntlpt19/LLM_training/TRAIN/CI/test"
save_folder = "/home/ntlpt19/LLM_training/TRAIN/CI/test_filter"


# process_images(input_folder, save_folder)


def get_final_data(parent_folder, child_folder1, child_folder2, third_folder):
    # Get the list of images in child_folder
    child_images1 = os.listdir(child_folder1)
    child_images2 = os.listdir(child_folder2)
    # Get the list of images in parent_folder
    parent_images = os.listdir(parent_folder)

    # Compare and copy images
    for image_name in parent_images:
        if image_name not in child_images1 and image_name not in child_images2:
            source_path = os.path.join(parent_folder, image_name)
            destination_path = os.path.join(third_folder, image_name)
            shutil.copyfile(source_path, destination_path)
            print(f"Copied {image_name} to {third_folder}")
            
parent_folder = '/home/ntlpt19/LLM_training/COO/Images'
child_folder1 = '/home/ntlpt19/LLM_training/TRAIN/COO/TEST/Images'

child_folder2 = '/home/ntlpt19/LLM_training/TRAIN/COO/TRAIN/Images'
third_folder = '/home/ntlpt19/LLM_training/EVAL/COO/Images'

get_final_data(parent_folder, child_folder1, child_folder2, third_folder)