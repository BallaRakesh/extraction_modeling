
import os
import shutil


def read_image_names(file_path):
    with open(file_path, 'r') as file:
        image_names = file.read().splitlines()
    return image_names



def separate_images(master_folder, image_names, destination_folder1, destination_folder2):


    matched_names = []
    unmatched_names = []

    for image_name in image_names:
        image_path = os.path.join(master_folder, image_name)
        if os.path.exists(image_path):
            shutil.move(image_path, destination_folder1)
            matched_names.append(image_name)
        else:
            unmatched_names.append(image_name)

    return matched_names, unmatched_names


# File paths
master_folder = '/home/ntlpt19/LLM_training/TRAIN/COO/Images'  # Replace with your master images folder path
destination_folder1 = '/home/ntlpt19/LLM_training/TRAIN/COO/train'  # Replace with your destination folder path
destination_folder2 = '/home/ntlpt19/LLM_training/TRAIN/COO/test'  # Replace with your destination folder path

file_path_image_names = '/home/ntlpt19/LLM_training/TRAIN/COO/train.txt'
image_names = read_image_names(file_path_image_names)
print(image_names)


# Step 2: Separate images from the master folder
matched_names, unmatched_names = separate_images(master_folder, image_names, destination_folder1, destination_folder2)


def copy_images_not_in_parent(child_folder, parent_folder, third_folder):
    if not os.path.exists(third_folder):
        os.makedirs(third_folder)

    # Get the list of images in child_folder
    child_images = os.listdir(child_folder)

    # Get the list of images in parent_folder
    parent_images = os.listdir(parent_folder)

    # Compare and copy images
    for image_name in parent_images:
        if image_name not in child_images:
            source_path = os.path.join(parent_folder, image_name)
            destination_path = os.path.join(third_folder, image_name)
            shutil.copyfile(source_path, destination_path)
            print(f"Copied {image_name} to {third_folder}")


# Call the function to copy images
copy_images_not_in_parent(destination_folder1, master_folder, destination_folder2)
