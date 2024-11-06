import shutil
from imagededup.methods import *
from imagededup.utils import plot_duplicates
import os
 

def remove_duplicate_images(master_directory, unique_docs_dir, unique_duplicates_images_directory):
    phasher = PHash()
    encodings = phasher.encode_images(image_dir=master_directory)
    duplicates = phasher.find_duplicates(encoding_map=encodings)
    unique_images = []
    duplicate_images = []
    SameDuplicatesInMoreThanOneImage = {}
    for key in duplicates.keys():
        if key not in duplicate_images:
            shutil.copy(os.path.join(master_directory, key), os.path.join(unique_docs_dir, key))
            unique_images.append(key)
            for image in duplicates[key]:
                shutil.copy(os.path.join(master_directory, image), os.path.join(unique_duplicates_images_directory, image))
                if image in duplicate_images:
                    SameDuplicatesInMoreThanOneImage.setdefault(image, []).append(key)
                else:
                    duplicate_images.append(image)

if __name__ == "__main__":
    master_directory = 'IMAGE DIRECTORY'
    output_path = master_directory + '/train_test_split'
    unique_duplicates_images_directory = os.path.join(output_path, 'Test')
    unique_docs_dir = os.path.join(output_path, "Train")
    for path in [unique_duplicates_images_directory, unique_docs_dir]:
        os.makedirs(path, exist_ok=True)
    
    remove_duplicate_images(master_directory, unique_docs_dir, unique_duplicates_images_directory)
