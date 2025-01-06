import os
import shutil
import random

def segregate_samples(image_folder, label_folder, ocr_folder, output_folder, sample_count=10):
    # Create separate output directories for images, labels, and OCR
    image_output_folder = os.path.join(output_folder, 'images')
    label_output_folder = os.path.join(output_folder, 'labels')
    ocr_output_folder = os.path.join(output_folder, 'ocr')

    os.makedirs(image_output_folder, exist_ok=True)
    os.makedirs(label_output_folder, exist_ok=True)
    os.makedirs(ocr_output_folder, exist_ok=True)

    # List all files in the image folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    
    # Select a random sample of files
    sampled_files = random.sample(image_files, min(sample_count, len(image_files)))

    for file_name in sampled_files:
        # Create the corresponding label and OCR filenames
        base_name = os.path.splitext(file_name)[0]
        label_file = base_name + '.txt'
        ocr_file = base_name + '_textAndCoordinates.txt'
        
        # Copy image
        shutil.copy(os.path.join(image_folder, file_name), os.path.join(image_output_folder, file_name))
        print(f'Copied {file_name} to {image_output_folder}')
        
        # Copy label if exists
        if os.path.exists(os.path.join(label_folder, label_file)):
            shutil.copy(os.path.join(label_folder, label_file), os.path.join(label_output_folder, label_file))
            print(f'Copied {label_file} to {label_output_folder}')
        else:
            print(f'{label_file} does not exist in {label_folder}')

        # Copy OCR file if exists
        if os.path.exists(os.path.join(ocr_folder, ocr_file)):
            shutil.copy(os.path.join(ocr_folder, ocr_file), os.path.join(ocr_output_folder, ocr_file))
            print(f'Copied {ocr_file} to {ocr_output_folder}')
        else:
            print(f'{ocr_file} does not exist in {ocr_folder}')

# Specify your folders here
image_folder = '/home/gpu1admin/rakesh/ingram_rakesh_data/images'
label_folder = '/home/gpu1admin/rakesh/ingram_rakesh_data/labels'
ocr_folder = '/home/gpu1admin/rakesh/ingram_rakesh_data/OCR'
output_folder = '/home/gpu1admin/rakesh/geo_testing/data_oct21'

# Call the function to segregate samples
segregate_samples(image_folder, label_folder, ocr_folder, output_folder)
