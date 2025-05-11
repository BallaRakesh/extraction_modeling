import os
from typing import List, Dict, Union
import numpy as np
from PIL import Image
from datasets import Dataset, DatasetDict
import datasets
import os
from PIL import Image


def convert_image_mode(img: Image.Image, target_mode: str = 'L') -> Image.Image:
    """
    Convert image to the specified mode, preserving as much original information as possible.
    
    Args:
        img (PIL.Image.Image): Input image
        target_mode (str): Desired image mode (default 'L' for grayscale)
    
    Returns:
        PIL.Image.Image: Converted image
    """
    # Map of conversion strategies
    conversion_strategies = {
        '1': lambda x: x.convert('L'),    # 1-bit pixels (black and white)
        'L': lambda x: x,                 # Grayscale 
        'P': lambda x: x.convert('L'),    # Palette-mapped 
        'RGB': lambda x: x.convert('L'),  # Color to grayscale
        'RGBA': lambda x: x.convert('L'), # Color with alpha to grayscale
    }
    
    # Get the current mode
    current_mode = img.mode
    
    # Choose conversion strategy
    if current_mode in conversion_strategies:
        return conversion_strategies[current_mode](img)
    
    # Fallback to direct conversion
    return img.convert(target_mode)

def create_custom_dataset(root_folder: str) -> Dataset:
    """
    Create a custom dataset from a root folder with class-specific subfolders.
    
    Args:
        root_folder (str): Path to the root folder containing class subfolders
    
    Returns:
        Dataset: A single dataset with all images
    """
    # Collect images and labels
    images = []
    labels = []
    class_names = []
    image_names = []
    # Iterate through class folders
    for class_name in sorted(os.listdir(root_folder)):
        class_path = os.path.join(root_folder, class_name)
        
        # Skip if not a directory
        if not os.path.isdir(class_path):
            continue
        
        # Collect images for this class
        class_images = []
        image_name = []
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            print(img_name)
            # Check if it's an image file
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                # Open and convert image mode
                img = Image.open(img_path)
                img = convert_image_mode(img, target_mode='L')  # Ensure grayscale mode
                img.load()  # Ensure the image is valid
                print(img)
                class_images.append(img)
                image_name.append(img_name)
        # Add images and their corresponding labels
        images.extend(class_images)
        image_names.extend(image_name)
        labels.extend([len(class_names)] * len(class_images))
        class_names.append(class_name)

    # Create dataset
    dataset = Dataset.from_dict({
        'image': images,
        'label': labels,
        'image_names':image_names
    })

    # Set features with class names
    dataset = dataset.cast_column(
        'label', 
        datasets.ClassLabel(names=class_names)
    )

    return dataset






def debug_image_modes(root_folder: str):
    """
    Debug and print detailed information about image modes and loading
    
    Args:
        root_folder (str): Path to the root folder containing class subfolders
    """
    print("Image Mode and Loading Debug Script")
    print("-" * 50)
    
    # Counters for different loading scenarios
    mode_counts = {}
    load_types = {}
    
    # Iterate through class folders
    for class_name in sorted(os.listdir(root_folder)):
        class_path = os.path.join(root_folder, class_name)
        
        # Skip if not a directory
        if not os.path.isdir(class_path):
            continue
        
        # Iterate through images in the class folder
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            
            # Check if it's an image file
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                try:
                    # Open the image without loading
                    img = Image.open(img_path)
                    
                    # Detailed image information
                    print(f"\nImage: {img_path}")
                    print(f"Image Type: {type(img)}")
                    print(f"Mode Before Load: {img.mode}")
                    print(f"Size Before Load: {img.size}")
                    
                    # Track mode counts
                    mode_counts[img.mode] = mode_counts.get(img.mode, 0) + 1
                    
                    # Try different loading methods
                    methods = [
                        ("Direct", lambda x: x),
                        ("Load()", lambda x: x.load()),
                        ("Convert('L')", lambda x: x.convert('L')),
                        ("Convert+Load", lambda x: x.convert('L').load())
                    ]
                    
                    for method_name, method in methods:
                        try:
                            # Create a fresh image object
                            test_img = Image.open(img_path)
                            method(test_img)
                            load_types[method_name] = load_types.get(method_name, 0) + 1
                            print(f"Method '{method_name}' successful")
                        except Exception as e:
                            print(f"Method '{method_name}' failed: {e}")
                
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")
    
    # Print summary
    print("\n--- Summary ---")
    print("Mode Counts:")
    for mode, count in mode_counts.items():
        print(f"{mode}: {count}")
    
    print("\nLoading Method Successes:")
    for method, count in load_types.items():
        print(f"{method}: {count}")

# Example usage
if __name__ == "__main__":
    # Specify your root folder path
    train_root_folder = "/home/data_science/geo_testing/Classification_root/BILLS"
    test_root_folder = "/home/data_science/geo_testing/Classification_root/BILLS"
        # Specify your root folder path
    
    # # Run the debug script
    # debug_image_modes(ROOT_FOLDER)
    # exit('OKOKOKOKOKO')
    # Create the dataset
    # custom_dataset = create_custom_dataset(ROOT_FOLDER)
    # Assuming you have your root directories for train and test
    train_dataset = create_custom_dataset(train_root_folder)
    test_dataset = create_custom_dataset(test_root_folder)

    # If you want to combine them into a DatasetDict
    dataset = datasets.DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })
    # Print dataset information
    print(dataset)
    print(dataset['train'].features)
    print(dataset['train']['image'][0])
    print(dataset['train']['label'][0])
    print(dataset['train'].features['label'].names)
    
    exit('OKOKOKO')
    print("\nDataset Features:")
    print(custom_dataset.features)
    print("\nClass Names:")
    print(custom_dataset.features['label'].names)
    
    # Optional: Verify first few samples
    print("\nFirst Image:")
    print(custom_dataset['image'][0])
    print("\nFirst Label:")
    print(custom_dataset['label'][0])