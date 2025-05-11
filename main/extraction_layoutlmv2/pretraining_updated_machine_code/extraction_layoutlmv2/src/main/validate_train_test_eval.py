import os
import glob
from configparser import ConfigParser
from src.main.extraction.config.prod_mapping import product_code_map, document_code_map

if __name__ == "__main__":
    
    # product config
    product_config = ConfigParser()
    product_config.read("src/main/extraction/config/config.ini")

    prod_code = product_code_map[product_config["Product"]["code"]]
    doc_code = document_code_map[product_config["Product"]["document_code"]]

    # data folder path
    product_wise_folder = ConfigParser()
    product_wise_folder.read("src/main/extraction/config/prod.ini")
    folder_path = product_wise_folder[prod_code][doc_code]

    print("==================Trade Finance Solutions===================")
    print(f"Product Code: {prod_code}")
    print(f"Documenry Code: {doc_code}")
    print(f"folder_path: {folder_path}")


    train_folder: str = f"{folder_path}/train_images"
    test_folder: str = f"{folder_path}/test_images"
    eval_folder: str = f"{folder_path}/eval_images"


    train_images = [image.split("/")[-1] for image in glob.glob(os.path.join(train_folder,"*.png"))]
    test_images = [image.split("/")[-1] for image in glob.glob(os.path.join(test_folder, "*.png"))]
    eval_images = [image.split("/")[-1] for image in glob.glob(os.path.join(eval_folder,"*.png"))]

    assert train_images
    assert test_images
    assert eval_images

    
    print(f"Number of training images: {len(train_images)}")
    print(f"Number of test images: {len(test_images)}")
    print(f"Number of eval images: {len(eval_images)}")
          
    
    # This assertion is to check if any duplicate images
    assert len(train_images) == len(set(train_images))
    assert len(test_images) == len(set(test_images))
    assert len(eval_images) == len(set(eval_images))

    # This assertion is to check any overlapping in train, test and eval images    
    aa = set(train_images).intersection(set(test_images)).intersection(set(eval_images))
    print(aa)
    
