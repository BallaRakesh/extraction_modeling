import os
import shutil
from configparser import ConfigParser

if __name__ == "__main__":
    folder_path: str = "/home/tarun/Downloads/FooterData/combined"
    
    print(f"folder_path: {folder_path}")

    train_file_path: str = f"{folder_path}/train.txt"
    test_file_path: str = f"{folder_path}/test.txt"

    main_image_folder: str = f"{folder_path}/Images"


    with open(f"{train_file_path}", "r") as f:
        train_data =  f.readlines()

    train_data = [file.strip() for file in train_data]
    print(train_data)


    image_names_train: list = [f'''{file.split(".txt")[0]}.png''' for file in train_data]
    print(f"Image names: {image_names_train}")

    with open(f"{test_file_path}", "r") as f:
        test_data =  f.readlines()

    test_data = [file.strip() for file in test_data]
    print(train_data)


    image_names_test: list = [f'''{file.split(".txt")[0]}.png''' for file in test_data]
    print(f"Image names: {image_names_test}")


    # create a new folder
    # train_images
    if not os.path.exists(f"{folder_path}/train_images"):
        os.makedirs(f"{folder_path}/train_images")


    # moving train images
    [
        shutil.copy(
            f"{main_image_folder}/{image}", f"{folder_path}/train_images"
        )
        for image in image_names_train
    ]

    # test_images
    if not os.path.exists(f"{folder_path}/test_images"):
        os.makedirs(f"{folder_path}/test_images")

    
    # moving test images
    [
    shutil.copy(
        f"{main_image_folder}/{image}", f"{folder_path}/test_images"
    )
    for image in image_names_test
    ]