"""
* *********************************************************************************
* Number Theory S/W Pvt. Ltd CONFIDENTIAL                                      *
* *
* [2016] - [2023] Number Theory S/W Pvt. Ltd Incorporated                       *
* All Rights Reserved.                                                          *
* *
* NOTICE:  All information contained herein is, and remains                     *
* the property of Number Theory S/W Pvt. Ltd Incorporated and its suppliers,    *
* if any.  The intellectual and technical concepts contained                    *
* herein are proprietary to Number Theory S/W Pvt. Ltd Incorporated             *
* and its suppliers and may be covered by India. and Foreign Patents,           *
* patents in process, and are protected by trade secret or copyright law.       *
* Dissemination of this information or reproduction of this material            *
* is strictly forbidden unless prior written permission is obtained             *
* from Number Theory S/W Pvt. Ltd Incorporated.                                 *
* *
* *********************************************************************************
"""

import os
import shutil
import pandas as pd
from configparser import ConfigParser

if __name__ == "__main__":

    # reading of the configuration
    product_config = ConfigParser()
    product_config.read("src/main/config/config.ini")
    folder_path: str = product_config["DataSource"]["folder_path"]
    print(f"folder_path: {folder_path}")

    # train and test file paths
    train_file_path: str = "training_set.csv"
    test_file_path: str = "testing_set.csv"

    # dataframe -> Series => List
    # Creating train images list
    train_images = pd.read_csv(train_file_path)["image_path"].tolist()
    assert len(train_images) > 0
    train_images = [image.strip() for image in train_images]    
    print(f"Number of training images: {len(train_images)}")

    # Creating test images list
    test_images = pd.read_csv(test_file_path)["image_path"]
    assert len(test_images) > 0
    test_images = [image.strip() for image in test_images]
    print(f"Number of test images: {len(test_images)}")

    # check for the overlap in train and test images
    assert not set(train_images).intersection(set(test_images))
        
    # create a new folder
    # train_images
    if not os.path.exists(f"{folder_path}/train_images"):
        os.makedirs(f"{folder_path}/train_images")

    # moving train images
    [
        shutil.copy(
            f"{image}", f"{folder_path}/train_images"
        )
        for image in train_images
    ]

    # test_images
    if not os.path.exists(f"{folder_path}/test_images"):
        os.makedirs(f"{folder_path}/test_images")


    # moving test images
    [
    shutil.copy(
        f"{image}", f"{folder_path}/test_images"
    )
    for image in test_images
    ]