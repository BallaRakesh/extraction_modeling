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
import glob
import random
import shutil
import os

if __name__ == "__main__":
	sample_size: int = 100
	image_folder: str = ""
	labels_folder: str = ""
	output_folder: str = ""

	sampled_image_folder = random.choices(population=glob.glob(f"{image_folder}/*.png"),
	                                      k=sample_size)

	if not os.path.exists(f"{output_folder}/Images"):
		os.makedirs(f"{output_folder}/Images")
	if not os.path.exists(f"{output_folder}/Labels"):
		os.makedirs(f"{output_folder}/Labels")

	for image in sampled_image_folder:
		image_name: str = image.split("/")[-1].split(".")[0]
		shutil.copy(image, f"{output_folder}/Images")
		shutil.copy(f"{labels_folder}/{image_name}.txt", f"{output_folder}/Images")
