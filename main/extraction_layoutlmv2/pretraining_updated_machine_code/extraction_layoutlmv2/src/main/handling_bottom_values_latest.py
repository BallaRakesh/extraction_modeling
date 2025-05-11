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
import json
from datetime import datetime


# Get the file name of the executed Python script
# file_name = os.path.basename(__file__)
def check_and_modify_labels(json_data, label):
    if label in json_data and bottom_value in json_data:
        json_data[bottom_value][0][0] = "~ " + json_data[bottom_value][0][0]
        joined_value = json_data[label] + json_data[bottom_value]
        json_data[label] = joined_value
        del json_data[bottom_value]
    else:
        if bottom_value in json_data:
            json_data[label] = json_data.pop(bottom_value)

    return json_data


if __name__ == "__main__":
    start_time = datetime.now()
    folder_path: str = r"/home/ntlpt19/Downloads/Trade_finance_imp_stage_2/CS_NEW_ROOT/CS_ROOT_stage2/CS_ROOT_108"
    top_value = "drawer_bank_address"
    bottom_value = "drawer_bank_bottom_address"

    result_path = os.path.join(folder_path, "Results_CS_validated")
    data_path = os.path.join(folder_path, "New_Master_Data_Merged")

    data_files = os.listdir(data_path)
    result_files = os.listdir(result_path)
    for file in data_files:
        print("file name is:", file)
        # print("resulted filename is", predicted_files)
        # continue
        with open(os.path.join(data_path, file), "r") as f:
            file_path = os.path.join(data_path, file)
            labels = json.load(f)
            modified_data = check_and_modify_labels(labels, top_value)
            with open(file_path, "w") as modified_file:
                json.dump(modified_data, modified_file)

    for file in result_files:
        print("file name is:", file)
        # print("resulted filename is", predicted_files)
        # continue
        try:
            print(os.path.join(result_path, file[0:-5] + "1.txt"))
            with open(os.path.join(result_path, file[0:-5] + "1.txt"), "r") as f2:
                file_result_path = os.path.join(result_path, file[0:-5] + "1.txt")
                predicted = json.load(f2)
                modified_data = check_and_modify_labels(predicted, top_value)
                with open(file_result_path, "w") as modified_file:
                    json.dump(modified_data, modified_file)
        except:
            # print("some problem opening file")
            try:
                print(f'''printing the path: {result_path, file[0:-5] + "_s_11.txt"}''')
                with open(os.path.join(result_path, file[0:-5] + "_s_11.txt"), "r") as f2:
                    file__result_path = os.path.join(result_path, file[0:-5] + "_s_11.txt")
                    predicted = json.load(f2)
                    modified_data = check_and_modify_labels(predicted, top_value)
                    with open(file__result_path, "w") as modified_file:
                        json.dump(modified_data, modified_file)
            # print("opened")
            except:
                print("still not opened")
                continue
