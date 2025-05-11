import os


def remove_lines_by_ids(file_path, ids_to_remove):
    for files_ in os.listdir(file_path):
        # Read the content of the file
        print('%%%%%%%%% >', files_)
        with open(os.path.join(file_path, files_), 'r') as file:
            lines = file.readlines()

        # Filter out lines with IDs in the list
        filtered_lines = [line for line in lines if int(line.split()[0]) not in ids_to_remove]

        # Write the filtered lines back to the file (or a new file)
        with open(os.path.join(file_path, files_), 'w') as file:
            file.writelines(filtered_lines)

# Example usage

if __name__ == '__main__':
    labels_folder = '/home/ntlpt19/Downloads/Final_Delivery_Training_itter_5/Eval_data/CS_eval/Labels'
    # keys_to_delete = {
    #     "CI": [75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98]
    # }
    keys_to_delete = {
        "CI": [0, 3, 6, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19, 20, 21, 26, 27, 28, 29, 33, 34, 36, 37, 39, 41, 42, 44, 52, 53, 55, 59, 60, 63, 67, 68, 70, 71, 72, 73, 74],
        "BOL": [0, 7, 8, 11, 12, 13, 17, 24, 27, 28, 31, 32, 36, 40, 41, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53],
        # "PI": [66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100],
        "IC": [3, 4, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 27, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64],
        "AWB": [0, 1, 2, 5, 6, 9, 12, 15, 20, 23, 25, 27, 30, 33, 34, 36, 37, 38, 42, 43, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55],
        "BOE": [6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 25, 28, 29, 34, 35, 37, 38, 39],
        "COO": [2, 13, 14, 15, 16, 19, 20, 25, 30],
        "CS": [4, 15, 16, 17, 18, 21, 22, 25, 27, 29, 33, 36, 38, 39, 41, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64],
        "PL": [0, 1, 4, 5, 6, 9, 10, 11, 12, 13, 14, 21, 22, 24, 27, 28, 31, 32, 33, 35, 39, 40, 41, 42, 45, 46, 47],
        "PO": [6, 8, 13, 15, 17, 18, 20, 21, 22, 23, 24, 25, 29, 30, 31, 32, 33, 34, 35, 38, 39, 41, 43],
        "PI": [2, 3, 12, 13, 14, 15, 16, 17, 21, 24, 25, 29, 30, 31, 32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 47, 48, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]
    }
    document_class = 'CS'
    list_of_keys = keys_to_delete.get(document_class, [])
    remove_lines_by_ids(labels_folder, list_of_keys)
