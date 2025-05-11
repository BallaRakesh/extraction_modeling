import os


def remove_lines_by_ids(file_path, ids_to_remove):
    for files_ in os.listdir(file_path):
        # Read the content of the file
        with open(os.path.join(file_path, files_), 'r') as file:
            lines = file.readlines()

        # Filter out lines with IDs in the list
        filtered_lines = [line for line in lines if int(line.split()[0]) not in ids_to_remove]

        # Write the filtered lines back to the file (or a new file)
        with open(os.path.join(file_path, files_), 'w') as file:
            file.writelines(filtered_lines)

# Example usage

if __name__ == '__main__':
    labels_folder = '/home/ntlpt19/Downloads/Evaluation_Data/finalEvaluationEvalData_itter3/BEFORE/CI_105_extra/labels'
    keys_to_delete = {
        "CI": [75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98]
    }
    document_class = 'CI'
    list_of_keys = keys_to_delete.get(document_class, [])
    remove_lines_by_ids(labels_folder, list_of_keys)
