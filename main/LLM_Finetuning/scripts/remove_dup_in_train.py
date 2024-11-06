import os
import re

def rename_and_remove_duplicates(directory):
    """
    Renames files in the specified directory by removing the "s_intvalue" part
    from the filenames and deletes duplicates, keeping only unique images.

    :param directory: Path to the directory containing the image files.
    """
    # Dictionary to keep track of unique base filenames
    unique_files = {}

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Check if the filename matches the pattern 'Purchase_Order_1_3_s_x.png'
        match = re.match(r'(.*)_s_\d+(\.png)', filename)
        if match:
            # Construct the base filename
            base_filename = match.group(1) + match.group(2)
            
            # Full path of the current file
            old_file_path = os.path.join(directory, filename)
            
            # Check if this base filename has been seen before
            if base_filename not in unique_files:
                # If not, add it to the dictionary and rename the file
                unique_files[base_filename] = old_file_path
                new_file_path = os.path.join(directory, base_filename)
                os.rename(old_file_path, new_file_path)
                print(f'Renamed: {filename} to {base_filename}')
            else:
                # If it has been seen before, it is a duplicate and should be deleted
                os.remove(old_file_path)
                print(f'Removed duplicate: {filename}')

if __name__ == "__main__":
    # Set the directory where your files are located
    directory = '/home/ntlpt-42/Downloads/train'
    
    # Call the function to rename files and remove duplicates
    rename_and_remove_duplicates(directory)
