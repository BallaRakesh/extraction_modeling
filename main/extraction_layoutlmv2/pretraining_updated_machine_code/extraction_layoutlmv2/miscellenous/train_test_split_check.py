import os
import pandas as pd

def remove_extension(filename):
    return os.path.splitext(filename)[0]

def create_dataframe(folder_path):
    files = os.listdir(folder_path)
    filenames_without_extension = [remove_extension(file) for file in files]
    counts = [filenames_without_extension.count(name) for name in filenames_without_extension]

    return pd.DataFrame(
        {"image_name": filenames_without_extension, "count": counts}
    )


def remove_suffix(filename):
    return filename.replace("_s_1", "").replace("_s_2", "").replace("_s_3", "").replace("_s_4", "").replace("_s_5", "").replace("_s_6", "").replace("_s_7", "").replace("_s_8", "").replace("_s_9", "").replace("_s_10", "").replace("_s_11", "")
    
def create_data_frame(folder_path):
    files = os.listdir(folder_path)
    filenames_without_extension = [remove_extension(file) for file in files]
    filenames_without_suffix = [remove_suffix(name) for name in filenames_without_extension]

    df = pd.DataFrame({"image_name": filenames_without_suffix})
    df['count'] = df.groupby('image_name')['image_name'].transform('size')
    df = df.drop_duplicates().reset_index(drop=True)

    return df




def find_common_elements(file_path1, file_path2):
    df1 = pd.read_csv(file_path1)
    df2 = pd.read_csv(file_path2)

    # Assuming the second column in both files is at index 1 (0-based indexing)
    column_index = 1
    if common_elements := set(df1.iloc[:, column_index]).intersection(
        set(df2.iloc[:, column_index])
    ):
        common_df = pd.DataFrame({"Common_Images_test_train": list(common_elements)})
        print("Common elements found:")
        print(common_df)
        return common_df
    else:
        print("No common elements found.")
        return None




#give the ROOT path name  

fols_name='/home/ntlpt19/Downloads/Trade_finance_imp_stage_2/COO_ROOT/ROT'

folder_path = os.path.join(fols_name,'Labels')
df = create_dataframe(folder_path)
print(df)
df.to_csv(os.path.join(fols_name,'cs_phase1'))     #############complete data


folder_path = os.path.join(fols_name,'train')
df = create_data_frame(folder_path)
print(df)
df.to_csv(os.path.join(fols_name,'cs_phase1_train'))  #################train data


folder_path = os.path.join(fols_name,'test')
df = create_data_frame(folder_path)
print(df)
df.to_csv(os.path.join(fols_name,'cs_phase1_test'))  ########test data

#give the path name where cs_phase1_train and cs_phase1_test has beed saved

file_path1 = os.path.join(fols_name,'cs_phase1_test')
file_path2 = os.path.join(fols_name,'cs_phase1_train')

common_df = find_common_elements(file_path1, file_path2)
if common_df is not None:
    common_df.to_csv('cs_phase1_common_train1_and_train2.csv', index=True)