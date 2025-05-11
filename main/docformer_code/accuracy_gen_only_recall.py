import pandas as pd

# Load the CSV file
file_path = '/home/ntlpt19/Desktop/TF_release/docformer/aggregated_results.csv'
df = pd.read_csv(file_path)
class_accuracy = df.groupby('gt_label')['pred_1_0'].apply(lambda x: x.sum() / len(x))

# Save the results to a text file
output_txt_path = 'class_wise_accuracy.txt'
with open(output_txt_path, 'w') as f:
    for class_name, accuracy in class_accuracy.items():
        f.write(f"{class_name}: {accuracy:.2f}\n")

print("Class-wise accuracy saved to:", output_txt_path)

