import pandas as pd

# Load the CSV file
file_path = '/home/ntlpt19/Desktop/TF_release/docformer/classification_output.csv'
df = pd.read_csv(file_path)

# Add the new column based on condition
df['correct_incorrect'] = df.apply(lambda row: 1 if row['Ground Truth'] == row['Predicted'] else 0, axis=1)

# Save the updated DataFrame to a new CSV file
output_path = 'classification_updated.csv'
df.to_csv(output_path, index=False)

print("Updated file saved to:", output_path)

# # Load the updated CSV file
# file_path = 'updated_file.csv'  # Path to your updated CSV
# df = pd.read_csv(file_path)

# Calculate class-wise accuracy
class_accuracy = df.groupby('Ground Truth')['correct_incorrect'].apply(lambda x: x.sum() / len(x))

# Save the results to a text file
output_txt_path = 'class_wise_accuracy.txt'
with open(output_txt_path, 'w') as f:
    for class_name, accuracy in class_accuracy.items():
        f.write(f"{class_name}: {accuracy:.2f}\n")

print("Class-wise accuracy saved to:", output_txt_path)

