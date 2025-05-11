import pandas as pd

# Load the CSV file
csv_file_path = "/home/ntlpt19/Desktop/TF_release/docformer/classification_output.csv"  # Replace with the path to your CSV file
output_file_path = "classification_chunk1_scores.csv"

# Read the CSV into a DataFrame
df = pd.read_csv(csv_file_path)

# Filter rows where 'Image Name' contains '_S_0' or does not contain 'S_'
filtered_df = df[df['Image Name'].str.contains(r'_S_0\.png$', regex=True) | ~df['Image Name'].str.contains(r'_S_', regex=True)]

filtered_df['correct_incorrect'] = filtered_df.apply(lambda row: 1 if row['Ground Truth'] == row['Predicted'] else 0, axis=1)

class_accuracy = filtered_df.groupby('Ground Truth')['correct_incorrect'].apply(lambda x: x.sum() / len(x))

# Save the results to a text file
output_txt_path = 'class_wise_accuracy_chunk1.txt'
with open(output_txt_path, 'w') as f:
    for class_name, accuracy in class_accuracy.items():
        f.write(f"{class_name}: {accuracy:.2f}\n")

print("Class-wise accuracy saved to:", output_txt_path)



# Save the filtered DataFrame to a new CSV file
filtered_df.to_csv(output_file_path, index=False)

print(f"Filtered rows saved to: {output_file_path}")
