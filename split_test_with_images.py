import pandas as pd

def split_csv_by_answer(input_file, output_yes_no_file, output_other_file):
    # Load the CSV file
    df = pd.read_csv(input_file)

    # Split the data into two parts
    yes_no_df = df[df['answer'].str.lower().isin(['yes', 'no'])]
    other_df = df[~df['answer'].str.lower().isin(['yes', 'no'])]

    # Save the two parts into separate files
    yes_no_df.to_csv(output_yes_no_file, index=False)
    other_df.to_csv(output_other_file, index=False)

# File paths
input_file = 'data/vqa_rad/test_with_images.csv'
output_yes_no_file = 'data/vqa_rad/test_yes_no.csv'
output_other_file = 'data/vqa_rad/test_other.csv'

# Split the file
split_csv_by_answer(input_file, output_yes_no_file, output_other_file)

print("Files have been split successfully!")
