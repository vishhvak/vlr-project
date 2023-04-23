import os

# Path to folder containing images
img_folder = 'MJ_train_5000_Adithya'

# Path to text file with labels
label_file = 'MJ_train_5000_labels_Adithya.txt'

# Path to output text file
output_file = 'labels.txt'

# Read in labels from text file
with open(label_file, 'r') as f:
    labels = f.read().splitlines()

# Get list of image filenames in folder
img_filenames = sorted(os.listdir(img_folder))

# Generate lines for output text file
output_lines = [f"./content/{img_folder}/{img_filename},{label}\n" for img_filename, label in zip(img_filenames, labels)]

# Write output lines to text file
with open(output_file, 'w') as f:
    f.writelines(output_lines)