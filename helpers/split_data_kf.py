# Script that splits a directory containing n subfolders with images into
# one that contains k sets with each folder containing a train and validation set

from sklearn.model_selection import KFold
import numpy as np
import shutil
import sys
import os
import glob

if __name__ == "__main__":

	data_dir = sys.argv[1]
	k = int(sys.argv[2])  # Number of folds
	output_directory = sys.argv[3]  # Directory to save the subsets
	
	image_files = []
	labels = []

	# Read images from both subfolders into a list
	for class_label, subfolder in enumerate(os.listdir(data_dir)):
		#print(class_label, subfolder)
		subfolder_path = os.path.join(data_dir, subfolder)
		if os.path.isdir(subfolder_path):
			subfolder_files = glob.glob(os.path.join(subfolder_path, '*.jpeg'))
			image_files += glob.glob(os.path.join(subfolder_path, '*.jpeg'))
			labels += [class_label] * len(subfolder_files)

	print(f"Total no. of images found: {len(image_files)}")

	X = np.array(image_files)
	Y = np.array(labels)

	kf = KFold(n_splits=k, shuffle=True, random_state=42)

	fold_index = 1

	for train_index, test_index in kf.split(image_files):
		#print(f"  Train: index={train_index}")
		#print(f"  Test:  index={test_index}")
		print(f"Size of k-1 train folds: {len(train_index)}, size of 1 fold: {len(test_index)}")

		ring_index = [i for i in Y[test_index] if i==0]
		non_ring_index = [i for i in Y[test_index] if i==1]

		print(f"In each validation split")
		print(f"Size of rings: {len(ring_index)}, Size of nonrings: {len(non_ring_index)}")
		print(f"Creating subset {fold_index}")

		# Create a new directory for the subset
		subset_directory = os.path.join(output_directory, f"subset_{fold_index}")
		os.makedirs(subset_directory, exist_ok=True)

		# Copy images to the subset directory while preserving the folder structure
		for index in train_index:
			source_file = image_files[index]
			relative_path = os.path.relpath(source_file, data_dir)
			destination_file = os.path.join(os.path.join(subset_directory,"train"), relative_path)
			os.makedirs(os.path.dirname(destination_file), exist_ok=True)
			shutil.copyfile(source_file, destination_file)

		for index in test_index:
			source_file = image_files[index]
			relative_path = os.path.relpath(source_file, data_dir)
			destination_file = os.path.join(os.path.join(subset_directory,"val"), relative_path)
			os.makedirs(os.path.dirname(destination_file), exist_ok=True)
			shutil.copyfile(source_file, destination_file)
    	
		fold_index += 1

