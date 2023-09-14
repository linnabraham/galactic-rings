#!/bin/env python


import sys
import os
import glob
import shutil
import numpy as np
np.random.seed(42)

if __name__=="__main__":


    base_dir = sys.argv[1]
    for class_label in os.listdir(base_dir):
        if class_label == "Rings":
            pass
        elif class_label == "NonRings":
            data_dir = os.path.join(base_dir,class_label) 
            print(data_dir)
            #sys.exit(0)
            image_files = []

            image_files += glob.glob(os.path.join(data_dir, '*.jpeg'))

            print(f"Total no. of images found: {len(image_files)}")

            sets = [int(arg) for arg in sys.argv[2:]]


            for num in sets:
                subset = np.random.choice(image_files, num, replace=False)
                print(len(subset))
                subset_dir = f"set_{num}"
                os.makedirs(subset_dir)
                ring_dir = os.path.join(base_dir, "Rings")
                nonring_dir = os.path.join(subset_dir, "NonRings")
                os.makedirs(nonring_dir)
                shutil.copytree(ring_dir, os.path.join(subset_dir,"Rings"))
                for path in subset:
                    base_path = os.path.basename(path)
                    dest_file = os.path.join(nonring_dir,base_path)
                    shutil.copyfile(path, dest_file)


