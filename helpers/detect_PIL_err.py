#!/bin/env python
# This script is used to check the integrity of the images used for training or prediction
# If PIL is not able to open an image, our training or prediction gets interrupted unexpectedly
import os
import sys
from PIL import Image

folder_path = sys.argv[1]
extensions = []
for fldr in os.listdir(folder_path):
    sub_folder_path = os.path.join(folder_path, fldr)
    for filee in os.listdir(sub_folder_path):
        file_path = os.path.join(sub_folder_path, filee)
        #print('** Path: {}  **'.format(file_path), end="\r", flush=True)
        #print(".",end="")
        try:
            im = Image.open(file_path)
            rgb_im = im.convert('RGB')
        except:
            print(f"{file_path} is broken")
        else:
            if filee.split('.')[1] not in extensions:
                extensions.append(filee.split('.')[1])
