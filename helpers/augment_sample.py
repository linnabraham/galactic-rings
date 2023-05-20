
## Rotate and save color image
# pass the directory containing jpg images and the output folder to write images
# and the step size as arguments

import sys
from PIL import Image
import glob
import os

imgFile = sys.argv[1]
output = sys.argv[2]
step = int(sys.argv[3])

# print(imgFile, output, step)

name, ext = os.path.splitext(os.path.basename(imgFile))
img  = Image.open(imgFile)

try:
    # Rotate and save
    for i in range(0,360,step):
            title = name +'-'+str(i)+ext
            rot = img.rotate(i, expand=0)
            filename = os.path.join(output,title)
            # print(filename)
            rot.save(filename)
except:
    print("error")
