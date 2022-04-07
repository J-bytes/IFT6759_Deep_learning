import os

import numpy as np
import pandas as pd
import shutil

#global params
img_dir="data/data_split2"
new_img_dir="data/data_upsampled"

#create required directories
# if not os.path.isdir(new_img_dir) :
#     shutil.copytree(img_dir,new_img_dir)
def upsampler(n_up) :

    count = 0

    #step 1 : find the file in the specific class
    for keyname in os.listdir(f"{new_img_dir}/train/images") :

        # location=img_path[len(self.img_dir)+1:len(self.img_dir)+3]

        # print("loc",location)
        # location=re.search("/[0-9][0-9]/",img_path).group()[1:-1]
        location_file = f"{new_img_dir}/train/labels/{keyname[-40:-4]}.txt"
        if os.path.exists(location_file):
            with open(location_file) as f:
                category_id = f.readlines()[0].split()[0]
                f.close()
            if int(category_id) in [5, 8, 12] :
                for i in range(n_up) :
                    count+=1
                    #we then need to make copies of this image
                    shutil.copy(f"{new_img_dir}/train/images/{keyname[:-4]}.jpg",f"{new_img_dir}/train/images/{keyname[:-4]}({i}).jpg")
                    shutil.copy(f"{new_img_dir}/train/labels/{keyname[:-4]}.txt",f"{new_img_dir}/train/labels/{keyname[:-4]}({i}).txt")

    print(count)
upsampler(n_up=1)