import os

import numpy as np
import pandas as pd
import shutil

#global params
img_dir="data/data_split2"
new_img_dir="data/data_split3"

#create required directories
if not os.path.isdir(new_img_dir) :
    shutil.copytree(img_dir,new_img_dir)
def upsampler(classes_id,n_up) :

    #step 1 : find the file in the specific class
    for keyname in os.listdir(f"{new_img_dir}/train/images") :

        # location=img_path[len(self.img_dir)+1:len(self.img_dir)+3]

        # print("loc",location)
        # location=re.search("/[0-9][0-9]/",img_path).group()[1:-1]
        try :
            category_id, new_x, new_y, new_width, new_height = np.loadtxt(f"{new_img_dir}/train/labels/{keyname}",
                                                                unpack=True)
        except:
            print(f"the txt file was not found for {keyname}")

        if int(category_id) in classes_id :
            for i in range(n_up) :
                #we then need to make copies of this image
                shutil.copy(f"{new_img_dir}/train/images/{keyname[:-4]}.jpg",f"{new_img_dir}/train/images/{keyname[:-4]}({i}).jpg")
                shutil.copy(f"{new_img_dir}/train/labels/{keyname[:-4]}.txt",f"{new_img_dir}/train/labels/{keyname[:-4]}({i}).txt")


upsampler([5,8,12],1)