import os
from cv2 import split

import numpy as np
import pandas as pd
import shutil

#global params
label_dir="data/train_augmented/train_augmented/labels"

def changer() :

    to_map = {0: 12, 1:5, 2: 8}

    for keyname in os.listdir(label_dir) :
        if 'rf' in keyname:
            location_file = f"{label_dir}/{keyname}"
            with open(location_file) as f:
                txt_content =  f.readlines()[0]
                category_id = txt_content.split()[0]
                # print('cat', category_id)
                # print('cont', txt_content)
                if int(category_id) in [0, 1, 2]:
                    new_cat = to_map[int(category_id)]
                    txt_content_new = []
                    txt_content_new.append(str(new_cat))
                    txt_content_new.extend(txt_content.split()[1:])
                    
                    with open(location_file, 'w') as n:
                        for item in txt_content_new:
                            n.write(item)
                            n.write(' ')
                        n.close()
                f.close()

changer()