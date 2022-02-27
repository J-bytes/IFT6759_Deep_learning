import pandas as pd
import numpy as np
import json
import shutil
import os
#data=json.load(open("data/caltech_images_20210113.json")) for all images
data=json.load(open("data/caltech_bboxes_20200316.json"))
image_final_directory="data/images"
image_initial_direcory="" # full path required

new_dict={}
for image in data["images"] : #gonna be freaking long

    location=image["location"]
    src=image_initial_direcory+"/"+image["file_name"]
    dst=image_final_directory+"/"+location+image["file_name"]
    shutil.copy(src,dst)
    os.remove(src)
    new_file_dict=image
    for annotation in data["annotations"] :
        if annotation["image_id"]==image["id"] :
            for category in data["categories"] :
                if annotation["category_id"]==category["id"] :
                    annotation["category"]=category["name"]
            new_file_dict=new_file_dict|annotation

    if location in new_dict :
        new_dict[location].append(new_file_dict)
    else :
        new_dict[location]=[new_file_dict]



for key in new_dict :
    # the json file where the output must be stored
    out_file = open(f"data/key/annotation.json", "w")
    location=new_dict[key]
    to_write={}
    for image in location :
        to_write[image["file_name"]]=image
    json.dump(to_write, out_file, indent=6)

    out_file.close()




