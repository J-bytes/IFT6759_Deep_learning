import pandas as pd
import numpy as np
import json
import shutil
import os
#data=json.load(open("data/caltech_images_20210113.json")) for all images
f=open("cct_transform_log.txt","w")
data=json.load(open("caltech_bboxes_20200316.json"))
image_final_directory="data/images"
image_initial_direcory="/mnt/g/cct_images" # full path required
if not os.path.exists(image_final_directory):
    os.makedirs(image_final_directory)
new_dict={}
for image in data["images"] : #gonna be freaking long
    try :
        location=image["location"]
        src=f"{image_initial_direcory}/{image['file_name']}"
        dst=f"{image_final_directory}/{location}/{image['file_name']}"
        if not os.path.exists(f"{image_final_directory}/{location}") :
            os.makedirs(f"{image_final_directory}/{location}")

        shutil.copy(src,dst)
        #os.remove(src)
        new_file_dict=image
        for annotation in data["annotations"] :
            if annotation["image_id"]==image["id"] :
                for category in data["categories"] :
                    if annotation["category_id"]==category["id"] :
                        annotation["category"]=category["name"]
                new_file_dict=new_file_dict|annotation
        if "category" in new_file_dict :
            if location in new_dict :
                new_dict[location].append(new_file_dict)
            else :
                new_dict[location]=[new_file_dict]
        else :
            os.remove(dst)
    except Exception as e :
        print(f"error {e}")
        f.write(f"{e} : {image['id']}")


for key in new_dict :
    # the json file where the output must be stored
    out_file = open(f"data/images/{key}/annotation.json", "w")
    location=new_dict[key]
    to_write={}
    for image in location :
        to_write[image["file_name"]]=image
    json.dump(to_write, out_file, indent=6)

    out_file.close()




f.close()