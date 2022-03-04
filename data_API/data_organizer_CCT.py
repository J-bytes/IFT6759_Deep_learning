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
for image in data["images"] : # boucle sur le fichier json ; liste des images
    try :
        location=image["location"]
        src=f"{image_initial_direcory}/{image['file_name']}" #fichier source
        dst=f"{image_final_directory}/{location}/{image['file_name']}" # destinatioon ou copier l'image "data/images/location/image.png
        if not os.path.exists(f"{image_final_directory}/{location}") : #create folder if location does not exist
            os.makedirs(f"{image_final_directory}/{location}")

        shutil.copy(src,dst)
        #os.remove(src) # remove the original to not overload disk space
        new_file_dict=image # copy the dictionnary with the different PARTIAL info about the image
        for annotation in data["annotations"] : #boucle sur la liste des annotations
            if annotation["image_id"]==image["id"] : #trouver l'annotation correspondant à l'image
                for category in data["categories"] : # trouver la catégorie correspondant à celle dans l'annotation
                    if annotation["category_id"]==category["id"] :
                        annotation["category"]=category["name"] # add the category key to the dictionnary annnotation
                new_file_dict=new_file_dict|annotation # merges the dict new_file_dict and annotation CAREFUL the operator | works only in python 3.9

        if "category" in new_file_dict : #empty IS a category
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