import os
import numpy as np
import pandas as pd
import json
import shutil
used_locations=os.listdir("data/images")
mapping={
    "bobcat" :0,
    "oppossum":1,
     "car": 2,
    "coyote" :3,
    "raccoon" :4,
    "bird" : 5,
    "dog" : 6,
    "cat" : 7,
    "squirrel" :8,
    "rabbit" :9,
    "skunk":10,
    "fox" :11,
    "rodent":12,
    "deer":13,
    "empty" :14,
}

used_category_names=mapping.keys()
image_count={}
for i in used_category_names :
    image_count[i]=0


data=json.load(open("caltech_images_20210113.json"))
data_dir="data/data"
new_data_dir="test_set"
# format data better cuz they did a shit job
annotations = {}
for file in data["annotation"]:
    annotations[file["image_id"]] = file

categories = {}
for item in data["categories"]:
    categories[item["id"]] = item["name"]

def part1() :
    if not os.path.isdir(new_data_dir) :
        os.mkdir(new_data_dir)
        os.mkdir(new_data_dir+"/images")
        os.mkdir(new_data_dir+"/labels")

    for file in data["images"]:

        if file["location"] not in used_locations : # new location

            category_id=annotations[file["id"]]["category_id"]

            if categories[category_id] in used_category_names :
                #add this image to the test set
                if image_count[categories[category_id]]<100 :
                    shutil.copy(f"{data_dir}/images/{file['id']}.jpg")
                    image_count[categories[category_id]]+=1


mapping2={

}

def part2() :
    #assuming roboflow annotation has been done and pushed to test/labels
    for file in os.listdir(new_data_dir+"/images") :

        file_id=file[:-4]
        annotation=np.loadtxt(f"{new_data_dir}/labels/{file_id}.txt")
        annotation[0]=mapping[categories[annotations[file_id]["category_id"]]]
        np.savetxt(f"{new_data_dir}/labels/{file_id}.txt",annotation)


