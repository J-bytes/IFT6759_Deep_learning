import json
import matplotlib.pyplot as plt
import os

import pandas as pd
import sklearn.model_selection

data_dir=".\data\images"
data={}

for dir in os.listdir(data_dir):


    annotation=json.load(open(data_dir+"/"+dir+"/annotation.json"))
    data=data|annotation


data=pd.DataFrame(data).T
import numpy as np
for animal_class in ["bat","insect","mountain_lion","lizard","badger"] :
    data.replace(animal_class,np.nan,inplace=True)
data.dropna(inplace=True)
data.reset_index(inplace=True)
X=data["file_name"]
y=data["category_id"].astype(int)
skf = sklearn.model_selection.StratifiedShuffleSplit(n_splits=1,test_size=0.1,random_state=11)
splits=skf.split(X, y)
for train,test in splits :
    test_set=data.loc[test]
    train_set=data.loc[train]


train_set.reset_index(inplace=True)
skf = sklearn.model_selection.StratifiedShuffleSplit(n_splits=1,test_size=0.3,random_state=11)
splits=skf.split(train_set["file_name"], train_set["category_id"])
for train,valid in splits :
    valid_set=train_set.loc[valid]
    train_set=train_set.loc[train]

def plot(data,title=None) :
    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                                   AutoMinorLocator)
    #plt.figure(figsize=(8, 6),dpi=1200)
    plt.rcParams["figure.figsize"]=(20,12)
    plt.rcParams["figure.dpi"]=400
    ax=data.sort_index().plot(kind="bar",stacked=True)
    ax.xaxis.set_major_locator(MultipleLocator(20))

    plt.semilogy()
    plt.title("Distribution of image categories by location")
    plt.legend(bbox_to_anchor=(1.11,1.),loc="upper right")
    plt.xticks(rotation=90)
    plt.xlabel("category")
    plt.ylabel("count")
    plt.savefig("test.png")

    plt.show()




titles=["train","valid","test"]
plt.rcParams["figure.figsize"] = (20, 12)
plt.rcParams["figure.dpi"] = 400
fig,ax=plt.subplots()
n_files=[train_set,valid_set,test_set]
for ex,data in enumerate(n_files) :

    data.to_csv(titles[ex])
    data=data.groupby(["category"]).sum()["category_id"] # test location after?
    data.plot(kind="bar",label=titles[ex],ax=ax,color=f"C{ex+1}")



plt.xticks(rotation=90)
plt.xlabel("locations")
plt.ylabel("count")
plt.legend()
plt.title("distribution of classes in the differents datasets")
plt.savefig("histogram_distribution_datasets.png")




#creating the new datasets :
id2number={6:0,1:1,33:2,9:3,3:4,11:5,8:6,16:7,5:8,10:9,7:10,51:11,99:12,34:13,30:14}
data_dir2="data_split2"
if not os.path.isdir(f"./data/{data_dir2}"):
    os.mkdir(f"./data/{data_dir2}")

import cv2 as cv
from PIL import Image
for ex,data in enumerate(n_files) :
    if not os.path.isdir(f"./data/{data_dir2}/{titles[ex]}") :
        os.mkdir(f"./data/{data_dir2}/{titles[ex]}")
    print(titles[ex])
    for image in data.iterrows() :
        image=image[1]
        file_name=image["image_id"]
        location=image["location"]
        img_path=f"{data_dir}/{location}/{file_name}.jpg"
        image_data = cv.imread(img_path)  # TODO verify dimension
        image_data = cv.resize(image_data, (608,608))

        cv.imwrite(f"./data/{data_dir2}/{titles[ex]}/images/{file_name}.jpg",image_data)

        bbox = image["bbox"]
        bbox_x0 = bbox[0]
        bbox_y0 = bbox[1]
        bbox_width0 = bbox[2]
        bbox_height0 = bbox[3]

        width_pic = image["width"]
        height_pic = image["height"]
        category_id = id2number[int(image["category_id"])]
        #print("category_id", category_id)

        new_x = (bbox_x0 + bbox_width0 / 2) / width_pic
        new_y = (bbox_y0 + bbox_height0 / 2) / height_pic

        new_width = bbox_width0 / width_pic
        new_height = bbox_height0 / height_pic

        to_save = f"./data/{data_dir2}/{titles[ex]}/labels/{file_name}.txt"

        f = open(to_save, "w+")
        to_write = str(
            str(category_id) + ' ' + str(new_x) + ' ' + str(new_y) + ' ' + str(new_width) + ' ' + str(new_height))
        f.write(to_write)
        f.close()

