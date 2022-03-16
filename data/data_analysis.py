import json
import matplotlib.pyplot as plt
import os

import pandas as pd

data_dir="data/images"
locations= {}

for dir in os.listdir(data_dir):

    categories = {}
    annotation=json.load(open(data_dir+"/"+dir+"/annotation.json"))
    for img_file in annotation :
        img_file=annotation[img_file]
        category=img_file["category"]
        if category in categories :
            categories[category]+=1
        else :
            categories[category]=1
    locations[dir]=categories

stop=1
data=pd.DataFrame(locations)
data=data.fillna(0).T

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
    plt.xlabel("locations")
    plt.ylabel("count")
    plt.savefig("test.png")

    plt.show()


#--------------------------------------------------------
#separation of data
import numpy as np

locations=list(data.axes[0].to_numpy())

np.random.shuffle(locations)

n=0
#step 1 : count images
for location in locations :
    n+=len([name for name in os.listdir(f"{data_dir}/{location}")])

#step 2 : split n images
n1=int(0.1*n)
training=7*n1
validation=2*n1
test=n1
sizes=[]


# step 3 : assign location to dataset
n_images=n
train_locations=[]
valid_locations=[]
test_location=[]
step=0

n_train=0
n_valid=0
n_test=0
for location in data.axes[0].to_numpy() :
    n_image=len([name for name in os.listdir(f"{data_dir}/{location}")])
    n_images-=n_image
    if n_images>(training+validation) :
        n_test+=n_image
        test_location.append(location)
    elif n_images>training :
        n_valid+=n_image
        valid_locations.append(location)
    else :
        n_train+=n_image
        train_locations.append(location)


stop=1
titles=["training","validation","test"]

plt.rcParams["figure.figsize"] = (20, 12)
plt.rcParams["figure.dpi"] = 400
fig,ax=plt.subplots()
n_files=[n_train,n_valid,n_test]
for ex,location_list in enumerate([train_locations,valid_locations,test_location]) :
    np.savetxt(f"{titles[ex]}.txt",np.array([str(n_files[ex])]+location_list).astype(int),fmt='%i', delimiter=",")
    data2=data.T[location_list].T.sum()
    data2.plot(kind="bar",label=titles[ex],ax=ax,color=f"C{ex+1}")



plt.xticks(rotation=90)
plt.xlabel("locations")
plt.ylabel("count")
plt.legend()
plt.title("distribution of classes in the differents datasets")
plt.savefig("histogram_distribution_datasets.png")




