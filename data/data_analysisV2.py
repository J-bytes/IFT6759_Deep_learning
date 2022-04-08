import json
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import sklearn.model_selection

data_dir="data/data_split3"
data=[{},{},{}]
datasets=["train","valid","test"]



#creating the new datasets :
# Mapping :
# 0 : bobcat         # 9 : rabbit
# 1 : opossum        # 10 : skunk
# 2 : car            # 11 : fox
# 3 : coyote         # 12 : rodent
# 4 : racoon         # 13 : deer
# 5 : bird           # 14 : empty
# 6 : dog
# 7 : cat
# 8 : squirrel

mapping={
    0 : "bobcat",
    1 : "oppossum",
    2 : "car",
    3 : "coyote",
    4 : "raccoon",
    5 : "bird",
    6 : "dog",
    7 : "cat",
    8 : "squirrel",
    9 : "rabbit",
    10 : "skunk",
    11 : "fox",
    12 : "rodent",
    13 : "deer",
    14 : "empty"
}
for dir in os.listdir(data_dir):

    for ex,dataset in enumerate(datasets) :
        for file in os.listdir(f"{data_dir}/{dataset}/labels") :
            category_id, new_x, new_y, new_width, new_height = np.loadtxt(f"{data_dir}/{dataset}/labels/{file}",
                                                                          unpack=True)
            category_id=int(category_id)
            data[ex][mapping[category_id]]=data[ex].get(mapping[category_id],0)+1





titles = ["train", "valid", "test"]
plt.rcParams["figure.figsize"] = (20, 12)
plt.rcParams["figure.dpi"] = 400
fig, ax = plt.subplots()
datas=data
for ex, data in enumerate(datas):
    data=pd.DataFrame(data.items())
    data.to_csv(titles[ex])
    data.plot(kind="bar",label=titles[ex],ax=ax,color=f"C{ex+1}")
plt.xticks(rotation=45, fontsize=25)
plt.xlabel("Classes")#, fontsize = 60)
plt.ylabel("Count")#, fontsize = 60)
plt.legend()#prop={'size':45})
#plt.title("Distribution of classes in the differents datasets", fontsize = 30)
plt.savefig("histogram_distribution_datasets3.png")

#creating the new datasets :
# Mapping :
# 0 : bobcat         # 9 : rabbit
# 1 : opossum        # 10 : skunk
# 2 : car            # 11 : fox
# 3 : coyote         # 12 : rodent
# 4 : racoon         # 13 : deer
# 5 : bird           # 14 : empty
# 6 : dog
# 7 : cat
# 8 : squirrel

