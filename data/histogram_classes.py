import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import sklearn.model_selection
import numpy as np

data_dir="data/data/images"
data={}

#liste des fichiers dannotation
for dir in os.listdir(data_dir):
    annotation=json.load(open(data_dir+"/"+dir+"/annotation.json"))
    data=data|annotation
temp = []
for key, value in data.items():
    temp.append((value['location'], value['category'], value['category_id']))
data = pd.DataFrame(temp, columns=['location','animal', 'category_id'])

def read_locations(file):
    temp = []
    with open(file) as my_file:
        for line in my_file:
            temp.append(line[:-1])
    return temp[1:]

train_list = read_locations("training.txt")
test_list = read_locations("test.txt")
valid_list = read_locations("validation.txt")


drops =  ["bat","insect","mountain_lion","lizard","badger"]
data = data[~data['animal'].isin(drops)]
conditions = [data['location'].isin(train_list) ,data['location'].isin(test_list),data['location'].isin(valid_list)]
choices = ['train', 'test', 'valid']
data['dataset'] = np.select(conditions,choices)
data.reset_index(inplace=True, drop=True)

test_set=data[data['dataset']=='test']
train_set=data[data['dataset']=='train']
valid_set=data[data['dataset']=='valid']
titles = ['train', 'valid','test']
fig, ax = plt.subplots()
n_files = [train_set, valid_set, test_set]
for ex, data in enumerate(n_files):
    data.to_csv(titles[ex])
    data = data.groupby(["animal"]).sum()["category_id"] # test location after?
    data.plot(kind="bar",label=titles[ex],ax=ax,color=f"C{ex+1}")
plt.xticks(rotation=45, fontsize=25)
plt.xlabel("Classes")#, fontsize = 60)
plt.ylabel("Count")#, fontsize = 60)
plt.legend()#prop={'size':45})
#plt.title("Distribution of classes in the differents datasets", fontsize = 30)
plt.savefig("histogram_distribution_datasets_first_split.png")
