import torch
from torch.utils.data import Dataset
import os
from imageio import imread
#import re
import json
import numpy as np
class CustomImageDataset(Dataset):
    def __init__(self, img_dir,locations, transform=None):
        self.locations=locations # feed only select locations
        self.img_dir = img_dir
        self.transform = transform
        self.length=0
        self.files=[]
        self.annotation_files={}
        for location in locations :
            for root, dirs, files in os.walk(f"{self.img_dir}/{location}", topdown=False):
                for name in files:
                    location_file=os.path.join(root, name)
                    if location_file[::-1][0:4]!="json"[::-1] :
                        self.files.append(location_file)
                        self.length+=1
                    else :
                        self.annotation_files[str(location)]=location_file
                break;

        self.categories={}

        data=json.load(open(f"{os.getcwd()}/IFT6759_Deep_learning/data_API/caltech_bboxes_20200316.json"))
        for ex,category in enumerate(data["categories"]) :
            self.categories[category["name"]]=ex
    def __len__(self):
        return self.length

    def label_transform(self,label): # encode one_hot

        one_hot= np.zeros((22))
        one_hot[self.categories[label]]=1
        return one_hot
    def __getitem__(self, idx):
        img_path=self.files[idx]
        location=img_path[len(self.img_dir)+1:len(self.img_dir)+3]
        loc_len=4
        if location[1]=="/" :
            location=location[0]
            loc_len=3

        #location=re.search("/[0-9][0-9]/",img_path).group()[1:-1]
        annotations=json.load(open(self.annotation_files[location]))
        image = imread(img_path)

        if self.transform:
            image = self.transform(image)

        image=torch.tensor(image)
        img_ann = annotations[img_path[len(self.img_dir)+loc_len:-4]+".jpg"]
        label=self.label_transform(img_ann["category"])
        bbox=img_ann["bbox"]

        return image, label#,bbox