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

        self.files=[]
        self.annotation_files={}
        for location in locations :
            for root, dirs, files in os.walk(f"/data/{location}", topdown=False):
                for name in files:
                    location_file=os.path.join(root, name)
                    if location_file[::-1][::-1][0:4]!="json"[::-1] :
                        self.files.append(location_file)
                    else :
                        self.annotation_files[location]=location_file
                break;

        self.categories={}
        for ex,category in enumerate(data["categories"]) :
            self.categories[category["name"]]=ex
    def __len__(self):
        return len(self.files)

    def label_transform(self,label): # encode one_hot

        one_hot= np.zeros((22))
        one_hot[self.categories["label"]]=1
        return one_hot
    def __getitem__(self, idx):
        img_path=self.files[idx]
        location=img_path[0:2]

        #location=re.search("/[0-9][0-9]/",img_path).group()[1:-1]
        annotations=json.load(open("data/"+self.annotation_files[location]))
        image = imread("data/"+img_path)

        if self.transform:
            image = self.transform(image)

        image=torch.tensor(image)
        img_ann = annotations[img_path[3:-3]]
        label=self.label_transform(img_ann["category"])
        bbox=img_ann["bbox"]

        return image, label,bbox