import sys

import torch
from torch.utils.data import Dataset
import os
from imageio import imread
#import re
import json
import numpy as np
import cv2 as cv
from PIL import Image
import re

id2number={6:0,1:1,33:2,9:3,3:4,11:5,8:6,16:7,5:8,10:9,7:10,51:11,99:12,39:13,34:14,37:15,30:16,14:17,21:18,40:19,66:20,97:21}


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, locations, transform=None):
        self.largeur = 320
        self.hauteur = 320
        self.locations=locations # feed only select locations
        self.img_dir = img_dir
        self.transform = transform
        self.length=0
        self.files=[]
        self.annotation_files={}

        for location in locations :
            annotation=json.load(open(f"{self.img_dir}/{str(location)}/annotation.json"))
            self.annotation_files[str(location)] = f"{self.img_dir}/{str(location)}/annotation.json"
            for file in annotation :

                    #if annotation[file]["category"] not in ["bat","insect","mountain_lion","lizard","badger"] :
                    self.files.append(f"{self.img_dir}/{str(location)}/{file}")
                    self.length += 1


        self.categories={}

        data=json.load(open(f"{os.getcwd()}/data_API/caltech_bboxes_20200316.json"))
        i=0
        for category in data["categories"] :
            #if category["name"] not in ["empty","bat","insect","mountain_lion","lizard","badger"] :
            self.categories[category["name"]]=id2number[int(category["id"])]
            i+=1
    def __len__(self):
        return self.length

    def label_transform(self,label): # encode one_hot
        if label=="empty" :
            return torch.zeros((22))
        one_hot= torch.zeros((22))
        one_hot[self.categories[label]]=1
        return one_hot.float()

    def __getitem__(self, idx):
        img_path=self.files[idx]

        patterns=img_path.split("/")[::-1]
        location = patterns[1]
        keyname = patterns[0]

        annotations=json.load(open(self.annotation_files[location]))
        image = cv.imread(img_path)
        image = cv.resize(image, (self.hauteur, self.largeur))




        if self.transform:
            image=Image.fromarray(np.uint8(image))
            image = self.transform(image)




        img_ann = annotations[keyname]
        label=self.label_transform(img_ann["category"])

        return image.float(), label