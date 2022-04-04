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
    def __init__(self, img_dir, transform=None):
        self.largeur = 320
        self.hauteur = 320

        self.img_dir = img_dir
        self.transform = transform
        self.length=0
        self.files=[]
        self.annotation_files={}


        for file in os.listdir(img_dir+"/images") :
            if file[::-1][0:3]=="jpg"[::-1] :
                category_id,new_x,new_y,new_width,new_height=np.loadtxt(f"{self.img_dir}/labels/{file[:-3]}txt",unpack=True)
                if int(category_id) not in [66,97,40,14,21] :
                    self.files.append(f"{self.img_dir}/images/{file}")


    def __len__(self):
        return len(self.files)

    def label_transform(self,label_id): # encode one_hot
        if int(label_id)==30 :
            return torch.zeros((16))
        one_hot= torch.zeros((16))
        one_hot[id2number[int(label_id)]]=1
        return one_hot.float()

    def __getitem__(self, idx):
        img_path=self.files[idx]
        if os.name=="nt" : #if on windows
            patterns = img_path.split("\\")[::-1]
            #print('patterns', patterns)
            location=patterns[0].split("/")[3]
            #print('location', location)
            keyname = patterns[0].split("/")[4]
        else :
            patterns=img_path.split("/")[::-1]
            location = patterns[1]
            keyname = patterns[0]
        #location=img_path[len(self.img_dir)+1:len(self.img_dir)+3]

        #print("loc",location)
        #location=re.search("/[0-9][0-9]/",img_path).group()[1:-1]
        category_id,new_x,new_y,new_width,new_height=np.loadtxt(f"{self.img_dir}/labels/{keyname[:-3]}txt",unpack=True)
        image = cv.imread(img_path) #TODO verify dimension
        image = cv.resize(image, (self.hauteur, self.largeur))
        




        if self.transform:
            image=Image.fromarray(np.uint8(image))
            image = self.transform(image)

        #image=torch.tensor(image).float()


        label=self.label_transform(category_id)


        return image.float(), label