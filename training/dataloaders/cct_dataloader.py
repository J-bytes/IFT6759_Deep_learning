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
class CustomImageDataset(Dataset):
    def __init__(self, img_dir,locations, transform=None):
        self.largeur = 600
        self.hauteur = 480
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

        data=json.load(open(f"{os.getcwd()}/data_API/caltech_bboxes_20200316.json"))
        for ex,category in enumerate(data["categories"]) :
            self.categories[category["name"]]=ex
    def __len__(self):
        return self.length

    def label_transform(self,label): # encode one_hot

        one_hot= torch.zeros((22))
        one_hot[self.categories[label]]=1
        return one_hot.float()
    def __getitem__(self, idx):
        img_path=self.files[idx]
        if os.name=="nt" : #if on windows
            patterns = img_path.split("\\")[::-1]
            location=patterns[1].split("/")[2]
        else :
            patterns=img_path.split("/")[::-1]
            location = patterns[1]
        #location=img_path[len(self.img_dir)+1:len(self.img_dir)+3]


        #location=re.search("/[0-9][0-9]/",img_path).group()[1:-1]
        annotations=json.load(open(self.annotation_files[location]))
        image = cv.imread(img_path) #TODO verify dimension
        annotation = annotations[patterns[0]]
        bbox = annotation["bbox"]
        bbox_x0 = bbox[0]
        bbox_y0 = bbox[1]
        bbox_width0 = bbox[2]
        bbox_height0 = bbox[3]

        width_pic = annotation["width"]
        height_pic = annotation["height"]
        category_id = annotation["category_id"]

        new_x = (bbox_x0 + bbox_width0 / 2) * self.largeur/width_pic
        new_y = (bbox_y0 + bbox_height0 / 2) * self.hauteur / height_pic

        new_width = (bbox_width0 * self.largeur)/width_pic
        new_height = (bbox_height0 * self.hauteur) / height_pic
        #f = open(img_path+".txt", "w+")
        #f.write(category_id + "," + new_x + "," + new_y + "," + new_width + "," + new_height)
        #f.close()

        #image=np.reshape(image,(3,self.largeur, self.hauteur))



        if self.transform:
            image=Image.fromarray(np.uint8(image))
            image = self.transform(image)

        #image=torch.tensor(image).float()

        try :
            img_ann = annotations[patterns[0]]
        except Exception as e:
            print(e,"\n")
            print(location)
            sys.exit()

        label=self.label_transform(img_ann["category"])
        bbox=img_ann["bbox"]

        return image.float(), label#,bbox