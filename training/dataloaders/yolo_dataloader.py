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



id2number={6:0,1:1,33:2,9:3,3:4,11:5,8:6,16:7,5:8,10:9,7:10,51:11,99:12,39:13,34:14,37:15,  30:16,14:17,21:18,40:19,66:20,97:21}


class CustomImageDataset(Dataset):
    def __init__(self, img_dir,locations, transform=None):
        self.largeur = 600
        self.hauteur = 600
        self.locations = locations  # feed only select locations
        self.img_dir = img_dir
        self.transform = transform
        self.length = 0
        self.files = []
        self.annotation_files = {}

        for location in locations:
            #print(self.img_dir)
            annotation = json.load(open(f"{self.img_dir}/{str(location)}/annotation.json"))
            self.annotation_files[str(location)] = f"{self.img_dir}/{str(location)}/annotation.json"
            for file in annotation:

                if annotation[file]["category"] not in ["bat", "insect", "mountain_lion", "lizard", "badger"]:
                    self.files.append(f"{self.img_dir}/{str(location)}/{file}")
                    self.length += 1

        self.categories = {}

        data = json.load(open(f"{os.getcwd()}/data_API/caltech_bboxes_20200316.json"))
        i = 0
        for category in data["categories"]:
            if category["name"] not in [ "bat", "insect", "mountain_lion", "lizard", "badger"]:
                self.categories[category["name"]] = id2number[int(category["id"])]
                i += 1


    def __len__(self):
        return self.length


    def label_transform(self, label):  # encode one_hot
        return self.categories[label]
    def __getitem__(self, idx):
        img_path=self.files[idx]
        if os.name=="nt" : #if on windows
            patterns = img_path.split("\\")[::-1]
            #print(patterns)
            location=patterns[0].split("/")[2]
            keyname = patterns[0].split("/")[3]
        else :
            patterns=img_path.split("/")[::-1]
            location = patterns[1]
            keyname = patterns[0]
        #location=img_path[len(self.img_dir)+1:len(self.img_dir)+3]

        #print("loc",location)
        #location=re.search("/[0-9][0-9]/",img_path).group()[1:-1]
        annotations=json.load(open(self.annotation_files[location]))
        image = cv.imread(img_path) #TODO verify dimension

        annotation = annotations[keyname]

        width_pic = annotation["width"]
        height_pic = annotation["height"]

        if self.transform:
            image=Image.fromarray(np.uint8(image))
            image = self.transform(image)

        #image=torch.tensor(image).float()

        try :

            img_ann = annotations[keyname]

        except Exception as e:
            print(e,"\n")
            print('loc',location)
            sys.exit()

        label=self.label_transform(img_ann["category"])
        bbox=img_ann["bbox"]

        boxes=[]
        labels=[label]

        # xmin = left corner x-coordinates
        xmin = bbox[0]
        # xmax = right corner x-coordinates
        xmax = bbox[0]+bbox[2]
        # ymin = left corner y-coordinates
        ymin = bbox[1]
        # ymax = right corner y-coordinates
        ymax = bbox[1]+bbox[3]

        # resize the bounding boxes according to the...
        # ... desired `width`, `height`
        xmin_final = (xmin / width_pic) * self.largeur
        xmax_final = (xmax / width_pic) * self.largeur
        ymin_final = (ymin / height_pic) * self.hauteur
        ymax_final = (ymax / height_pic) * self.hauteur

        boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])

        # bounding box to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # area of the bounding boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # no crowd instances
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        # labels to tensor
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # prepare the final `target` dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id


        return image.float(), target