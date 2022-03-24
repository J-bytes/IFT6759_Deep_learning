import sys

import torch
from torch.utils.data import Dataset
import os
from imageio import imread
import json
import numpy as np
import cv2 as cv
import re
import shutil

class YoloDataTransformer:
    def __init__(self, img_dir_orig, img_dir_yolo, locations_train, locations_valid, locations_test):
        self.largeur = 600
        self.hauteur = 480
        self.locations_train = locations_train
        self.locations_valid = locations_valid
        self.locations_test = locations_test
        self.img_dir_orig = img_dir_orig
        self.img_dir_yolo = img_dir_yolo

    def transform(self, ):
        datasets = {'train': self.locations_train, 'valid': self.locations_valid, 'test': self.locations_test}
        for dataset, values in datasets.items():
            for loc in values:
                folder = str(self.img_dir_orig + '/' + str(loc))
                # print(folder)
                annotation_file = json.load(open(str(folder + '/annotation.json')))
                print(len(annotation_file))
                for root, dirs, files in os.walk(folder, topdown=False):
                    for name in files:
                        location_file = os.path.join(root, name)

                        if location_file[::-1][0:4] != "json"[::-1]:
                            destination = str(self.img_dir_yolo + '/' + dataset + '/images/' + name)
                            image = cv.resize(cv.imread(location_file), (self.largeur, self.hauteur))  # TODO verify dimension
                            cv.imwrite(destination, image)
                            annotation = annotation_file[name]
                            bbox = annotation["bbox"]
                            bbox_x0 = bbox[0]
                            bbox_y0 = bbox[1]
                            bbox_width0 = bbox[2]
                            bbox_height0 = bbox[3]

                            width_pic = annotation["width"]
                            height_pic = annotation["height"]
                            category_id = annotation["category_id"]

                            new_x = (bbox_x0 + bbox_width0 / 2) / width_pic
                            new_y = (bbox_y0 + bbox_height0 / 2) / height_pic

                            new_width = bbox_width0 / width_pic
                            new_height = bbox_height0 / height_pic

                            to_save = str(self.img_dir_yolo + '/' + dataset + '/labels/' + name[:-4] + '.txt')

                            f = open(to_save, "w+")
                            to_write = str(str(category_id) + ' ' + str(new_x) + ' ' + str(new_y) + ' ' + str(new_width) + ' ' + str(new_height))
                            f.write(to_write)
                            f.close()

        return 'na'


def main():
    original_data_path = f"{os.getcwd()}/data/images"
    yolo_data_path = f"{os.getcwd()}/data/yolo"

    # print("original_data_path:", original_data_path)
    # print("yolo_data_path:", yolo_data_path)

    training = np.loadtxt(str(yolo_data_path + "/training.txt"), dtype=int)
    num_img_training = training[0]
    locations_train = training[1:]

    validation = np.loadtxt(str(yolo_data_path + "/validation.txt"), dtype=int)
    num_img_validation = validation[0]
    locations_valid = validation[1:]

    test = np.loadtxt(str(yolo_data_path + "/test.txt"), dtype=int)
    num_img_test = test[0]
    locations_test = test[1:]

    transformer = YoloDataTransformer(original_data_path, yolo_data_path, locations_train, locations_valid, locations_test)
    transformer.transform()


if __name__ == "__main__":
    main()
