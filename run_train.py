#------python import------------------------------------

import torch

import os
import argparse

#----------- parse arguments----------------------------------
def init_parser() :
    parser = argparse.ArgumentParser(description='Launch training for a specific model')

    parser.add_argument('--model',
                        default='alexnet',
                        const='all',
                        type=str,
                        nargs='?',
                        choices=["alexnet","resnext50_32x4d","vgg19"],
                        required=True,
                        help='Choice of the model')
    parser.add_argument('--dataset',
                        default='2',
                        const='all',
                        type=str,
                        nargs='?',
                        choices=['1', '2', '3',"4"],
                        required=True,
                        help='Version of the dataset')
    parser.add_argument('--img_size',
                        default=320,
                        const='all',
                        type=int,
                        nargs='?',
                        required=False,
                        help='width and length to resize the images to. Choose a value between 320 and 608.')

    parser.add_argument('--wandb',
                        default=False,
                        const='all',
                        type=bool,
                        nargs='?',
                        choices=[True,False],
                        required=False,
                        help='True or False, do you wish (and did you setup) wandb? You will need to add the project name in the initialization of wandb in train.py')

    return parser


def main() :
    parser = init_parser()
    args = parser.parse_args()
    os.system(f"source venv/bin/activate")
    if args.model!="yolo" :

        os.system(f"python train.py --model {args.model} --dataset {args.dataset} --img_size {args.img_size} --wandb {args.wandb}")

    else :
        data_folder=f"data/data_split{args.dataset}/train"
        device= "cuda" if torch.cuda.is_available() else "cpu"
        os.system(f"python models/yolov5/train.py \
        --img 320 \
        --batch-size -1 \
        --epoch 50 \
        --workers 4 \
        --name exp \
        --weights yolov5m.pt \
        --device {device} \
        --exist-ok True --data {data_folder}") #add args
if __name__=="__main__" :
    main()