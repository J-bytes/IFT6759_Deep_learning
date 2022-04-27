#------python import------------------------------------

import torch
from data.animal_class_scraper import AnimalsClassScraper
import os
import argparse
import shutil
#----------- parse arguments----------------------------------
def init_parser() :
    parser = argparse.ArgumentParser(description='Launch training for a specific model')

    parser.add_argument('--model',
                        default='alexnet',
                        const='all',
                        type=str,
                        nargs='?',
                        choices=["alexnet","resnext50_32x4d","vgg19","yolo"],
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
                        
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='do you wish (and did you setup) wandb? You will need to add the project name in the initialization of wandb in train.py')

    parser.add_argument('--epoch',
                        default=50,
                        const='all',
                        type=int,
                        nargs='?',
                        required=False,
                        help="Number of epochs to train ; a patiance of 5 is implemented by default")

    return parser


def main() :
    parser = init_parser()
    args = parser.parse_args()
    os.system(f"source venv/bin/activate")
    acs=AnimalsClassScraper()

    if not os.path.isdir(f"data/data_split{args.dataset}") :
        print("We need to change our dataset")

        if not os.path.isdir(f"data/data_split2") :
            print("You need to download the dataset data split 2!\
             Please see the jupyter notebook or README.")
        else :
            if args.dataset==3 :
                #we need to upsample
                acs.upsample(classes=[5,8,12])
                os.renames("data/data_split2","data/data_split3")
            else :
                # we need to augment the data
                acs.augment(classes=[5,8,12])
                os.renames("data/data_split2", "data/data_split4")




    if args.model!="yolo" :
        wandb_arg= "--wandb" if args.wandb else "--no-wandb"
        os.system(f"python train.py --model {args.model} --dataset {args.dataset} --img_size {args.img_size} {wandb_arg} --epoch {args.epoch}")

    else :
        data_folder=os.path.join(os.getcwd(),f"data/data_split{args.dataset}/data_split2.yaml")
        device= "cuda:0" if torch.cuda.is_available() else "cpu"
        os.system(f"python {os.getcwd()}/models/yolov5/train.py \
        --img 320 \
        --batch-size 32 \
        --epoch {args.epoch} \
        --workers 4 \
        --name exp \
        --weights yolov5m.pt \
        --device {device} \
        --exist-ok \
        --nosave \
        --data {data_folder} \
         --patience 5") #add args

        weights_path=f"models/models_weights/yolov5m/v{args.dataset}"
        os.mkdir(weights_path,exist_ok=True,parents=True)
        shutil.move(f"{os.getcwd()}/models/yoov5/runs/train/exp/weights/best.pt",weights_path+"/yolov5m.pt")
if __name__=="__main__" :
    main()