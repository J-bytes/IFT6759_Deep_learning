#------python import------------------------------------

import torch

import os
import argparse
from yolo_testing import yolo_testing,plot
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


    parser.add_argument("-t", '--testset',
                        default="unseen",

                        type=str,

                        choices=["seen", "unseen"],
                        required=True,
                        help='Choice of the test set 1-seen locations 2-unseen locations')
    return parser


def main() :
    parser = init_parser()
    args = parser.parse_args()
    os.system(f"source venv/bin/activate")
    if args.testset=="seen" :
        test_folder = f"data/data_split{args.version}/test"
    else :
        test_folder = "data/unseen_test_set/test"
    if args.model!="yolo" :

        os.system(f"python detect.py --model {args.model} --dataset {args.dataset} --testset {args.testset}")

    else :
        os.system(f"python models/yolov5/detect.py \
            --weights models/models_weights/yolov5m/yolov5m.pt \
            --img 320 --conf 0.25 \
            --source {os.getcwd()+test_folder+'/images'} \
            --save_txt True \
            --save_conf \
            --exist-ok")
        exp_folder="models/yolov5/runs/detect/exp/labels"
        results=yolo_testing(exp_folder,label_folder=test_folder+"/labels")
        plot(results)
if __name__=="__main__" :
    main()