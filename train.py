#------python import------------------------------------
import warnings
import torch
import tqdm
import copy
#from comet_ml import Experiment

import os
import numpy as np
import torchvision
import argparse
#-----local imports---------------------------------------
from training.training import training

from custom_utils import set_parameter_requires_grad,Experiment,preprocess,metrics




# -----------cuda optimization tricks-------------------------
# DANGER ZONE !!!!!
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)
torch.backends.cudnn.benchmark = True

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

    return parser

def main() :
    parser=init_parser()
    args = parser.parse_args()

    version = args.dataset
    if int(version)>1 :
        num_classes=14
        from training.dataloaders.cct_dataloader_V2 import CustomImageDataset
        train_dataset = CustomImageDataset(f"data/data/data_split{version}/train", transform=preprocess)
        val_dataset = CustomImageDataset(f"data/data/data_split{version}/valid", transform=preprocess)


    # else :
    #     num_classes = 19
    #     from training.dataloaders.cct_dataloader import CustomImageDataset
    #     train_list = np.loadtxt(f"data/training.txt")[1::].astype(int)
    #     val_list = np.loadtxt(f"data/validation.txt")[1::].astype(int)
    #     train_dataset = CustomImageDataset("data/data/images",locations=train_list, transform=preprocess)
    #     val_dataset = CustomImageDataset("data/data/images",locations=val_list, transform=preprocess)
    #-----------defining metrics - -------------------------------------------
    # import sklearn
    # from sklearn.metrics import top_k_accuracy_score
    #
    # def top1(true, pred):
    #     true = np.argmax(true, axis=1)
    #     # labels=np.unique(true)
    #     labels = np.arange(0, num_classes)
    #
    #     return top_k_accuracy_score(true, pred, k=1, labels=labels)
    #
    # def top5(true, pred):
    #     true = np.argmax(true, axis=1)
    #     labels = np.arange(0, num_classes)
    #
    #     return top_k_accuracy_score(true, pred, k=5, labels=labels)
    #
    # def f1(true, pred):
    #     true = np.argmax(true, axis=1)
    #     pred = np.argmax(pred, axis=1)
    #
    #     return sklearn.metrics.f1_score(true, pred, average='macro')  # weighted??
    #
    # def precision(true, pred):
    #     true = np.argmax(true, axis=1)
    #     pred = np.argmax(pred, axis=1)
    #     return sklearn.metrics.precision_score(true, pred, average='macro')
    #
    # def recall(true, pred):
    #     true = np.argmax(true, axis=1)
    #     pred = np.argmax(pred, axis=1)
    #     return sklearn.metrics.recall_score(true, pred, average='macro')
    #
    # metrics = {
    #     "f1": f1,
    #
    #     "precision": precision,
    #     "recall": recall,
    #     "top-1": top1,
    #     "top-5": top5
    # }
    criterion = torch.nn.CrossEntropyLoss()

    # -----------model initialisation------------------------------

    model=torch.hub.load('pytorch/vision:v0.10.0', args.model, pretrained=True)
    if args.model in ["vgg19","alexnet"] :
        if args.model=="vgg19" :
            batch_size = 8
            #set_parameter_requires_grad(model, feature_extract=True)
        else :
            batch_size = 168
        #model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes, bias=True)

    else :#for resnext
        model.fc = torch.nn.Linear(2048, num_classes)
        batch_size = 8
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        warnings.warn("No gpu is available for the computation")


    print("The model has now been successfully loaded into memory")





    # -------data initialisation-------------------------------
    #os.environ["WANDB_MODE"] = "offline"




    training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=int(os.cpu_count()/3),pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=int(batch_size*2), shuffle=False, num_workers=int(os.cpu_count()/3),pin_memory=True)
    print("The data has now been loaded successfully into memory")
    # ------------training--------------------------------------------
    print("Starting training now")




    model = model.to(device)
    experiment = Experiment(f"{model._get_name()}/v{version}")

    optimizer = torch.optim.AdamW(model.parameters())

    training(model,optimizer,criterion,training_loader,validation_loader,device,minibatch_accumulate=1,epoch_max=50,patience=5,experiment=experiment,metrics=metrics)

if __name__ == "__main__":
     main()