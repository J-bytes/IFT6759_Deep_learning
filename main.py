#------python import------------------------------------
import warnings
import torch
import tqdm
import copy
#from comet_ml import Experiment

import os
import numpy as np
import torchvision
#-----local imports---------------------------------------
from training.training import training
from training.dataloaders.cct_dataloader_V2 import CustomImageDataset
from utils import set_parameter_requires_grad,Experiment,preprocess




# -----------cuda optimization tricks-------------------------
#
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)
torch.backends.cudnn.benchmark = True


# -----------model initialisation------------------------------
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    warnings.warn("No gpu is available for the computation")

# image size input 600x480
# model=Rcnn(features=[6300,2,22],channels=[3,64,32,1]).to(device)
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', classes=22).to(device)
# ---------------------------------------------------


# vgg = torchvision.models.vgg19(pretrained=True)
# # set_parameter_requires_grad(vgg, feature_extract=True)
# vgg.classifier[6] = torch.nn.Linear(vgg.classifier[6].in_features, 14, bias=True)
# ---------------------------------------------------
# alexnet
alexnet = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
# set_parameter_requires_grad(alexnet, feature_extract=True)
alexnet.classifier[6] = torch.nn.Linear(alexnet.classifier[6].in_features, 14, bias=True)
##---------------------------------------------------

resnext50 = torchvision.models.resnext50_32x4d(pretrained=True)
resnext50.fc = torch.nn.Linear(2048, 14)



criterion = torch.nn.CrossEntropyLoss()  # to replace..?
print("The model has now been successfully loaded into memory")

#------------defining metrics--------------------------------------------
import sklearn
from sklearn.metrics import top_k_accuracy_score


def top1(true, pred):
    true = np.argmax(true, axis=1)
    # labels=np.unique(true)
    labels = np.arange(0, 14)

    return top_k_accuracy_score(true,pred,k=1,labels=labels)


def top5(true, pred):

    true = np.argmax(true, axis=1)
    labels = np.arange(0, 14)

    return top_k_accuracy_score(true, pred, k=5, labels=labels)


def macro(true,pred) :
    true=np.argmax(true,axis=1)
    pred=np.argmax(pred,axis=1)

    return sklearn.metrics.f1_score(true,pred,average='macro') #weighted??

def f1(true,pred) :
    true=np.argmax(true,axis=1)
    pred=np.argmax(pred,axis=1)
    return sklearn.metrics.f1_score(true,pred,average='weighted') #weighted??

def f1(true, pred):
    true = np.argmax(true, axis=1)
    pred = np.argmax(pred, axis=1)
    return sklearn.metrics.f1_score(true, pred, average='macro')  # weighted??


def auc(true, pred):
    print('auc true pred', true, pred)

    true = np.argmax(true, axis=1)
    labels = np.arange(0, 14)
    return sklearn.metrics.roc_auc_score(true, pred, multi_class="ovo", labels=labels)  # ovo???

def precision(true,pred):
    true = np.argmax(true, axis=1)
    pred=np.argmax(pred,axis=1)
    return sklearn.metrics.precision_score(true,pred,average='macro') 

def recall(true,pred):
    true = np.argmax(true, axis=1)
    pred=np.argmax(pred,axis=1)
    return sklearn.metrics.recall_score(true,pred,average='macro')  



metrics={
    "f1"    :    f1,
    "macro":     macro,
    "precision": precision,
    "recall":    recall,
    "top-1" :    top1,
    "top-5" :    top5
}

if __name__ == "__main__":
    # -------data initialisation-------------------------------
    #os.environ["WANDB_MODE"] = "offline"
    batch_size = 16
    data_path = f"data/data/images"


    # train_list = np.loadtxt(f"data/training.txt")[1::].astype(int)
    # val_list = np.loadtxt(f"data/validation.txt")[1::].astype(int)
    # test_list = np.loadtxt(f"data/test.txt")[1::].astype(int)



    # train_list = np.loadtxt(f"data/training.txt")[1::].astype(int)
    # val_list = np.loadtxt(f"data/validation.txt")[1::].astype(int)
    # test_list = np.loadtxt(f"data/test.txt")[1::].astype(int)
    # train_list = np.loadtxt(f"data/training.txt")[1::].astype(int)
    # val_list = np.loadtxt(f"data/validation.txt")[1::].astype(int)
    # train_dataset = CustomImageDataset("data/data/images",locations=train_list, transform=preprocess)
    # val_dataset = CustomImageDataset("data/data/images",locations=val_list, transform=preprocess)


    train_dataset = CustomImageDataset("data/data/data_split3/train", transform=preprocess)
    val_dataset = CustomImageDataset("data/data/data_split3/valid", transform=preprocess)
    # test_dataset = CustomImageDataset("data/data/data_split2/test", transform=preprocess)
    # train_dataset = CustomImageDataset("data/data/animals-detection-mini.v1-mini.yolov5pytorch/train", transform=preprocess)
    # val_dataset = CustomImageDataset("data/data/animals-detection-mini.v1-mini.yolov5pytorch/valid", transform=preprocess)
    # test_dataset = CustomImageDataset("data/data/animals-detection-mini.v1-mini.yolov5pytorch/test", transform=preprocess)

    training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                                                  pin_memory=True)  # num_worker>0 not working on windows
    validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                                                    pin_memory=True)
    print("The data has now been loaded successfully into memory")
    # ------------training--------------------------------------------
    print("Starting training now")

   # if input("do you want to clear old log files? (yes/no)").lower()=="yes" :



    for model in [resnext50] :

        model = model.to(device)
        experiment = Experiment(f"log/{model._get_name()}/v3")
        optimizer = torch.optim.AdamW(model.parameters())
        training(model,optimizer,criterion,training_loader,validation_loader,device,verbose=False,epoch_max=50,patience=5,experiment=experiment,metrics=metrics)
