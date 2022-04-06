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
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)
torch.backends.cudnn.benchmark = True



# -----------cuda optimization tricks-------------------------
#
# torch.autograd.set_detect_anomaly(False)
# torch.autograd.profiler.profile(False)
# torch.autograd.profiler.emit_nvtx(False)
# torch.backends.cudnn.benchmark = True


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


vgg = torchvision.models.vgg19(pretrained=True)
# set_parameter_requires_grad(vgg, feature_extract=True)
vgg.classifier[6] = torch.nn.Linear(vgg.classifier[6].in_features, 16, bias=True)
# ---------------------------------------------------
# alexnet
alexnet = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
# set_parameter_requires_grad(alexnet, feature_extract=True)
alexnet.classifier[6] = torch.nn.Linear(alexnet.classifier[6].in_features, 16, bias=True)
##---------------------------------------------------
# frcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
#
# # replace the classifier with a new one, that has
# # num_classes which is user-defined
# num_classes = 22  # 1 class (person) + background
# # get number of input features for the classifier
# in_features = frcnn.roi_heads.box_predictor.cls_score.in_features
# # replace the pre-trained head with a new one
# frcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
#


criterion = torch.nn.CrossEntropyLoss()  # to replace..?
print("The model has now been successfully loaded into memory")

#------------defining metrics--------------------------------------------
import sklearn
from sklearn.metrics import top_k_accuracy_score


def top1(true, pred):
    true = np.argmax(true, axis=1)
    # labels=np.unique(true)
    labels = np.arange(0, 16)

    return top_k_accuracy_score(true,pred,k=1,labels=labels)


def top5(true, pred):

    true = np.argmax(true, axis=1)
    labels = np.arange(0, 16)

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
    labels = np.arange(0, 16)
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
    batch_size = 4
    data_path = f"data/data/images"


    # train_list = np.loadtxt(f"data/training.txt")[1::].astype(int)
    # val_list = np.loadtxt(f"data/validation.txt")[1::].astype(int)
    # test_list = np.loadtxt(f"data/test.txt")[1::].astype(int)



    # train_list = np.loadtxt(f"data/training.txt")[1::].astype(int)
    # val_list = np.loadtxt(f"data/validation.txt")[1::].astype(int)
    # test_list = np.loadtxt(f"data/test.txt")[1::].astype(int)
    # train_list = np.loadtxt(f"data/test_test.txt")[1::].astype(int)
    # val_list = np.loadtxt(f"data/test_test.txt")[1::].astype(int)
    # test_list = np.loadtxt(f"data/test_test.txt")[1::].astype(int)

    train_dataset = CustomImageDataset("data/data/data_split2/train", transform=preprocess)
    val_dataset = CustomImageDataset("data/data/data_split2/valid", transform=preprocess)
    test_dataset = CustomImageDataset("data/data/data_split2/test", transform=preprocess)

    training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                                  pin_memory=True)  # num_worker>0 not working on windows
    validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                                    pin_memory=True)
    print("The data has now been loaded successfully into memory")
    # ------------training--------------------------------------------
    print("Starting training now")

   # if input("do you want to clear old log files? (yes/no)").lower()=="yes" :



    for model in [vgg, alexnet] :
        model = model.to(device)

        experiment = Experiment(f"log/{model._get_name()}/v2")
        optimizer = torch.optim.AdamW(model.parameters())
        training(model,optimizer,criterion,training_loader,validation_loader,device,verbose=False,epoch_max=50,patience=5,experiment=experiment,metrics=metrics)
