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
from training.dataloaders.frcnn_dataloader import CustomImageDataset
from training.frcnn_training import training
from utils import set_parameter_requires_grad,Experiment,preprocess,collate_fn



# -----------cuda optimization tricks-------------------------

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


##---------------------------------------------------
frcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

#frcnn=torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

in_features = frcnn.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
frcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, 17)
#
# # replace the classifier with a new one, that has
# # num_classes which is user-defined
# num_classes = 22  # 1 class (person) + background
# # get number of input features for the classifier

#


criterion = torch.nn.CrossEntropyLoss()  # to replace..?
print("The model has now been successfully loaded into memory")

#------------defining metrics--------------------------------------------
import sklearn
from sklearn.metrics import top_k_accuracy_score
num_classes=17
def top1(true,pred) :
    true = np.argmax(true, axis=1)
    #labels=np.unique(true)
    labels = np.arange(0, num_classes-1)
    return top_k_accuracy_score(true,pred,k=1,labels=labels)
def top5(true,pred) :
    true = np.argmax(true, axis=1)
    labels = np.arange(0,num_classes-1)
    return top_k_accuracy_score(true,pred,k=5,labels=labels)

def f1(true, pred) :
    #print("f1 true and pred", true, pred)
    # true=np.argmax(true,axis=1)
    # pred=np.argmax(pred,axis=1)
    return sklearn.metrics.f1_score(true,pred,average='weighted') #weighted??

def macro(true, pred) :
    #print("macro true and pred", true, pred)
    # true=np.argmax(true,axis=1)
    # pred=np.argmax(pred,axis=1)
    return sklearn.metrics.f1_score(true,pred,average='macro') #weighted??

def auc(true,pred) :
    # true = np.argmax(true, axis=1)
    labels = np.arange(0, num_classes-1)
    return sklearn.metrics.roc_auc_score(true,pred,multi_class="ovo",labels=labels) #ovo???

metrics={
    "f1"    :    f1,
    "macro":    macro,
    # "top-1" :    top1,
    # "top-5" :    top5,
    # "auc"   :   auc,

}

if __name__ == "__main__":
    # -------data initialisation-------------------------------
    batch_size = 6
    data_path = f"data/data/images"
   



    train_list = np.loadtxt(f"data/training.txt")[1::].astype(int)
    val_list = np.loadtxt(f"data/validation.txt")[1::].astype(int)
    test_list = np.loadtxt(f"data/test.txt")[1::].astype(int)
    # train_list = np.loadtxt(f"data/test_test.txt")[1::].astype(int)
    # val_list = np.loadtxt(f"data/test_test.txt")[1::].astype(int)
    # test_list = np.loadtxt(f"data/test_test.txt")[1::].astype(int)


    for model in [frcnn]:

        train_dataset = CustomImageDataset(data_path, locations=train_list, transform=preprocess)
        val_dataset = CustomImageDataset(data_path, locations=val_list, transform=preprocess)
        test_dataset = CustomImageDataset(data_path, locations=test_list, transform=preprocess)
    
        training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6,pin_memory=True,collate_fn=collate_fn)  # num_worker>0 not working on windows
        validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=6,pin_memory=True,collate_fn=collate_fn)
        print("The data has now been loaded successfully into memory")
    
        model = model.to(device)
        experiment = Experiment(f"log/{model._get_name()}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
        # ------------training--------------------------------------------
        print("Starting training now")
    
        training(model, optimizer, criterion, training_loader, validation_loader, device, verbose=False, epoch_max=50, patience=5, experiment=experiment, metrics=metrics, batch_size=batch_size)