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
from training.dataloaders.cct_dataloader import CustomImageDataset
from utils import set_parameter_requires_grad,Experiment,preprocess
from main import metrics,criterion,vgg,alexnet,device
from training.training import validation_loop
from tqdm import tqdm
test_list=np.loadtxt(f"data/test.txt")[1::].astype(int)
data_path=f"data/images"

vgg.load_state_dict(torch.load("models/models_weights/VGG_3.pt"))
alexnet.load_state_dict(torch.load("models/models_weights/AlexNet_1.pt"))
models=[vgg,alexnet]
final_results={}
for model in models :
    model=model.to(device)
    for location in test_list :
        test_dataset = CustomImageDataset(data_path, locations=test_list, transform=preprocess)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=50, shuffle=True, num_workers=0,  pin_memory=True)
        val_loss, results=validation_loop(model,tqdm(test_loader),criterion,device)
        metrics_results={}
        for key in metrics :
            metrics_results[key]=metrics[key](results[0].numpy(),results[1].numpy())
        final_results[str(location)]=metrics_results

import pandas as pd
import matplotlib.pyplot as plt
data=pd.DataFrame(final_results).T
data.to_csv("test_results.csv")

plt.rcParams["figure.figsize"] = (20, 12)
plt.rcParams["figure.dpi"] = 400
fig,ax=plt.subplots()
#data.T[location_list].T.sum()
data.plot(kind="bar",ax=ax)
plt.xticks(rotation=90)
plt.xlabel("locations")
plt.ylabel("metrics_results")
plt.legend()
plt.title("test set results per location")
plt.savefig("test_set_results.png")
