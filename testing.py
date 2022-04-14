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
from custom_utils import preprocess,Experiment
from main import metrics,criterion,device
from training.training import validation_loop
from tqdm import tqdm

num_classes=14
batch_size=16
# model = torchvision.models.resnext50_32x4d(pretrained=True)
# model.fc = torch.nn.Linear(2048, num_classes)
# model.load_state_dict(torch.load("models/models_weights/ResNet/v3/ResNet.pt"))


test_dataset = CustomImageDataset("data/data/test_set3/test", transform=preprocess)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0,pin_memory=True) #keep

running_loss,results=validation_loop(model=model,loader=test_loader,criterion=criterion,device=device)
experiment=Experiment(f"{model._get_name()}/test")
if experiment:
    for key in metrics:
        experiment.log_metric(key, metrics[key](results[1].numpy(), results[0].numpy()), epoch=0)
        #wandb.log({key: metrics[key](results[1].numpy(), results[0].numpy())})

from sklearn.metrics import confusion_matrix

def answer(v) :
    v=v.numpy()
    return np.where(np.max(v,axis=1)>0.5,np.argmax(v,axis=1),15)
y_true,y_pred=answer(results[0]),answer(results[1])
m=confusion_matrix(y_true, y_pred,normalize="pred").round(2)*100
m=m.round(0)
x = ['bobcat', 'opossum', 'car', 'coyote', 'raccoon', 'bird', 'dog', 'cat', 'squirrel', 'rabbit', 'skunk', 'fox', 'rodent', 'deer',"empty"]
z_text = [[str(y) for y in x] for x in m]


import plotly.figure_factory as ff
fig = ff.create_annotated_heatmap(m, x=x, y=x, annotation_text=z_text, colorscale='Viridis')
fig.update_layout(margin=dict(t=50, l=200))
fig['data'][0]['showscale'] = True
fig.show()