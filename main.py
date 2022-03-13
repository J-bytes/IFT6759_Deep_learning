#------python import------------------------------------
import warnings
import torch
import tqdm
import copy
from comet_ml import Experiment
import os

#-----local imports---------------------------------------
from training.training import training
from training.dataloaders.cct_dataloader import CustomImageDataset
from models.Rcnn import Rcnn


#-------data initialisation-------------------------------
print("dd", os.getcwd())
#data_path=f"{os.getcwd()}/data/data/images"
# train_dataset=CustomImageDataset(data_path,locations=list(range(65,70)))
# val_dataset=CustomImageDataset(data_path,locations=list(range(0,5)))
# val_dataset.method="val"
# training_loader=torch.utils.data.DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=5,pin_memory=True)
# validation_loader=torch.utils.data.DataLoader(val_dataset, batch_size=6, shuffle=True, num_workers=5,pin_memory=True)
train_dataset=CustomImageDataset(data_path,locations=11)
training_loader=torch.utils.data.DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=5,pin_memory=True)

print("The data has now been loaded successfully into memory")
#-----------model initialisation------------------------------
if torch.cuda.is_available() :
    device="cuda:0"
else :
    device="cpu"
    warnings.warn("No gpu is available for the computation")

#image size input 600x480
model=Rcnn(features=[6300,2,22],channels=[3,64,32,1]).to(device)
optimizer=torch.optim.AdamW(model.parameters())
criterion=torch.nn.KLDivLoss() # to replace
print("The model has now been successfully loaded into memory")

#---comet logger initialisation
# Create an experiment with your api key
#experiment = Experiment(
#    api_key="",
#    project_name="ift6759",
#    workspace="bariljeanfrancois",
#)
print("Starting training now")
training(model,optimizer,criterion,training_loader,validation_loader,device,verbose=False)
