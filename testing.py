#------python import------------------------------------

import torch
import numpy as np

#-----local imports---------------------------------------

from custom_utils import preprocess,Experiment
from train import metrics,criterion,device
from training.training import validation_loop

import time
import argparse

#----------- parse arguments----------------------------------

parser = argparse.ArgumentParser(description='Launch testing for a specific model')

parser.add_argument('model',
                    default='alexnet',
                    const='all',
                    type=str,
                    nargs='?',
                    choices=["alexnet","resnext50_32x4d","vgg19"],
                    required=True,
                    help='Choice of the model')

parser.add_argument('testset',
                    default='1',
                    const='all',
                    type=str,
                    nargs='?',
                    choices=['1', '2'],
                    required=True,
                    help='Version of the test set ; 1-seen locations 2-unseen locations')


parser.add_argument('dataset',
                    default='2',
                    const='all',
                    type=str,
                    nargs='?',
                    choices=['1', '2', '3',"4"],
                    required=True,
                    help='Version of the training dataset used')


args = parser.parse_known_args()
version=args.dataset


if int(args.testset)>1 :
    num_classes=14
    from training.dataloaders.cct_dataloader_V2 import CustomImageDataset
    test_dataset = CustomImageDataset(f"data/data/data_split{args.testset}/train", transform=preprocess)



else :
    num_classes = 19
    from training.dataloaders.cct_dataloader import CustomImageDataset
    test_list = np.loadtxt(f"data/test.txt")[1::].astype(int)

    test_dataset = CustomImageDataset("data/data/data_split1",locations=test_list, transform=preprocess)

model=torch.hub.load('pytorch/vision:v0.10.0', args.model, pretrained=True)
if args.model in ["vgg19","alexnet"] :
    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes, bias=True)
else :#for resnext
    model.fc = torch.nn.Linear(2048, num_classes)

batch_size=16

model.load_state_dict(torch.load(f"models/models_weights/{model._get_name()}/v{args.dataset}/{model._get_name()}.pt"))
model=model.to(device)


test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0,pin_memory=True) #keep

a=time.time()
running_loss,results=validation_loop(model=model,loader=test_loader,criterion=criterion,device=device)
print((time.time()-a)/len(test_dataset))
experiment=Experiment(f"{model._get_name()}/test")

if experiment:
    for key in metrics:
        experiment.log_metric(key, metrics[key](results[1].numpy(), results[0].numpy()), epoch=0)
        #wandb.log({key: metrics[key](results[1].numpy(), results[0].numpy())})

from sklearn.metrics import confusion_matrix

def answer(v) :
    v=v.numpy()
    return np.where(np.max(v,axis=1)>0.6,np.argmax(v,axis=1),15)
y_true,y_pred=answer(results[0]),answer(results[1])



#------------defining metrics--------------------------------------------
import sklearn
from sklearn.metrics import top_k_accuracy_score


def macro(true,pred) :


    return sklearn.metrics.f1_score(true,pred,average='macro') #weighted??

def f1(true,pred) :

    return sklearn.metrics.f1_score(true,pred,average='weighted') #weighted??

def precision(true,pred):

    return sklearn.metrics.precision_score(true,pred,average='macro')

def recall(true,pred):

    return sklearn.metrics.recall_score(true,pred,average='macro')



metrics={
    "f1"    :    f1,
    "macro":     macro,
    "precision": precision,
    "recall":    recall,

}



for metric in metrics.keys() :
    print(metric+" : ",metrics[metric](y_true,y_pred))

a=np.where(y_true.astype(int)==15,0,1)
b=np.where(y_pred.astype(int)==15,0,1)
print("identification results :",np.mean(np.where(a==b,1,0)))
m=confusion_matrix(y_true, y_pred,normalize="pred").round(2)

print("avg class : ",np.mean(np.diag(m)))
x = ['bobcat', 'opossum', 'car', 'coyote', 'raccoon', 'bird', 'dog', 'cat', 'squirrel', 'rabbit', 'skunk', 'fox', 'rodent', 'deer',"empty"]
z_text = [[str(y) for y in x] for x in m]


import plotly.figure_factory as ff
fig = ff.create_annotated_heatmap(m, x=x, y=x, annotation_text=z_text, colorscale='Viridis')

fig.update_layout(
    margin=dict(t=50, l=200),

    #title="ResNext50 3.0",
    xaxis_title="True labels",
    yaxis_title="Predictions",

)

fig['data'][0]['showscale'] = True
fig.show()