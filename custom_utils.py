from torchvision import transforms
import os
import torch
import pathlib
import sklearn
import numpy as np
from  sklearn.metrics import top_k_accuracy_score
import wandb
#-----------------------------------------------------------------------------------
class Experiment() :
    def __init__(self,directory,is_wandb=False):
        self.is_wandb=is_wandb
        self.directory="log/"+directory
        self.weight_dir="models/models_weights/"+directory
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        path=pathlib.Path(self.weight_dir)
        path.mkdir(parents=True,exist_ok=True)

        root,dir,files = list(os.walk(self.directory))[0]
        for f in files:
            os.remove(root+"/"+f)



    def log_metric(self,metric_name,value,epoch):

        f=open(f"{self.directory}/{metric_name}.txt","a")
        if type(value)==list :
            f.write("\n".join(str(item) for item in value))
        else :
            f.write(f"{epoch} , {str(value)}")

        if self.is_wandb :
            wandb.log({metric_name: value})
    def save_weights(self,model):

        torch.save(model.state_dict(), f"{self.weight_dir}/{model._get_name()}.pt")

#-----------------------------------------------------------------------------------
def set_parameter_requires_grad(model, feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False
#-----------------------------------------------------------------------------------
class preprocessing() :
    def __init__(self,img_size,other=None):
        self.img_size=img_size
        self.added_transform=other

    def preprocessing(self):
        preprocess = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            self.added_transform
        ])
        return preprocess
#-----------------------------------------------------------------------------------
def collate_fn(batch):
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))

#-----------------------------------------------------------------------------------
num_classes = 14 #+empty
def top1(true, pred):
    true = np.argmax(true, axis=1)
    # labels=np.unique(true)
    labels = np.arange(0, num_classes)

    return top_k_accuracy_score(true, pred, k=1, labels=labels)


def top5(true, pred):
    true = np.argmax(true, axis=1)
    labels = np.arange(0, num_classes)

    return top_k_accuracy_score(true, pred, k=5, labels=labels)


def f1(true, pred):
    true = np.argmax(true, axis=1)
    pred = np.argmax(pred, axis=1)

    return sklearn.metrics.f1_score(true, pred, average='macro')  # weighted??


def precision(true, pred):
    true = np.argmax(true, axis=1)
    pred = np.argmax(pred, axis=1)
    return sklearn.metrics.precision_score(true, pred, average='macro')


def recall(true, pred):
    true = np.argmax(true, axis=1)
    pred = np.argmax(pred, axis=1)
    return sklearn.metrics.recall_score(true, pred, average='macro')


metrics = {
    "f1": f1,
    "precision": precision,
    "recall": recall,
    "top-1": top1,
    "top-5": top5
}
#-----------------------------------------------------------------------------------
def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)