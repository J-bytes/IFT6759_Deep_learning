from torchvision import transforms
import os
import torch
import pathlib
class Experiment() :
    def __init__(self,directory):
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

    def save_weights(self,model):

        torch.save(model.state_dict(), f"{self.weight_dir}/{model._get_name()}.pt")


def set_parameter_requires_grad(model, feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False


preprocess = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(320),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def collate_fn(batch):
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))

