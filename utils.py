from torchvision import transforms
import os

class Experiment() :
    def __init__(self,directory):
        self.directory=directory
        if not os.path.exists(directory):
            os.makedirs(directory)

        root,dir,files = list(os.walk(directory))[0]
        for f in files:
            os.remove(root+"/"+f)

    def log_metric(self,metric_name,value,epoch):

        f=open(f"{self.directory}/{metric_name}.txt","a")
        if type(value)==list :
            f.write("\n".join(str(item) for item in value))
        else :
            f.write(f"{epoch} , {str(value)}")


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