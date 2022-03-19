from torchvision import transforms
import os

class Experiment() :
    def __init__(self,directory):
        self.directory=directory
        if not os.path.exists(directory):
            os.makedirs(directory)

        files = list(os.walk(directory))[0][2]
        for f in files:
            os.remove(f)

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
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])