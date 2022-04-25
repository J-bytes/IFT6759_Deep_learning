#------python import------------------------------------
import warnings
import torch
import wandb
import os
import argparse
#-----local imports---------------------------------------
from training.training import training
from training.dataloaders.cct_dataloader import CustomImageDataset
from custom_utils import set_parameter_requires_grad,Experiment,preprocessing,metrics




# -----------cuda optimization tricks-------------------------
# DANGER ZONE !!!!!
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)
torch.backends.cudnn.benchmark = True

#----------- parse arguments----------------------------------
def init_parser() :
    parser = argparse.ArgumentParser(description='Launch training for a specific model')

    parser.add_argument('--model',
                        default='alexnet',
                        const='all',
                        type=str,
                        nargs='?',
                        choices=["alexnet","resnext50_32x4d","vgg19"],
                        required=True,
                        help='Choice of the model')
    parser.add_argument('--dataset',
                        default='2',
                        const='all',
                        type=str,
                        nargs='?',
                        choices=['1', '2', '3',"4"],
                        required=True,
                        help='Version of the dataset')
    parser.add_argument('--img_size',
                        default=320,
                        const='all',
                        type=int,
                        nargs='?',
                        required=False,
                        help='width and length to resize the images to. Choose a value between 320 and 608.')

    parser.add_argument('--wandb',
                        default=False,
                        const='all',
                        type=bool,
                        nargs='?',
                        choices=[True,False],
                        required=False,
                        help='True or False, do you wish (and did you setup) wandb? You will need to add the project name in the initialization of wandb in train.py')

    return parser

def main() :
    parser=init_parser()
    args = parser.parse_args()

    prepro=preprocessing(img_size=args.img_size)
    preprocess=prepro.preprocessing()
    version = args.dataset

    num_classes=14

    train_dataset = CustomImageDataset(f"data/data/data_split{version}/train", transform=preprocess)
    val_dataset = CustomImageDataset(f"data/data/data_split{version}/valid", transform=preprocess)

    criterion = torch.nn.CrossEntropyLoss()

    # -----------model initialisation------------------------------

    model=torch.hub.load('pytorch/vision:v0.10.0', args.model, pretrained=True)
    if args.model in ["vgg19","alexnet"] :
        if args.model=="vgg19" :
            batch_size = 8
            #set_parameter_requires_grad(model, feature_extract=True)
        else :
            batch_size = 168
        #model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes, bias=True)

    else :#for resnext
        model.fc = torch.nn.Linear(2048, num_classes)
        batch_size = 8
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        warnings.warn("No gpu is available for the computation")


    print("The model has now been successfully loaded into memory")





    # -------data initialisation-------------------------------
    #os.environ["WANDB_MODE"] = "offline"




    training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=int(os.cpu_count()/3),pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=int(batch_size*2), shuffle=False, num_workers=int(os.cpu_count()/3),pin_memory=True)
    print("The data has now been loaded successfully into memory")
    # ------------training--------------------------------------------
    print("Starting training now")



    #send model to gpu
    model = model.to(device)

    #initialize metrics loggers
    if args.wandb :
        wandb.init(project='mila-prof-master-gang', tags=[args.model,args.version])
        wandb.watch(model)

    experiment = Experiment(f"{model._get_name()}/v{version}",wandb=args.wandb)

    optimizer = torch.optim.AdamW(model.parameters())

    training(model,optimizer,criterion,training_loader,validation_loader,device,minibatch_accumulate=1,epoch_max=50,patience=5,experiment=experiment,metrics=metrics)

if __name__ == "__main__":
     main()