
import torch
import tqdm
import numpy as np


# training loop

def training_loop(model,loader,optimizer,device,verbose,epoch) :
    running_loss=0
    i=0
    results=[torch.tensor([]),torch.tensor([])]
    model.train()

    for images,targets in loader:
        optimizer.zero_grad()

        image_H = images[0].shape[2]
        x = torch.tensor([])
        for image in images:
            x = torch.cat((x, image.view(1, 3, image_H, image_H)), dim=0)

        images=x
        images = images.to(device)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images,targets)

        losses = sum(loss for loss in loss_dict.values())



        losses.backward()
        optimizer.step()
        running_loss+=losses.detach()

        if verbose and i % 20 == 0:
            print(f" epoch : {epoch} , iteration :{i} ,running_loss : {running_loss}")


        #ending loop
        #del inputs,labels,loss,outputs #garbage management sometimes fails with cuda
        i+=1
    return running_loss,results



def validation_loop(model,loader,device):
    running_loss=0
    i=0
    model.eval()
    results = [torch.tensor([]), torch.tensor([])]
    with torch.no_grad() :
        for images,labels in loader:
            # get the inputs; data is a list of [inputs, labels]

            image_H=images[0].shape[2]
            x = torch.tensor([])
            for image in images:
                x = torch.cat((x, image.view(1, 3, image_H, image_H)), dim=0)

            images = x
            images = images.to(device)

            # forward + backward + optimize
            outputs = model(images)
            outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
            if len(outputs[0]['boxes']) != 0:
                boxes = outputs[0]['boxes'].data.numpy()
                scores = outputs[0]['scores'].data.numpy()
                # filter out boxes according to `detection_threshold`
                arg=np.argmax(scores)
                if scores[arg]>0.5 : #define some other threshold?
                box = boxes[arg].copy()

                # get all the predicited class names
                pred_classes = [i for i in outputs[0]['labels'].cpu().numpy()]





            #
            # if metrics :
            #     for key in metrics:
            #         metrics_results[key] += metrics[key](labels.cpu().numpy(), torch.nn.functional.softmax(
            #             outputs).cpu().detach().numpy()) / len(inputs)
            #ending loop
            #del inputs,labels,outputs,loss #garbage management sometimes fails with cuda
            i+=1
    return running_loss,results



def training(model,optimizer,training_loader,validation_loader,device="cpu",metrics=None,verbose=False,experiment=None,patience=5,epoch_max=50) :

    epoch=0



    train_loss_list=[]
    val_loss_list=[]
    best_loss=np.inf
    while patience>0 and epoch<epoch_max:  # loop over the dataset multiple times

        if not verbose:
            train_loss,results = training_loop(model, tqdm.tqdm(training_loader), optimizer, device, verbose,
                                                        epoch)
            val_loss, results = validation_loop(model, tqdm.tqdm(validation_loader), device
                                                        )

        else :
            train_loss,results = training_loop(model, training_loader, optimizer, device, verbose,
                                                        epoch)
            val_loss, results = validation_loop(model, validation_loader, device)



        #LOGGING DATA
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        if experiment :
            experiment.log_metric("training_loss",train_loss.tolist(),epoch=epoch)
            experiment.log_metric("validation_loss", val_loss.tolist(),epoch=epoch)
            for key in metrics :
                experiment.log_metric(key,metrics[key](results[1].numpy(),results[0].numpy()),epoch=epoch)



        if val_loss<best_loss :
            best_loss=val_loss
            #save the model after XX iterations : TODO : adjust when to save weights
            torch.save(model.state_dict(), f"models/models_weights/{model._get_name()}_{epoch}.pt")
        else :
            patience-=1
            print("patience has been reduced by 1")
        #Finishing the loop
        epoch+=1
    print('Finished Training')