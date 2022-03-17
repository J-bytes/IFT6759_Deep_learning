
import torch
import tqdm
import numpy as np


# training loop

def training_loop(model,loader,optimizer,criterion,device,verbose,epoch,metrics) :
    running_loss=0
    i=0
    metrics_results = {}
    model.train()
    if metrics :
        for key in metrics:
            metrics_results[key] = 0
    for inputs,labels in loader:
        # get the inputs; data is a list of [inputs, labels]

        inputs,labels=inputs.to(device),labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.detach()

        if verbose and i % 20 == 0:
            print(f" epoch : {epoch} , iteration :{i} ,running_loss : {running_loss}")

        if metrics :
            for key in metrics:
                metrics_results[key]+= metrics[key](outputs,labels)/len(inputs)

        #ending loop
        del inputs,labels,loss,outputs #garbage management sometimes fails with cuda
        i+=1
    return running_loss,metrics_results



def validation_loop(model,loader,criterion,device,verbose,epoch,metrics):
    running_loss=0
    i=0
    model.eval()
    metrics_results={}
    if metrics :
        for key in metrics :
            metrics_results[key]=0
    with torch.no_grad() :
        for inputs,labels in loader:
            # get the inputs; data is a list of [inputs, labels]

            inputs,labels=inputs.to(device),labels.to(device)

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss+=loss.detach()

            if verbose and i%20==0 :
                print(f" epoch : {epoch} , iteration :{i} ,running_loss : {running_loss}")

            if metrics :
                for key in metrics:
                    metrics_results[key] += metrics[key](outputs, labels) / len(inputs)
            #ending loop
            del inputs,labels,outputs,loss #garbage management sometimes fails with cuda
            i+=1
    return running_loss,metrics_results



def training(model,optimizer,criterion,training_loader,validation_loader,device="cpu",metrics=None,verbose=False,experiment=None,patience=5,epoch_max=50) :

    epoch=0



    train_loss_list=[]
    val_loss_list=[]
    best_loss=np.inf
    while patience>0 and epoch<epoch_max:  # loop over the dataset multiple times

        if not verbose:
            train_loss, metrics_results = training_loop(model, tqdm.tqdm(training_loader), optimizer, criterion, device, verbose,
                                                        epoch, metrics)
            val_loss, metrics_results = validation_loop(model, tqdm.tqdm(validation_loader), criterion, device, verbose, epoch,
                                                        metrics)

        else :
            train_loss, metrics_results = training_loop(model, training_loader, optimizer, criterion, device, verbose,
                                                        epoch, metrics)
            val_loss, metrics_results = validation_loop(model, validation_loader, criterion, device, verbose, epoch,
                                                        metrics)


        #other evaluation metrics to display :


        #log the results :
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        if experiment :
            experiment.log_metric("training_loss",train_loss.tolist(),epoch=epoch)
            experiment.log_metric("validation_loss", val_loss.tolist(),epoch=epoch)
            for key in metrics_results :
                experiment.log_metric(key,metrics_results[key],epoch=epoch)



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