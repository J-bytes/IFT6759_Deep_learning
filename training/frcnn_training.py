import torch
import tqdm
import numpy as np
import wandb


# training loop
from torch.optim.lr_scheduler import StepLR
from training.engine import train_one_epoch, evaluate
import training.utils as utils


def training_loop(model,loader,optimizer,device,verbose,epoch) :
    running_loss=0
    i=0

    model.train()

    for images,targets in loader:
        for param in model.parameters() :
            param.grad=None

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
    return running_loss

@torch.no_grad()
def validation_loop(model,loader,device,verbose,epoch) :
    running_loss=0
    i=0

    model.train()

    for images,targets in loader:
        for param in model.parameters() :
            param.grad=None

        image_H = images[0].shape[2]
        x = torch.tensor([])
        for image in images:
            x = torch.cat((x, image.view(1, 3, image_H, image_H)), dim=0)

        images=x
        images = images.to(device)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images,targets)
        losses = sum(loss for loss in loss_dict.values())
        running_loss+=losses

        if verbose and i % 20 == 0:
            print(f" epoch : {epoch} , iteration :{i} ,running_loss : {running_loss}")


        #ending loop
        #del inputs,labels,loss,outputs #garbage management sometimes fails with cuda
        i+=1
    return running_loss

@torch.no_grad()
def test_loop(model,loader,device,verbose, epoch):

    i=0
    model.eval()
    # results = [torch.tensor([]), torch.tensor([])]
    pred_label_list = []
    true_label_list = []

    for images, targets in loader:
        image_H = images[0].shape[2]
        x = torch.tensor([])
        for image in images:
            x = torch.cat((x, image.view(1, 3, image_H, image_H)), dim=0)

        images = x.to(device)
        #loss_dict = model(images, targets)

        #losses = sum(loss for loss in loss_dict.values())

        #running_loss += losses.detach()
        true_label = int(targets[0]['labels'][0])
        true_label_list.append(true_label)
        # results[1] = torch.cat((results[1], true_label))
        outputs = model(images)
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        if len(outputs[0]['boxes']) != 0:
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            # filter out boxes according to `detection_threshold`
            arg = np.argmax(scores)
            print('scores', scores)
            if scores[arg] > 0.05:  # define some other threshold?
                box = boxes[arg].copy()
                pred = outputs[0]['labels'][arg]
                print('pred', int(pred), scores[arg])
                # results[0] = torch.cat((results[0], pred))
                pred_label_list.append(int(pred))
        else:
            # results[0].cat(results[0],0)
            pred_label_list.append(0)

            # get all the predicited class names
            pred_classes = [i for i in outputs[0]['labels'].cpu().numpy()]


        # ending loop
        # del inputs,labels,loss,outputs #garbage management sometimes fails with cuda
        i += 1
    results=[torch.tensor(pred_label_list), torch.tensor(true_label_list)]
    return results



def training(model,optimizer,criterion,training_loader,validation_loader,device="cpu",metrics=None,verbose=False,experiment=None,patience=5,epoch_max=50, batch_size=1) :
    wandb.init(project='mila-prof-master-gang', tags=[model._get_name()])
    wandb.config = {
        "epochs": epoch_max,
        "batch_size": batch_size,
        "model": model,
        "metrics": metrics,
        "optimizer": optimizer,
        "patience": patience
    }
    wandb.watch(model, log_freq=batch_size)

    epoch=0

    train_loss_list=[]
    val_loss_list=[]
    best_loss=np.inf

    while patience>0 and epoch<epoch_max:  # loop over the dataset multiple times
        if not verbose:
            train_loss= training_loop(model, tqdm.tqdm(training_loader), optimizer, device, verbose, epoch)
            val_loss = validation_loop(model, tqdm.tqdm(validation_loader), device, verbose, epoch)
        else :
            train_loss = training_loop(model, training_loader, optimizer, device, verbose, epoch)
            val_loss = validation_loop(model, validation_loader, device, verbose, epoch)

        results=test_loop(model, validation_loader, device, verbose, epoch) #return the results for metrics evaluation
        #LOGGING DATA _ COMET
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        if experiment :
            experiment.log_metric("training_loss",train_loss.tolist(),epoch=epoch)
            wandb.log({"train_loss": train_loss,
                       "val_loss": val_loss})

            for key in metrics :
                experiment.log_metric(key,metrics[key](results[1].numpy(),results[0].numpy()),epoch=epoch)
                wandb.log({key : metrics[key](results[1].numpy(),results[0].numpy())})


        if val_loss<best_loss :
            best_loss=val_loss
            #save the model after XX iterations : TODO : adjust when to save weights
            torch.save(model.state_dict(), f"models/models_weights/best_{model._get_name()}_{epoch}.pt")
            print('Saved Weights coz best loss found so far')
        else :
            patience-=1
            print("patience has been reduced by 1")
        #Finishing the loop
        epoch+=1

    # LOGGING DATA _ WANDB
    wandb.run.summary["best_loss"] = best_loss
    print('Finished Training', best_loss)
    torch.save(model.state_dict(), f"models/models_weights/last_{model._get_name()}_{epoch}.pt")
    print('Final model saved')



def training_pytorch(model,optimizer,training_loader,validation_loader,test_loader, device="cpu",metrics=None,verbose=False,experiment=None,patience=5,epoch_max=50, batch_size=1) :
    wandb.init(project='mila-prof-master-gang', tags=[model._get_name()])
    wandb.config = {
        "epochs": epoch_max,
        "batch_size": batch_size,
        "model": model,
        "metrics": metrics,
        "optimizer": optimizer,
        "patience": patience
    }
    wandb.watch(model, log_freq=batch_size)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    all_train_logs = []
    all_valid_logs = []
    epoch = 0
    while patience>0 and epoch<epoch_max:
        # train for one epoch, printing every 10 iterations
        train_logs = train_one_epoch(model, optimizer, training_loader, device, epoch, print_freq=100)
        all_train_logs.append(train_logs)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, validation_loader, device=device)


        for images, targets in validation_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.no_grad():
                loss_dict = model(images, targets)
                loss_dict = [{k: loss.to('cpu')} for k, loss in loss_dict.items()]
                all_valid_logs.append(loss_dict)

        # for images, targets in validation_loader:  # can do batch of 10 prob.
        #     images = [image.to(device) for image in images]
        #     targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        #
        #     with torch.no_grad():
        #         cis_loss_dict = model(images, targets)
        #         cis_loss_dict = [{k: loss.to('cpu')} for k, loss in cis_loss_dict.items()]
        #         all_cis_valid_logs.append(cis_loss_dict)
                #LOGGING DATA _ COMET

        # if experiment :
        #     experiment.log_metric("training_loss",train_loss.tolist(),epoch=epoch)
        #     wandb.log({"train_loss": train_loss,
        #                "val_loss": val_loss})
        #
        #     for key in metrics :
        #         experiment.log_metric(key,metrics[key](results[1].numpy(),results[0].numpy()),epoch=epoch)
        #         wandb.log({key : metrics[key](results[1].numpy(),results[0].numpy())})

        epoch += 1
    print("---- EVALUATION ON TEST SET: ----")

    torch.save(model.state_dict(), f"models/models_weights/pytorch_training/last_{model._get_name()}_{epoch}.pt")
    evaluate(model, test_loader, device=device)