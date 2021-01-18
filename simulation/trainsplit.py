import torch
import torchvision
import torch.nn as nn
import numpy as np
import json
from tqdm import tqdm
import utils
import validate
import argparse
import models.vgg
import models.densenet
import validate
from torch.utils.data import Dataset, DataLoader
from ray import tune
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument("json_path", type=str)


def train(modela, modelb, modelc, device, data_loader, optimizer, loss_fn):
    modela.train()
    modelb.train()
    modelc.train()

    loss_avg = utils.RunningAverage()

    with tqdm(total=len(data_loader)) as t:
        for batch_idx, data in enumerate(data_loader):

            inputs = data[0].to(device)
            target = data[1].to(device)

            outputa = modela(inputs)
            outputanew = outputa.detach()
            outputanew.requires_grad = True
            outputb = modelb(outputanew)
            outputbnew = outputb.detach()
            outputbnew.requires_grad = True
            outputs = modelc(outputbnew)            

            loss = loss_fn(outputs, target)            

            ######### BEGIN SPLIT C ##########
            optimizer["c"].zero_grad()
            loss.backward()
            optimizer["c"].step()            

            ########### END SPLIT C ##############            

            ########## BEGIN SPLIT B ############
            targetb = outputb - outputbnew.grad
            lossb = nn.MSELoss(reduction='sum')(outputb , targetb.detach())    #This is the MSE objective function            

            optimizer["b"].zero_grad()
            lossb.backward()
            optimizer["b"].step()            

            ######### END SPLIT B ##############            

            ######## BEGIN SPLIT A #############
            targeta = outputa - outputanew.grad
            lossa = nn.MSELoss(reduction='sum')(outputa, targeta.detach())            

            optimizer["a"].zero_grad()
            lossa.backward()
            optimizer["a"].step()
            ###### END SPLIT A ###############

            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()
    return loss_avg()


def train_and_evaluate(modela, modelb, modelc, device, train_loader, val_loader, optimizer, loss_fn, params, scheduler):
    best_acc = 0.0

    for epoch in range(params.epochs):
        avg_loss = train(modela, modelb, modelc, device, train_loader, optimizer, loss_fn)
        print("Epoch {}/{} Loss:{}".format(epoch, params.epochs, avg_loss))

        acc = validate.evaluate(modela, modelb, modelc, device, val_loader)

        tune.report(mean_accuracy=acc)

        is_best = (acc > best_acc)
        if is_best:
            best_acc = acc
        if scheduler:
            scheduler["a"].step()
            scheduler["b"].step()
            scheduler["c"].step()

def trainconfig(config):
    params = utils.Params(args.json_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(160)

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_dataset = torchvision.datasets.CIFAR10(root="~/cifar10", train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = torchvision.datasets.CIFAR10(root='~/cifar10', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers, pin_memory=True)

    if config["model"] == "VGG":
        modela = models.vgg.SplitA().to(device)
        modelb = models.vgg.SplitB().to(device)
        modelc = models.vgg.SplitC().to(device)

        optimizera = torch.optim.SGD(modela.parameters(), params.lr, momentum=params.momentum, weight_decay=params.weight_decay)
        optimizerb = torch.optim.SGD(modelb.parameters(), params.lr, momentum=params.momentum, weight_decay=params.weight_decay)
        optimizerc = torch.optim.SGD(modelc.parameters(), params.lr, momentum=params.momentum, weight_decay=params.weight_decay)
        optimizer = {"a":optimizera, "b":optimizerb, "c":optimizerc}

        schedulera = torch.optim.lr_scheduler.MultiStepLR(optimizera, [150, 250, 350], gamma=0.1)
        schedulerb = torch.optim.lr_scheduler.MultiStepLR(optimizerb, [150, 250, 350], gamma=0.1)
        schedulerc = torch.optim.lr_scheduler.MultiStepLR(optimizerc, [150, 250, 350], gamma=0.1)
        scheduler = {"a": schedulera, "b": schedulerb, "c":schedulerc}

    elif config["model"] == "DenseNet":
        modela = models.densenet.SplitA().to(device)
        modelb = models.densenet.SplitB().to(device)
        modelc = models.densenet.SplitC().to(device)

        optimizera = torch.optim.SGD(modela.parameters(), params.lr, momentum=params.momentum, weight_decay=params.weight_decay)
        optimizerb = torch.optim.SGD(modelb.parameters(), params.lr, momentum=params.momentum, weight_decay=params.weight_decay)
        optimizerc = torch.optim.SGD(modelc.parameters(), params.lr, momentum=params.momentum, weight_decay=params.weight_decay)
        optimizer = {"a":optimizera, "b":optimizerb, "c":optimizerc}

        schedulera = torch.optim.lr_scheduler.MultiStepLR(optimizera, [150, 250, 350], gamma=0.1)
        schedulerb = torch.optim.lr_scheduler.MultiStepLR(optimizerb, [150, 250, 350], gamma=0.1)
        schedulerc = torch.optim.lr_scheduler.MultiStepLR(optimizerc, [150, 250, 350], gamma=0.1)
        scheduler = {"a": schedulera, "b": schedulerb, "c":schedulerc}

    loss_fn = nn.CrossEntropyLoss()

    train_and_evaluate(modela, modelb, modelc, device, train_loader, val_loader, optimizer, loss_fn, params, scheduler)

if __name__ == "__main__":
    args = parser.parse_args()

    analysis = tune.run(trainconfig, config={
        "model": tune.grid_search(['VGG', 'DenseNet']),
        },resources_per_trial={
            "cpu": 1,
            "gpu": 0.5
        },name="split_setup",local_dir="splitlearning_results",queue_trials=True)
    
    print("Best configuration", analysis.get_best_config(metric="mean_accuracy"))
  

