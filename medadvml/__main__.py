# tasks = ['pathmnist', 'dermamnist', 'octmnist', 'pneumoniamnist', 'retinamnist', 'breastmnist', 'bloodmnist', 'tissuemnist', 'organamnist', 'organcmnist', 'organsmnist']

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from medadvml.data.loader import get_dataloader
from medadvml.models import ModelWrapper 
from medadvml.config import Config
from medadvml.utilities.losses import FocalLoss

def parse_opts():
    parser = argparse.ArgumentParser(description='MedLearn')
    parser.add_argument("-t", "--task", type=str, required=True)
    parser.add_argument("-e", '--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument("-a", "--analysis", action="store_true")
    parser.add_argument("-f", "--fitting", action="store_true")
    parser.add_argument("-m", "--matrix", action="store_true")
    parser.add_argument("-v", "--visualisation", action="store_true")
    return parser.parse_args()

def get_soup_for_task(model, name):
    def pathmnist(model):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.model.parameters(), lr=0.001, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        return criterion, optimizer, scheduler
    
    def dermamnist(model):
        criterion = FocalLoss(gamma=2.0)
        optimizer = torch.optim.AdamW(model.model.parameters(), lr=0.001, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        return criterion, optimizer, scheduler
    
    def brestmnist(model):
        criterion = FocalLoss(gamma=2.0)
        optimizer = torch.optim.AdamW(model.model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        return criterion, optimizer, scheduler
    
    def pneumoniamnist(model):
        criterion = FocalLoss(gamma=2.0)
        optimizer = torch.optim.AdamW(model.model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        return criterion, optimizer, scheduler
    
    def retinamnist(model):
        criterion = FocalLoss(gamma=2.0)
        optimizer = torch.optim.AdamW(model.model.parameters(), lr=0.001, weight_decay=1e-2)
        scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=0.0005, anneal_strategy="cos")
        return criterion, optimizer, scheduler

    if name in locals().keys():
        print(f"loading soup for {name}")
        return locals().get(name)(model)
    else:
        criterion = FocalLoss(gamma=2.0)
        optimizer = torch.optim.AdamW(model.model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        return criterion, optimizer, scheduler
        
    

def main(opts):
    config  = Config(opts.task)
    dataloaders, class_names, dataset_sizes = get_dataloader(config)
    model = ModelWrapper(config)

    if opts.analysis:
        model.analyze_data(dataloaders, class_names)
    if opts.fitting:
        criterion, optimizer, scheduler = get_soup_for_task(model, config.name)
        model.fit(dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=opts.epochs)
    if opts.matrix:
        model.confusion_matrix(dataloaders, class_names)
    if opts.visualisation:
        model.visualize(dataloaders, class_names)

if __name__ == "__main__":
    opts = parse_opts()
    main(opts)



