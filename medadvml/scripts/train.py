# tasks = ['pathmnist', 'dermamnist', 'octmnist', 'pneumoniamnist', 'retinamnist', 'breastmnist', 'bloodmnist', 'tissuemnist', 'organamnist', 'organcmnist', 'organsmnist']

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from medadvml.data.loader import get_dataloader
from medadvml.models import ModelWrapper 
from medadvml.config import Config

def parse_opts():
    parser = argparse.ArgumentParser(description='MedLearn')
    parser.add_argument("-t", "--task", type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    return parser.parse_args()

def main(opts):
    config  = Config(opts.task)
    dataloaders, class_names, dataset_sizes = get_dataloader(config)
    model = ModelWrapper(config)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.model.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # model.fit(dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=opts.epochs)

    model.visualize(dataloaders, class_names)

if __name__ == "__main__":
    opts = parse_opts()
    main(opts)



