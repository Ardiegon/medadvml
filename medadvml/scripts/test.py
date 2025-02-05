import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader,  random_split
from models.simple import SimpleLinearModel
import data.datasets as ds 
import  data.generator as gn
import matplotlib.pyplot as plt

def visualize_results(model, dataset, epoch, plot_path, device):
    model.eval()
    
    x = torch.tensor([item[0] for item in dataset.dataset], dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        y_pred = model(x)

    y_true = torch.tensor([item[1] for item in dataset.dataset], dtype=torch.float32).to(device)

    plt.figure(figsize=(10, 5))
    plt.scatter(x.cpu().numpy(), y_true.cpu().numpy(), label='GT', color='blue', alpha=0.5)
    plt.plot(x.cpu().numpy(), y_pred.cpu().numpy(), label='Predicted', color='red')
    plt.title('Sinusoidal Data vs. Model Prediction - Epoch {}'.format(epoch))
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.legend()

    plt.savefig(plot_path)
    plt.close() 

def main():
    generator = gn.LinspaceSampleGenerator(num_samples=10000).set_val_function(gn.GenerationType.NoisySinus).create()
    spiker  = gn.Spiker(np.pi*3, 0.1)
    dataset = ds.SpikedSignalDataset(generator, spiker)
    
    epoch = 1000
    model = SimpleLinearModel()
    model.load_state_dict(torch.load(f'./src/checkpoints/model_epoch_{epoch}.pth'))
    visualize_results(model, dataset)

if __name__ == "__main__":
    main()
