
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader, random_split
from models.simple import SimpleLinearModel
from PIL import Image

from torch.utils.tensorboard import SummaryWriter 
from torch.optim.lr_scheduler import StepLR
from scripts.test import visualize_results
import torchvision.transforms as transforms
from utilities.platform import get_torch_device
from torch.nn.utils import clip_grad_norm_



def parse_opts():
    parser = argparse.ArgumentParser(description='Train a simple linear model on sinusoidal data')
    parser.add_argument("-n", "--n-data-pairs", type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=2**16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--val_split', type=float, default=0.1, help='Fraction of data to use for validation')
    return parser.parse_args()

def train(model, dataloader, optimizer, criterion, writer, epoch, device):
    model.train()
    pbar = tqdm(dataloader, leave=False)
    for x_batch, y_batch in pbar:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch.unsqueeze(1))
        loss = criterion(outputs, y_batch.unsqueeze(1))
        loss.backward()
        clip_grad_norm_(model.parameters(), 3.0)
        optimizer.step()

        writer.add_scalar('Loss/Train', loss.item(), epoch)
        pbar.set_description(f"Training - Loss: {loss.item()}")

def validate(model, dataloader, criterion, writer, epoch, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation", leave=False)
        for i, (x_batch, y_batch) in enumerate(pbar):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(x_batch.unsqueeze(1))
            loss = criterion(outputs, y_batch.unsqueeze(1))
            total_loss += loss.item()
            avg_loss = total_loss / (i + 1)
            
            writer.add_scalar('Loss/Validation', avg_loss, epoch)
        pbar.set_description(f"Validation - Loss: {avg_loss}")

def main(opts):
    device = get_torch_device()
    generator = gn.LinspaceSampleGenerator(num_samples=opts.n_data_pairs).set_val_function(gn.GenerationType.NoisySinus).create()
    spiker = gn.Spiker(np.pi , 0.05)
    dataset = ds.SpikedSignalDataset(generator, spiker)

    val_size = int(len(dataset) * opts.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=opts.batch_size, shuffle=False)

    model = SimpleLinearModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=opts.learning_rate)
    criterion = nn.MSELoss()
    scheduler = StepLR(optimizer, step_size=1, gamma=0.998)
    writer = SummaryWriter(log_dir='./src/logs') 

    for epoch in tqdm(range(opts.epochs), desc="Epochs"):
        train(model, train_loader, optimizer, criterion, writer, epoch, device)
        validate(model, val_loader, criterion, writer, epoch, device)

        # scheduler.step()

        if (epoch + 1) % 10 == 0:
            plot_dir = './src/logs/plots'
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = f'./src/logs/plots/epoch_{epoch + 1}.png'
            visualize_results(model, dataset, epoch, plot_path, device)
            img = Image.open(plot_path)
            img_tensor = transforms.ToTensor()(img) 
            writer.add_image('Model Predictions', img_tensor, epoch)

        if (epoch + 1) % 100 == 0:
            checkpoint_path = f'./src/checkpoints/model_epoch_{epoch + 1}.pth'
            torch.save(model.state_dict(), checkpoint_path)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)


    writer.close()  

if __name__ == "__main__":
    opts = parse_opts()
    main(opts)
