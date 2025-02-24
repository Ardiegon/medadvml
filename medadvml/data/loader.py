# C:\Users\oskbs\.medmnist path

import medmnist
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from medadvml.config import Config, BATCH_SIZE


def get_transform(phase):
    if phase == "train":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    if phase in ["val", "test"]:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

def get_dataloader(config: Config):
    size=128
    DataClass = getattr(medmnist, config['python_class'])
    train_dataset = DataClass(split='train', transform=get_transform("train"), size=size, download=True)
    val_dataset = DataClass(split='val', transform=get_transform("val"), size=size, download=True)
    test_dataset = DataClass(split='test', transform=get_transform("test"), size=size, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)

    class_names = list(config["label"].values())
    dataset_sizes = config["n_samples"]
    
    dataloaders = {"train": train_loader,
                   "val": val_loader,
                   "test": test_loader}
    
    
    return dataloaders, class_names, dataset_sizes

