# C:\Users\oskbs\.medmnist path

import medmnist
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from imblearn.over_sampling import SMOTE
import numpy as np

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

def undersample(dataset, number_of_indices):
    labels = torch.tensor(dataset.labels).squeeze()
    unique_classes = torch.unique(labels).tolist()
    indices = []

    for cls in unique_classes:
        cls_indices = (labels == cls).nonzero(as_tuple=True)[0].tolist()
        sampled_indices = np.random.choice(cls_indices, min(len(cls_indices), number_of_indices), replace=False)
        indices.extend(sampled_indices)
    
    return Subset(dataset, indices) 

def oversample(dataset, number_of_indices):
    data = torch.stack([dataset[i][0] for i in range(len(dataset))])
    labels = torch.tensor(dataset.labels).squeeze().numpy()
    data_flat = data.view(data.shape[0], -1).numpy()

    smote = SMOTE(sampling_strategy={cls: number_of_indices for cls in np.unique(labels) if sum(labels == cls) < number_of_indices}, k_neighbors=3)
    resampled_data, resampled_labels = smote.fit_resample(data_flat, labels)
    resampled_data = torch.tensor(resampled_data).view(-1, *data.shape[1:])
    resampled_labels = torch.tensor(resampled_labels)

    final_indices = []
    for cls in np.unique(resampled_labels.numpy()):
        cls_indices = np.where(resampled_labels.numpy() == cls)[0]
        sampled_cls_indices = np.random.choice(cls_indices, min(len(cls_indices), number_of_indices), replace=False)
        final_indices.extend(sampled_cls_indices)

    resampled_data = resampled_data[final_indices]
    resampled_labels = resampled_labels[final_indices]

    return TensorDataset(resampled_data, resampled_labels)


def modify(dataset, name):
    def dermamnist(dataset):
        return oversample(dataset, number_of_indices=1400)
    
    if name in locals().keys():
        return locals().get(name)(dataset)
    else:
        return dataset

def get_dataloader(config: Config):
    size = 128
    DataClass = getattr(medmnist, config['python_class'])
    train_dataset = DataClass(split='train', transform=get_transform("train"), size=size, download=True)
    val_dataset = DataClass(split='val', transform=get_transform("val"), size=size, download=True)
    test_dataset = DataClass(split='test', transform=get_transform("test"), size=size, download=True)

    train_dataset = modify(train_dataset, config.name)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)

    class_names = list(config["label"].values())
    dataset_sizes = config["n_samples"]
    
    dataloaders = {"train": train_loader,
                   "val": val_loader,
                   "test": test_loader}
    
    return dataloaders, class_names, dataset_sizes
