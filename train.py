import argparse
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from model import Net

argparser = argparse.ArgumentParser()
argparser.add_argument("--data-path", type=str, required=True)
args = argparser.parse_args()

def load_file(path):
    return np.load(path).astype(np.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device}')

# Parameters
batch_size = 64
num_workers = 4
l_rate = 1e-3
num_epochs = 7

train_transforms = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(0.49, 0.248),
                                    transforms.RandomAffine(
                                        degrees=(-5, 5), translate=(0, 0.05), scale=(0.9, 1.1)),
                                        transforms.RandomResizedCrop((224, 224), scale=(0.35, 1)
                                    )
])

val_transforms = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.49], [0.248]),
])

train_dataset = torchvision.datasets.DatasetFolder(
    f"{args.data_path}/train/",
    loader=load_file, 
    extensions="npy", 
    transform=train_transforms
)

val_dataset = torchvision.datasets.DatasetFolder(
    f"{args.data_path}/val/",
    loader=load_file, 
    extensions="npy", 
    transform=val_transforms
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

print(f"There are {len(train_dataset)} train images and {len(val_dataset)} val images")
