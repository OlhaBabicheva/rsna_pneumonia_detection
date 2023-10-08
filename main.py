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

# Parameters
in_dim = 1
out_dim = 1
batch_size = 32
num_workers = 4
l_rate = 1e-3
num_epochs = 7

train_transforms = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(0.49, 0.248),
                                    transforms.RandomAffine(degrees=(-5, 5), translate=(0, 0.05), scale=(0.9, 1.1)),
                                    transforms.RandomResizedCrop((224, 224), scale=(0.35, 1), antialias=True)
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

train_loader = DataLoader(train_dataset, batch_size = batch_size, num_workers = num_workers, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = batch_size, num_workers = num_workers, shuffle = False)

model = Net(in_dim, out_dim).to(device)

loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = l_rate)

def train(dataloader, model):
    size = len(dataloader.dataset)
    best_accuracy = -1

    model.train()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}\n---------------')

        for batch, (X, y) in enumerate(train_loader):
            X = X.to(device)
            y = y.to(device, dtype=torch.float32)

            # Compute predictions and loss
            pred = model(X)[:, 0]
            pred = pred.to(dtype=torch.float32)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            prediction = pred.round()
            n_correct = (prediction == y).sum()
            training_acc = n_correct/X.shape[0]
            print(f"accuracy: {training_acc.item()*100}%")
    
    torch.save(model.state_dict(), 'model.pt')

if __name__ == '__main__':
    print(f'Using {device}')
    print(f"There are {len(train_dataset)} images in training set and {len(val_dataset)} images in validation set\n")
    train(train_loader, model)
