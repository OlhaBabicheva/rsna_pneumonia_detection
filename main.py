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

def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)
    
def resume(model, filename):
    model.load_state_dict(torch.load(filename))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
in_dim = 1
out_dim = 1
batch_size = 64
num_workers = 8
l_rate = 0.0001
num_epochs = 30
e_stop_thresh = 5

train_transforms = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(0.49, 0.248),
                                    transforms.RandomAffine(degrees=(-5, 5), translate=(0, 0.05), scale=(0.9, 1.1)),
                                    transforms.RandomResizedCrop((64, 64), scale=(0.35, 1), antialias=True)
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

def train_val(train_loader, val_loader, model):
    size = len(train_loader.dataset)
    best_acc = -1
    best_epoch = -1

    model.train()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}\n---------------')

        for batch, (X_train, y_train) in enumerate(train_loader):
            X_train = X_train.to(device)
            y_train = y_train.to(device, dtype=torch.float32)

            # Compute predictions and loss
            pred = model(X_train)[:, 0]
            pred = pred.to(dtype=torch.float32)
            loss = loss_fn(pred, y_train)

            # Backpropagation
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss, current = loss.item(), (batch + 1) * len(X_train)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            n_correct = (pred.round() == y_train).sum()
            training_acc = n_correct/X_train.shape[0]
            print(f"training accuracy: {training_acc.item()*100}%")
        
        model.eval()
        with torch.no_grad():
            X_val, y_val = next(val_loader)
            X_val = X_val.to(device)
            y_val = y_val.to(device, dtype=torch.float32)

            pred = model(X_val)[:, 0]
            n_correct = (pred.round() == y_val).sum()
            val_acc = n_correct/X_val.shape[0]
            print(f"End of epoch {epoch+1}: validation accuracy = {val_acc.item()*100}%")

            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                checkpoint(model, "best_model.pth")
            elif epoch - best_epoch > e_stop_thresh:
                print(f"Early stopped training at epoch {epoch+1}")
                break

if __name__ == '__main__':
    print(f'Using {device}')
    print(f"There are {len(train_dataset)} images in training set and {len(val_dataset)} images in validation set\n")
    val_iter = iter(val_loader)
    train_val(train_loader, val_iter, model)
