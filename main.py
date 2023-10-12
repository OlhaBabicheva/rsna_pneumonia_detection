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
out_dim = 2
batch_size = 64
num_workers = 4
num_epochs = 50
e_stop_thresh = 10

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

model = VGGNet(in_dim, out_dim).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

def train_val(train_loader, val_loader, model):
    size = len(train_loader.dataset)
    best_acc = -1
    best_epoch = -1

    model.train()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}')
        print('-'*15)

        for batch, (X_train, y_train) in enumerate(train_loader):
            X_train = X_train.to(device)
            y_train = y_train.to(device)

            # Compute predictions and loss
            pred = model(X_train)
            loss = loss_fn(pred, y_train)

            # Backpropagation
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            # Running training accuracy
            _, prediction = pred.max(1)
            
            loss, current = loss.item(), (batch + 1) * len(X_train)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            n_correct = (prediction == y_train).sum()
            training_acc = n_correct/X_train.shape[0]
            print(f"training accuracy: {training_acc.item()*100}%")
        
        model.eval()
        with torch.no_grad():
            X_val, y_val = next(val_loader)
            X_val = X_val.to(device)
            y_val = y_val.to(device)

            pred = model(X_val)
            _, prediction = pred.max(1)
            n_correct = (prediction == y_val).sum()
            validation_acc = n_correct/X_val.shape[0]
            print(f"validation accuracy: {validation_acc.item()*100}%")
            print('-'*15)

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
