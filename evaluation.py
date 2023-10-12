import argparse
import torch
import torchvision
import torchmetrics
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from model import VGGNet

argparser = argparse.ArgumentParser()
argparser.add_argument("--data-path", type=str, required=True)
argparser.add_argument("--model-path", type=str, required=True)
args = argparser.parse_args()


def load_file(path):
    return np.load(path).astype(np.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
batch_size = 64
num_workers = 0

val_transforms = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.49], [0.248]),
])

val_dataset = torchvision.datasets.DatasetFolder(
    f"{args.data_path}/val/",
    loader=load_file, 
    extensions="npy", 
    transform=val_transforms
)

val_loader = DataLoader(val_dataset, batch_size = batch_size, num_workers = num_workers, shuffle = False)

model = VGGNet()
model.load_state_dict(torch.load(args.model_path))
model.eval()
model.to(device)

predictions = []
labels = []


for batch, (X_val, y_val) in enumerate(val_loader):
    with torch.no_grad():
        X_val = X_val.to(device)
        y_val = y_val.to(device)
        pred = model(X_val)
        _, prediction = pred.max(1)
        predictions.append(prediction)
        labels.append(y_val)

predictions = torch.cat([pred for pred in predictions], dim=0)
labels = torch.cat([label for label in labels], dim=0)

accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=2).to(device)(predictions, labels)
precision = torchmetrics.Precision(task='multiclass', num_classes=2).to(device)(predictions, labels)
recall = torchmetrics.Recall(task='multiclass', num_classes=2).to(device)(predictions, labels)
cmatrix = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=2).to(device)(predictions, labels)
cmatrix_threshed = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=2, threshold=0.25).to(device)(predictions, labels)

print(f"Val Accuracy: {accuracy}")
print(f"Val Precision: {precision}")
print(f"Val Recall: {recall}")
print(f"Confusion Matrix:\n {cmatrix}")
print(f"Confusion Matrix 2:\n {cmatrix_threshed}")

fig, axis = plt.subplots(3, 3, figsize=(9, 9))

for i in range(3):
    for j in range(3):
        rnd_idx = np.random.randint(0, len(predictions))
        axis[i][j].imshow(val_dataset[rnd_idx][0][0], cmap="bone")
        axis[i][j].set_title(f"Pred:{int(predictions[rnd_idx] > 0.5)}, Label:{labels[rnd_idx]}")
        axis[i][j].axis("off")

plt.show()