import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, in_dim, num_classes):
        super(Net, self).__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_dim, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(256*64*64, 256),
            nn.ReLU(),
            nn.Linear(256, 128, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc_layers(x)
        return x

def test():
    batch_size, in_channels, img_size = 64, 1, 64
    model = Net(1, 1)
    x = torch.randn(batch_size, in_channels, img_size, img_size)
    print(model(x).shape)
    assert model(x).shape == (batch_size, 1), "Generator test failed"
    print("Success, tests passed!")

if __name__ == "__main__":
    test()