import torch
import torch.nn as nn

VGG16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

class VGGNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super(VGGNet, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG16)
        self.fcs = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        for x in architecture:
            if type(x) == int:
                out_channels = x
                layers += [nn.Conv2d(in_channels=in_channels, 
                                     out_channels=out_channels,
                                     kernel_size=(3,3), 
                                     stride=(1,1), 
                                     padding=(1,1)),
                            nn.BatchNorm2d(x),
                            nn.ReLU()]
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]

        return nn.Sequential(*layers)

def test():
    batch_size, in_channels, img_size = 64, 1, 224
    model = VGGNet(1, 2)
    x = torch.randn(batch_size, in_channels, img_size, img_size)
    print(model(x).shape)
    assert model(x).shape == (batch_size, 2), "Generator test failed"
    print("Success, tests passed!")

if __name__ == "__main__":
    test()