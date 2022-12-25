import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear

# 一般在model里面进行验证，验证输出尺寸即可

class NNModel(nn.Module):
    def __init__(self):
        super(NNModel, self).__init__()
        self.model = nn.Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(in_features=1024, out_features=64),
            Linear(in_features=64, out_features=10)
        )

    def forward(self, input):
        return self.model(input)


