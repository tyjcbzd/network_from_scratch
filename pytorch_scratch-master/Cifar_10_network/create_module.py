import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms
from torch.nn import Sequential, Flatten, Conv2d, MaxPool2d, Linear
from torch.utils.tensorboard import SummaryWriter

train_set = torchvision.datasets.CIFAR10("../data", train=True, download=True, transform=torchvision.transforms.ToTensor())

'''
以 VGG-16为例：
'''
class NNModel(nn.Module):
    def __init__(self):
        super(NNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=1024, out_features=64)
        self.linear2 = nn.Linear(in_features=64, out_features=10)

        # 等价于上面
        self.model1 = nn.Sequential(
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
        return self.model1(input)


input = torch.randn([64, 3, 32, 32])
model = NNModel()

print(model)
output = model(input)
print(output.shape)

writer = SummaryWriter("../logs")
writer.add_graph(model, input)
writer.close()


