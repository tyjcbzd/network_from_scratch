import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms
from torch.nn import Sequential, Flatten, Conv2d, MaxPool2d, Linear
from torch.utils.data import DataLoader
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


dataset = DataLoader(train_set, batch_size=16, )

loss = nn.CrossEntropyLoss()
model = NNModel()
opti = torch.optim.SGD(model.parameters(), lr=0.01) # 定义优化器

for epoch in range(20): # 总的要循环的轮次
    running_loss = 0.0
    for data in dataset: # 每一轮中其中对所有数据遍历的误差loss
        imgs, targets = data
        outputs = model(imgs)
        result_loss = loss(outputs, targets)
        opti.zero_grad()  # 每次梯度下降之后都要重新将优化器归零
        result_loss.backward() # 反向传播求出每个节点参数的梯度
        opti.step() # 对参数进行调节
        running_loss = running_loss + result_loss # 每一轮的整体误差loss
    print(running_loss)


