import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, MaxPool2d, Linear, ReLU, Flatten, Softmax


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # self.conv1 = nn.Conv2d(3, 16, 5)
        # self.pool1 = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(16, 32, 5)
        # self.pool2 = nn.MaxPool2d(2, 2)
        # self.fc1 = nn.Linear(32*5*5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

        self.model = nn.Sequential(
            Conv2d(in_channels=3, out_channels=16, kernel_size=5),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(in_channels=16, out_channels=32, kernel_size=5),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            # 展平
            Flatten(),
            Linear(in_features=32 * 5 * 5, out_features=120),
            ReLU(),
            Linear(in_features=120, out_features=84),
            ReLU(),
            Linear(in_features=84, out_features=10),
            # Softmax()
            # CrossEntropy中包含Softmax
        )



    def forward(self, x):
        # x = F.relu(self.conv1(x))    # input(3, 32, 32) output(16, 28, 28)
        # x = self.pool1(x)            # output(16, 14, 14)
        # x = F.relu(self.conv2(x))    # output(32, 10, 10)
        # x = self.pool2(x)            # output(32, 5, 5)
        # x = x.view(-1, 32*5*5)       # output(32*5*5)
        # x = F.relu(self.fc1(x))      # output(120)
        # x = F.relu(self.fc2(x))      # output(84)
        # x = self.fc3(x)              # output(10)
        return self.model(x)



# model = LeNet()
# data = torch.randn([1, 3, 32, 32])
# print(model(data).shape)
# print(model(data))