import os

import torch
import torchvision.transforms
from PIL import Image
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear

# os.open("dog.png")

image = Image.open("dog.png")
print(image)
image = image.convert('RGB')

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)
image = torch.FloatTensor(image)

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




# 如果电脑上没有gpu，那么可以换成cpu
if torch.cuda.is_available():
    model = torch.load("model_29.pth")
else:
    model = torch.load("model_29.pth", map_location=torch.device("cpu"))

print(model)
model.eval()

'''
训练模型报错：
RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) 
should be the same or input should be a MKLDNN tensor and weight is a dense tensor

错误原因：get this error because your model is on the GPU, but your data is on the CPU. 
因为model使用gpu训练出来的，但是我输入的照片dog的image是cpu的，所以需要调用,cuda（）
'''

# 禁止梯度下降, 可以减少使用内存，提高性能
with torch.no_grad():
    # 注意实际训练中需要多一个 batch_size
    image = torch.reshape(image, [1, 3, 32, 32])
    if torch.cuda.is_available():
        image = image.cuda()
    output = model(image)
print(output)