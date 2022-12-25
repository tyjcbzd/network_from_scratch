# imageNet 100多个G，电脑存储不支持实现
import torchvision.models
from torchvision import datasets, models
import torch

'''
   VGG16 Model
class NNModel(nn.Module):
    def __init__(self):
        super(NNModel, self).__init__()
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
'''
# 相当与于从网络上下载下来了模型，参数是初始化的并没有训练
vgg16_false = torchvision.models.vgg16(pretrained=False)

# 保存方式一：将 模型结构+参数 保存到本地文件
torch.save(vgg16_false, "vgg16_method.pth")

# # 模型加载 --对保存方式一
# model = torch.load("vgg16_method.pth")
# print(model)


# 保存方式二： 模型参数(官方推荐) 存储更小
torch.save(vgg16_false.state_dict(), "vgg16_method2.pth")

# 模型加载 --对保存方式二 : 先创建一个初始化参的模型，再将参数字典加载进去
vgg16 = torchvision.models.vgg16(pretrained=False)
model_parameters_dict = torch.load("vgg16_method2.pth")
print(model_parameters_dict)
vgg16.load_state_dict(model_parameters_dict)
print(vgg16)

# # 网络中的模型还有参数是训练好的
# vgg16_true = torchvision.models.vgg16(pretrained=True)
#
# # print(vgg16_true)
# # 相当于直接下载好了模型，我只需要调用就可以。如果下载的包含训练好的模型，那么只需要
# input = torch.randn([64, 3, 32, 32])
# output = vgg16_true(input)
# print(output.shape)


# 方式一 也有不好的地方，如果是自定义的模型，那么在加载的那个py文件里面需要把自定义的模型写在里面，或者是py文件中的模型引入到加载的那个文件下面