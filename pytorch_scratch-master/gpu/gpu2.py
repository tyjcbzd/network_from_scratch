import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


'''
    定义训练的设备
    选择gpu
    .device()
    
    PS：model.to(device)，loss_fn.to(device)可以不用重新赋值，但是imgs和targets必须重新赋值
'''

# device = torch.device("cpu")
# 还有一种写法
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 假设有多个显卡
device1 = torch.device("cuda:0")
device2 = torch.device("cuda:1")




# 搭建神经网络
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

# 下载数据集
train_set = torchvision.datasets.CIFAR10("../data", train=True, transform=torchvision.transforms.ToTensor(),download=True)
test_set = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(),download=True)

# 记录训练集和测试集的样本长度
train_length = len(train_set)
test_length = len(test_set)

print(f'训练集样本长度：{train_length}')
print(f'测试集样本长度：{test_length}')

# 利用DataLoader加载数据集
train_data_loader = DataLoader(train_set, batch_size=64)
test_data_loader = DataLoader(test_set, batch_size=64)

# 创建模型
model = NNModel()

# 转移到设备
# model.to(device)
model = model.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 转移到设备
loss_fn = loss_fn.to(device)

# 优化器
learning_rate = 1e-2 # 1e-2 = 1 x (10)^(-2)
# 这里使用随机梯度下降SGD
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# 设置训练网络的一些参数
## 记录训练的次数
total_train_step = 0
## 记录测试的次数
total_test_step = 0
## 训练的轮数
epoches = 10

# 添加tensorboard
writer = SummaryWriter("../logs")

'''
关于 model的train、eval函数，涉及到BN层和dropout的时候需要添加（不涉及的时候也添加也没有任何问题）
为什么？ https://blog.csdn.net/qq_38410428/article/details/101102075
'''


for epoch in range(epoches):
    print(f'-----第{epoch+1}轮训练开始-----') # 设置为 epoch+1 是因为range函数是从 0 开始的


    # 训练步骤开始
    model.train()
    for data in train_data_loader:
        imgs, targets = data

        # 转移到设备
        imgs = imgs.to(device)
        targets = targets.to(device)

        outputs = model(imgs)
        loss = loss_fn(outputs, targets)

        # 将优化器清零
        optimizer.zero_grad()
        # 反向传播求出每个节点参数的梯度
        loss.backward()
        # 对参数进行调节
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            # 输出loss.item() 是将tensor数据类ixng转化成真实的数字 tensor(5) --.item()--> 5
            print("训练次数：{}, Loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    model.eval()
    # 测试数据集
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad(): # 禁止梯度下降
        for data in test_data_loader:
            imgs, targets = data

            # 转移到设备
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy/test_length))
    writer.add_scalar("test_loss", total_test_loss, total_test_step + 1)
    writer.add_scalar("test_accuracy",total_accuracy/test_length, total_test_step+1)
    total_test_step = total_test_step + 1

    # 保存每一轮训练的模型 方式一
    torch.save(model, "model_{}.pth".format(epoch+1))
    # 保存方式二（推荐）
    torch.save(model.state_dict(), "model_{}.pth".format(epoch+1))

writer.close()

