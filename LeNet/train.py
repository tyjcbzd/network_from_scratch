import sys

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim
from model import *
from tqdm import tqdm
import torch
import torchvision


# 以类的方式定义参数，还有很多方法，config文件等等
class Args:
    def __init__(self) -> None:
        self.batch_size = 1
        self.learning_rate = 0.001
        self.epochs = 20
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


args = Args()


def main():
    # 模型保存文件
    save_path = './Lenet.pth'
    # 是否可用GPU
    device = args.device
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 数据集
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             transform=torchvision.transforms.ToTensor(),
                                             download=True)
    valid_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                             transform=torchvision.transforms.ToTensor(),
                                             download=True)
    # size of training data 50000
    # print(len(train_set))
    train_length = len(train_set)
    valid_length = len(valid_set)

    # 利用DataLoader加载数据集
    train_data_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    valid_data_loader = DataLoader(valid_set, batch_size=4, shuffle=True)

    # initialize model,loss,optimizer
    net = LeNet()
    net.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    best_acc = 0.0
    # number of train batches        number of training /batch_size = 50000/4=12500
    # print(train_steps)

    train_epochs_loss = []
    valid_epochs_loss = []
    train_acc = []
    val_acc = []

    for epoch in range(args.epochs):
        # ================train============================
        print(f'================Epoch:{epoch + 1} starts training============================')
        net.train()
        train_epoch_loss = []
        acc = 0
        train_bar = tqdm(train_data_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # 清零梯度
            optimizer.zero_grad()
            # 预测结果
            outputs = net(inputs)
            # 计算损失
            loss = loss_fn(outputs, labels)
            # 反向梯度计算
            loss.backward()
            # 利用反向传播得到的梯度更新参数
            optimizer.step()

            train_epoch_loss.append(loss.item())
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, labels).sum().item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     args.epochs,
                                                                     loss)
        train_epochs_loss.append(np.average(train_epoch_loss))

        # 所有train epoch汇总，用于绘图
        train_acc.append(acc / train_length)
        print("train acc = {:.3f}, loss = {}".format(acc / train_length, np.average(train_epoch_loss)))

        # ================ validation ============================
        # 训练集的所有数据全部训练完了再进行测验
        with torch.no_grad():
            net.eval()
            test_bar = tqdm(valid_data_loader, file=sys.stdout)
            val_epoch_loss = []
            acc, num = 0, 0

            for data in test_bar:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                # 测试误差
                loss = loss_fn(outputs, labels.to(device))
                val_epoch_loss.append(loss.item())
                # 每一个预测类
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, labels).sum().item()

            valid_epochs_loss.append(np.average(val_epoch_loss))
            val_acc.append(acc/valid_length)
            print("epoch = {}, valid acc = {:.2f}, loss = {}".format(epoch + 1,
                                                                     acc/valid_length,
                                                                     np.average(val_epoch_loss)))
            # =========================save model=====================
            # 验证集上正确率更高就保存模型参数
            if acc/valid_length > best_acc:
                best_acc = acc/valid_length
                torch.save(net.state_dict(), save_path)

    # =========================plot==========================
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(train_acc, '-o', label="train_acc")
    plt.plot(val_acc, '-o', label="validation_acc")
    plt.title("epochs_accuracy")
    plt.legend()
    plt.subplot(122)
    plt.plot(train_epochs_loss, '-o', label="train_loss")
    plt.plot(valid_epochs_loss, '-o', label="valid_loss")
    plt.title("epochs_loss")
    plt.legend()
    plt.show()

    print("finish training")


if __name__ == "__main__":
    main()
