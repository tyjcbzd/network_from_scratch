import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms

'''

Function ----> transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

"""Normalize a tensor image with mean and standard deviation.
    This transform does not support PIL Image.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``channels, 
    this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``


'''
def main():

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # 50000张训练图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
                                               shuffle=True, num_workers=0)

    # 10000张验证图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=10000,
                                             shuffle=False, num_workers=0)

    # iter ---> 将 val_loader转化成可迭代的迭代器
    # 可用可不用，也可以在下面的with torch.no_grad()当中写 for data in val_loader: 然后再赋值等等
    val_data_iter = iter(val_loader)
    # 用next 可以获取一批数据
    val_image, val_label = val_data_iter.next()
    print(val_label)
    # print(val_image.shape)
    # print(val_label.shape)
    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet()

    # 在我们自定义的LeNet中并没有添加Softmax是因为CrossEntropy中已经包含了Softmax
    # Note that this case is equivalent to the combination of :class:`~torch.nn.LogSoftmax` and
    #       :class:`~torch.nn.NLLLoss`.

    # define the loss function
    loss_function = nn.CrossEntropyLoss()
    # set the optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(5):  # loop over the dataset multiple times
        # train error
        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            # 所以梯度累积的目的是实现低显存跑大batchsize
            # 具体见： https://www.zhihu.com/question/303070254，这样不更新参数，而是累加梯度，变相变大了训练 batchsize
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            # Backpropagation
            loss.backward()
            # ipdate the weights and biases
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if step % 500 == 499:  # print every 500 mini-batches
                # with 一个上下文管理器
                with torch.no_grad():
                    outputs = net(val_image)  # [batch, 10]
                    # 这个max函数会返回(最大值，索引)，我只需要索引
                    predict_y = torch.max(outputs, dim=1)[1]
                    # 使用 .item()拿到Tensor中的数值
                    # val_label 中写入的是[10000]个数据，都是标签的索引值 [3,5,8,3...,...]
                    accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)

                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))
                    running_loss = 0.0

    print('Finished Training')

    save_path = './Lenet.pth'
    torch.save(net.state_dict(), save_path)

if __name__ == '__main__':
    main()