"""
# @Time : 2022/12/5 11:26
# @Author : ruetrash
# @File : mnist_train.py
"""

# AlexNet & MNIST


import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
import time
from ModelAndLayers.AlexNetAndMnist.AlexNet import AlexNet

# transform
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(),
    transforms.ToTensor(),

])

transform1 = transforms.Compose([
    transforms.ToTensor()
])

# device : GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载数据
trainset = torchvision.datasets.MNIST(root='ModelAndLayers/Data', train=True, download=True,
                                      transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True,
                                          num_workers=0)  # windows下num_workers设置为0，不然有bug

testset = torchvision.datasets.MNIST(root='ModelAndLayers/Data', train=False, download=True,
                                     transform=transform1)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

# net
net = AlexNet()

#损失函数:这里用交叉熵
criterion = nn.CrossEntropyLoss()

#优化器 这里用SGD
optimizer = optim.SGD(net.parameters(),lr=1e-3, momentum=0.9)


net.to(device)

print("Start Training!")

num_epochs = 20 #训练次数

for epoch in range(num_epochs):
    running_loss = 0
    batch_size = 100

    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('[%d, %5d] loss:%.4f'%(epoch+1, (i+1)*100, loss.item()))

print("Finished Traning")


#保存训练模型
torch.save(net.state_dict(), 'ModelAndLayers/AlexNetAndMnist/MNIST_bak.pkl')

net.load_state_dict(torch.load('ModelAndLayers/AlexNetAndMnist/MNIST_bak.pkl'))
# 开始识别
with torch.no_grad():
    # 在接下来的代码中，所有Tensor的requires_grad都会被设置为False
    correct = 0
    total = 0

    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        out = net(images)
        _, predicted = torch.max(out.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 100 test images:{}%'.format(100 * correct / total))  # 输出识别准确率
