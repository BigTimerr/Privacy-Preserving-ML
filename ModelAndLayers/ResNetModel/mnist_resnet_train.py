"""
# @Time : 2022/12/5 11:26
# @Author : ruetrash
# @File : mnist_train.py
"""

# AlexNet & MNIST

import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.utils.data
import time
from ModelAndLayers.ResNetModel.ResNet import resnet18

# transform
transform = transforms.Compose([
    transforms.ToTensor(),
])

# device : GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # 加载数据
train_set = torchvision.datasets.MNIST(root='ModelAndLayers/Data/', train=True, download=True,
                                       transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True)

test_set = torchvision.datasets.MNIST(root='ModelAndLayers/Data/', train=False, download=True,
                                      transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False)

# net
net = resnet18()

# 损失函数:这里用交叉熵
criterion = nn.CrossEntropyLoss()

# 优化器 这里用SGD
optimizer = optim.Adam(net.parameters(), lr=1e-3)

net.to(device)

print("Start Training!")

num_epochs = 10  # 训练次数

for epoch in range(num_epochs):
    running_loss = 0
    batch_size = 100

    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('[%d, %5d] loss:%.4f' % (epoch + 1, (i + 1) * 100, loss.item()))

print("Finished Training")

# 保存训练模型
net.eval()
torch.save(net.state_dict(), 'ModelAndLayers/ResNetModel/ResNet18.pkl')

net.load_state_dict(torch.load('ModelAndLayers/ResNetModel/ResNet18.pkl'))
start_time = time.time()
# 开始识别
with torch.no_grad():
    # 在接下来的代码中，所有Tensor的requires_grad都会被设置为False
    total_correct = 0
    total_total = 0

    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        out = net(images)
        _, predicted = torch.max(out.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()

        total_total += total
        total_correct += correct
        print(correct)
        print('Accuracy of the network on the 100 test images:{}%'.format(100 * correct / total))  # 输出识别准确率
end_time = time.time()
print("时间", end_time - start_time)
print(correct)
print('Accuracy of the network on the 10000 test images:{}%'.format(100 * total_correct / total_total))  # 输出识别准确率
