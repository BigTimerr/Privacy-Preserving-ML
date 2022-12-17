"""
# @Time : 2022/12/5 14:41
# @Author : ruetrash
# @File : AlexNet.py
"""

import torch.nn as nn
import torch.nn.functional as F


# 定义网络结构
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        # 由于MNIST为28x28， 而最初AlexNet的输入图片是227x227的。所以网络层数和参数需要调节
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # AlexCONV1(3,96, k=11,s=4,p=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # AlexPool1(k=3, s=2)
        self.relu1 = nn.ReLU()

        # self.conv2 = nn.Conv2d(96, 256, kernel_size=5,stride=1,padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # AlexCONV2(96, 256,k=5,s=1,p=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # AlexPool2(k=3,s=2)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # AlexCONV3(256,384,k=3,s=1,p=1)
        # self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)  # AlexCONV4(384, 384, k=3,s=1,p=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # AlexCONV5(384, 256, k=3, s=1,p=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # AlexPool3(k=3,s=2)
        self.relu3 = nn.ReLU()

        self.fc6 = nn.Linear(256 * 3 * 3, 1024)  # AlexFC6(256*6*6, 4096)
        self.fc7 = nn.Linear(1024, 512)  # AlexFC6(4096,4096)
        self.fc8 = nn.Linear(512, 10)  # AlexFC6(4096,1000)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        # print("pool1", x)
        # print("***")
        # print("***")
        # print("***")
        # print("***")
        x = self.relu1(x)
        # print("relu1", x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)
        x = self.relu3(x)
        x = x.view(-1, 256 * 3 * 3)  # Alex: x = x.view(-1, 256*6*6)
        x = self.fc6(x)
        x = F.relu(x)
        x = self.fc7(x)
        x = F.relu(x)
        x = self.fc8(x)
        return x
