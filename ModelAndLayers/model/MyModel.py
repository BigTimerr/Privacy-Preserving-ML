"""
# @Time : 2022/9/8 14:50
# @Author : ruetrash
# @File : MyModel.py
"""
import torch
import torch.nn as nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear,AvgPool2d,ReLU


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
        self.maxpool1 = MaxPool2d(kernel_size=2)
        self.conv2 = Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.maxpool2 = MaxPool2d(kernel_size=2)
        self.conv3 = Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.maxpool3 = MaxPool2d(kernel_size=2)
        self.flatten = Flatten()
        self.linear1 = Linear(1024, 64)
        self.linear2 = Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        print("cov1", x)
        print("bias", self.conv1.bias)
        x = self.maxpool1(x)
        print("maxpool1",x)
        x = self.conv2(x)
        print("conv2", x)
        print("bias", self.conv2.bias)
        x = self.maxpool2(x)
        print("maxpool2",x)
        x = self.conv3(x)
        print("conv3", x)
        x = self.maxpool3(x)
        print("maxpool3", x)
        x = self.flatten(x)
        x = self.linear1(x)
        print("linear1",x)
        x = self.linear2(x)
        print("linear2",x)

        return x

class MyModel1(nn.Module):
    def __init__(self):
        super(MyModel1, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
        self.avgPool1 = AvgPool2d(kernel_size=2)
        self.relu1 = ReLU(inplace=True)

        self.conv2 = Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.avgPool2 = AvgPool2d(kernel_size=2)
        # self.relu2 = ReLU(inplace=True)

        self.conv3 = Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.avgPool3 = AvgPool2d(kernel_size=2)

        self.flatten = Flatten()
        self.linear1 = Linear(1024, 64)
        self.linear2 = Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        print("con1", x)
        x = self.avgPool1(x)
        print("avgPool1", x)
        x = self.relu1(x)
        print("relu1",x)

        x = self.conv2(x)
        print("con2", x)
        x = self.avgPool2(x)
        print("avgPool2", x)
        # x = self.relu2(x)

        x = self.conv3(x)
        x = self.avgPool3(x)

        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x

class MyModel2(nn.Module):
    def __init__(self):
        super(MyModel2, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=2)
        self.conv2 = Conv2d(in_channels=6, out_channels=3, kernel_size=6, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        print(x)
        x = self.conv2(x)
        print(x)

        return x


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 32*28*28
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32*14*14
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64*14*14
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64*7*7
            nn.ReLU(inplace=True),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 128*7*7
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 256*7*7
        )

        self.layer5 = nn.Sequential(

            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 256*7*7
            nn.MaxPool2d(kernel_size=3, stride=2),  # 256*3*3
            nn.ReLU(inplace=True),
        )

        self.fc1 = nn.Linear(256 * 3 * 3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        x = self.layer5(x)

        x = x.view(-1, 256 * 3 * 3)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    AlexNet = AlexNet()
    print(AlexNet)
    torch.save(AlexNet.state_dict(), "ModelAndLayers/model/myModel.pth")

    input = torch.ones(1, 3, 28, 28)
    output = AlexNet(input)
    print(repr(output))


    # AlexNet = AlexNet2()
    # torch.save(AlexNet.state_dict(), "ModelAndLayers/model/AlexNet.pth")
    # input = torch.ones(16, 3, 32, 32)
    # output = AlexNet(input)
    # print(output.shape)
    # print(output)
