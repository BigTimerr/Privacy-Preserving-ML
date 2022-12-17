"""
# @Time : 2022/9/8 10:03
# @Author : ruetrash
# @File : function.py
"""
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

def getOneImage():
    batch_size = 1
    test_data = torchvision.datasets.CIFAR10(root="ModelAndLayers/build_secure_model/dataset", train=False,
                                             transform=transforms.ToTensor(), download=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

    img, target = next(iter(test_loader))
    img = img.numpy().astype(np.int32)
    return img, target
