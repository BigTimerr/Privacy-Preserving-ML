"""
# @Time : 2022/10/17 20:18
# @Author : ruetrash
# @File : pytorch_test.py
"""

import torch
import numpy as np

value = torch.tensor([[[[7, 7, 8, 4, 1],
                    [8, 8, 2, 8, 1],
                    [1, 6, 9, 0, 5],
                    [6, 8, 4, 8, 9],
                    [0, 2, 3, 8, 0]],

                   [[0, 7, 5, 3, 2],
                    [-2, -4, -7, -5, -6],
                    [5, 9, 4, 6, 2],
                    [-3, -4, -7, -6, 0],
                    [7, 6, 2, 8, 8]],

                   [[4, 9, 7, 9, 9],
                    [2, 1, 1, 2, 8],
                    [-3, -6, -2, -8, -2],
                    [2, 7, 5, 7, 4],
                    [6, 0, 2, 6, 0]]]], dtype=torch.float)

kernel = torch.tensor([[[[3, 4], [5, 6]],
                        [[4, 6], [7, 8]],
                        [[6, 9], [4, 7]]],

                       [[[2, 1], [2, 8]],
                        [[4, 10], [1, 4]],
                        [[6, 4], [2, 4]]],

                       [[[7, 8], [6, 3]],
                        [[2, 2], [2, 2]],
                        [[4, 2], [8, 1]]]], dtype=torch.float)

MaxPool2d = torch.nn.MaxPool2d(2, stride=1)
out = MaxPool2d(value)

print(out)



