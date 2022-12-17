"""
# @Time : 2022/9/7 10:42
# @Author : ruetrash
# @File : plaintext_pytorch_test.py
"""
import torch.nn

img = torch.tensor([[[[  0,   7,   0,   7,  -4],
         [  5,   0,   6,  -7,  -9],
         [  5,  -3,   2,  -9,   7],
         [  5,  -1,  -5,   3,   0],
         [  8,   9,   2,   7,  -3]],

        [[  0,  -8,  -9,   5,  -4],
         [ -8,   2,   1,   0,  -4],
         [ -6,  -9,  -3,   3,   7],
         [ -7,   4,  -3,  -2,  -4],
         [  0,  -7,   7,  -3,   3]],

        [[ -1,  -5,   2,  -7,  -3],
         [  6,   3,  -9,   1,   5],
         [  6,  -6,   0,  -5,  -4],
         [  0,   9,   9, -10,   0],
         [  2,  -7,  -4,   3,  -8]]]],dtype=torch.float)
kernel = torch.tensor([[[[3, 4], [5, 6]],
                        [[4, 6], [7, 8]],
                        [[6, 9], [4, 7]]],

                       [[[2, 1], [2, 8]],
                        [[4, 10], [1, 4]],
                        [[6, 4], [2, 4]]],

                       [[[7, 8], [6, 3]],
                        [[2, 2], [2, 2]],
                        [[4, 2], [8, 1]]]], dtype=torch.float)

'''Conv2D'''
out = torch.nn.functional.conv2d(img, kernel, stride=1, padding=1)


'''SecMaxPool2D'''
AvgPool2d = torch.nn.MaxPool2d(kernel_size=2, stride=1)
out = AvgPool2d(out)


'''ReLu'''
RelU = torch.nn.ReLU()
out = RelU(out)

'''SecAvgPool2D'''
AvgPool2d = torch.nn.AvgPool2d(kernel_size=2, stride=1)
out = AvgPool2d(out)

# '''Conv2D'''
# out = torch.nn.functional.conv2d(out, kernel,stride=1,padding=1)
#
# '''SecAvgPool2D'''
# AvgPool2d = torch.nn.AvgPool2d(kernel_size=2, stride=1)
# out = AvgPool2d(out)

'''flatten'''
out = torch.nn.Flatten(out)
print(out)