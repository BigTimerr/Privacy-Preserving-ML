"""
# @Time : 2022/10/17 20:06
# @Author : ruetrash
# @File : single_test_client.py
"""
import time

import TCP.tcp as tcp
import torch
import ProtocolOnRing.param as param
import numpy as np
import ProtocolOnRing.secret_sharing_fixpoint as ssf
import ModelAndLayers.layers.layers_of_fixpoints as layers
import ModelAndLayers.model.modeloflayers as model

# 初始化客户端进程
client = tcp.TCPClient("127.0.0.1", 9999, 40960)
client.run()

if torch.cuda.is_available() is True:
    torch.device = 'GPU'

p = 1

value_pre = torch.randn(size=(500, 3, 28, 28))
# value = np.ones(shape=(1, 3, 5, 5))
# value = np.array([[[[7, 7, 8, 4, 1],
#                     [8, 8, 2, 8, 1],
#                     [1, 6, 9, 0, 5],
#                     [6, 8, 4, 8, 9],
#                     [0, 2, 3, 8, 0]],
#
#                    [[0, 7, 5, 3, 2],
#                     [-2, -4, -7, -5, -6],
#                     [5, 9, 4, 6, 2],
#                     [-3, -4, -7, -6, 0],
#                     [7, 6, 2, 8, 8]],
#
#                    [[4, 9, 7, 9, 9],
#                     [2, 1, 1, 2, 8],
#                     [-3, -6, -2, -8, -2],
#                     [2, 7, 5, 7, 4],
#                     [6, 0, 2, 6, 0]]]])

kernel_1 = torch.tensor([[[[2, 3], [4, 5]],
                          [[3, 5], [6, 7]],
                          [[5, 8], [3, 6]]],

                         [[[1, 0], [1, 7]],
                          [[3, 9], [0, 3]],
                          [[5, 3], [1, 3]]],

                         [[[6, 7], [5, 2]],
                          [[1, 1], [1, 1]],
                          [[3, 1], [7, 0]]]])

value = ssf.encode(value_pre)
value_0, value_1 = ssf.share_float(value)
client.send_torch_array(value_0)

img = ssf.ShareFloat(value_1, p, client)

# 创建模型
client_model = model.ModelOfLayers()

'''Conv2'''
conv2D = layers.SecConv2d(kernel_1, stride=1, padding=0)
client_model.add(conv2D)

'''MaxPool'''
# conv2D = layers.SecMaxPool2D(2, stride=1, padding=0)
# client_model.add(conv2D)


'''Relu'''
# Relu = layers.SecReLu()
# client_model.add(Relu)

'''avgPool'''
# avgPool = layers.SecAvgPool2D(2,stride=1,padding=0)
# client_model.add(avgPool)


# 将图片信息和信息输入到模型中
client_model.set_input(img)

# 启动 模型
client_model.predict()

res = ssf.restore_float(client_model.input, party=2)
res = ssf.decode(res)

