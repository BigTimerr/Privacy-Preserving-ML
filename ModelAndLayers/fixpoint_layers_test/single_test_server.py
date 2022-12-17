"""
# @Time : 2022/10/17 20:06
# @Author : ruetrash
# @File : single_test_server.py
"""
import TCP.tcp as tcp
import torch
import ProtocolOnRing.param as param
import numpy as np
import ProtocolOnRing.secret_sharing_fixpoint as ssf
import ModelAndLayers.layers.layers_of_fixpoints as layers
import ModelAndLayers.model.modeloflayers as model

# 初始化服务器端进程
server = tcp.TCPServer("127.0.0.1", 9999, 40960)
server.run()

if torch.cuda.is_available() is True:
    torch.device = 'GPU'

p = 0

kernel_0 = torch.tensor([[[[1, 1], [1, 1]],
                      [[1, 1], [1, 1]],
                      [[1, 1], [1, 1]]],

                     [[[1, 1], [1, 1]],
                      [[1, 1], [1, 1]],
                      [[1, 1], [1, 1]]],

                     [[[1, 1], [1, 1]],
                      [[1, 1], [1, 1]],
                      [[1, 1], [1, 1]]]])

value_0 = server.receive_torch_array()
img = ssf.ShareFloat(value_0, p, server)

# 创建模型
server_model = model.ModelOfLayers()

'''Conv2'''
conv2D = layers.SecConv2d(kernel_0, stride=1, padding=0)
server_model.add(conv2D)

'''MaxPool'''
# conv2D = layers.SecMaxPool2D(2, stride=1, padding=0)
# server_model.add(conv2D)


'''ReLU'''
# Relu = layers.SecReLu()
# server_model.add(Relu)

'''avgPool'''
# avgPool = layers.SecAvgPool2D(2,stride=1,padding=0)
# server_model.add(avgPool)

# 将图片信息和信息输入到模型中
server_model.set_input(img)


# 启动模型
server_model.predict()

res = ssf.restore_float(server_model.input, party=2)
res = ssf.decode(res)

