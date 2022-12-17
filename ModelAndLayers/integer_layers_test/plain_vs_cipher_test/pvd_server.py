"""
# @Time : 2022/9/7 10:09
# @Author : ruetrash
# @File : single_server.py
"""
import torch
import TCP.tcp as tcp
import numpy as np
import ProtocolOnRing.param as param
import ModelAndLayers.model.modeloflayers as model
import ModelAndLayers.layers.layers as layers
from ProtocolOnRing.secret_sharing_vector_onring import ShareV
import ProtocolOnRing.secret_sharing_vector_onring as ssv

# 初始化服务器端进程
server = tcp.TCPServer("127.0.0.1", 9999, 4096)
server.run()

# 初始化参数
p = 0  # party 0:server
Ring = param.Ring

# 初始化图片信息
img_shape = (1, 3, 5, 5)
kernel_shape = (3, 3, 2, 2)  # 卷积层的kernel 形状

# 接受传入的数据
img_0 = server.receive_np_array()
kernel_0 = server.receive_np_array()

img = ShareV(img_0, p, server)

# 创建模型
server_model = model.ModelOfLayers()

'''BatchNormalization'''


'''Conv2D'''
Conv2D = layers.SecConv2d(kernel_0, stride=1, padding=1)
server_model.add(Conv2D)


'''SecMaxPool2D'''
SecMaxPool2D = layers.SecMaxPool2D(kernel_size=2, stride=1, padding=0)
server_model.add(SecMaxPool2D)

'''ReLu'''
Relu = layers.SecReLu()
server_model.add(Relu)

'''SecAvgPool2D'''
SecAvgPool2D = layers.SecAvgPool2D(kernel_size=2, stride=1, padding=0)
server_model.add(SecAvgPool2D)


# '''Conv2d'''
# Conv2D = layers.Conv2D(kernel_0, kernel_shape, stride=1, padding=1)
# server_model.add(Conv2D)
#
# '''SecMaxPool2D'''
# SecAvgPool2D = layers.SecMaxPool2D(kernel_size=2, stride=1, padding=0)
# server_model.add(SecAvgPool2D)


'''flatten'''
flatten = layers.Flatten()
server_model.add(flatten)

# 启动模型
server_model.set_input(img)
server_model.predict()
res = ssv.restore_nparray(server_model.input, party=2)
print(res)
