"""
# @Time : 2022/8/30 14:38
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

if torch.cuda.is_available() is True:
    torch.device = 'GPU'

# 初始化参数
p = 0  # party 0:server
Ring = param.Ring

img_shape = (1, 3, 5, 5)
kernel_shape = (3, 3, 2, 2)

# 接收参数,将自己的分享份额构建成ShareV类型
img_0 = server.receive_np_array()
# kernel_0 = server.receive_np_array()
img = ShareV(img_0, p, server)

# 卷积核
kernel_0 = np.array([[[[1, 1], [1, 1]],
                      [[1, 1], [1, 1]],
                      [[1, 1], [1, 1]]],

                     [[[1, 1], [1, 1]],
                      [[1, 1], [1, 1]],
                      [[1, 1], [1, 1]]],

                     [[[1, 1], [1, 1]],
                      [[1, 1], [1, 1]],
                      [[1, 1], [1, 1]]]])

# 创建模型
server_model = model.ModelOfLayers()

'''
        卷积层
'''

# conv2D = layers.SecConv2d(kernel_0, stride=1, padding=0)
# server_model.add(conv2D)
#
# # 将图片信息和信息输入到模型中
# server_model.set_input(img)

'''
        ReLu层
'''
# Relu = layers.SecReLu()
# server_model.add(Relu)
#
# # 将图片信息和信息输入到模型中
# server_model.set_input(img)


'''
        SecMaxPool2D
'''

# MaxPoll2D = layers.SecMaxPool2D(kernel_size=2, stride=1)
# server_model.add(MaxPoll2D)
#
# # 将图片信息和信息输入到模型中
# server_model.set_input(img)

'''
    SecAvgPool2D
'''
# SecAvgPool2D = layers.SecAvgPool2D((2, 2), stride=1,padding=0)
# server_model.add(SecAvgPool2D)
# server_model.set_input(img)


'''
        BatchNormalization
'''
# gamma_0 = server.receive_np_array()
# beta_0 = server.receive_np_array()
#
# BatchNormalization = layers.BatchNormalization(gamma_0,beta_0)
# server_model.add(BatchNormalization)
# server_model.set_input(img)

'''
        flatten
'''
# flatten = layers.Flatten(img_shape)
# server_model.add(flatten)
# server_model.set_input(img)

'''
    SecLinear
'''
# x = server.receive_np_array()
# weight = server.receive_np_array()
#
# x = ShareV(x,p,server)
#
# bias = np.zeros((3, 10),dtype=np.int32)
# secLinear = layers.SecLinear(weight, bias)
# server_model.add(secLinear)
#
# server_model.set_input(x)





# 启动模型
server_model.predict()

res = ssv.restore_nparray(server_model.input, party=2)

print(res)
