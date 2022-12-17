"""
# @Time : 2022/8/30 14:38
# @Author : ruetrash
# @File : single_client.py
"""

import torch
import TCP.tcp as tcp
import ProtocolOnRing.param as param
import numpy as np
import ModelAndLayers.model.modeloflayers as model
import ModelAndLayers.layers.layers as layers
from ProtocolOnRing.secret_sharing_vector_onring import ShareV
import ProtocolOnRing.secret_sharing_vector_onring as ssv

# 初始化客户端进程
client = tcp.TCPClient("127.0.0.1", 9999, 4096)
client.run()

if torch.cuda.is_available() is True:
    torch.device = 'GPU'

# 初始化参数
p = 1  # party 0:client
Ring = param.Ring
ksize = 2

# 随机创建图片
img_shape = (1, 3, 5, 5)
kernel_shape = (3, 3, 2, 2)

# value = np.random.randint(0, 10, size=img_shape, dtype=np.int32)
# kernel = np.random.randint(0, 10, size=kernel_shape, dtype=np.int32)

value = np.array([[[[7, 7, 8, 4, 1],
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
                    [6, 0, 2, 6, 0]]]])


# 将图片进行分享
img_0, img_1 = ssv.share_nparray(value)
# kernel_0, kernel_1 = ssv.share_nparray(kernel)
client.send_np_array(img_0)
# client.send_np_array(kernel_0)

# 使用自己的分享份额构建 ShareV类型
img = ShareV(img_1, p, client)

# 卷积核
kernel_1 = np.array([[[[2, 3], [4, 5]],
                      [[3, 5], [6, 7]],
                      [[5, 8], [3, 6]]],

                     [[[1, 0], [1, 7]],
                      [[3, 9], [0, 3]],
                      [[5, 3], [1, 3]]],

                     [[[6, 7], [5, 2]],
                      [[1, 1], [1, 1]],
                      [[3, 1], [7, 0]]]])

# 创建模型
client_model = model.ModelOfLayers()

'''
        卷积层   
'''

# conv2D = layers.SecConv2d(kernel_1, stride=1, padding=0)
# client_model.add(conv2D)
#
# # 将图片信息和信息输入到模型中
# client_model.set_input(img)

'''
        ReLu层 
'''
# Relu = layers.SecReLu()
# client_model.add(Relu)
#
# # 将图片信息和信息输入到模型中
# client_model.set_input(img)


'''
        SecMaxPool2D    
'''

# MaxPoll2D = layers.SecMaxPool2D(kernel_size=2, stride=1)
# client_model.add(MaxPoll2D)
#
# # 将图片信息和信息输入到模型中
# client_model.set_input(img)

'''
    SecAvgPool2D    损失精度
'''
# SecAvgPool2D = layers.SecAvgPool2D(kernel_size=(2, 2), stride=1, padding=0)
# client_model.add(SecAvgPool2D)
# client_model.set_input(img)

'''
        BatchNormalization
'''
# gamma = np.ones(img_shape, dtype=np.int32)
# beta = np.ones(img_shape, dtype=np.int32)
# gamma_0, gamma_1 = ssv.share_nparray(gamma)
# beta_0, beta_1 = ssv.share_nparray(beta)
# client.send_np_array(gamma_0)
# client.send_np_array(beta_0)
#
# BatchNormalization = layers.BatchNormalization(gamma_1, beta_1)
# client_model.add(BatchNormalization)
# client_model.set_input(img)


'''
        flatten
'''
# flatten = layers.Flatten(img_shape)
# client_model.add(flatten)
# client_model.set_input(img)



'''
    SecLinear
'''
# # x = np.random.randint(0, 10, size=(3, 13), dtype=np.int32)
# # weight = np.random.randint(0, 10, size=(10, 13), dtype=np.int32)
#
# x = np.array([[1, 5, 0, 5, 4, 1, 3, 4, 4, 2, 5, 9, 9],
#               [2, 0, 5, 9, 4, 4, 8, 0, 6, 6, 7, 7, 0],
#               [9, 5, 5, 2, 3, 4, 8, 1, 7, 4, 0, 8, 5]], dtype=np.int32)
# #
# weight = np.array([[9, 6, 7, 1, 6, 9, 8, 9, 3, 8, 8, 4, 6],
#                    [7, 4, 6, 7, 6, 1, 0, 5, 4, 6, 5, 9, 1],
#                    [8, 8, 0, 8, 1, 5, 4, 6, 6, 8, 6, 0, 7],
#                    [7, 8, 0, 8, 8, 5, 1, 4, 4, 4, 7, 1, 9],
#                    [6, 5, 5, 0, 1, 2, 7, 3, 2, 5, 0, 9, 2],
#                    [5, 6, 1, 1, 2, 0, 7, 6, 5, 4, 7, 1, 5],
#                    [4, 8, 7, 2, 5, 6, 7, 3, 9, 6, 3, 9, 9],
#                    [2, 2, 4, 4, 4, 8, 2, 0, 3, 9, 9, 9, 4],
#                    [3, 2, 4, 7, 1, 2, 4, 4, 3, 4, 7, 3, 1],
#                    [8, 1, 6, 0, 7, 2, 8, 0, 3, 1, 1, 9, 2]], dtype=np.int32)
#
# print(x)
# print(weight)
#
# x_0, x_1 = ssv.share_nparray(x)
# weight_0, weight_1 = ssv.share_nparray(weight)
#
# client.send_np_array(x_0)
# client.send_np_array(weight_0)
#
# x = ShareV(x_1, p, client)
#
# bias = np.zeros((3, 10), dtype=np.int32)
# secLinear = layers.SecLinear(weight_1, bias)
# client_model.add(secLinear)
#
# client_model.set_input(x)



# 启动 模型
client_model.predict()

res = ssv.restore_nparray(client_model.input, party=2)
print(res.shape)
print(res)
