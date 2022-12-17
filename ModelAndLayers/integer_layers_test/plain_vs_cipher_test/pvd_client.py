"""
# @Time : 2022/9/7 10:08
# @Author : ruetrash
# @File : single_client.py
"""
import torch
import TCP.tcp as tcp
import numpy as np
import ProtocolOnRing.param as param
import ModelAndLayers.model.modeloflayers as model
import ModelAndLayers.layers.layers as layers
from ProtocolOnRing.secret_sharing_vector_onring import ShareV
import ProtocolOnRing.secret_sharing_vector_onring as ssv

# 初始化客户端进程
client = tcp.TCPClient("127.0.0.1", 9999, 4096)
client.run()

# 初始化参数
p = 1  # party 0:client
Ring = param.Ring

# 初始化图片信息
img_shape = (1, 3, 5, 5)
kernel_shape = (3, 3, 2, 2)  # 卷积层的kernel 形状

img = np.random.randint(-10, 10, size=img_shape, dtype=np.int32)

# 卷积核为了和在明文上的训练一致，所以直接固定
kernel_0 = np.array([[[[1, 1], [1, 1]],
                      [[1, 1], [1, 1]],
                      [[1, 1], [1, 1]]],

                     [[[1, 1], [1, 1]],
                      [[1, 1], [1, 1]],
                      [[1, 1], [1, 1]]],

                     [[[1, 1], [1, 1]],
                      [[1, 1], [1, 1]],
                      [[1, 1], [1, 1]]]])
kernel_1 = np.array([[[[2, 3], [4, 5]],
                      [[3, 5], [6, 7]],
                      [[5, 8], [3, 6]]],

                     [[[1, 0], [1, 7]],
                      [[3, 9], [0, 3]],
                      [[5, 3], [1, 3]]],

                     [[[6, 7], [5, 2]],
                      [[1, 1], [1, 1]],
                      [[3, 1], [7, 0]]]])
print(repr(img))
# 分享数据
img_0, img_1 = ssv.share_nparray(img)
client.send_np_array(img_0)
client.send_np_array(kernel_0)

img = ShareV(img_1, p, client)

# 创建模型
client_model = model.ModelOfLayers()

'''BatchNormalization'''


'''Conv2D'''
Conv2D = layers.SecConv2d(kernel_1, stride=1, padding=1)
client_model.add(Conv2D)

'''SecMaxPool2D'''
SecMaxPool2D = layers.SecMaxPool2D(kernel_size=2, stride=1, padding=0)
client_model.add(SecMaxPool2D)

'''ReLu'''
Relu = layers.SecReLu()
client_model.add(Relu)

'''SecAvgPool2D'''
SecAvgPool2D = layers.SecAvgPool2D(kernel_size=2, stride=1, padding=0)
client_model.add(SecAvgPool2D)


# '''Conv2D'''
# Conv2D = layers.Conv2D(kernel_1, kernel_shape, stride=1, padding=1)
# client_model.add(Conv2D)
#
#
# '''SecMaxPool2D'''
# SecAvgPool2D = layers.SecMaxPool2D(kernel_size=2, stride=1, padding=0)
# client_model.add(SecAvgPool2D)


'''flatten'''
flatten = layers.Flatten()
client_model.add(flatten)


# 启动 模型
client_model.set_input(img)
client_model.predict()
res = ssv.restore_nparray(client_model.input, party=2)
print(res.shape)
print(res)
