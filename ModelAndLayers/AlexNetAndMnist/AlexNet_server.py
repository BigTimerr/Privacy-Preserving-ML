"""
# @Time : 2022/12/5 14:35
# @Author : ruetrash
# @File : AlexNet_server.py
"""

import torch
import TCP.tcp as tcp
from ProtocolOnRing.secret_sharing_fixpoint import ShareFloat
from ModelAndLayers.AlexNetAndMnist import AlexNet
from ModelAndLayers.build_secure_model.build_secure_model_process import buildSecureModelProcessOnServer
import ProtocolOnRing.secret_sharing_fixpoint as ssf

# 初始化参数
server = tcp.TCPServer("127.0.0.1", 9999, 40960)
server.run()
p = 0

net = AlexNet.AlexNet()
net.load_state_dict(torch.load('ModelAndLayers/AlexNetAndMnist/MNIST_bak.pkl'))
secure_model = buildSecureModelProcessOnServer(model=net, server=server)

while 1:
    img_0 = server.receive_torch_array()
    input_image = ShareFloat(img_0, p, server)

    secure_model.set_input(input_image)
    secure_model.predict()

    res_ = ssf.restore_float(secure_model.input)
    res = ssf.decode(res_)
