"""
# @Time : 2022/9/7 23:04
# @Author : ruetrash
# @File : read_model_test_server.py
"""

import numpy as np
import torch
import torch.nn as nn
import TCP.tcp as tcp
from ModelAndLayers.build_secure_model.build_secure_model_process import buildSecureModelProcessOnServer
from ModelAndLayers.model.MyModel import MyModel,AlexNet
import ModelAndLayers.build_secure_model.function as function
import ProtocolOnRing.secret_sharing_fixpoint as ssf
from ProtocolOnRing.secret_sharing_fixpoint import ShareFloat

# 初始化参数
server = tcp.TCPServer("127.0.0.1", 9999, 4096)
server.run()
p = 0

img_0 = server.receive_torch_array()
input = ShareFloat(img_0, p, server)


model = AlexNet()
model.load_state_dict(torch.load("ModelAndLayers/model/myModel.pth"))


secure_model = buildSecureModelProcessOnServer(model=model, server=server)
secure_model.set_input(input)
secure_model.predict()

res = ssf.restore_float(secure_model.input, party=2)
res = ssf.decode(res)
print(res)
print(res.shape)
