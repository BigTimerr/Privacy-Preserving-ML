"""
# @Time : 2022/9/7 15:27
# @Author : ruetrash
# @File : read_model_test_client.py
"""
import numpy as np
import torch
import torch.nn as nn
import TCP.tcp as tcp
from ModelAndLayers.build_secure_model.build_secure_model_process import buildSecureModelProcessOnClient
from ModelAndLayers.model.MyModel import MyModel,AlexNet
import ModelAndLayers.build_secure_model.function as function
import ProtocolOnRing.secret_sharing_fixpoint as ssf
from ProtocolOnRing.secret_sharing_fixpoint import ShareFloat

# 初始化参数
client = tcp.TCPClient("127.0.0.1", 9999, 4096)
client.run()
p = 1

# img = np.random.random((1, 3, 32, 32))
img = torch.ones((1, 3, 28, 28))

# img, label = function.getOneImage()
img_0, img_1 = ssf.share_float(ssf.encode(img))

client.send_torch_array(img_0)

input = ShareFloat(img_1, p, client)

model = AlexNet()
model.load_state_dict(torch.load("ModelAndLayers/model/myModel.pth"))


secure_model = buildSecureModelProcessOnClient(model=model, client=client)

print(secure_model)
secure_model.set_input(input)
secure_model.predict()

res = ssf.restore_float(secure_model.input, party=2)
res = ssf.decode(res)
print(res)
print(res.shape)

