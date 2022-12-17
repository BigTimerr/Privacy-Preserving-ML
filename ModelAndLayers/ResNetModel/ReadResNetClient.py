"""
# @Time : 2022/12/12 21:17
# @Author : ruetrash
# @File : ReadResNetClient.py
"""

import time

import TCP.tcp as tcp
import torch
import ModelAndLayers.ResNetModel.ResNet as ResNet
from ModelAndLayers.build_secure_model.build_secure_model_process import buildSecureModelProcessOnClient
import ProtocolOnRing.secret_sharing_fixpoint as ssf
from ProtocolOnRing.secret_sharing_fixpoint import ShareFloat

# 初始化参数
client = tcp.TCPClient("127.0.0.1", 9999, 40960)
client.run()
p = 1


net = ResNet.ResNet18()
net.load_state_dict(torch.load('ModelAndLayers/ResNetModel/ResNet18.pth'))

secure_model = buildSecureModelProcessOnClient(model=net, client=client)

print(secure_model)

# 开始识别
images = torch.ones(size=(1, 3, 32, 32))
img_0, img_1 = ssf.share_float(ssf.encode(images))
client.send_torch_array(img_0)
input_images = ShareFloat(img_1, p, client)

secure_model.set_input(input_images)
secure_model.predict()

res = ssf.restore_float(secure_model.input)
res = ssf.decode(res)


