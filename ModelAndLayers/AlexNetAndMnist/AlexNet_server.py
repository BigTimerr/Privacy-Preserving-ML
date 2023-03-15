"""
# @Time : 2022/12/5 14:35
# @Author : ruetrash
# @File : AlexNet_server.py
"""

import torch
import TCP.tcp as tcp
from ProtocolOnRing.secret_sharing_fixpoint import ShareFloat
from ModelAndLayers.AlexNetAndMnist import AlexNet
import ProtocolOnRing.secret_sharing_fixpoint as ssf
from ModelAndLayers.onnx_converter import onnx_converter

# 初始化参数
server = tcp.TCPServer("127.0.0.1", 9999, 40960)
server.run()
p = 0

device = "cpu"

net = AlexNet.AlexNet()
net.load_state_dict(torch.load('ModelAndLayers/AlexNetAndMnist/MNIST_bak.pkl'))
# secure_model = buildSecureModelProcessOnServer(model=net, server=server)
dummy_input = torch.ones(100, 1, 28, 28)
secure_model = onnx_converter.from_pytorch(net, dummy_input, party=server, device=device)

while 1:
    img_0 = server.receive_torch_array(device)
    input_image = ShareFloat(img_0, p, server, device)

    secure_model.set_input(input_image)
    secure_model.predict()

    output = secure_model.output['output']
    res_ = ssf.restore_float(output)
    # res_ = ssf.restore_float(secure_model.input)
    res = ssf.decode(res_)
