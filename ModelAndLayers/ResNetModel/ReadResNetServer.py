"""
# @Time : 2022/12/12 21:17
# @Author : ruetrash
# @File : ReadResNetServer.py
"""

import torch
import TCP.tcp as tcp
from ProtocolOnRing.secret_sharing_fixpoint import ShareFloat
import ModelAndLayers.ResNetModel.ResNet as ResNet
from ModelAndLayers.build_secure_model.build_secure_model_process import buildSecureModelProcessOnServer
import ProtocolOnRing.secret_sharing_fixpoint as ssf
from ModelAndLayers.onnx_converter import onnx_converter
import warnings

warnings.filterwarnings('ignore')
# 初始化参数
server = tcp.TCPServer("127.0.0.1", 9999, 40960)
server.run()
p = 0
device = "cpu"

net = ResNet.resnet18()
net.load_state_dict(torch.load("ModelAndLayers/ResNetModel/ResNet18.pkl"))
net.eval()

dummy_input = torch.ones(10, 1, 28, 28)
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


# pytorch_model = ResNet.resnet18()
# pytorch_model.load_state_dict(torch.load("ModelAndLayers/ResNetModel/ResNet18_noBN.pkl"))
# dummy_input = torch.ones((3, 1, 28, 28))
# pytorch_model.eval()
# secure_model = onnx_converter.from_pytorch(pytorch_model, dummy_input, party=server)
#
#
# img_0 = server.receive_torch_array()
# input = ssf.ShareFloat(img_0, p, server)
#
# secure_model.set_input(input)
# secure_model.predict()
#
# output = secure_model.output['output']
#
# res_ = ssf.restore_float(output)
# res = ssf.decode(res_)