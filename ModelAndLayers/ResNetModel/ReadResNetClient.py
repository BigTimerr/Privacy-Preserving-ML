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
from ModelAndLayers.onnx_converter import onnx_converter
import warnings
import torchvision.transforms as transforms
import torchvision

# warnings.filterwarnings('ignore')
# 初始化参数
client = tcp.TCPClient("127.0.0.1", 9999, 40960)
client.run()
p = 1

device = "cpu"


# # 加载数据
transform1 = transforms.Compose([transforms.ToTensor()])
test_set = torchvision.datasets.MNIST(root='ModelAndLayers/Data', train=False, download=False,
                                      transform=transform1)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=10, shuffle=False)

net = ResNet.resnet18()
net.load_state_dict(torch.load("ModelAndLayers/ResNetModel/ResNet18.pkl"))
net.eval()

dummy_input = torch.ones(10, 1, 28, 28)
secure_model = onnx_converter.from_pytorch(net, dummy_input, party=client, device=device)


# 开始识别
correct_total = 0
total_total = 0

for data in test_loader:
    correct = 0
    total = 0
    images, labels = data
    labels = labels.to(device)

    img_0, img_1 = ssf.share_float(ssf.encode(images))
    client.send_torch_array(img_0)
    input = ShareFloat(img_1, p, client, device)

    start_time = time.time()
    secure_model.set_input(input)
    secure_model.predict()

    output = secure_model.output['output']
    res_ = ssf.restore_float(output)
    # res_ = ssf.restore_float(secure_model.input)
    res = ssf.decode(res_)
    end_time = time.time()

    _, predicted = torch.max(res, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

    total_total += total
    correct_total += correct

    print("测试图片总数", total)
    print('Accuracy of the network on test images:{}%'.format(100 * correct / total))  # 输出识别准确率
    print("总共用时：", end_time - start_time)

print("测试图片总数", correct_total)
print('Accuracy of the network on test images:{}%'.format(100 * correct_total / total_total))  # 输出识别准确率



# pytorch_model = ResNet.resnet18()
# pytorch_model.load_state_dict(torch.load("ModelAndLayers/ResNetModel/ResNet18_noBN.pkl"))
#
# dummy_input = torch.ones((3, 1, 28, 28))
# pytorch_model.eval()
# secure_model = onnx_converter.from_pytorch(pytorch_model, dummy_input, party=client)
#
# img = torch.ones((3, 1, 28, 28))
# # out = pytorch_model(img)
# # print(out)
#
# img_0, img_1 = ssf.share_float(ssf.encode(img))
# client.send_torch_array(img_0)
# input = ssf.ShareFloat(img_1, p, client)
#
# secure_model.set_input(input)
# secure_model.predict()
#
# output = secure_model.output['output']
#
# res_ = ssf.restore_float(output)
# res = ssf.decode(res_)
#
# print(res)

