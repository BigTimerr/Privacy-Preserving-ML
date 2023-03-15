"""
# @Time : 2022/12/5 14:34
# @Author : ruetrash
# @File : AlexNet_client.py
"""
import time

import TCP.tcp as tcp
import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
from ModelAndLayers.AlexNetAndMnist import AlexNet
import ProtocolOnRing.secret_sharing_fixpoint as ssf
from ProtocolOnRing.secret_sharing_fixpoint import ShareFloat
from ModelAndLayers.onnx_converter import onnx_converter

# 初始化参数
client = tcp.TCPClient("127.0.0.1", 9999, 40960)
client.run()
p = 1
device = "cpu"

# 加载数据
transform1 = transforms.Compose([transforms.ToTensor()])
test_set = torchvision.datasets.MNIST(root='ModelAndLayers/Data/', train=False, download=True,
                                      transform=transform1)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=0)

net = AlexNet.AlexNet()
net.load_state_dict(torch.load('ModelAndLayers/AlexNetAndMnist/MNIST_bak.pkl'))


dummy_input = torch.ones(100, 1, 28, 28)
secure_model = onnx_converter.from_pytorch(net, dummy_input, party=client, device=device)


# 开始识别
correct_total = 0
total_total = 0

for data in test_loader:
    correct = 0
    total = 0
    images, labels = data

    images = images.to(device)
    labels = labels.to(device)

    img_0, img_1 = ssf.share_float(ssf.encode(images))
    client.send_torch_array(img_0)
    input = ShareFloat(img_1, p, client, device)

    start_time = time.time()
    secure_model.set_input(input)
    secure_model.predict()

    output = secure_model.output['output']
    res_ = ssf.restore_float(output)
    res = ssf.decode(res_)
    end_time = time.time()

    _, predicted = torch.max(res, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

    total_total += total
    correct_total += correct

    print('Accuracy of the network on test images:{}%'.format(100 * correct / total))  # 输出识别准确率
    print("总共用时：", end_time - start_time)

