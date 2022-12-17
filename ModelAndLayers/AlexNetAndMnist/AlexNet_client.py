"""
# @Time : 2022/12/5 14:34
# @Author : ruetrash
# @File : AlexNet_client.py
"""
import time

import TCP.tcp as tcp
import torch
import torchvision
import torchvision.transforms as transforms
from ModelAndLayers.AlexNetAndMnist import AlexNet
from ModelAndLayers.build_secure_model.build_secure_model_process import buildSecureModelProcessOnClient
import ProtocolOnRing.secret_sharing_fixpoint as ssf
from ProtocolOnRing.secret_sharing_fixpoint import ShareFloat

# 初始化参数
client = tcp.TCPClient("127.0.0.1", 9999, 40960)
client.run()
p = 1

# 加载数据
transform1 = transforms.Compose([transforms.ToTensor()])
test_set = torchvision.datasets.MNIST(root='ModelAndLayers/AlexNetAndMnist/data', train=False, download=True,
                                      transform=transform1)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=0)

net = AlexNet.AlexNet()
net.load_state_dict(torch.load('ModelAndLayers/AlexNetAndMnist/MNIST_bak.pkl'))

secure_model = buildSecureModelProcessOnClient(model=net, client=client)

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

# 开始识别
correct_total = 0
total_total = 0

for data in test_loader:

    correct = 0
    total = 0
    images, labels = data

    img_0, img_1 = ssf.share_float(ssf.encode(images))
    client.send_torch_array(img_0)
    input = ShareFloat(img_1, p, client)

    start_time = time.time()
    secure_model.set_input(input)
    secure_model.predict()

    res = ssf.restore_float(secure_model.input)
    res = ssf.decode(res)
    end_time = time.time()

    _, predicted = torch.max(res, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

    total_total += total
    correct_total += correct

    print('Accuracy of the network on test images:{}%'.format(100 * correct / total))  # 输出识别准确率
    print("总共用时：", end_time - start_time)
