import numpy as np
import torch
import ProtocolOnRing.secret_sharing_vector_onring as ssv
import TCP.tcp as tcp
import MSB.utils_new_vector as utilsNV
from ProtocolOnRing import param

# 初始化参数
client = tcp.TCPClient("127.0.0.1", 9999, 4096)
client.run()

p = 1  # party 1:client
Ring = param.Ring
n = param.n

# 计算目标
X = torch.randint(-10, 10, size=(n,), dtype=torch.int64)


# 首先将X Y分享，然后发送给两个服务器（模拟）
x_0, x_1 = utilsNV.share_value_vector(X, n)
client.send_torch_array(x_0)

x = ssv.ShareV(value=x_1, p=p, tcp=client)

###############################初始值################################
#
print()
print("/******************************************************/")
print("x:", X)
print("/******************************************************/")
print()
#
# # ****************************得到MSB*****************************
#
cb = utilsNV.get_carry_bit_new(x, n)
res = utilsNV.get_MSB(cb, x)
#
print()
print("/******************************************************/")
print("计算 MSB 其结果为:", res)
print("/******************************************************/")
print()
#
