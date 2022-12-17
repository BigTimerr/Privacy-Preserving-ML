import TCP.tcp as tcp
import MSB.utils_vector as utilsV
from ProtocolOnRing import param
import ProtocolOnRing.secret_sharing_vector_onring as ssv
import numpy as np
import torch

def share_value_vector(t, n):
    t = t & 0xffffffff
    t_0 = torch.randint(0, int(Ring / 2), t.shape, dtype=torch.int64) & 0xffffffff
    t_1 = (t - t_0) % Ring
    t_0 = int2bite_arr(t_0, n)
    t_1 = int2bite_arr(t_1, n)
    return t_0, t_1

def int2bite_arr(value, n):
    arr = torch.zeros(size=(n, 32)).bool()
    for i in range(0, 32):
        arr[:, i] = ((value >> i) & 0x01).reshape(1, n)
    return arr

# 初始化参数
client = tcp.TCPClient("127.0.0.1", 9998, 4096)
client.run()

p = 1  # party 1:client
Ring = param.Ring
n = param.n

# 计算目标
X = torch.randint(-10, 10, size=(n,), dtype=torch.int64)
# X = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

# 首先将X Y分享，然后发送给两个服务器（模拟）
x_0, x_1 = share_value_vector(X, n)
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
cb = utilsV.get_carry_bit_sonic(x, n)
res = utilsV.get_MSB(cb, x)
#
print()
print("/******************************************************/")
print("计算 MSB 其结果为:", res)
print("/******************************************************/")
print()
#
