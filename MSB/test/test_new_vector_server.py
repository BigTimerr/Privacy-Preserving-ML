import numpy as np
import ProtocolOnRing.secret_sharing_vector_onring as ssv
import TCP.tcp as tcp
import MSB.utils_new_vector as utilsNV
from ProtocolOnRing import param

# 初始化参数
server = tcp.TCPServer("127.0.0.1", 9999, 4096)
server.run()

p = 0  # party 0:server
Ring = param.Ring
n = param.n

# 首先接受所有的参数
x_0 = server.receive_torch_array()
x = ssv.ShareV(value=x_0, p=p, tcp=server)



# # ****************************得到MSB*****************************
#
cb = utilsNV.get_carry_bit_new(x, n)
res = utilsNV.get_MSB(cb, x)
print()
print("/******************************************************/")
print("计算 MSB 其结果为:", res)
print("/******************************************************/")
print()
#
