import TCP.tcp as tcp
import MSB.utils_vector as utilsV
from ProtocolOnRing import param
import ProtocolOnRing.secret_sharing_vector_onring as ssv

# 初始化参数
server = tcp.TCPServer("127.0.0.1", 9998, 4096)
server.run()

p = 0  # party 0:server
Ring = param.Ring
n = param.n

# 首先接受所有的参数
x_0 = server.receive_torch_array()
# y_0 = server.receive_np_array()
#
x = ssv.ShareV(value=x_0, p=p, tcp=server)
# y = utils.Share(value=y_0, p=p, TCP=server)


# # ****************************得到MSB*****************************
#
cb = utilsV.get_carry_bit_sonic(x, n)
res = utilsV.get_MSB(cb, x)
print()
print("/******************************************************/")
print("计算 MSB 其结果为:", res)
print("/******************************************************/")
print()
#
