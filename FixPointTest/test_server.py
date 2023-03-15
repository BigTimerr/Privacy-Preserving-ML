"""
# @Time : 2022/9/15 16:02
# @Author : ruetrash
# @File : test_server.py
"""
import TCP.tcp as tcp
import ProtocolOnRing.secret_sharing_fixpoint as ssf
from ProtocolOnRing.secret_sharing_fixpoint import ShareFloat
import ProtocolOnRing.param as param

# 初始化参数
Ring = param.Ring
p = 0
device = param.device
server = tcp.TCPServer("127.0.0.1", 9999, 4096)
server.run()

x_0 = server.receive_torch_array(device)
y_0 = server.receive_torch_array(device)
t_0 = server.receive_torch_array(device)

x = ShareFloat(x_0, p, server, device)
y = ShareFloat(y_0, p, server, device)
t = ShareFloat(t_0, p, server, device)

print("****************************************")
z = x + y
z = ssf.decode(ssf.restore_float(z))
print("x + y", z)
print("****************************************")
#
print("****************************************")
z = x - y
z = ssf.decode(ssf.restore_float(z))
print("x - y", z)
print("****************************************")



print("****************************************")
z = x * y
res = ssf.decode(ssf.restore_float(z))
print("x*y", res)

z = z * t
res = ssf.decode(ssf.restore_float(z))
print("x * y * t", res)
print("****************************************")


print("****************************************")
z = x @ y

res = ssf.decode(ssf.restore_float(z))
print("x @ y", res)

z = z @ t
res = ssf.decode(ssf.restore_float(z))
print("x @ y @ T", res)
print("****************************************")

print("****************************************")
z = x >= y
z = ssf.restore_float(z)
# z = ssf.decode(ssf.restore_float(z))
print("x >= y", z)
print("****************************************")

print("****************************************")
z = x <= y
z = ssf.restore_float(z)
# z = ssf.decode(ssf.restore_float(z))
print("x >= y", z)
print("****************************************")

print("****************************************")
z = x > y
z = ssf.restore_float(z)
# z = ssf.decode(ssf.restore_float(z))
print("x > y", z)
print("****************************************")

print("****************************************")
z = x < y
z = ssf.restore_float(z)
# z = ssf.decode(ssf.restore_float(z))
print("x < y", z)
print("****************************************")
