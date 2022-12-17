"""
# @Time : 2022/9/15 16:02
# @Author : ruetrash
# @File : test_server.py
"""
import TCP.tcp as tcp
import ProtocolOnRing.secret_sharing_fixpoint as ssf
from ProtocolOnRing.secret_sharing_fixpoint import ShareFloat
import ProtocolOnRing.param as param



Q = param.Q
LEN_INTEGER = param.LEN_INTEGER
LEN_DECIMAL = param.LEN_DECIMAL
INVERSE = param.INVERSE
p = 0

# 初始化参数
server = tcp.TCPServer("127.0.0.1", 9999, 4096)
server.run()

x_0 = server.receive_torch_array()
y_0 = server.receive_torch_array()
t_0 = server.receive_torch_array()

x = ShareFloat(x_0, p, server)
y = ShareFloat(y_0, p, server)
t = ShareFloat(t_0, p, server)

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
