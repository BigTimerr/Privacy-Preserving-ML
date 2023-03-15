"""
# @Time : 2022/9/15 16:02
# @Author : ruetrash
# @File : test_client.py
"""
import torch

import TCP.tcp as tcp
import ProtocolOnRing.secret_sharing_fixpoint as ssf
from ProtocolOnRing.secret_sharing_fixpoint import ShareFloat
import ProtocolOnRing.param as param
import numpy as np

Ring = param.Ring
p = 1
device = param.device

# 初始化参数
client = tcp.TCPClient("127.0.0.1", 9999, 4096)
client.run()

size = 8

X = torch.randn(size=(size, size))
Y = torch.randn(size=(size, size))
T = torch.randn(size=(size, size))

print("X", X)
print("Y", Y)
# print("T", T)
# print("X + Y:", X + Y)
# print("X - Y:", X - Y)
# print("X * Y:", X * Y)
# print("X * Y * T:", X * Y * T)
print("X @ Y", torch.matmul(X, Y))
print("X @ Y @ T", torch.matmul(torch.matmul(X, Y), T))

# print(" ")
# print(" ")

x_0, x_1 = ssf.share_float(ssf.encode(X))
y_0, y_1 = ssf.share_float(ssf.encode(Y))
t_0, t_1 = ssf.share_float(ssf.encode(T))

# print("x_0, x_1", x_0, x_1)
# print("y_0, y_1", y_0, y_1)

client.send_torch_array(x_0)
client.send_torch_array(y_0)
client.send_torch_array(t_0)

x = ShareFloat(x_1, p, client, device)
y = ShareFloat(y_1, p, client, device)
t = ShareFloat(t_1, p, client, device)

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
print("损失值", res - (X * Y))

z = z * t
res = ssf.decode(ssf.restore_float(z))
print("损失值", res - (X * Y * T))

print("****************************************")

print("****************************************")
z = x @ y

res = ssf.decode(ssf.restore_float(z))
print("损失值", res - np.matmul(X, Y))

z = z @ t
res = ssf.decode(ssf.restore_float(z))
temp = res - np.matmul(np.matmul(X, Y), T)
print("损失值", temp.max())
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
