#
import random

import TCP.tcp as tcp
import ProtocolOnRing.secret_sharing_vector_onring as ssv
from ProtocolOnRing.secret_sharing_vector_onring import ShareV
import ProtocolOnRing.param as param
import numpy as np
import time
import csv
import torch

# 初始化参数
client = tcp.TCPClient("127.0.0.1", 9999, 4096)
client.run()

Ring = param.Ring
p = 1  # party 1:client
n = param.n


def showData(X, Y, T, x, y, t):
    # 打印一下 看看情况
    print("/******************************************************/")
    print("x 的原始值:", X)
    print("y 的原始值:", Y)
    print("t 的原始值:", T)
    # print("x 的分享份额:", x.value)
    # print("y 的分享份额:", y.value)
    # print("t 的分享份额:", t.value)
    print("/******************************************************/")
    print()


def sec_ge(x: ShareV, y: ShareV):
    z = x >= y
    res = ssv.restore_tensor(z, party=2)
    print()
    print("/******************************************************/")
    print("计算 x >= y 其结果为:", res)
    print("/******************************************************/")
    print()


def sec_le(x: ShareV, y: ShareV):
    z = x <= y
    res = ssv.restore_tensor(z, party=2)
    print()
    print("/******************************************************/")
    print("计算 x <= y 其结果为:", res)
    print("/******************************************************/")
    print()


def sec_gt(x: ShareV, y: ShareV):
    z = x > y
    res = ssv.restore_tensor(z, party=2)
    print()
    print("/******************************************************/")
    print("计算 x > y 其结果为:", res)
    print("/******************************************************/")
    print()


def sec_lt(x: ShareV, y: ShareV):
    z = x < y
    res = ssv.restore_tensor(z, party=2)
    print()
    print("/******************************************************/")
    print("计算 x < y 其结果为:", res)
    print("/******************************************************/")
    print()


def secADD(x: ShareV, y: ShareV):
    # ****************************加法*****************************
    z = x + y
    res = ssv.restore_tensor(z, party=2)
    print("/******************************************************/")
    print("计算 x+y 其结果为:", res)
    print("/******************************************************/")
    print()


def secIntADD(x: ShareV, t):
    # ****************************加法*****************************
    z = x + t
    res = ssv.restore_tensor(z, party=2)
    print("/******************************************************/")
    print("计算 x+t(常数) 其结果为:", res)
    print("/******************************************************/")
    print()


def secDec(x: ShareV, y: ShareV):
    # ****************************减法*****************************
    z = x - y
    res = ssv.restore_tensor(z, party=2)
    print("/******************************************************/")
    print("计算 x-y 其结果为:", res)
    print("/******************************************************/")
    print()


def secIntDec(x: ShareV, t):
    # ****************************int减法*****************************
    z = x - t
    res = ssv.restore_tensor(z, party=2)
    print("/******************************************************/")
    print("计算 x-t(常数) 其结果为:", res)
    print("/******************************************************/")
    print()


def secMul(x: ShareV, y: ShareV):
    # ****************************乘法*****************************
    z = x * y
    res = ssv.restore_tensor(z, party=2)
    print("/******************************************************/")
    print("计算 x*y 其结果为:", res)
    print("/******************************************************/")
    print()


def secFooldMul(x: ShareV, t):
    # ****************************泛洪乘法*****************************
    z = x * t
    res = ssv.restore_tensor(z, party=2)
    print("/******************************************************/")
    print("计算 x*t(常数) 其结果为:", res)
    print("/******************************************************/")
    print()


def sec_mat_mul(x: ShareV, y: ShareV):
    # ****************************矩阵点乘********************************
    z = x @ y
    res = ssv.restore_tensor(z, party=2)
    print("/******************************************************/")
    print("计算 矩阵乘法 其结果为:", res, sep=",")
    print("/******************************************************/")
    print()


if __name__ == '__main__':
    # input_shape = (1, 1, 1, 10)
    # kernel_shape = (1, 1, 1, 10)
    input_shape = (4, 4)
    kernel_shape = (4, 4)

    X = torch.randint(-4, 4, size=input_shape, dtype=torch.int64)
    Y = torch.randint(-4, 4, size=input_shape, dtype=torch.int64)
    T = torch.ones(1, dtype=torch.int64) * random.randint(1, 5)

    # 首先将X Y分享，然后发送给两个服务器（模拟）
    x_0, x_1 = ssv.share_tensor(X)
    y_0, y_1 = ssv.share_tensor(Y)
    t_0, t_1 = ssv.share_value(T)

    client.send_torch_array(x_0)
    client.send_torch_array(y_0)
    client.send_torch_array(t_0)

    # 创建自身的分享份额
    x = ShareV(value=x_1, p=p, tcp=client)
    y = ShareV(value=y_1, p=p, tcp=client)
    t = ShareV(value=t_1, p=p, tcp=client)

    showData(X, Y, T, x, y, t)
    secADD(x, y)
    secIntADD(x, t)
    secDec(x, y)
    secIntDec(x, t)
    secMul(x, y)
    secFooldMul(x, t)
    sec_mat_mul(x, y)
    sec_ge(x, y)
    sec_le(x, y)
    sec_gt(x, y)
    sec_lt(x, y)

    client.close()
