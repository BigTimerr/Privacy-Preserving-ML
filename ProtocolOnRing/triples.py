import numpy as np
import torch
import ProtocolOnRing.param as param

Ring = param.Ring
data_path = param.data_path

"""
    运行此文件生成三元组住,需要将下面的注释打开,运行成功
    并且生成数据之后,需要将下面代码重新注释掉
    
"""


# ##################################################################################
# def share_tensor(t):
#     t_0 = torch.randint(0, int(Ring / 2), size=t.shape, dtype=torch.int64) & 0xffffffff
#     t_1 = (t - t_0) % Ring
#     return t_0, t_1
#
#
# k = 50000
#
# a = torch.randint(0, int(Ring / 2), (k,), dtype=torch.int64) & 0xffffffff
# b = torch.randint(0, int(Ring / 2), (k,), dtype=torch.int64) & 0xffffffff
# c = (a * b) % Ring
#
# a_0, a_1 = share_tensor(a)
# b_0, b_1 = share_tensor(b)
# c_0, c_1 = share_tensor(c)
#
# torch.save(a, data_path + "a.pth")
# torch.save(b, data_path + "b.pth")
# torch.save(c, data_path + "c.pth")
# torch.save(a_0, data_path + "a_0.pth")
# torch.save(a_1, data_path + "a_1.pth")
# torch.save(b_0, data_path + "b_0.pth")
# torch.save(b_1, data_path + "b_1.pth")
# torch.save(c_0, data_path + "c_0.pth")
# torch.save(c_1, data_path + "c_1.pth")

# ###############################################################################


a = torch.load(data_path + "a.pth")
b = torch.load(data_path + "b.pth")
c = torch.load(data_path + "c.pth")
a_0 = torch.load(data_path + "a_0.pth")
b_0 = torch.load(data_path + "b_0.pth")
c_0 = torch.load(data_path + "c_0.pth")
a_1 = torch.load(data_path + "a_1.pth")
b_1 = torch.load(data_path + "b_1.pth")
c_1 = torch.load(data_path + "c_1.pth")

# print(a)
# print(type(a))
# print((a_0 + a_1) % Ring)
# print(a_0)
# print(a_1)


def get_triples(p, ptr):
    # print("triples: a: %d, b: %d ,c :%d" % (a[ptr].item(), b[ptr].item(), c[ptr].item()))

    if p == 0:
        # print("triples: a0: %d, b0: %d ,c0 :%d" % (a_0[ptr].item(), b_0[ptr].item(), c_0[ptr].item()))
        return (a_0[ptr].item(), b_0[ptr].item(), c_0[ptr].item())
    else:
        # print("triples: a1: %d, b1: %d ,c1 :%d" % (a_1[ptr].item(), b_1[ptr].item(), c_1[ptr].item()))
        return (a_1[ptr].item(), b_1[ptr].item(), c_1[ptr].item())


def get_mul_triples(p, ptr):
    if p == 0:
        return (a[ptr].item(), c_0[ptr].item())
    else:
        return (b[ptr].item(), c_1[ptr].item())
