"""
# @Time : 2022/10/12 19:45
# @Author : ruetrash
# @File : triples_fixpoint.py
"""
import numpy as np
import torch
import ProtocolOnRing.param as param
import random

Q = param.Q
Ring = param.Ring
data_path = "./ProtocolOnRing/triples_fixpoint/"

rd = np.random.RandomState(888)

"""
    运行此文件生成三元组,需要将下面的注释打开,运行成功
    并且生成数据之后,需要将下面代码重新注释掉
    因为目前定点数的环大小和整数大小还不一样，因此与整数部分的三元组生成分为两个文件
"""


# ##################################################################################
#
# def share_float(t):
#     # t_0 = np.random.randint(0, Q, t.shape)
#     # t_0 = rd.randint(0, Ring, t.shape)
#     t_0 = random.randrange(Q)
#     t_1 = (t - t_0) % Q
#     return t_0, t_1
#
#
# k = 50000
#
# # a = rd.randint(0, Ring, k)
# # b = rd.randint(0, Ring, k)
# a = random.randrange(Q)
# b = random.randrange(Q)
# c = (a * b) % Q
#
# a_0, a_1 = share_float(a)
# b_0, b_1 = share_float(b)
# c_0, c_1 = share_float(c)
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
# #
# print(a)
# print((a_0 + a_1) % Q)
# print(b)
# print((b_0 + b_1) % Q)
# print(a*b)
# print(c)
# print((a * b) % Q)

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
# print((a_0 + a_1) % Q)
# print(b)
# print((b_0 + b_1) % Q)
# print(c)
# print(a*b)
# print((a * b) % Q)


# def get_triples(p, ptr):
#     # print("triples: a: %d, b: %d ,c :%d" % (a[ptr].item(), b[ptr].item(), c[ptr].item()))
#
#     if p == 0:
#         # print("triples: a0: %d, b0: %d ,c0 :%d" % (a_0[ptr].item(), b_0[ptr].item(), c_0[ptr].item()))
#         return (a_0[ptr].item(), b_0[ptr].item(), c_0[ptr].item())
#     else:
#         # print("triples: a1: %d, b1: %d ,c1 :%d" % (a_1[ptr].item(), b_1[ptr].item(), c_1[ptr].item()))
#         return (a_1[ptr].item(), b_1[ptr].item(), c_1[ptr].item())


def get_triples(p, ptr):
    # print("triples: a: %d, b: %d ,c :%d" % (a[ptr].item(), b[ptr].item(), c[ptr].item()))

    if p == 0:
        # print("triples: a0: %d, b0: %d ,c0 :%d" % (a_0[ptr].item(), b_0[ptr].item(), c_0[ptr].item()))
        return (a_0, b_0, c_0)
    else:
        # print("triples: a1: %d, b1: %d ,c1 :%d" % (a_1[ptr].item(), b_1[ptr].item(), c_1[ptr].item()))
        return (a_1, b_1, c_1)
