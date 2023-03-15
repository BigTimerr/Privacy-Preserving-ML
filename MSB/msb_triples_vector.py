"""
# @Time : 2022/7/26 15:33
# @Author : ruetrash
# @File : msb_triples_vector.py
"""
import time
import numpy as np
import ProtocolOnRing.param as param
import torch

"""
    使用到MSB之前必须打开下面的注释运行程序，只需要打开一层注释部分即可
"""


Ring = param.Ring
data_path = "./MSB/MSB_triples_data/"
k = 35
n = 20000000

# ################################################################################
# # def share_bool(t):
# #     t_0 = np.random.randint(0, 2, size=(n, k), dtype=bool)
# #     t_1 = t ^ t_0
# #     return t_0, t_1
# #
# #
# # a = np.random.randint(0, 2, size=(n, k), dtype=bool)
# # b = np.random.randint(0, 2, size=(n, k), dtype=bool)
# #
# # c = a & b
# #
# # a_0, a_1 = share_bool(a)
# # b_0, b_1 = share_bool(b)
# # c_0, c_1 = share_bool(c)
#
# a_0 = b_0 = c_0 = torch.ones(size=(n,k)).bool()
# a_1 = b_1 = c_1 = torch.zeros(size=(n,k)).bool()
#
#
# # torch.save(a, data_path + "a.pth")
# # torch.save(b, data_path + "b.pth")
# # torch.save(c, data_path + "c.pth")
# torch.save(a_0, data_path + "a_0.pth")
# torch.save(a_1, data_path + "a_1.pth")
# torch.save(b_0, data_path + "b_0.pth")
# torch.save(b_1, data_path + "b_1.pth")
# torch.save(c_0, data_path + "c_0.pth")
# torch.save(c_1, data_path + "c_1.pth")
# #############################################################################

# a = torch.load(data_path + "a.pth")
# b = torch.load(data_path + "b.pth")
# c = torch.load(data_path + "c.pth")
a_0 = torch.load(data_path + "a_0.pth")
b_0 = torch.load(data_path + "b_0.pth")
c_0 = torch.load(data_path + "c_0.pth")
a_1 = torch.load(data_path + "a_1.pth")
b_1 = torch.load(data_path + "b_1.pth")
c_1 = torch.load(data_path + "c_1.pth")


def get_triples_msb(p, ptr, n, l):
    """
    :param p: 客户端还是服务器
    :param ptr: 从第几个数据开始取
    :param n: 取几维的数据
    :param l: 每一维度去多少数据
    :return: a,b,c三元组
    """

    if p == 0:
        return a_0[:n, ptr:ptr + l], b_0[:n, ptr:ptr + l], c_0[:n, ptr:ptr + l]
    else:
        return a_1[:n, ptr:ptr + l], b_1[:n, ptr:ptr + l], c_1[:n, ptr:ptr + l]

