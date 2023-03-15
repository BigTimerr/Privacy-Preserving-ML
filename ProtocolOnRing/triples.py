"""
# @Time : 2022/10/12 19:45
# @Author : ruetrash
# @File : triples_fixpoint.py
"""
import numpy as np
import torch
import ProtocolOnRing.param as param
import random

Ring = param.Ring
data_path = "./ProtocolOnRing/triples_data/"

# rd = np.random.RandomState(888)

"""
    运行此文件生成三元组,需要将下面的注释打开,运行成功 (需要在根目录下运行)
    并且生成数据之后,需要将下面代码重新注释掉
"""


# ##################################################################################
#
# def share_float(t):
#     t_0 = random.randrange(Ring)
#     t_1 = (t - t_0) % Ring
#     return t_0, t_1
#
#
# k = 50000
#
# a = random.randrange(Ring)
# b = random.randrange(Ring)
# c = (a * b) % Ring
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


def get_triples(p, ptr):
    if p == 0:
        return a_0, b_0, c_0
    else:
        return a_1, b_1, c_1
