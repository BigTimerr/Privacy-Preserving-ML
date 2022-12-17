"""
# @Time : 2022/7/26 15:33
# @Author : ruetrash
# @File : msb_triples_vector.py
"""
import time
import numpy as np
from multiprocessing import shared_memory
import ProtocolOnRing.param as param
import torch

Ring = param.Ring
data_path = "./MSB/data_vector/"
k = 100
n = 20000

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
    '''

    :param p: 客户端还是服务器
    :param ptr: 从第几个数据开始取
    :param n: 取几维的数据
    :param l: 每一维度去多少数据
    :return: a,b,c三元组
    '''
    # print("triples: a: %d, b: %d ,c :%d" % (a[ptr].item(), b[ptr].item(), c[ptr].item()))

    if p == 0:
        # print("triples: a0: %d, b0: %d ,c0 :%d" % (a_0[ptr].item(), b_0[ptr].item(), c_0[ptr].item()))
        return a_0[:n, ptr:ptr + l], b_0[:n, ptr:ptr + l], c_0[:n, ptr:ptr + l]
    else:
        # print("triples: a1: %d, b1: %d ,c1 :%d" % (a_1[ptr].item(), b_1[ptr].item(), c_1[ptr].item()))
        return a_1[:n, ptr:ptr + l], b_1[:n, ptr:ptr + l], c_1[:n, ptr:ptr + l]


#
# Ring = param.Ring_2
# k = 4096
# global buffer
#
#
# def share_value(t):
#     t_0 = np.random.randint(0, 2, size=k, dtype=bool)
#     t_1 = t ^ t_0
#     return t_0, t_1
#
#
# def generate_triples():
#     a = np.random.randint(0, 2, size=k, dtype=bool)
#     b = np.random.randint(0, 2, size=k, dtype=bool)
#     c = a ^ b
#     return a, b, c
#
#
# def close_shm(shm):
#     shm.close()
#     shm.unlink()
#
#
# def get_triples(p, ptr):
#     if p == 0:
#         shm_a_0 = shared_memory.SharedMemory(name="a_0", size=k)
#         a_0 = shm_a_0.buf[ptr]
#         shm_b_0 = shared_memory.SharedMemory(name="b_0", size=k)
#         b_0 = shm_b_0.buf[ptr]
#         shm_c_0 = shared_memory.SharedMemory(name="c_0", size=k)
#         c_0 = shm_c_0.buf[ptr]
#         close_shm(shm_a_0)
#         close_shm(shm_b_0)
#         close_shm(shm_c_0)
#         return a_0, b_0, c_0
#     else:
#         shm_a_1 = shared_memory.SharedMemory(name="a_1", size=k)
#         a_1 = shm_a_1.buf[ptr]
#         shm_b_1 = shared_memory.SharedMemory(name="b_1", size=k)
#         b_1 = shm_b_1.buf[ptr]
#         shm_c_1 = shared_memory.SharedMemory(name="c_1", size=k)
#         c_1 = shm_c_1.buf[ptr]
#         close_shm(shm_a_1)
#         close_shm(shm_b_1)
#         close_shm(shm_c_1)
#         return a_1, b_1, c_1
#
#
# def showData(shm):
#     print(shm.buf.tolist())
#
#
# if __name__ == '__main__':
#     shm_a = shared_memory.SharedMemory(create=True, name="a", size=k)
#     shm_b = shared_memory.SharedMemory(create=True, name="b", size=k)
#     shm_c = shared_memory.SharedMemory(create=True, name="c", size=k)
#     shm_a_0 = shared_memory.SharedMemory(create=True, name="a_0", size=k)
#     shm_a_1 = shared_memory.SharedMemory(create=True, name="a_1", size=k)
#     shm_b_0 = shared_memory.SharedMemory(create=True, name="b_0", size=k)
#     shm_b_1 = shared_memory.SharedMemory(create=True, name="b_1", size=k)
#     shm_c_0 = shared_memory.SharedMemory(create=True, name="c_0", size=k)
#     shm_c_1 = shared_memory.SharedMemory(create=True, name="c_1", size=k)
#
#     # 每分钟刷新一次a,b,c 的值
#     while 1:
#         a, b, c = generate_triples()
#
#         a_0, a_1 = share_value(a)
#         b_0, b_1 = share_value(b)
#         c_0, c_1 = share_value(c)
#
#         shm_a.buf[:] = bytearray(a)
#         shm_b.buf[:] = bytearray(b)
#         shm_c.buf[:] = bytearray(c)
#         shm_a_0.buf[:] = bytearray(a_0)
#         shm_a_1.buf[:] = bytearray(a_1)
#         shm_b_0.buf[:] = bytearray(b_0)
#         shm_b_1.buf[:] = bytearray(b_1)
#         shm_c_0.buf[:] = bytearray(c_0)
#         shm_c_1.buf[:] = bytearray(c_1)
#
#         showData(shm_a)
#         showData(shm_a_0)
#         showData(shm_a_1)
#         time.sleep(60)
