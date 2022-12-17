# import triples as tp
#
#
# for i in range(5):
#     a_0 = tp.get_a0(i)
#     a_1 = tp.get_a1(i)
#
#     b_0 = tp.get_b0(i)
#     b_1 = tp.get_b1(i)
#
#     c_0 = tp.get_c0(i)
#     c_1 = tp.get_c1(i)
#
#     print(((a_0 + a_1) * (b_0 + b_1)) % tp.Ring)
#     print((c_0 + c_1 ) % tp.Ring)


# a = 6
#
# b = 8
#
# r = 10
#
# print((a-b)%r)
#
#
# print((a-b%r)%r)
#
#
# print((1-8)%20)

#
# import torch  # 命令行是逐行立即执行的
# content = torch.load('./triples_data/1/a.pth')
# print(content)   # keys()
# # 之后有其他需求比如要看 key 为 model 的内容有啥
#
#
# print("==================================================================")
#
# content = torch.load('./triples_data/2/a.pth')
# print(content)   # keys()
# # 之后有其他需求比如要看 key 为 model 的内容有啥

from multiprocessing import shared_memory, sharedctypes, managers

import numpy as np
import torch
import ProtocolOnRing.param as param
import triples_queue as tp
from multiprocessing import shared_memory

Ring = 1000
k = 10
n = 5

if __name__ == '__main__':
    X = np.array([[[[-2, 6, 4], [-3, -4, -7]]]])

    Y = np.array([[[[9, 7, -9, 8], [-5, 5, -6, 8], [9, 4, -7, -5]]]])


    print(np.matmul(X, Y))

    print(X@Y)
