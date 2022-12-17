import numpy as np
from ProtocolOnRing import param
import torch

"""
    使用的刘新的方法来求取numpy数组的的MSB
"""

Ring = param.Ring

n = param.n


# 准备阶段
x_j = torch.zeros(size=(n, 32)).bool()
a_0 = torch.zeros(size=(n, 32)).bool()
b_0 = torch.zeros(size=(n, 32)).bool()
c_0 = torch.zeros(size=(n, 32)).bool()
a_1 = torch.ones(size=(n, 32)).bool()
b_1 = torch.ones(size=(n, 32)).bool()
c_1 = torch.ones(size=(n, 32)).bool()

# layer1
P1P2P3P4_new_0 = torch.zeros(size=(n, 32)).bool()
P1P2P3P4_new_1 = torch.ones(size=(n, 32)).bool()
G_all_0_new = torch.zeros(size=(n, 23)).bool()
G_all_1_new = torch.ones(size=(n, 23)).bool()
P1_a0_new = P2_a0_new = P3_a0_new = P4_a0_new = G1_a0_new = G2_a0_new = G3_a0_new = torch.zeros(size=(n, 7)).bool()
P1_a1_new = P2_a1_new = P3_a1_new = P4_a1_new = G1_a1_new = G2_a1_new = G3_a1_new = torch.ones(size=(n, 7)).bool()
a0_12 = a0_13 = a0_14 = a0_23 = a0_24 = a0_34 = a0_123 = a0_124 = a0_134 = a0_234 = a0_1234 = torch.zeros(size=(n, 7)).bool()
a1_12 = a1_13 = a1_14 = a1_23 = a1_24 = a1_34 = a1_123 = a1_124 = a1_134 = a1_234 = a1_1234 = torch.ones(size=(n, 7)).bool()
Ga0_12 = Ga0_13 = Ga0_23 = Ga0_123 = c0_G3P4 = torch.zeros(size=(n, 7)).bool()
Ga1_12 = Ga1_13 = Ga1_23 = Ga1_123 = c1_G3P4 = torch.ones(size=(n, 7)).bool()
P1_last_a0 = P2_last_a0 = P3_last_a0 = torch.zeros(size=(1, n)).bool()
P1_last_a1 = P2_last_a1 = P3_last_a1 = torch.ones(size=(1, n)).bool()
a0_12_last = a0_13_last = a0_23_last = a0_123_last = G1_last_a0 = G2_last_a0 = c0_G2P3_last = torch.zeros(size=(1, n)).bool()
a1_12_last = a1_13_last = a1_23_last = a1_123_last = G1_last_a1 = G2_last_a1 = c1_G2P3_last = torch.ones(size=(1, n)).bool()
G_i_layer2_new = torch.zeros(size=(n, 8)).bool()
P_i_layer2_new = torch.zeros(size=(n, 8)).bool()

# layer2
allP_0_layer2 = torch.zeros(size=(n, 8)).bool()
allP_1_layer2 = torch.ones(size=(n, 8)).bool()
allG_0_layer2 = torch.zeros(size=(n, 6)).bool()
allG_1_layer2 = torch.ones(size=(n, 6)).bool()
P1_a0_layer2_new = P2_a0_layer2_new = P3_a0_layer2_new = P4_a0_layer2_new = G1_a0_layer2_new = G2_a0_layer2_new = G3_a0_layer2_new = torch.zeros(size=(n, 2)).bool()
P1_a1_layer2_new = P2_a1_layer2_new = P3_a1_layer2_new = P4_a1_layer2_new = G1_a1_layer2_new = G2_a1_layer2_new = G3_a1_layer2_new = torch.ones(size=(n, 2)).bool()
a0_12_2 = a0_13_2 = a0_14_2 = a0_23_2 = a0_24_2 = a0_34_2 = a0_123_2 = a0_124_2 = a0_134_2 = a0_234_2 = a0_1234_2 = torch.zeros(size=(n, 2)).bool()
a1_12_2 = a1_13_2 = a1_14_2 = a1_23_2 = a1_24_2 = a1_34_2 = a1_123_2 = a1_124_2 = a1_134_2 = a1_234_2 = a1_1234_2 = torch.ones(size=(n, 2)).bool()
Ga0_12_2 = Ga0_13_2 = Ga0_23_2 = Ga0_24_2 = Ga0_34_2 = Ga0_123_2 = torch.zeros(size=(n, 2)).bool()
Ga1_12_2 = Ga1_13_2 = Ga1_23_2 = Ga1_24_2 = Ga1_34_2 = Ga1_123_2 = torch.ones(size=(n, 2)).bool()
c0_G3P4_layer2 = torch.zeros(size=(n, 2)).bool()
c1_G3P4_layer2 = torch.ones(size=(n, 2)).bool()

# layer3
P2_b0_layer3_new = P2_b1_layer3_new = G1_a0_layer3_new = G1_a1_layer3_new = c0_G1P2_layer3_new = c1_G1P2_layer3_new = torch.zeros(size=(1, n)).bool()


def int2bite(value):
    arr = torch.zeros(size=(32,)).bool()
    for i in range(0, 32):
        x = (value >> i) & 0x01
        if x:
            arr[i] = True

    return arr


def int2bite_arr(value, n):
    arr = torch.zeros(size=(n, 32)).bool()
    for i in range(0, 32):
        arr[:, i] = ((value >> i) & 0x01).reshape(1, n)
    return arr


def share_value_vector(t, n):
    t = t & 0xffffffff
    t_0 = torch.randint(0, int(Ring / 2), t.shape, dtype=torch.int64) & 0xffffffff
    t_1 = (t - t_0) % Ring
    t_0 = int2bite_arr(t_0, n)
    t_1 = int2bite_arr(t_1, n)
    return t_0, t_1




def get_carry_bit_new(x, n) -> bool:

    # init
    is_client = x.p
    P_i_layer1 = x.value

    if is_client == 1:
        E_i = P_i_layer1 ^ a_0
        F_i = x_j ^ b_0
    else:
        E_i = x_j ^ a_1
        F_i = P_i_layer1 ^ b_1

    x.tcp.send_torch_array(torch.cat((E_i, F_i), dim=0))
    get_array = x.tcp.receive_torch_array()
    E_of_G = get_array[:n] ^ E_i
    F_of_G = get_array[n:] ^ F_i



    if is_client == 1:
        G_i_layer1 = Cand_2_client(E_of_G, F_of_G, a_0, b_0, c_0)
    else:
        G_i_layer1 = Cand_2_server(E_of_G, F_of_G, a_1, b_1, c_1)

    # layer1
    G4_i_layer1 = G_i_layer1[:, [3, 7, 11, 15, 19, 23, 27]]
    if is_client == 1:
        Ei_of_allP = P_i_layer1 ^ P1P2P3P4_new_0
        Ei_of_allG = G_i_layer1[:, [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28,
                                    29]] ^ G_all_0_new
    else:
        Ei_of_allP = P_i_layer1 ^ P1P2P3P4_new_1
        Ei_of_allG = G_i_layer1[:, [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28,
                                    29]] ^ G_all_1_new

    temp1 = torch.cat((Ei_of_allP, Ei_of_allG), dim=1)
    x.tcp.send_torch_array(temp1)
    get_array_layer1 = x.tcp.receive_torch_array()

    E_of_P1arr = Ei_of_allP[:, [0, 4, 8, 12, 16, 20, 24]] ^ get_array_layer1[:, [0, 4, 8, 12, 16, 20, 24]]
    E_of_P1 = Ei_of_allP[:, 28] ^ get_array_layer1[:, 28]
    E_of_P2arr = Ei_of_allP[:, [1, 5, 9, 13, 17, 21, 25]] ^ get_array_layer1[:, [1, 5, 9, 13, 17, 21, 25]]
    E_of_P2 = Ei_of_allP[:, 29] ^ get_array_layer1[:, 29]
    E_of_P3arr = Ei_of_allP[:, [2, 6, 10, 14, 18, 22, 26]] ^ get_array_layer1[:, [2, 6, 10, 14, 18, 22, 26]]
    E_of_P3 = Ei_of_allP[:, 30] ^ get_array_layer1[:, 30]
    E_of_P4arr = Ei_of_allP[:, [3, 7, 11, 15, 19, 23, 27]] ^ get_array_layer1[:, [3, 7, 11, 15, 19, 23, 27]]
    E_of_G1arr = Ei_of_allG[:, [0, 3, 6, 9, 12, 15, 18]] ^ get_array_layer1[:, [31, 34, 37, 40, 43, 46, 49]]
    E_of_G1 = Ei_of_allG[:, 21] ^ get_array_layer1[:, 52]
    E_of_G2arr = Ei_of_allG[:, [1, 4, 7, 10, 13, 16, 19]] ^ get_array_layer1[:, [32, 35, 38, 41, 44, 47, 50]]
    E_of_G2 = Ei_of_allG[:, 22] ^ get_array_layer1[:, 53]
    E_of_G3arr = Ei_of_allG[:, [2, 5, 8, 11, 14, 17, 20]] ^ get_array_layer1[:, [33, 36, 39, 42, 45, 48, 51]]

    if is_client == 1:
        P_pre = Cand_4_client(E_of_P1arr, E_of_P2arr, E_of_P3arr, E_of_P4arr, P1_a0_new, P2_a0_new, P3_a0_new,
                              P4_a0_new, a0_12, a0_13, a0_14, a0_23, a0_24, a0_34, a0_123, a0_124, a0_134, a0_234,
                              a0_1234)
        P_last = Cand_3_client(E_of_P1, E_of_P2, E_of_P3, P1_last_a0, P2_last_a0, P3_last_a0, a0_12_last, a0_13_last,
                               a0_23_last, a0_123_last)
        G_pre = Cand_4_client(E_of_G1arr, E_of_P2arr, E_of_P3arr, E_of_P4arr, G1_a0_new, P2_a0_new, P3_a0_new,
                              P4_a0_new, a0_12, a0_13, a0_14, a0_23, a0_24, a0_34, a0_123, a0_124, a0_134, a0_234,
                              a0_1234) ^ Cand_3_client(E_of_G2arr, E_of_P3arr, E_of_P4arr, G2_a0_new, P3_a0_new,
                                                       P4_a0_new, Ga0_12, Ga0_13, Ga0_23, Ga0_123) ^ Cand_2_client(
            E_of_G3arr, E_of_P4arr,
            G3_a0_new, P4_a0_new, c0_G3P4) ^ G4_i_layer1

        G_last = Cand_3_client(E_of_G1, E_of_P2, E_of_P3, G1_last_a0, P2_last_a0, P3_last_a0, a0_12_last, a0_13_last,
                               a0_23_last, a0_123_last) ^ Cand_2_client(E_of_G2, E_of_P3, G2_last_a0, P3_last_a0,
                                                                        c0_G2P3_last) ^ G_i_layer1[:, 30]

    else:
        P_pre = Cand_4_server(E_of_P1arr, E_of_P2arr, E_of_P3arr, E_of_P4arr, P1_a1_new, P2_a1_new, P3_a1_new,
                              P4_a1_new, a1_12, a1_13, a1_14, a1_23, a1_24, a1_34, a1_123, a1_124, a1_134, a1_234,
                              a1_1234)
        P_last = Cand_3_server(E_of_P1, E_of_P2, E_of_P3, P1_last_a1, P2_last_a1, P3_last_a1, a1_12_last, a1_13_last,
                               a1_23_last, a1_123_last)
        G_pre = Cand_4_server(E_of_G1arr, E_of_P2arr, E_of_P3arr, E_of_P4arr, G1_a1_new, P2_a1_new, P3_a1_new,
                              P4_a1_new, a1_12, a1_13, a1_14, a1_23, a1_24, a1_34, a1_123, a1_124, a1_134, a1_234,
                              a1_1234) ^ Cand_3_server(E_of_G2arr, E_of_P3arr, E_of_P4arr, G2_a1_new, P3_a1_new,
                                                       P4_a1_new, Ga1_12, Ga1_13, Ga1_23, Ga1_123) ^ Cand_2_server(
            E_of_G3arr, E_of_P4arr, G3_a1_new, P4_a1_new, c1_G3P4) ^ G4_i_layer1

        G_last = Cand_3_server(E_of_G1, E_of_P2, E_of_P3, G1_last_a1, P2_last_a1, P3_last_a1, a1_12_last, a1_13_last,
                               a1_23_last, a1_123_last) ^ Cand_2_server(E_of_G2, E_of_P3, G2_last_a1, P3_last_a1,
                                                                        c1_G2P3_last) ^ G_i_layer1[:, 30]

    G_i_layer2_new[:, :7] = G_pre
    G_i_layer2_new[:, 7] = G_last
    P_i_layer2_new[:, :7] = P_pre
    P_i_layer2_new[:, 7] = P_last

    # layer2
    G4_i_layer2 = G_i_layer2_new[:, [3, 7]]
    if is_client == 1:
        Ei_of_allP_layer2 = P_i_layer2_new ^ allP_0_layer2
        Ei_of_allG_layer2 = G_i_layer2_new[:, [0, 1, 2, 4, 5, 6]] ^ allG_0_layer2
    else:
        Ei_of_allP_layer2 = P_i_layer2_new ^ allP_1_layer2
        Ei_of_allG_layer2 = G_i_layer2_new[:, [0, 1, 2, 4, 5, 6]] ^ allG_1_layer2

    temp2 = torch.cat((Ei_of_allP_layer2, Ei_of_allG_layer2), dim=1)
    x.tcp.send_torch_array(temp2)
    get_array_layer2 = x.tcp.receive_torch_array()

    E_of_P1arr_2 = Ei_of_allP_layer2[:, [0, 4]] ^ get_array_layer2[:, [0, 4]]
    E_of_P2arr_2 = Ei_of_allP_layer2[:, [1, 5]] ^ get_array_layer2[:, [1, 5]]
    E_of_P3arr_2 = Ei_of_allP_layer2[:, [2, 6]] ^ get_array_layer2[:, [2, 6]]
    E_of_P4arr_2 = Ei_of_allP_layer2[:, [3, 7]] ^ get_array_layer2[:, [3, 7]]
    E_of_G1arr_2 = Ei_of_allG_layer2[:, [0, 3]] ^ get_array_layer2[:, [8, 11]]
    E_of_G2arr_2 = Ei_of_allG_layer2[:, [1, 4]] ^ get_array_layer2[:, [9, 12]]
    E_of_G3arr_2 = Ei_of_allG_layer2[:, [2, 5]] ^ get_array_layer2[:, [10, 13]]

    if is_client == 1:
        P_i_layer3_new = Cand_4_client(E_of_P1arr_2, E_of_P2arr_2, E_of_P3arr_2, E_of_P4arr_2, P1_a0_layer2_new,
                                       P2_a0_layer2_new, P3_a0_layer2_new, P4_a0_layer2_new, a0_12_2, a0_13_2, a0_14_2,
                                       a0_23_2, a0_24_2, a0_34_2, a0_123_2, a0_124_2, a0_134_2, a0_234_2, a0_1234_2)

        G_i_layer3_new = Cand_4_client(E_of_G1arr_2, E_of_P2arr_2, E_of_P3arr_2, E_of_P4arr_2, G1_a0_layer2_new,
                                       P2_a0_layer2_new, P3_a0_layer2_new, P4_a0_layer2_new, a0_12_2, a0_13_2, a0_14_2,
                                       a0_23_2, a0_24_2, a0_34_2, a0_123_2, a0_124_2, a0_134_2, a0_234_2,
                                       a0_1234_2) ^ Cand_3_client(E_of_G2arr_2, E_of_P3arr_2, E_of_P4arr_2,
                                                                  G2_a0_layer2_new, P3_a0_layer2_new, P4_a0_layer2_new,
                                                                  Ga0_12_2, Ga0_13_2, Ga0_23_2,
                                                                  Ga0_123_2) ^ Cand_2_client(E_of_G3arr_2, E_of_P4arr_2,
                                                                                             G3_a0_layer2_new,
                                                                                             P4_a0_layer2_new,
                                                                                             c0_G3P4_layer2) ^ G4_i_layer2
    else:
        P_i_layer3_new = Cand_4_server(E_of_P1arr_2, E_of_P2arr_2, E_of_P3arr_2, E_of_P4arr_2, P1_a1_layer2_new,
                                       P2_a1_layer2_new, P3_a1_layer2_new, P4_a1_layer2_new, a1_12_2, a1_13_2, a1_14_2,
                                       a1_23_2, a1_24_2, a1_34_2, a1_123_2, a1_124_2, a1_134_2, a1_234_2, a1_1234_2)

        G_i_layer3_new = Cand_4_server(E_of_G1arr_2, E_of_P2arr_2, E_of_P3arr_2, E_of_P4arr_2, G1_a1_layer2_new,
                                       P2_a1_layer2_new, P3_a1_layer2_new, P4_a1_layer2_new, a1_12_2, a1_13_2, a1_14_2,
                                       a1_23_2, a1_24_2, a1_34_2, a1_123_2, a1_124_2, a1_134_2, a1_234_2,
                                       a1_1234_2) ^ Cand_3_server(E_of_G2arr_2, E_of_P3arr_2, E_of_P4arr_2,
                                                                  G2_a1_layer2_new, P3_a1_layer2_new, P4_a1_layer2_new,
                                                                  Ga1_12_2, Ga1_13_2, Ga1_23_2,
                                                                  Ga1_123_2) ^ Cand_2_server(E_of_G3arr_2, E_of_P4arr_2,
                                                                                             G3_a1_layer2_new,
                                                                                             P4_a1_layer2_new,
                                                                                             c1_G3P4_layer2) ^ G4_i_layer2

    # layer3
    G2_i_layer3_new = G_i_layer3_new[:, 1]
    if is_client == 1:
        E_i_P2_layer3_new = P_i_layer3_new[:, 1] ^ P2_b0_layer3_new
        E_i_G1_layer3_new = G_i_layer3_new[:, 0] ^ G1_a0_layer3_new
    else:
        E_i_P2_layer3_new = P_i_layer3_new[:, 1] ^ P2_b1_layer3_new
        E_i_G1_layer3_new = G_i_layer3_new[:, 0] ^ G1_a1_layer3_new

    temp3 = torch.stack((E_i_P2_layer3_new, E_i_G1_layer3_new), dim=1)
    x.tcp.send_torch_array(temp3)
    get_array_layer3 = x.tcp.receive_torch_array()

    E_of_P2_layer3_new = E_i_P2_layer3_new ^ get_array_layer3[:, 0]
    E_of_G1_layer3_new = E_i_G1_layer3_new ^ get_array_layer3[:, 1]

    if is_client == 1:
        Cb = Cand_2_client(E_of_G1_layer3_new, E_of_P2_layer3_new, G1_a0_layer3_new, P2_b0_layer3_new,
                           c0_G1P2_layer3_new) ^ G2_i_layer3_new
    else:
        Cb = Cand_2_server(E_of_G1_layer3_new, E_of_P2_layer3_new, G1_a1_layer3_new, P2_b1_layer3_new,
                           c1_G1P2_layer3_new) ^ G2_i_layer3_new

    return Cb


def get_MSB(cb, x):
    res = x.value[:, 31] ^ cb
    x.tcp.send_torch_array(res)
    res = x.tcp.receive_torch_array() ^ res

    return res


def Cand_2_client(E, F, a0, b0, c0):
    return (E & b0) ^ (F & a0) ^ c0


def Cand_2_server(E, F, a1, b1, c1):
    return (E & F) ^ (E & b1) ^ (F & a1) ^ c1


def Cand_3_client(E_arr1, E_arr2, E_arr3, a0_1, a0_2, a0_3, a0_12, a0_13, a0_23, a0_123):
    arr = (E_arr1 & E_arr2 & a0_3) ^ (E_arr1 & E_arr3 & a0_2) ^ (E_arr2 & E_arr3 & a0_1) ^ (E_arr1 & a0_23) ^ (
            E_arr2 & a0_13) ^ (E_arr3 & a0_12) ^ a0_123
    return arr


def Cand_3_server(E_arr1, E_arr2, E_arr3, a1_1, a1_2, a1_3, a1_12, a1_13, a1_23, a1_123):
    arr = (E_arr1 & E_arr2 & E_arr3) ^ (E_arr1 & E_arr2 & a1_3) ^ (E_arr1 & E_arr3 & a1_2) ^ (
            E_arr2 & E_arr3 & a1_1) ^ (E_arr1 & a1_23) ^ (
                  E_arr2 & a1_13) ^ (E_arr3 & a1_12) ^ a1_123
    return arr


def Cand_4_client(E_arr1, E_arr2, E_arr3, E_arr4, a0_1, a0_2, a0_3, a0_4, a0_12, a0_13, a0_14, a0_23, a0_24, a0_34,
                  a0_123, a0_124, a0_134, a0_234, a0_1234):
    arr = (E_arr1 & E_arr2 & E_arr3 & a0_4) ^ (E_arr1 & E_arr2 & E_arr4 & a0_3) ^ (E_arr1 & E_arr4 & E_arr3 & a0_2) ^ (
            E_arr4 & E_arr2 & E_arr3 & a0_1) ^ (E_arr1 & E_arr2 & a0_34) ^ (E_arr1 & E_arr3 & a0_24) ^ (
                  E_arr1 & E_arr4 & a0_23) ^ (E_arr2 & E_arr3 & a0_14) ^ (E_arr2 & E_arr4 & a0_13) ^ (
                  E_arr3 & E_arr4 & a0_12) ^ (E_arr1 & a0_234) ^ (E_arr2 & a0_134) ^ (E_arr3 & a0_124) ^ (
                    E_arr4 & a0_123) ^ a0_1234
    return arr


def Cand_4_server(E_arr1, E_arr2, E_arr3, E_arr4, a0_1, a0_2, a0_3, a0_4, a0_12, a0_13, a0_14, a0_23, a0_24, a0_34,
                  a0_123, a0_124, a0_134, a0_234, a0_1234):
    arr = (E_arr1 & E_arr2 & E_arr3 & E_arr4) ^ (E_arr1 & E_arr2 & E_arr3 & a0_4) ^ (
            E_arr1 & E_arr2 & E_arr4 & a0_3) ^ (E_arr1 & E_arr4 & E_arr3 & a0_2) ^ (
                  E_arr4 & E_arr2 & E_arr3 & a0_1) ^ (E_arr1 & E_arr2 & a0_34) ^ (
                  E_arr1 & E_arr3 & a0_24) ^ (E_arr1 & E_arr4 & a0_23) ^ (E_arr2 & E_arr3 & a0_14) ^ (
                  E_arr2 & E_arr4 & a0_13) ^ (E_arr3 & E_arr4 & a0_12) ^ (E_arr1 & a0_234) ^ (E_arr2 & a0_134) ^ (
                  E_arr3 & a0_124) ^ (E_arr4 & a0_123) ^ a0_1234

    return arr
