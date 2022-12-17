import numpy as np
from ProtocolOnRing import param
from MSB import msb_triples_vector as triples

"""
    使用的sonic的方法来求取numpy数组的的MSB
"""
n = param.n
Ring = param.Ring
global ptr
ptr = 0


def int2bite(value):
    arr = np.random.choice(a=[False], size=32)
    for i in range(0, 32):
        x = (value >> i) & 0x01
        if x:
            arr[i] = True

    return arr


def int2bite_arr(value, n):
    arr = np.random.choice(a=[False], size=(n, 32))
    for i in range(0, 32):
        arr[:, i] = ((value >> i) & 0x01).reshape(1, n)

    arr = arr.astype(bool)
    return arr


def share_value_vector(t, n):
    t_0 = np.random.randint(-Ring, Ring, size=n)
    t_1 = (t - t_0)
    t_0 = int2bite_arr(t_0, n)
    t_1 = int2bite_arr(t_1, n)
    return t_0, t_1


class ShareV(object):
    def __init__(self, value, p, tcp):
        self.value = value
        self.p = p
        if p == 0:
            self.party = 'server'
        else:
            self.party = 'client'
        self.tcp = tcp


def get_carrt_bit_sonic(x: ShareV, n) -> bool:
    global ptr
    is_client = x.p
    # 生成 P和G。    P 只需要进行异或操作，所以直接继承过来。 G需要与操作，需要做转化
    P_i_layer1 = x.value
    G_i_layer1 = get_G_array(x)

    # layer 1
    P_pre, G_pre = get_P_and_G_array(x, 31, 15, P_i_layer1, G_i_layer1, True)
    P_i_2 = np.random.choice(a=[False], size=(n, 16))
    G_i_2 = np.random.choice(a=[False], size=(n, 16))

    P_i_2[:, 1:] = P_pre
    P_i_2[:, 0] = P_i_layer1[:, 0]

    G_i_2[:, 1:] = G_pre
    G_i_2[:, 0] = G_i_layer1[:, 0]

    # layer2
    P_i_3, G_i_3 = get_P_and_G_array(x, 16, 8, P_i_2, G_i_2, False)

    # layer3
    P_i_4, G_i_4 = get_P_and_G_array(x, 8, 4, P_i_3, G_i_3, False)

    # layer4
    P_i_5, G_i_5 = get_P_and_G_array(x, 4, 2, P_i_4, G_i_4, False)

    # layer5
    G2_i_layer5 = P_i_5[:, 1]
    a, b, c = triples.get_triples(x.p, ptr, 1)
    ptr += 1

    F_i_P2_layer5 = P_i_5[:, 1] ^ b
    E_i_G1_layer5 = G_i_5[:, 0] ^ a

    x.tcp.send_np_array(np.column_stack((F_i_P2_layer5, E_i_G1_layer5)))
    get_arr_layer5 = x.tcp.receive_np_array()

    F_of_P2_layer5 = F_i_P2_layer5 ^ get_arr_layer5[:, 0]
    E_of_G1_layer5 = E_i_G1_layer5 ^ get_arr_layer5[:, 1]

    Cb_i = C_and_2party(x.p, E_of_G1_layer5, F_of_P2_layer5, a, b, c) ^ G2_i_layer5

    return Cb_i


def C_and_2party(p, E, F, a, b, c):
    if p == 1:
        return (E & b) ^ (F & a) ^ c
    else:
        return (E & F) ^ (E & b) ^ (F & a) ^ c


def get_G_array(x: ShareV):
    global ptr

    a, b, c = triples.get_triples(x.p, ptr, 32)
    ptr += 32
    x_j = np.random.choice(a=[False], size=(n, 32))

    if x.p == 1:
        E_i = x.value ^ a
        F_i = x_j ^ b
    else:
        E_i = x_j ^ a
        F_i = x.value ^ b

    x.tcp.send_np_array(np.row_stack((E_i, F_i)))
    get_array = x.tcp.receive_np_array()
    E_of_G = get_array[:n] ^ E_i
    F_of_G = get_array[n:] ^ F_i

    G_i_layer1 = C_and_2party(x.p, E_of_G, F_of_G, a, b, c)

    return G_i_layer1


def get_P_and_G_array(x: ShareV, end, l, P_pre, G_pre, is_layer1):
    global ptr
    if is_layer1:
        begin = 1
    else:
        begin = 0

    G2_pre = G_pre[:, begin + 1:end:2]

    P1_a, P2_bP, c_P1P2 = triples.get_triples(x.p, ptr, l)
    G1_A, P2_bG, c_G1P2 = triples.get_triples(x.p, ptr + l, l)
    ptr += (l * 2)

    E_i_P1 = P_pre[:, begin:end:2] ^ P1_a
    E_i_G1 = G_pre[:, begin:end:2] ^ G1_A
    FP_i_P2 = P_pre[:, begin + 1:end:2] ^ P2_bP
    FG_i_P2 = P_pre[:, begin + 1:end:2] ^ P2_bG

    x.tcp.send_np_array(np.column_stack((E_i_P1, FP_i_P2, E_i_G1, FG_i_P2)))
    get_arr = x.tcp.receive_np_array()

    len = int(np.size(get_arr, 1) / 4)

    E_of_P1 = get_arr[:, 0:len] ^ E_i_P1
    FP_of_P2 = get_arr[:, len:len * 2] ^ FP_i_P2
    E_of_G1 = get_arr[:, len * 2:len * 3] ^ E_i_G1
    FG_of_P2 = get_arr[:, len * 3:len * 4] ^ FG_i_P2

    P = C_and_2party(x.p, E_of_P1, FP_of_P2, P1_a, P2_bP, c_P1P2)
    G = C_and_2party(x.p, E_of_G1, FG_of_P2, G1_A, P2_bG, c_G1P2) ^ G2_pre

    return P, G


def get_MSB(cb, x):
    res = x.value[:, 31] ^ cb
    x.tcp.send_np_array(res)
    res = x.tcp.receive_np_array() ^ res

    return res
