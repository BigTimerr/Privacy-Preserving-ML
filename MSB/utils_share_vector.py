import numpy as np
import numpy.random

Ring = 10
n = 10

def int2bit_arr(value, n):
    arr = np.random.choice(a=[False], size=(n, 32))
    for i in range(0, 32):
        arr[:, i] = ((value >> i) & 0x01).reshape(1, n)
    arr = arr.astype(bool)
    return arr

def share_value_vector(sum_arr, n):
    share1 = np.random.randint(-Ring, Ring, size=n)
    share2 = (sum_arr - share1)
    return share1, share2

def calculate_c_i_next(ai, bi, ci):
    return (ai & bi) ^ (ci & (ai ^ bi))

def getMSB(a_h, b_h, c_h):
    return a_h ^ b_h ^ c_h

if __name__ == "__main__":
    sum_arr = np.random.randint(-Ring, Ring, size=n)
    share1, share2 = share_value_vector(sum_arr, n)

    share1_arr = int2bit_arr(share1, n)
    share2_arr = int2bit_arr(share2, n)

    i = 0
    c_i = np.asarray([False] * n)
    c_i_next = share1_arr[:, 0] & share2_arr[:, 0]
    while i < 32:
        c_i = c_i_next
        c_i_next = calculate_c_i_next(share1_arr[:, i], share1_arr[:, i], c_i)
        i += 1
    print(getMSB(share1_arr[:, 31], share1_arr[:, 31], c_i_next))
