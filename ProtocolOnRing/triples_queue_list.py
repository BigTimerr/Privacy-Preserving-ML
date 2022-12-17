"""
# @Time : 2022/8/23 20:52
# @Author : ruetrash
# @File : triples_queue_list.py
"""
import time
from multiprocessing import shared_memory
import ProtocolOnRing.param as param
import numpy as np

Ring = param.Ring
k = 50
global buffer


def share_tensor(t):
    t_0 = np.random.randint(-Ring, Ring - 1, t.shape, dtype=np.int32) & 0xffffffff
    t_1 = (t - t_0) % (Ring * 2)

    t_0 = t_0.tolist()
    t_1 = t_1.tolist()

    return t_0, t_1


def generate_triples():
    a = np.random.randint(-Ring, Ring - 1, size=k, dtype=np.int32) & 0xffffffff
    b = np.random.randint(-Ring, Ring - 1, size=k, dtype=np.int32) & 0xffffffff
    c = (a * b) % (Ring * 2)
    return a, b, c


def close_shm(shm):
    shm.close()
    shm.unlink()


def get_triples(p, ptr):
    if p == 0:
        shm_a_0 = shared_memory.ShareableList(name="a_0")
        array_a_0 = np.array(shm_a_0)
        a_0 = array_a_0[ptr]

        shm_b_0 = shared_memory.ShareableList(name="b_0")
        array_b_0 = np.array(shm_b_0)
        b_0 = array_b_0[ptr]

        shm_c_0 = shared_memory.ShareableList(name="c_0")
        array_c_0 = np.array(shm_c_0)
        c_0 = array_c_0[ptr]

        shm_a_0.shm.close()
        shm_b_0.shm.close()
        shm_c_0.shm.close()
        return a_0, b_0, c_0
    else:
        shm_a_1 = shared_memory.ShareableList(name="a_1")
        array_a_1 = np.array(shm_a_1)
        a_1 = array_a_1[ptr]

        shm_b_1 = shared_memory.ShareableList(name="b_1")
        array_b_1 = np.array(shm_b_1)
        b_1 = array_b_1[ptr]

        shm_c_1 = shared_memory.ShareableList(name="c_1")
        array_c_1 = np.array(shm_c_1)
        c_1 = array_c_1[ptr]

        shm_a_1.shm.close()
        shm_b_1.shm.close()
        shm_c_1.shm.close()
        return a_1, b_1, c_1


def get_triples_msb(p, ptr, n, l):
    if p == 0:
        shm_a_0 = shared_memory.ShareableList(name="a_0")
        array_a_0 = np.array(shm_a_0)
        a_0 = array_a_0[ptr]

        shm_b_0 = shared_memory.ShareableList(name="b_0")
        array_b_0 = np.array(shm_b_0)
        b_0 = array_b_0[ptr]

        shm_c_0 = shared_memory.ShareableList(name="c_0")
        array_c_0 = np.array(shm_c_0)
        c_0 = array_c_0[ptr]

        shm_a_0.shm.close()
        shm_b_0.shm.close()
        shm_c_0.shm.close()
        return a_0, b_0, c_0
    else:
        shm_a_1 = shared_memory.ShareableList(name="a_1")
        array_a_1 = np.array(shm_a_1)
        a_1 = array_a_1[ptr]

        shm_b_1 = shared_memory.ShareableList(name="b_1")
        array_b_1 = np.array(shm_b_1)
        b_1 = array_b_1[ptr]

        shm_c_1 = shared_memory.ShareableList(name="c_1")
        array_c_1 = np.array(shm_c_1)
        c_1 = array_c_1[ptr]

        shm_a_1.shm.close()
        shm_b_1.shm.close()
        shm_c_1.shm.close()
        return a_1, b_1, c_1


if __name__ == '__main__':

    # 每分钟刷新一次a,b,c 的值
    while 1:

        a, b, c = generate_triples()

        a_0, a_1 = share_tensor(a)
        b_0, b_1 = share_tensor(b)
        c_0, c_1 = share_tensor(c)

        shm_a_0 = shared_memory.ShareableList(a_0, name='a_0')
        shm_a_1 = shared_memory.ShareableList(a_1, name='a_1')
        shm_b_0 = shared_memory.ShareableList(b_0, name='b_0')
        shm_b_1 = shared_memory.ShareableList(b_1, name='b_1')
        shm_c_0 = shared_memory.ShareableList(c_0, name='c_0')
        shm_c_1 = shared_memory.ShareableList(c_1, name='c_1')

        print(a)
        print(a_0)
        print(a_1)

        time.sleep(60)

        shm_a_0.shm.close()
        shm_a_1.shm.close()
        shm_b_0.shm.close()
        shm_b_1.shm.close()
        shm_c_0.shm.close()
        shm_c_1.shm.close()
