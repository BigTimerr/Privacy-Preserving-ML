import time
from multiprocessing import shared_memory
import torch
import ProtocolOnRing.param as param

Ring = param.Ring
k = 50
global buffer


def share_tensor(t):
    t_0 = torch.randint(0, Ring, t.shape)
    t_1 = (t - t_0) % Ring
    return t_0, t_1


def generate_triples():
    a = torch.randint(-Ring, Ring, (k,))
    b = torch.randint(-Ring, Ring, (k,))
    c = (a * b) % Ring
    return a, b, c


def close_shm(shm):
    shm.close()
    shm.unlink()


def get_triples(p, ptr):
    if p == 0:
        shm_a_0 = shared_memory.SharedMemory(name="a_0", size=k)
        a_0 = shm_a_0.buf[ptr]
        shm_b_0 = shared_memory.SharedMemory(name="b_0", size=k)
        b_0 = shm_b_0.buf[ptr]
        shm_c_0 = shared_memory.SharedMemory(name="c_0", size=k)
        c_0 = shm_c_0.buf[ptr]
        close_shm(shm_a_0)
        close_shm(shm_b_0)
        close_shm(shm_c_0)
        return a_0, b_0, c_0
    else:
        shm_a_1 = shared_memory.SharedMemory(name="a_1", size=k)
        a_1 = shm_a_1.buf[ptr]
        shm_b_1 = shared_memory.SharedMemory(name="b_1", size=k)
        b_1 = shm_b_1.buf[ptr]
        shm_c_1 = shared_memory.SharedMemory(name="c_1", size=k)
        c_1 = shm_c_1.buf[ptr]
        close_shm(shm_a_1)
        close_shm(shm_b_1)
        close_shm(shm_c_1)
        return a_1, b_1, c_1


def showData(shm):
    print(shm.buf.tolist())


if __name__ == '__main__':
    shm_a = shared_memory.SharedMemory(create=True, name="a", size=k)
    shm_b = shared_memory.SharedMemory(create=True, name="b", size=k)
    shm_c = shared_memory.SharedMemory(create=True, name="c", size=k)
    shm_a_0 = shared_memory.SharedMemory(create=True, name="a_0", size=k)
    shm_a_1 = shared_memory.SharedMemory(create=True, name="a_1", size=k)
    shm_b_0 = shared_memory.SharedMemory(create=True, name="b_0", size=k)
    shm_b_1 = shared_memory.SharedMemory(create=True, name="b_1", size=k)
    shm_c_0 = shared_memory.SharedMemory(create=True, name="c_0", size=k)
    shm_c_1 = shared_memory.SharedMemory(create=True, name="c_1", size=k)

    # 每分钟刷新一次a,b,c 的值
    while 1:
        a, b, c = generate_triples()

        a_0, a_1 = share_tensor(a)
        b_0, b_1 = share_tensor(b)
        c_0, c_1 = share_tensor(c)


        shm_a.buf[:] = bytearray(a)
        shm_b.buf[:] = bytearray(b)
        shm_c.buf[:] = bytearray(c)
        shm_a_0.buf[:] = bytearray(a_0)
        shm_a_1.buf[:] = bytearray(a_1)
        shm_b_0.buf[:] = bytearray(b_0)
        shm_b_1.buf[:] = bytearray(b_1)
        shm_c_0.buf[:] = bytearray(c_0)
        shm_c_1.buf[:] = bytearray(c_1)

        showData(shm_a)
        showData(shm_a_0)
        showData(shm_a_1)
        time.sleep(60)
