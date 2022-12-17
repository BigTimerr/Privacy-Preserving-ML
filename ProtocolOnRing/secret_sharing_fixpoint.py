"""
# @Time : 2022/10/12 16:32
# @Author : ruetrash
# @File : secret_sharing_fixpoint.py
"""
import random
import torch
import ProtocolOnRing.param as param
import ProtocolOnRing.triples_fixpoint as tf
from MSB import utils_vector


Ring = param.Ring

BASE = param.BASE
Q = param.Q
scaled = param.scaled
LEN_INTEGER = param.LEN_INTEGER
LEN_DECIMAL = param.LEN_DECIMAL
INVERSE = param.INVERSE
KAPPA = param.KAPPA
PRECISION = LEN_INTEGER + LEN_DECIMAL

assert (Q > BASE ** LEN_INTEGER + LEN_DECIMAL)
assert ((INVERSE * scaled) % Q == 1)
assert (Q > BASE ** (2 * (LEN_INTEGER + LEN_DECIMAL) + KAPPA))

global ptr
ptr = 0


class ShareFloat(object):
    def __init__(self, value, p, tcp):
        self.value = value
        self.p = p
        self.tcp = tcp
        if p == 0:
            self.party = 'server'
        else:
            self.party = 'client'

    def __str__(self):
        return "[{} value:{},\n party:{}]".format(self.__class__.__name__, self.value, self.party)

    def __add__(self, other):
        if isinstance(other, ShareFloat):
            return sec_add(self, other)
        elif isinstance(other, int):
            if self.p == 0:
                y_share = ShareFloat(value=other, p=self.p, tcp=self.tcp)
            else:
                y_share = ShareFloat(value=0, p=self.p, tcp=self.tcp)
            return sec_add(self, y_share)

    def __neg__(self):
        neg_value = (-self.value) % Q
        return ShareFloat(value=neg_value, p=self.p, tcp=self.tcp)

    def __sub__(self, other):
        if isinstance(other, ShareFloat):
            return sec_add(self, -other)
        elif isinstance(other, int):
            if self.p == 0:
                y_share = ShareFloat(value=-other, p=self.p, tcp=self.tcp)
            else:
                y_share = ShareFloat(value=0, p=self.p, tcp=self.tcp)
            return sec_add(self, y_share)

    def __mul__(self, other):
        if isinstance(other, ShareFloat):
            return sec_mul_float(self, other)
        elif isinstance(other, int):
            res = ShareFloat(value=(self.value * other), p=self.p, tcp=self.tcp)
            return res
        else:
            res = ShareFloat(value=(self.value * other), p=self.p, tcp=self.tcp)
            return res

    def __gt__(self, other):
        le = sec_less_eq_MSB(self, other)
        if le.p == 0:
            w = -le.value + 1
        else:
            w = -le.value
        return ShareFloat(w, self.p, self.tcp)

    def __lt__(self, other):
        ge = sec_great_eq_MSB(self, other)
        if ge.p == 0:
            w = -ge.value + 1
        else:
            w = -ge.value
        return ShareFloat(value=w, p=self.p, tcp=self.tcp)

    def __ge__(self, other):
        if isinstance(other, ShareFloat):
            return sec_great_eq_MSB(self, other)

    def __le__(self, other):
        if isinstance(other, ShareFloat):
            return sec_less_eq_MSB(self, other)

    def __getitem__(self, item):
        r = ShareFloat(value=self.value[item], p=self.p, tcp=self.tcp)
        return r

    def __setitem__(self, key, value):
        self.value[key] = value.value

    def __matmul__(self, other):
        if isinstance(other, ShareFloat):
            return sec_mat_mul(self, other)


def encode(t):
    t = (t * scaled)
    t = torch.floor(t).long() % Q
    return t


def decode(t):
    t = torch.where(t > Q / 2, t - Q, t)
    t = t / scaled

    return t


def share_float(t):
    t_0 = torch.randint(0, int(Ring / 2), t.shape, dtype=torch.int64)
    # t_0 = torch.randint(0, Q, t.shape)
    t_1 = (t - t_0) % Q

    return t_0, t_1


def restore_float(t):
    t.tcp.send_torch_array(t.value)
    other_share = t.tcp.receive_torch_array()
    res = (other_share + t.value) % Q

    return res


def restore_tensor(t):
    t.tcp.send_torch_array(t.value)
    other_share = t.tcp.receive_torch_array()
    res = (other_share + t.value) % Q
    res = torch.where(res > Q / 2, res - Q, res)

    return res


def int2bite_arr(z, size):
    arr = torch.zeros(size=(size, 32)).bool()
    for i in range(0, 32):
        arr[:, i] = ((z.value >> i) & 0x01).reshape(1, size)

    return arr


def img2col(img, k_size, stride=1):
    N, C, H, W = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    out_h = (H - k_size) // stride + 1
    out_w = (W - k_size) // stride + 1
    kw = kh = k_size
    out_size = out_w * out_h
    col = torch.zeros(size=(N, C, kw * kh, out_size), dtype=torch.int64)
    for y in range(out_h):
        y_start = y * stride
        y_end = y_start + kh
        for x in range(out_w):
            x_start = x * stride
            x_end = x_start + kw
            col[:, :, 0:, y * out_w + x] = img[:, :, y_start:y_end, x_start:x_end].reshape(N, C, kh * kw)
    return col.reshape((N, -1, out_size))


def truncate(res):
    if res.p == 0:
        # apply mask known only by P0, and reconstruct masked b to P1 or P2
        mask = (torch.ones(res.value.shape, dtype=torch.int64) * random.randrange(2**31)).long()
        mask_low = (mask % scaled)

        res.tcp.send_torch_array((res.value + mask) % Q)
        b_masked_low = res.tcp.receive_torch_array()

        b_low = (b_masked_low - mask_low) % Q
        res.value = (res.value - b_low) % Q
        z = (res.value * INVERSE) % Q

        return ShareFloat(z, res.p, res.tcp)

    if res.p == 1:
        mask_low = res.tcp.receive_torch_array()
        b_masked = (mask_low + res.value) % Q
        b_masked_low = b_masked % scaled
        res.tcp.send_torch_array(b_masked_low)

        z = (res.value * INVERSE) % Q

        return ShareFloat(z, res.p, res.tcp)


def sec_add(x, y) -> ShareFloat:
    return ShareFloat(value=(x.value + y.value) % Q, p=x.p, tcp=x.tcp)


def sec_mul_float(x, y):
    global ptr
    p = x.p

    (a, b, c) = tf.get_triples(p, ptr)

    a = ShareFloat(value=a, p=p, tcp=x.tcp)
    b = ShareFloat(value=b, p=p, tcp=x.tcp)
    c = ShareFloat(value=c, p=p, tcp=x.tcp)

    e_h = ShareFloat(value=(x.value - a.value) % Q, p=p, tcp=x.tcp)
    f_h = ShareFloat(value=(y.value - b.value) % Q, p=p, tcp=x.tcp)

    e = restore_float(e_h)
    f = restore_float(f_h)

    res = (p * e * f + e * b.value + a.value * f + c.value) % Q
    res = ShareFloat(res, p, x.tcp)

    res = truncate(res)
    # res.value = res.value * BASE ** (-1 * LEN_DECIMAL)

    return res


def sec_mat_mul(x: ShareFloat, y: ShareFloat):
    global ptr
    p = x.p

    (a, b, c) = tf.get_triples(p, ptr)
    tem_value = x.value @ y.value
    m_0 = x.value.shape[-1]
    c = (c * m_0) % Q

    a_v = torch.ones(x.value.shape, dtype=x.value.dtype) * a
    b_v = torch.ones(y.value.shape, dtype=x.value.dtype) * b
    c_v = torch.ones(tem_value.shape, dtype=x.value.dtype) * c

    e_h = ShareFloat(value=x.value - a, p=p, tcp=x.tcp)
    f_h = ShareFloat(value=y.value - b, p=p, tcp=x.tcp)

    e = restore_float(e_h)
    f = restore_float(f_h)

    res1 = torch.matmul(e, f) % Q
    res2 = torch.matmul(a_v, f) % Q
    res3 = torch.matmul(e, b_v) % Q

    res = (x.p * res1 + res2 + res3 + c_v) % Q
    res = ShareFloat(value=res, p=x.p, tcp=x.tcp)

    res = truncate(res)
    # print(decode(restore_float(res)))
    return res


def B2A(arr, p, tcp):
    k, h = shape = arr.shape

    a0_para = torch.zeros((k, h), dtype=torch.int32)
    a1_para = torch.ones((k, h), dtype=torch.int32)
    b0_para = torch.zeros((k, h), dtype=torch.int32)
    b1_para = torch.ones((k, h), dtype=torch.int32)
    c0_para = torch.zeros((k, h), dtype=torch.int32)
    c1_para = torch.ones((k, h), dtype=torch.int32)

    if p == 1:
        arr_a = arr.long()
        arr_b = torch.zeros(shape, dtype=torch.int32)
        e_i = arr_a - a0_para
        f_i = arr_b - b0_para
    else:
        arr_a = torch.zeros(shape, dtype=torch.int32)
        arr_b = arr.long()
        e_i = arr_a - a1_para
        f_i = arr_b - b1_para

    # get a*b
    e = restore_tensor(ShareFloat(e_i, p, tcp))
    f = restore_tensor(ShareFloat(f_i, p, tcp))

    if p == 1:
        ab_i = ((e * b0_para) + (f * a0_para) + c0_para) % Q
    else:
        ab_i = ((e * f) + (e * b1_para) + (f * a1_para) + c1_para) % Q

    res = (arr_a + arr_b - 2 * ab_i) % Q

    return res


def sec_less_eq_MSB(x: ShareFloat, y: ShareFloat) -> ShareFloat:

    return sec_great_eq_MSB(y, x)


def sec_great_eq_MSB(x: ShareFloat, y: ShareFloat) -> ShareFloat:
    z = x - y

    size = z.value.numel()

    z.value = z.value.reshape(1, size)
    z.value = int2bite_arr(z, size)

    cb = utils_vector.get_carry_bit_sonic(z, size)
    cb = cb.reshape(1, cb.numel())

    res = B2A(cb, x.p, x.tcp)
    res = res.reshape(x.value.shape)
    res = ShareFloat(res, x.p, x.tcp)

    return res


def debug(x):
    if isinstance(x, ShareFloat):
        res = decode(restore_float(x))
        print(res)


def debug_restore(x):
    if isinstance(x, ShareFloat):
        res = restore_float(x)
        print(res)
