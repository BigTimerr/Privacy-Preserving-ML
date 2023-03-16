"""
# @Time : 2022/10/12 16:32
# @Author : ruetrash
# @File : secret_sharing_fixpoint.py
"""
import random
import torch
import ProtocolOnRing.param as param
import ProtocolOnRing.triples as tf
from MSB import utils_vector


BASE = param.BASE
Ring = param.Ring
scaled = param.scaled
LEN_INTEGER = param.LEN_INTEGER
LEN_DECIMAL = param.LEN_DECIMAL
INVERSE = param.INVERSE
KAPPA = param.KAPPA
PRECISION = LEN_INTEGER + LEN_DECIMAL

assert (Ring > BASE ** LEN_INTEGER + LEN_DECIMAL)
assert ((INVERSE * scaled) % Ring == 1)
assert (Ring > BASE ** (2 * (LEN_INTEGER + LEN_DECIMAL) + KAPPA))

global ptr
ptr = 0


class ShareFloat(object):
    def __init__(self, value, p, tcp, device):
        self.value = value
        self.p = p
        self.tcp = tcp
        self.device = device
        if p == 0:
            self.party = 'server'
        else:
            self.party = 'client'

        if self.device == "cuda":
            self.value = self.value.to(device)

    def __str__(self):
        return "[{} value:{},\n party:{}]".format(self.__class__.__name__, self.value, self.party)

    def __add__(self, other):
        if isinstance(other, ShareFloat):
            return sec_add(self, other)
        elif isinstance(other, int):
            if self.p == 0:
                y_share = ShareFloat(value=other, p=self.p, tcp=self.tcp, device=self.device)
            else:
                y_share = ShareFloat(value=0, p=self.p, tcp=self.tcp, device=self.device)
            return sec_add(self, y_share)

    def __neg__(self):
        neg_value = (-self.value) % Ring
        return ShareFloat(value=neg_value, p=self.p, tcp=self.tcp, device=self.device)

    def __sub__(self, other):
        if isinstance(other, ShareFloat):
            return sec_add(self, -other)
        elif isinstance(other, int):
            if self.p == 0:
                y_share = ShareFloat(value=-other, p=self.p, tcp=self.tcp, device=self.device)
            else:
                y_share = ShareFloat(value=0, p=self.p, tcp=self.tcp, device=self.device)
            return sec_add(self, y_share)

    def __mul__(self, other):
        if isinstance(other, ShareFloat):
            return sec_mul_float(self, other)
        elif isinstance(other, int):
            res = ShareFloat(value=(self.value * other), p=self.p, tcp=self.tcp, device=self.device)
            return res
        else:
            res = ShareFloat(value=(self.value * other), p=self.p, tcp=self.tcp, device=self.device)
            return res

    def __gt__(self, other):
        le = sec_less_eq_MSB(self, other)
        if le.p == 0:
            w = -le.value + 1
        else:
            w = -le.value
        return ShareFloat(w, self.p, self.tcp, device=self.device)

    def __lt__(self, other):
        ge = sec_great_eq_MSB(self, other)
        if ge.p == 0:
            w = -ge.value + 1
        else:
            w = -ge.value
        return ShareFloat(value=w, p=self.p, tcp=self.tcp, device=self.device)

    def __ge__(self, other):
        if isinstance(other, ShareFloat):
            return sec_great_eq_MSB(self, other)

    def __le__(self, other):
        if isinstance(other, ShareFloat):
            return sec_less_eq_MSB(self, other)

    def __getitem__(self, item):
        r = ShareFloat(value=self.value[item], p=self.p, tcp=self.tcp, device=self.device)
        return r

    def __setitem__(self, key, value):
        self.value[key] = value.value

    def __matmul__(self, other):
        if isinstance(other, ShareFloat):
            return sec_mat_mul(self, other)


def encode(t):
    t = (t * scaled)
    t = torch.floor(t).long() % Ring
    return t


def decode(t):
    t = torch.where(t > Ring / 2, t - Ring, t)
    t = t / scaled

    return t


def share_float(t):
    t_0 = torch.randint(0, int(Ring / 2), t.shape, dtype=torch.int64, device=t.device)
    t_1 = (t - t_0) % Ring

    return t_0, t_1


def restore_float(t):
    t.tcp.send_torch_array(t.value)
    other_share = t.tcp.receive_torch_array(t.device)
    other_share = other_share.to(t.device)
    res = (other_share + t.value) % Ring

    return res


def restore_tensor(t):
    t.tcp.send_torch_array(t.value)
    other_share = t.tcp.receive_torch_array(t.device)
    res = (other_share + t.value) % Ring
    res = torch.where(res > Ring / 2, res - Ring, res)

    return res


def int2bite_arr(z, size):
    arr = torch.zeros(size=(size, 32), device=z.device).bool()
    for i in range(0, 32):
        arr[:, i] = ((z.value >> i) & 0x01).reshape(1, size)

    return arr


def img2col(img, k_size, stride, device):
    N, C, H, W = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    out_h = (H - k_size) // stride + 1
    out_w = (W - k_size) // stride + 1
    kw = kh = k_size
    out_size = out_w * out_h
    col = torch.zeros(size=(N, C, kw * kh, out_size), dtype=torch.int64, device=device)
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
        mask = mask.to(res.device)
        mask_low = (mask % scaled)

        res.tcp.send_torch_array((res.value + mask) % Ring)
        b_masked_low = res.tcp.receive_torch_array(res.device)

        b_low = (b_masked_low - mask_low) % Ring
        res.value = (res.value - b_low) % Ring
        z = (res.value * INVERSE) % Ring

        return ShareFloat(z, res.p, res.tcp, res.device)

    if res.p == 1:
        mask_low = res.tcp.receive_torch_array(res.device)

        b_masked = (mask_low + res.value) % Ring
        b_masked_low = b_masked % scaled
        res.tcp.send_torch_array(b_masked_low)

        z = (res.value * INVERSE) % Ring

        return ShareFloat(z, res.p, res.tcp, res.device)


def sec_add(x, y) -> ShareFloat:
    return ShareFloat(value=(x.value + y.value) % Ring, p=x.p, tcp=x.tcp, device=x.device)


def sec_mul_float(x, y):
    global ptr
    p = x.p

    (a, b, c) = tf.get_triples(p, ptr)

    a = torch.ones(x.value.shape, dtype=torch.int64, device=x.device) * a
    b = torch.ones(x.value.shape, dtype=torch.int64, device=x.device) * b
    c = torch.ones(x.value.shape, dtype=torch.int64, device=x.device) * c

    a = ShareFloat(value=a, p=p, tcp=x.tcp, device=x.device)
    b = ShareFloat(value=b, p=p, tcp=x.tcp, device=x.device)
    c = ShareFloat(value=c, p=p, tcp=x.tcp, device=x.device)

    e_h = ShareFloat(value=(x.value - a.value) % Ring, p=p, tcp=x.tcp, device=x.device)
    f_h = ShareFloat(value=(y.value - b.value) % Ring, p=p, tcp=x.tcp, device=x.device)

    e = restore_float(e_h)
    f = restore_float(f_h)

    res = (p * e * f + e * b.value + a.value * f + c.value) % Ring
    res = ShareFloat(res, p, x.tcp, x.device)

    res = truncate(res)

    return res


def sec_mat_mul(x: ShareFloat, y: ShareFloat):
    global ptr
    p = x.p

    (a, b, c) = tf.get_triples(p, ptr)
    # tem_value = x.value @ y.value
    tem_value = torch.ones(x.value.shape) @ torch.ones(y.value.shape)
    m_0 = x.value.shape[-1]
    c = (c * m_0) % Ring

    a_v = torch.ones(x.value.shape, dtype=x.value.dtype, device=x.device) * a
    b_v = torch.ones(y.value.shape, dtype=x.value.dtype, device=x.device) * b
    c_v = torch.ones(tem_value.shape, dtype=x.value.dtype, device=x.device) * c

    e_h = ShareFloat(value=x.value - a_v, p=p, tcp=x.tcp, device=x.device)
    f_h = ShareFloat(value=y.value - b_v, p=p, tcp=x.tcp, device=x.device)

    e = restore_float(e_h)
    f = restore_float(f_h)

    if x.device == "cuda":

        tag = 2 ** 24

        e_high = torch.floor(e / tag).to(torch.float64)
        e_low = (e - e_high * tag).to(torch.float64)

        f_high = torch.floor(f / tag).to(torch.float64)
        f_low = (f - f_high * tag).to(torch.float64)

        a_v_high = torch.floor(a_v / tag).to(torch.float64)
        a_v_low = (a_v - a_v_high * tag).to(torch.float64)

        b_v_high = torch.floor(b_v / tag).to(torch.float64)
        b_v_low = (b_v - b_v_high * tag).to(torch.float64)

        res1 = (torch.matmul(e_high, f_high) * tag * tag % Ring +
                torch.matmul(e_high, f_low) * tag % Ring +
                torch.matmul(e_low, f_high) * tag % Ring +
                torch.matmul(e_low, f_low)) % Ring

        res2 = (torch.matmul(a_v_high, f_high) * tag * tag % Ring +
                torch.matmul(a_v_high, f_low) * tag % Ring +
                torch.matmul(a_v_low, f_high) * tag % Ring +
                torch.matmul(a_v_low, f_low)) % Ring

        res3 = (torch.matmul(e_high, b_v_high) * tag * tag % Ring +
                torch.matmul(e_high, b_v_low) * tag % Ring +
                torch.matmul(e_low, b_v_high) * tag % Ring +
                torch.matmul(e_low, b_v_low)) % Ring

        res1 = res1.to(torch.int64)
        res2 = res2.to(torch.int64)
        res3 = res3.to(torch.int64)

    else:
        res1 = torch.matmul(e, f) % Ring
        res2 = torch.matmul(a_v, f) % Ring
        res3 = torch.matmul(e, b_v) % Ring

    res = (x.p * res1 + res2 + res3 + c_v) % Ring
    res = ShareFloat(value=res, p=x.p, tcp=x.tcp, device=x.device)

    res = truncate(res)
    return res


def B2A(arr, p, tcp, device):
    k, h = shape = arr.shape

    a0_para = torch.zeros((k, h), dtype=torch.int32, device=device)
    a1_para = torch.ones((k, h), dtype=torch.int32, device=device)
    b0_para = torch.zeros((k, h), dtype=torch.int32, device=device)
    b1_para = torch.ones((k, h), dtype=torch.int32, device=device)
    c0_para = torch.zeros((k, h), dtype=torch.int32, device=device)
    c1_para = torch.ones((k, h), dtype=torch.int32, device=device)

    if p == 1:
        arr_a = arr.long()
        arr_b = torch.zeros(shape, dtype=torch.int32, device=device)
        e_i = arr_a - a0_para
        f_i = arr_b - b0_para
    else:
        arr_a = torch.zeros(shape, dtype=torch.int32, device=device)
        arr_b = arr.long()
        e_i = arr_a - a1_para
        f_i = arr_b - b1_para

    # get a*b
    e = restore_tensor(ShareFloat(e_i, p, tcp, device))
    f = restore_tensor(ShareFloat(f_i, p, tcp, device))

    if p == 1:
        ab_i = ((e * b0_para) + (f * a0_para) + c0_para) % Ring
    else:
        ab_i = ((e * f) + (e * b1_para) + (f * a1_para) + c1_para) % Ring

    res = (arr_a + arr_b - 2 * ab_i) % Ring

    return res


def sec_less_eq_MSB(x: ShareFloat, y: ShareFloat) -> ShareFloat:

    return sec_great_eq_MSB(y, x)


def sec_great_eq_MSB(x: ShareFloat, y: ShareFloat) -> ShareFloat:
    z = x - y

    size = z.value.numel()

    z.value = z.value.reshape(1, size)
    z.value = int2bite_arr(z, size)

    carry_bit = utils_vector.get_carry_bit_sonic(z, size, z.device)
    carry_bit = carry_bit.reshape(1, carry_bit.numel())

    res = B2A(carry_bit, x.p, x.tcp, x.device)
    res = res.reshape(x.value.shape)
    res = ShareFloat(res, x.p, x.tcp, x.device)

    return res


def debug(x):
    if isinstance(x, ShareFloat):
        res = decode(restore_float(x))
        print(res)
        return res

def debug_no_print(x):
    if isinstance(x, ShareFloat):
        res = decode(restore_float(x))
        # print(res)
        return res


def debug_restore(x):
    if isinstance(x, ShareFloat):
        res = restore_float(x)
        print(res)
