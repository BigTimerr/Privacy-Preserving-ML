import random
import ProtocolOnRing.param as param
import ProtocolOnRing.triples as tp
from MSB import utils_vector as utils
import ProtocolOnRing.triples_queue_list as tqlist
import torch

Ring = param.Ring
n = param.n

global tp_ptr
tp_ptr = 0

global s_ptr
s_ptr = 0


class ShareV(object):
    def __init__(self, value, p, tcp):
        self.value = value
        self.p = p

        if p == 0:
            self.party = 'server'
        else:
            self.party = 'client'

        self.tcp = tcp

    def __str__(self):
        return "[{} value:{},\n party:{}]".format(self.__class__.__name__, self.value, self.party)

    def __add__(self, other):
        if isinstance(other, ShareV):
            return sec_add(self, other)
        elif isinstance(other, int):
            if self.p == 0:
                y_share = ShareV(value=other, p=self.p, tcp=self.tcp)
            else:
                y_share = ShareV(value=0, p=self.p, tcp=self.tcp)
            return sec_add(self, y_share)

    def __mul__(self, other):
        if isinstance(other, ShareV):
            return sec_mul(self, other)
        elif isinstance(other, int):
            res = ShareV(value=(self.value * other), p=self.p, tcp=self.tcp)
            return res
        else:
            res = ShareV(value=(self.value * other), p=self.p, tcp=self.tcp)
            return res

    def __neg__(self):
        neg_value = (-self.value) % Ring
        return ShareV(value=neg_value, p=self.p, tcp=self.tcp)

    def __sub__(self, other):
        if isinstance(other, ShareV):
            return sec_add(self, -other)
        elif isinstance(other, int):
            if self.p == 0:
                y_share = ShareV(value=-other, p=self.p, tcp=self.tcp)
            else:
                y_share = ShareV(value=0, p=self.p, tcp=self.tcp)
            return sec_add(self, y_share)

    def __gt__(self, other):
        le = sec_less_eq_MSB(self, other)
        if le.p == 0:
            w = -le.value + 1
        else:
            w = -le.value
        return ShareV(w, self.p, self.tcp)

    def __lt__(self, other):
        ge = sec_great_eq_MSB(self, other)
        if ge.p == 0:
            w = -ge.value + 1
        else:
            w = -ge.value
        return ShareV(value=w, p=self.p, tcp=self.tcp)

    def __ge__(self, other):
        if isinstance(other, ShareV):
            return sec_great_eq_MSB(self, other)

    def __le__(self, other):
        if isinstance(other, ShareV):
            return sec_less_eq_MSB(self, other)

    def __getitem__(self, item):
        r = ShareV(value=self.value[item], p=self.p, tcp=self.tcp)
        return r

    def __setitem__(self, key, value):
        self.value[key] = value.value

    def __matmul__(self, other):
        if isinstance(other, ShareV):
            return sec_mat_mul(self, other)



def share_tensor(t: torch.Tensor):
    t = t & 0xffffffff
    # 在进行MSB算法时，需要两个分享值的位数小于 30
    # 因为生成的数据是30位(环的大小是31位)的有符号数,这里直接生成31的无符号数,不用再进行转化了
    t_0 = torch.randint(0, 2 ** 30, t.shape, dtype=t.dtype)
    t_1 = (t - t_0) % Ring
    return t_0, t_1


def share_value(t):
    t = t & 0xffffffff
    # 因为生成的数据是30位(环的大小是31位)的有符号数,这里直接生成31的无符号数,不用再进行转化了
    t_0 = torch.ones(t.shape,dtype=t.dtype) * random.randint(0, 2 ** 30)
    t_1 = (t - t_0) % Ring
    return t_0, t_1


def int2bite_arr(z, size):
    arr = torch.zeros(size=(size, 32)).bool()
    for i in range(0, 32):
        arr[:, i] = ((z.value >> i) & 0x01).reshape(1, size)

    return arr


def img2col(img, ksize, stride=1):
    N, C, H, W = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    out_h = (H - ksize) // stride + 1
    out_w = (W - ksize) // stride + 1
    kw = kh = ksize
    out_size = out_w * out_h
    col = torch.empty(size=(N, C, kw * kh, out_size), dtype=torch.int64)
    for y in range(out_h):
        y_start = y * stride
        y_end = y_start + kh
        for x in range(out_w):
            x_start = x * stride
            x_end = x_start + kw
            col[:, :, 0:, y * out_w + x] = img[:, :, y_start:y_end, x_start:x_end].reshape(N, C, kh * kw)
    return col.reshape((N, -1, out_size))


def restore_tensor(t: ShareV, party=2):
    p = t.p
    if party == 0:
        if p == 0:
            other_share = t.tcp.receive_torch_array()
            res = (other_share + t.value) % Ring
            res = torch.where(res > Ring / 2, res - Ring, res)
            return res
        else:
            t.tcp.send_torch_array(t.value)

    elif party == 1:
        if p == 0:
            t.tcp.send_torch_array(t.value)
            return t.value
        else:
            other_share = t.tcp.receive_torch_array()
            res = (other_share + t.value) % Ring
            res = torch.where(res > Ring / 2, res - Ring, res)
            return res
    else:
        t.tcp.send_torch_array(t.value)
        other_share = t.tcp.receive_torch_array()
        res = (other_share + t.value) % Ring
        if isinstance(res, torch.Tensor):
            res = torch.where(res > Ring / 2, res - Ring, res)
        if isinstance(res, int):
            if res > Ring / 2:
                return res - Ring
        return res


def sec_add(x, y) -> ShareV:
    return ShareV(value=(x.value + y.value) % Ring, p=x.p, tcp=x.tcp)


def sec_mul(x, y) -> ShareV:
    global tp_ptr
    p = x.p
    (a, b, c) = tp.get_triples(p, tp_ptr)
    # a_v = torch.ones_like(x.value) * a
    # b_v = torch.ones_like(x.value) * b
    # c_v = torch.ones_like(x.value) * c

    a = ShareV(value=a, p=p, tcp=x.tcp)
    b = ShareV(value=b, p=p, tcp=x.tcp)
    c = ShareV(value=c, p=p, tcp=x.tcp)

    # e_h = x - a
    e_h = ShareV(value=x.value - a.value, p=p, tcp=x.tcp)
    # f_h = y - b
    f_h = ShareV(value=y.value - b.value, p=p, tcp=x.tcp)

    e = restore_tensor(e_h, party=2)
    f = restore_tensor(f_h, party=2)

    res = (p * e * f + e * b.value + a.value * f + c.value) % Ring
    res = ShareV(value=res, p=p, tcp=x.tcp)

    tp_ptr += 1

    return res


def B2A(arr, p, tcp):
    k, h = shape = arr.shape

    a0_para = torch.zeros((k, h), dtype=torch.int64)
    a1_para = torch.ones((k, h), dtype=torch.int64)
    b0_para = torch.zeros((k, h), dtype=torch.int64)
    b1_para = torch.ones((k, h), dtype=torch.int64)
    c0_para = torch.zeros((k, h), dtype=torch.int64)
    c1_para = torch.ones((k, h), dtype=torch.int64)

    if p == 1:
        arr_a = arr.long()
        arr_b = torch.zeros(shape, dtype=torch.int64)
        e_i = arr_a - a0_para
        f_i = arr_b - b0_para
    else:
        arr_a = torch.zeros(shape, dtype=torch.int64)
        arr_b = arr.long()
        e_i = arr_a - a1_para
        f_i = arr_b - b1_para

    # get a*b
    e = restore_tensor(ShareV(e_i, p, tcp), party=2)
    f = restore_tensor(ShareV(f_i, p, tcp), party=2)

    if p == 1:
        ab_i = ((e * b0_para) + (f * a0_para) + c0_para) % Ring
    else:
        ab_i = ((e * f) + (e * b1_para) + (f * a1_para) + c1_para) % Ring

    res = (arr_a + arr_b - 2 * ab_i) % Ring

    return res


def sec_less_eq_MSB(x: ShareV, y: ShareV) -> ShareV:
    z = y - x

    size = z.value.numel()

    z.value = z.value.reshape(1, size)
    z.value = int2bite_arr(z, size)

    cb = utils.get_carry_bit_sonic(z, size)
    cb = cb.reshape(1, cb.numel())

    res = B2A(cb, x.p, x.tcp)

    res = res.reshape(x.value.shape)
    res = ShareV(res, x.p, x.tcp)

    return res


def sec_great_eq_MSB(x: ShareV, y: ShareV) -> ShareV:
    z = x - y

    size = z.value.numel()

    z.value = z.value.reshape(1, size)
    z.value = int2bite_arr(z, size)

    cb = utils.get_carry_bit_sonic(z, size)
    cb = cb.reshape(1,cb.numel())

    res = B2A(cb, x.p, x.tcp)
    res = res.reshape(x.value.shape)
    res = ShareV(res, x.p, x.tcp)

    return res


def sec_mat_mul(x: ShareV, y: ShareV):
    global tp_ptr
    p = x.p

    (a, b, c) = tp.get_triples(p, tp_ptr)
    tem_value = x.value @ y.value
    m_0 = x.value.shape[-1]
    c = c * m_0

    a_v = torch.ones(x.value.shape, dtype=x.value.dtype) * a
    b_v = torch.ones(y.value.shape, dtype=x.value.dtype) * b
    c_v = torch.ones(tem_value.shape, dtype=x.value.dtype) * c

    e_h = ShareV(value=x.value - a, p=p, tcp=x.tcp)
    f_h = ShareV(value=y.value - b, p=p, tcp=x.tcp)

    e = restore_tensor(e_h, party=2)
    f = restore_tensor(f_h, party=2)

    res1 = torch.matmul(e, f)
    res2 = torch.matmul(a_v, f)
    res3 = torch.matmul(e, b_v)

    res = (x.p * res1 + res2 + res3 + c_v) % Ring
    res = ShareV(value=res, p=x.p, tcp=x.tcp)

    return res


def debug(x):
    if isinstance(x, ShareV):
        res = restore_tensor(x, party=2)
        print(res)
