"""
    url：https://mortendahl.github.io/2017/04/17/private-deep-learning-with-mpc/

"""


import random
import time

import gmpy2

BASE = 2
# PRECISION_INTEGRAL = 3
# PRECISION_FRACTIONAL = 3
# Q = 65537
# INVERSE = 57345
#
# # Q = 65535
# # INVERSE = 8192  # inverse of BASE**FRACTIONAL_PRECISION

Q = 4294967311
Q_half = 2147483655.5
PRECISION_INTEGRAL = 6
PRECISION_FRACTIONAL = 6
INVERSE = 1140850692
KAPPA = 4


PRECISION = PRECISION_INTEGRAL + PRECISION_FRACTIONAL
assert (Q > BASE ** PRECISION)

KAPPA = 1  # leave room for five digits overflow before leakage

import numpy as np

random.seed(888)

def encode(rational):
    upscaled = int(rational * BASE ** PRECISION_FRACTIONAL)
    field_element = upscaled % Q
    print("field_element", field_element)
    return field_element


def decode(field_element):
    upscaled = field_element if field_element <= Q / 2 else field_element - Q
    rational = upscaled / BASE ** PRECISION_FRACTIONAL
    return rational


def share(secret):
    x_0 = random.randrange(Q)
    x_1 = (secret - x_0) % Q
    return [x_0, x_1]

def share_t(secret, num):
    # x_0 = random.randrange(Q)
    x_0 = num
    x_1 = (secret - x_0) % Q
    return [x_0, x_1]

def generate_triples():
    a = random.randrange(Q)
    b = random.randrange(Q)
    c = (a * b) % Q

    a_0, a_1 = share(a)
    b_0, b_1 = share(b)
    c_0, c_1 = share(c)
    return a_0, b_0, c_0, a_1, b_1, c_1


def mul(x, y):
    x_0, x_1, y_0, y_1 = x[0], x[1], y[0], y[1]
    a_0, b_0, c_0, a_1, b_1, c_1 = 1789889384,4040808671,732328179,1893908210,3398975060,1209556153

    e_0 = (x_0 - a_0) % Q
    e_1 = (x_1 - a_1) % Q

    f_0 = (y_0 - b_0) % Q
    f_1 = (y_1 - b_1) % Q

    e = (e_0 + e_1) % Q
    f = (f_0 + f_1) % Q


    res_0 = (e * b_0 + a_0 * f + c_0) % Q
    res_1 = (e * f + e * b_1 + a_1 * f + c_1) % Q
    print(res_0)
    print(res_1)

    Z = [share(res_0), share(res_1)]
    w = [sum(row) % Q for row in zip(*Z)]

    # print("w", w)
    v = truncate(w)

    return v


def reconstruct(sharing):
    return sum(sharing) % Q


def reshare(x):
    Y = [share(x[0]), share(x[1])]
    return [sum(row) % Q for row in zip(*Y)]


def add(x, y):
    return [(xi + yi) % Q for xi, yi in zip(x, y)]


def sub(x, y):
    return [(xi - yi) % Q for xi, yi in zip(x, y)]


def imul(x, k):
    return [(xi * k) % Q for xi in x]


def truncate(a):
    assert ((INVERSE * BASE ** PRECISION_FRACTIONAL) % Q == 1)
    assert (Q > BASE ** (2 * PRECISION + KAPPA))

    # map to the positive range
    # Q =
    b = a

    # apply mask known only by P0, and reconstruct masked b to P1 or P2
    # mask = random.randrange(Q) % BASE ** (PRECISION + PRECISION_FRACTIONAL + KAPPA)
    mask = 797182
    mask_low = mask % BASE ** PRECISION_FRACTIONAL
    b_masked = reconstruct(add(b, [mask, 0]))

    # extract lower digits
    b_masked_low = b_masked % BASE ** PRECISION_FRACTIONAL
    b_low = sub(share(b_masked_low), share(mask_low))

    print("a", reconstruct(a))
    # remove lower digits
    c = sub(a, b_low)
    print("b_low", reconstruct(b_low))
    print("c", reconstruct(c))

    # # remove extra scaling factor
    d = imul(c, INVERSE)

    return d


class SecureRational(object):

    def secure(secret, num):
        z = SecureRational()
        z.shares = share_t(encode(secret),num)
        return z

    def reveal(self):
        return decode(reconstruct(reshare(self.shares)))

    def __repr__(self):
        return "SecureRational(%f)" % self.reveal()

    def __add__(x, y):
        z = SecureRational()
        z.shares = add(x.shares, y.shares)
        return z

    def __sub__(x, y):
        z = SecureRational()
        z.shares = sub(x.shares, y.shares)
        return z

    def __mul__(x, y):
        z = SecureRational()
        z.shares = mul(x.shares, y.shares)
        return z

    def __pow__(x, e):
        z = SecureRational.secure(1)
        for _ in range(e):
            z = z * x
        return z


# x = np.random.randint(-10, 10, 1) / 10
# y = np.random.randint(-10, 10, 1) / 10

x = -0.3
y = 0.2

print(x)
print(y)

X = SecureRational.secure(x, 1544301054)
Y = SecureRational.secure(y, 531728282)

print(X)
print(Y)
Z = X * Y


print("Z.reveal():", Z.reveal())
print("x * y = ", x * y)

print()
print()
print()

time.sleep(3)










