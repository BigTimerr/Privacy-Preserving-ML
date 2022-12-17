"""
# @Time : 2022/10/9 15:20
# @Author : ruetrash
# @File : simple_secmul_for_fixpoint_3p.py
"""

import random
import numpy as np

#
# Use e.g. https://www.compilejava.net/
#
#import java.util.*;
#import java.math.*;
#
#public class Entrypoint
#{
#  public static void main(String[] args)
#  {
#    BigInteger q = BigInteger.probablePrime(128, new Random());
#    BigInteger inverse = new BigInteger("10000000000").modInverse(q);
#    System.out.println(q);
#    System.out.println(inverse);
#  }
#}

BASE = 10

#PRECISION_INTEGRAL = 1
#PRECISION_FRACTIONAL = 6
#Q = 10000019

PRECISION_INTEGRAL = 8
PRECISION_FRACTIONAL = 8
Q = 293973345475167247070445277780365744413

PRECISION = PRECISION_INTEGRAL + PRECISION_FRACTIONAL

assert(Q > BASE**PRECISION)

def encode(rational):
    upscaled = int(rational * BASE**PRECISION_FRACTIONAL)
    field_element = upscaled % Q
    return field_element

def decode(field_element):
    upscaled = field_element if field_element <= Q/2 else field_element - Q
    rational = upscaled / BASE**PRECISION_FRACTIONAL
    return rational

def share(secret):
    first  = random.randrange(Q)
    second = random.randrange(Q)
    third  = (secret - first - second) % Q
    return [first, second, third]

def reconstruct(sharing):
    return sum(sharing) % Q

def reshare(x):
    Y = [ share(x[0]), share(x[1]), share(x[2]) ]
    return [ sum(row) % Q for row in zip(*Y) ]


def add(x, y):
    return [(xi + yi) % Q for xi, yi in zip(x, y)]


def sub(x, y):
    return [(xi - yi) % Q for xi, yi in zip(x, y)]


def imul(x, k):
    return [(xi * k) % Q for xi in x]

INVERSE = 104491423396290281423421247963055991507 # inverse of BASE**FRACTIONAL_PRECISION
KAPPA = 6 # leave room for five digits overflow before leakage

assert((INVERSE * BASE**PRECISION_FRACTIONAL) % Q == 1)
assert(Q > BASE**(2*PRECISION + KAPPA))

def truncate(a):
    # map to the positive range
    b = add(a, [BASE**(2*PRECISION+1), 0, 0])
    # apply mask known only by P0, and reconstruct masked b to P1 or P2
    mask = random.randrange(Q) % BASE**(PRECISION + PRECISION_FRACTIONAL + KAPPA)
    mask_low = mask % BASE**PRECISION_FRACTIONAL
    b_masked = reconstruct(add(b, [mask, 0, 0]))
    # extract lower digits
    b_masked_low = b_masked % BASE**PRECISION_FRACTIONAL
    b_low = sub(share(b_masked_low), share(mask_low))
    # remove lower digits
    c = sub(a, b_low)
    # remove extra scaling factor
    d = imul(c, INVERSE)
    return d

def mul(x, y):
    # local computation
    z0 = (x[0]*y[0] + x[0]*y[1] + x[1]*y[0]) % Q
    z1 = (x[1]*y[1] + x[1]*y[2] + x[2]*y[1]) % Q
    z2 = (x[2]*y[2] + x[2]*y[0] + x[0]*y[2]) % Q
    # reshare and distribute
    Z = [ share(z0), share(z1), share(z2) ]
    w = [ sum(row) % Q for row in zip(*Z) ]
    # bring precision back down from double to single
    v = truncate(w)
    return v


class SecureRational(object):

    def secure(secret):
        z = SecureRational()
        z.shares = share(encode(secret))
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

if __name__ == '__main__':

    x = .126
    y = -.115

    X = SecureRational.secure(x)
    Y = SecureRational.secure(y)

    z = X * Y
    print("z", z)
    print(z.reveal())
    print("x*y",x*y)
    # assert (z.reveal() == (x) * (y))

