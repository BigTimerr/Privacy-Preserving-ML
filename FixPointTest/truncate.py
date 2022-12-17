import random
import time
#
RING = 2 ** 32
scaled = 1009
INVERSE = 2668924177

# RING = 2 ** 32
# scaled = 509
# INVERSE = 58197

def encode(num):
    upscaled = int(num *scaled)
    ring_element = upscaled % RING
    return ring_element

def decode(ring_element):
    ring_element = ring_element if ring_element <= RING/2 else ring_element - RING
    num = ring_element / scaled
    return num


def share(plaintext):
    x_0 = random.randrange(RING)
    x_1 = (plaintext - x_0) % RING
    return [x_0, x_1]


def reconstruct(sharing):
    return sum(sharing) % RING


def generate_triples():
    a = random.randrange(RING)
    b = random.randrange(RING)
    c = (a * b) % RING

    a_0, a_1 = share(a)
    b_0, b_1 = share(b)
    c_0, c_1 = share(c)
    return a_0, b_0, c_0, a_1, b_1, c_1


def mul(x, y):
    x_0, x_1, y_0, y_1 = x[0], x[1], y[0], y[1]
    a_0, b_0, c_0, a_1, b_1, c_1 = generate_triples()

    e_0 = (x_0 - a_0) % RING
    e_1 = (x_1 - a_1) % RING

    f_0 = (y_0 - b_0) % RING
    f_1 = (y_1 - b_1) % RING

    e = (e_0 + e_1) % RING
    f = (f_0 + f_1) % RING


    res_0 = (e * b_0 + a_0 * f + c_0) % RING
    res_1 = (e * f + e * b_1 + a_1 * f + c_1) % RING

    w = [res_0, res_1]

    v = truncate(w)

    return v


def add(x, y):
    return [ (xi + yi) % RING for xi, yi in zip(x, y) ]

def sub(x, y):
    return [ (xi - yi) % RING for xi, yi in zip(x, y) ]

def imul(x, k):
    return [(xi * k) % RING for xi in x]


def truncate(a):
    # map to positive range
    #b = add(a, share(BASE**(2*PRECISION + 1)))
    # apply mask known only by P0, and reconstruct masked b to P1 or P2
    mask = random.randrange(RING)
    mask_low = mask % scaled
    a_masked = add(a, share(mask))
    a_masked_recons = reconstruct(a_masked)
    #b_masked = reconstruct(add(b, share(mask)))
    # extract lower digits
    b_masked_low = a_masked_recons % scaled
    b_low = sub(share(b_masked_low), share(mask_low))
    # remove lower digits
    c = sub(a, b_low)
    # division
    d = imul(c, INVERSE)
    return d


x = 0.919
y = 0.8202

encode_x = encode(x)
encode_y = encode(y)
shared_x = share(encode_x)
shared_y = share(encode_y)

z = mul(shared_x, shared_y)

print(decode(reconstruct(z)))
