# """
# # @Time : 2022/10/11 18:47
# # @Author : ruetrash
# # @File : inverse.py
# """

import gmpy2


def EX_GCD(a, b, arr):  # 扩展欧几里得
    if b == 0:
        arr[0] = 1
        arr[1] = 0
        return a
    g = EX_GCD(b, a % b, arr)
    t = arr[0]
    arr[0] = arr[1]
    arr[1] = t - int(a / b) * arr[1]
    return g


def ModReverse(a, n):  # ax=1(mod n) 求a模n的乘法逆x
    arr = [0, 1, ]
    gcd = EX_GCD(a, n, arr)
    if gcd == 1:
        return (arr[0] % n + n) % n
    else:
        return -1


a = 1048573
b = 2 ** 64
arr = [0, 1, ]
inverse = ModReverse(a, b)
print(a, '模', b, '的乘法逆：', inverse)
print(a, '和', b, '的最大公约数：', EX_GCD(a, b, arr))

print(f"验证 ({inverse} * {a}) % {b} =", (inverse * a) %b)



# for i in range(65535, 65550):
#     print(f"{i}是不是素数", gmpy2.is_prime(i))

