import TCP.tcp as tcp
import ProtocolOnRing.secret_sharing_vector_onring as ssv
from ProtocolOnRing.secret_sharing_vector_onring import ShareV
import ProtocolOnRing.param as param

# 初始化参数
server = tcp.TCPServer("127.0.0.1", 9999, 4096)
server.run()

p = 0  # party 0:server
Ring = param.Ring
device = param.device

def showData(x: ShareV, y: ShareV, t: ShareV):
    print("/******************************************************/")
    print("x 的分享份额:", x.value)
    print("y 的分享份额:", y.value)
    print("t 的分享份额:", t.value)
    print("/******************************************************/")
    print()


def sec_ge(x: ShareV, y: ShareV):
    z = x >= y
    res = ssv.restore_tensor(z, party=2)
    # print()
    # print("/******************************************************/")
    # print("计算 x >= y 其结果为:", res)
    # print("/******************************************************/")
    # print()


def sec_le(x: ShareV, y: ShareV):
    z = x <= y
    res = ssv.restore_tensor(z, party=2)
    # print()
    # print("/******************************************************/")
    # print("计算 x <= y 其结果为:", res)
    # print("/******************************************************/")
    # print()


def sec_gt(x: ShareV, y: ShareV):
    z = x > y
    res = ssv.restore_tensor(z, party=2)
    # print()
    # print("/******************************************************/")
    # print("计算 x >= y 其结果为:", res)
    # print("/******************************************************/")
    # print()


def sec_lt(x: ShareV, y: ShareV):
    z = x < y
    res = ssv.restore_tensor(z, party=2)
    # print()
    # print("/******************************************************/")
    # print("计算 x > y 其结果为:", res)
    # print("/******************************************************/")
    # print()


def secADD(x: ShareV, y: ShareV):
    # ****************************加法*****************************
    z = x + y
    res = ssv.restore_tensor(z, party=2)
    # print("/******************************************************/")
    # print("计算 x+y 其结果为:", res)
    # print("/******************************************************/")
    # print()


def secIntADD(x: ShareV, t):
    # ****************************加法*****************************
    z = x + t
    res = ssv.restore_tensor(z, party=2)
    # print("/******************************************************/")
    # print("计算 x+t(常数) 其结果为:", res)
    # print("/******************************************************/")
    # print()


def secDec(x: ShareV, y: ShareV):
    # ****************************减法*****************************
    z = x - y
    res = ssv.restore_tensor(z, party=2)
    # print("/******************************************************/")
    # print("计算 x-y 其结果为:", res)
    # print("/******************************************************/")
    # print()


def secIntDec(x: ShareV, t):
    # ****************************int减法*****************************
    z = x - t
    res = ssv.restore_tensor(z, party=2)
    # print("/******************************************************/")
    # print("计算 x-t(常数) 其结果为:", res)
    # print("/******************************************************/")
    # print()


def secMul(x: ShareV, y: ShareV):
    # ****************************乘法*****************************
    z = x * y
    res = ssv.restore_tensor(z, party=2)
    print("/******************************************************/")
    print("计算 x*y 其结果为:", res)
    print("/******************************************************/")
    print()


def secFooldMul(x: ShareV, t):
    # ****************************泛洪乘法*****************************
    #
    z = x * t
    res = ssv.restore_tensor(z, party=2)
    # print("/******************************************************/")
    # print("计算 x*t(常数) 其结果为:", res)
    # print("/******************************************************/")
    # print()
    #
    # ************************************************************


def sec_mat_mul(x: ShareV, y: ShareV):
    # ****************************矩阵点乘********************************
    z = x @ y
    res = ssv.restore_tensor(z, party=2)
    print("/******************************************************/")
    print("计算 矩阵乘法 其结果为:", res, sep=",")
    print("/******************************************************/")
    print()


if __name__ == '__main__':
    # 首先接受所有的参数
    x_0 = server.receive_torch_array(device)
    y_0 = server.receive_torch_array(device)
    t_0 = server.receive_torch_array(device)

    # 得到自身的分享份额
    x = ShareV(value=x_0, p=p, tcp=server)
    y = ShareV(value=y_0, p=p, tcp=server)
    t = ShareV(value=t_0, p=p, tcp=server)

    # showData(x, y, t)
    secADD(x, y)
    secIntADD(x, t)
    secDec(x, y)
    secIntDec(x, t)
    secMul(x, y)
    secFooldMul(x, t)
    sec_mat_mul(x, y)
    sec_ge(x, y)
    sec_le(x, y)
    sec_gt(x, y)
    sec_lt(x, y)


    server.close()
